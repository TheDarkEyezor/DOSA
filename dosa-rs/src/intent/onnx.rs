//! ONNX-based intent classification using DistilBERT
//!
//! Uses a trained DistilBERT model for joint intent classification
//! and slot filling (named entity extraction).

use super::types::*;
use super::{IntentClassifier, IntentContext};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use ndarray::{Array1, ArrayView1, ArrayView2};
use ort::session::Session;
use ort::value::Tensor;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::Tokenizer;

/// Configuration loaded from the NLU model's config.json
#[derive(Debug, Deserialize)]
pub struct NluConfig {
    pub intent2id: HashMap<String, usize>,
    pub id2intent: HashMap<String, String>,
    pub slot2id: HashMap<String, usize>,
    pub id2slot: HashMap<String, String>,
    pub max_length: usize,
}

/// Extracted slot/entity from the input
#[derive(Debug, Clone)]
pub struct ExtractedSlot {
    pub label: String,  // e.g., "PER", "DATE", "TIME", "LOC", "ORG", "EVENT"
    pub value: String,  // The extracted text
    pub start: usize,   // Start position in original text
    pub end: usize,     // End position in original text
}

/// Result from the ONNX classifier including slots
#[derive(Debug, Clone)]
pub struct OnnxClassifyResult {
    pub intent: String,
    pub confidence: f32,
    pub slots: Vec<ExtractedSlot>,
    pub all_intents: Vec<(String, f32)>,
}

/// ONNX-based classifier for intent detection and slot filling
pub struct OnnxClassifier {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    config: NluConfig,
    max_length: usize,
}

impl OnnxClassifier {
    /// Create a new ONNX classifier from model files
    ///
    /// # Arguments
    /// * `model_path` - Path to the .onnx model file
    /// * `config_path` - Path to config.json with label mappings
    /// * `vocab_path` - Path to vocab.txt for the tokenizer
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        config_path: P,
        vocab_path: P,
    ) -> Result<Self> {
        // Load ONNX model using the file path directly
        let session = Session::builder()
            .map_err(|e| anyhow!("Failed to create session builder: {:?}", e))?
            .with_intra_threads(1)
            .map_err(|e| anyhow!("Failed to set intra threads: {:?}", e))?
            .commit_from_file(model_path.as_ref().to_str().unwrap())
            .map_err(|e| anyhow!("Failed to load ONNX model: {:?}", e))?;

        // Load config
        let config_str = std::fs::read_to_string(config_path.as_ref())?;
        let config: NluConfig = serde_json::from_str(&config_str)?;
        let max_length = config.max_length;

        // Load tokenizer from vocab file
        let tokenizer = Self::create_tokenizer(vocab_path.as_ref())?;

        Ok(OnnxClassifier {
            session: Mutex::new(session),
            tokenizer,
            config,
            max_length,
        })
    }

    /// Create a WordPiece tokenizer from vocab.txt
    fn create_tokenizer(vocab_path: &Path) -> Result<Tokenizer> {
        use tokenizers::models::wordpiece::WordPiece;
        use tokenizers::normalizers::BertNormalizer;
        use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
        use tokenizers::processors::template::TemplateProcessing;
        use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;

        // Build WordPiece model from vocab
        let wordpiece = WordPiece::from_file(vocab_path.to_str().unwrap())
            .unk_token("[UNK]".to_string())
            .build()
            .map_err(|e| anyhow!("Failed to build WordPiece: {}", e))?;

        let mut tokenizer = Tokenizer::new(wordpiece);

        // Add BERT normalizer (lowercase, strip accents, etc.)
        tokenizer.with_normalizer(Some(BertNormalizer::default()));

        // Add BERT pre-tokenizer (whitespace + punctuation)
        tokenizer.with_pre_tokenizer(Some(BertPreTokenizer));

        // Add post-processor for [CLS] and [SEP] tokens
        let template = TemplateProcessing::builder()
            .try_single("[CLS] $A [SEP]")
            .map_err(|e| anyhow!("Template error: {}", e))?
            .special_tokens(vec![
                ("[CLS]", 101),
                ("[SEP]", 102),
            ])
            .build()
            .map_err(|e| anyhow!("Failed to build template: {}", e))?;
        tokenizer.with_post_processor(Some(template));

        // Add decoder
        tokenizer.with_decoder(Some(WordPieceDecoder::default()));

        Ok(tokenizer)
    }

    /// Classify input text and extract slots
    pub fn classify(&self, input: &str) -> Result<OnnxClassifyResult> {
        // Tokenize input
        let encoding = self.tokenizer
            .encode(input, true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let tokens = encoding.get_ids();
        let attention_mask = encoding.get_attention_mask();
        let offsets = encoding.get_offsets();

        // Pad/truncate to max_length
        let seq_len = tokens.len().min(self.max_length);
        let mut input_ids = vec![0i64; self.max_length];
        let mut attn_mask = vec![0i64; self.max_length];

        for i in 0..seq_len {
            input_ids[i] = tokens[i] as i64;
            attn_mask[i] = attention_mask[i] as i64;
        }

        // Create tensors using ort's Tensor type
        let input_ids_tensor = Tensor::<i64>::from_array((
            [1usize, self.max_length],
            input_ids.into_boxed_slice()
        )).map_err(|e| anyhow!("Failed to create input_ids tensor: {:?}", e))?;
        
        let attn_mask_tensor = Tensor::<i64>::from_array((
            [1usize, self.max_length],
            attn_mask.into_boxed_slice()
        )).map_err(|e| anyhow!("Failed to create attention_mask tensor: {:?}", e))?;

        // Run inference with locked session
        let mut session = self.session.lock()
            .map_err(|e| anyhow!("Failed to lock session: {:?}", e))?;
        
        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attn_mask_tensor,
        ]).map_err(|e| anyhow!("Inference failed: {:?}", e))?;

        // Extract intent logits (shape: [1, num_intents])
        let intent_output = &outputs["intent_logits"];
        let (_, intent_logits_flat) = intent_output.try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract intent tensor: {:?}", e))?;

        // Extract slot logits (shape: [1, seq_len, num_slots])
        let slot_output = &outputs["slot_logits"];
        let (_, slot_logits_flat) = slot_output.try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract slot tensor: {:?}", e))?;
        let num_slots = self.config.slot2id.len();

        // Create array views from slices
        let intent_logits = ArrayView1::from(intent_logits_flat);

        // Softmax for intent probabilities
        let intent_probs = Self::softmax_view(&intent_logits);

        // Get top intent
        let (intent_id, confidence) = intent_probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, &p)| (i, p))
            .unwrap_or((0, 0.0));

        let intent = self.config.id2intent
            .get(&intent_id.to_string())
            .cloned()
            .unwrap_or_else(|| "unknown".to_string());

        // Get all intents with probabilities
        let all_intents: Vec<(String, f32)> = intent_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let name = self.config.id2intent
                    .get(&i.to_string())
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                (name, p)
            })
            .collect();

        // Reshape slot logits for processing
        let slot_logits_2d = ArrayView2::from_shape(
            (self.max_length, num_slots),
            slot_logits_flat
        )?;

        // Extract slots using BIO tags
        let slots = self.extract_slots(
            input,
            &slot_logits_2d,
            offsets,
            seq_len,
        );

        Ok(OnnxClassifyResult {
            intent,
            confidence,
            slots,
            all_intents,
        })
    }

    /// Softmax function for array view
    fn softmax_view(logits: &ArrayView1<f32>) -> Array1<f32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp: Array1<f32> = logits.mapv(|x| (x - max).exp());
        let sum: f32 = exp.sum();
        exp / sum
    }

    /// Extract slots from slot logits using BIO tagging
    fn extract_slots(
        &self,
        input: &str,
        slot_logits: &ArrayView2<f32>,
        offsets: &[(usize, usize)],
        seq_len: usize,
    ) -> Vec<ExtractedSlot> {
        let mut slots = Vec::new();
        let mut current_slot: Option<(String, usize, usize)> = None; // (label, start, end)

        for i in 0..seq_len {
            // Skip special tokens (offset 0,0)
            if i >= offsets.len() || (offsets[i].0 == 0 && offsets[i].1 == 0 && i > 0) {
                // End any current slot
                if let Some((label, start, end)) = current_slot.take() {
                    if end > start && end <= input.len() {
                        slots.push(ExtractedSlot {
                            label,
                            value: input[start..end].trim().to_string(),
                            start,
                            end,
                        });
                    }
                }
                continue;
            }

            // Get argmax for this token
            let token_logits = slot_logits.row(i);
            let slot_id = token_logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);

            let slot_label = self.config.id2slot
                .get(&slot_id.to_string())
                .cloned()
                .unwrap_or_else(|| "O".to_string());

            let (start_offset, end_offset) = offsets[i];

            if slot_label == "O" {
                // End any current slot
                if let Some((label, start, end)) = current_slot.take() {
                    if end > start && end <= input.len() {
                        slots.push(ExtractedSlot {
                            label,
                            value: input[start..end].trim().to_string(),
                            start,
                            end,
                        });
                    }
                }
            } else if slot_label.starts_with("B-") {
                // End previous slot if any
                if let Some((label, start, end)) = current_slot.take() {
                    if end > start && end <= input.len() {
                        slots.push(ExtractedSlot {
                            label,
                            value: input[start..end].trim().to_string(),
                            start,
                            end,
                        });
                    }
                }
                // Start new slot
                let label = slot_label[2..].to_string();
                current_slot = Some((label, start_offset, end_offset));
            } else if slot_label.starts_with("I-") {
                // Continue current slot if label matches
                let label = &slot_label[2..];
                if let Some((ref curr_label, start, _)) = current_slot {
                    if curr_label == label {
                        current_slot = Some((curr_label.clone(), start, end_offset));
                    }
                }
                // If no current slot or mismatch, treat as B-
                if current_slot.is_none() {
                    current_slot = Some((label.to_string(), start_offset, end_offset));
                }
            }
        }

        // End any remaining slot
        if let Some((label, start, end)) = current_slot.take() {
            if end > start && end <= input.len() {
                slots.push(ExtractedSlot {
                    label,
                    value: input[start..end].trim().to_string(),
                    start,
                    end,
                });
            }
        }

        slots
    }

    /// Convert ONNX result to the unified Intent type
    pub fn to_intent(&self, result: &OnnxClassifyResult, raw_input: &str) -> Intent {
        match result.intent.as_str() {
            "calendar_create" => Intent::CalendarCreate {
                description: raw_input.to_string(),
            },
            "calendar_query" => {
                // Determine query type from slots
                let query_type = self.infer_calendar_query_type(raw_input, &result.slots);
                Intent::CalendarQuery { query_type }
            }
            "calendar_update" => Intent::CalendarUpdate {
                description: raw_input.to_string(),
            },
            "calendar_delete" => Intent::CalendarDelete {
                description: raw_input.to_string(),
            },
            "email_compose" => Intent::EmailCompose {
                description: raw_input.to_string(),
            },
            "email_reply" => Intent::EmailReply {
                description: raw_input.to_string(),
            },
            "email_attendees" => Intent::EmailAttendees {
                description: raw_input.to_string(),
            },
            "email_query" => {
                let query_type = self.infer_email_query_type(raw_input, &result.slots);
                Intent::EmailQuery { query_type }
            }
            "contact_update" => Intent::ContactUpdate {
                description: raw_input.to_string(),
            },
            "knowledge_query" => Intent::KnowledgeQuery {
                query: raw_input.to_string(),
            },
            "command" => Intent::Command(raw_input.to_string()),
            "conversation" | _ => Intent::Conversation {
                input: raw_input.to_string(),
            },
        }
    }

    /// Infer calendar query type from input and slots
    fn infer_calendar_query_type(&self, input: &str, slots: &[ExtractedSlot]) -> CalendarQueryType {
        let lower = input.to_lowercase();

        // Check slots for DATE
        for slot in slots {
            if slot.label == "DATE" {
                let date_lower = slot.value.to_lowercase();
                if date_lower.contains("tomorrow") {
                    return CalendarQueryType::Tomorrow;
                }
                if date_lower.contains("today") {
                    return CalendarQueryType::Today;
                }
                if date_lower.contains("week") {
                    return CalendarQueryType::Week;
                }
            }
        }

        // Fallback to keyword matching
        if lower.contains("tomorrow") {
            CalendarQueryType::Tomorrow
        } else if lower.contains("week") {
            CalendarQueryType::Week
        } else if lower.contains("upcoming") || lower.contains("next") {
            CalendarQueryType::Upcoming
        } else {
            CalendarQueryType::Today
        }
    }

    /// Infer email query type from input and slots
    fn infer_email_query_type(&self, input: &str, slots: &[ExtractedSlot]) -> EmailQueryType {
        let lower = input.to_lowercase();

        // Check for person in slots (from X)
        for slot in slots {
            if slot.label == "PER" && lower.contains("from") {
                return EmailQueryType::From(slot.value.clone());
            }
        }

        if lower.contains("unread") || lower.contains("new") {
            EmailQueryType::Unread
        } else if lower.contains("summary") || lower.contains("summarize") {
            EmailQueryType::Summary
        } else {
            EmailQueryType::List
        }
    }

    /// Get extracted entities as a HashMap for easy access
    #[allow(dead_code)]
    pub fn get_entities(&self, result: &OnnxClassifyResult) -> HashMap<String, Vec<String>> {
        let mut entities: HashMap<String, Vec<String>> = HashMap::new();
        for slot in &result.slots {
            entities
                .entry(slot.label.clone())
                .or_default()
                .push(slot.value.clone());
        }
        entities
    }
}

#[async_trait]
impl IntentClassifier for OnnxClassifier {
    async fn classify(&self, input: &str, _ctx: &IntentContext) -> Result<IntentResult> {
        // Check for explicit commands first
        if input.starts_with('/') {
            return Ok(IntentResult::new(
                Intent::Command(input.to_string()),
                1.0,
                input,
            ));
        }

        // Run ONNX classification
        let result = OnnxClassifier::classify(self, input)?;

        // Convert to Intent
        let intent = self.to_intent(&result, input);

        // Build alternatives from other high-scoring intents
        let alternatives: Vec<(Intent, f32)> = result
            .all_intents
            .iter()
            .filter(|(name, conf)| name != &result.intent && *conf > 0.1)
            .take(3)
            .map(|(name, conf)| {
                let alt_result = OnnxClassifyResult {
                    intent: name.clone(),
                    confidence: *conf,
                    slots: result.slots.clone(),
                    all_intents: vec![],
                };
                (self.to_intent(&alt_result, input), *conf)
            })
            .collect();

        Ok(IntentResult::new(intent, result.confidence, input).with_alternatives(alternatives))
    }

    fn name(&self) -> &str {
        "onnx"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let probs = OnnxClassifier::softmax_view(&logits.view());
        
        // Sum should be ~1.0
        let sum: f32 = probs.sum();
        assert!((sum - 1.0).abs() < 0.001);
        
        // Last element should be highest
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }
}
