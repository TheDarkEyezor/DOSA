//! Hybrid intent classifier combining ONNX neural model with keyword matching
//!
//! Strategy:
//! 1. Run fast ONNX classifier first (if available)
//! 2. If confidence is high (>0.8), use ONNX result
//! 3. If ambiguous, cross-check with keyword classifier
//! 4. Use LLM as final fallback for disambiguation

use super::keyword::KeywordClassifier;
use super::onnx::OnnxClassifier;
use super::types::*;
use super::IntentClassifier;
use crate::llm::OllamaClient;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;

/// Configuration for the hybrid classifier
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Use LLM only when keyword confidence is below this
    pub llm_threshold: f32,
    /// Minimum confidence to accept a classification
    pub min_confidence: f32,
    /// Whether to use LLM at all (can disable for speed)
    pub use_llm: bool,
    /// Whether to use ONNX model (if available)
    pub use_onnx: bool,
    /// ONNX confidence threshold - above this we trust the model
    pub onnx_threshold: f32,
}

impl Default for HybridConfig {
    fn default() -> Self {
        HybridConfig {
            llm_threshold: 0.8,
            min_confidence: 0.4,
            use_llm: true,
            use_onnx: true,
            onnx_threshold: 0.85,
        }
    }
}

/// LLM response for intent classification
#[derive(Debug, Serialize, Deserialize)]
struct LLMIntentResponse {
    intent_type: String,
    confidence: f32,
    #[serde(default)]
    explanation: Option<String>,
}

/// Hybrid classifier combining keyword matching with LLM fallback
pub struct HybridClassifier {
    keyword: KeywordClassifier,
    onnx: Option<Arc<OnnxClassifier>>,
    llm: Arc<OllamaClient>,
    config: HybridConfig,
}

impl HybridClassifier {
    pub fn new(llm: Arc<OllamaClient>) -> Self {
        HybridClassifier {
            keyword: KeywordClassifier::new(),
            onnx: None,
            llm,
            config: HybridConfig::default(),
        }
    }

    /// Create a new hybrid classifier with ONNX model
    pub fn with_onnx<P: AsRef<Path>>(
        llm: Arc<OllamaClient>,
        model_path: P,
        config_path: P,
        vocab_path: P,
    ) -> Result<Self> {
        let onnx = OnnxClassifier::new(model_path, config_path, vocab_path)?;
        Ok(HybridClassifier {
            keyword: KeywordClassifier::new(),
            onnx: Some(Arc::new(onnx)),
            llm,
            config: HybridConfig::default(),
        })
    }

    /// Try to load ONNX model from default data directory
    pub fn try_with_onnx(llm: Arc<OllamaClient>, data_dir: &Path) -> Self {
        let nlu_dir = data_dir.join("nlu");
        let model_path = nlu_dir.join("model.onnx");
        let config_path = nlu_dir.join("config.json");
        let vocab_path = nlu_dir.join("vocab.txt");

        if model_path.exists() && config_path.exists() && vocab_path.exists() {
            match OnnxClassifier::new(&model_path, &config_path, &vocab_path) {
                Ok(onnx) => {
                    eprintln!("✓ Loaded NLU model from {:?}", nlu_dir);
                    return HybridClassifier {
                        keyword: KeywordClassifier::new(),
                        onnx: Some(Arc::new(onnx)),
                        llm,
                        config: HybridConfig::default(),
                    };
                }
                Err(e) => {
                    eprintln!("⚠ Failed to load NLU model: {}", e);
                }
            }
        }

        // Fallback to keyword-only
        HybridClassifier {
            keyword: KeywordClassifier::new(),
            onnx: None,
            llm,
            config: HybridConfig::default(),
        }
    }
    
    pub fn with_config(mut self, config: HybridConfig) -> Self {
        self.config = config;
        self
    }

    /// Check if ONNX model is available
    pub fn has_onnx(&self) -> bool {
        self.onnx.is_some()
    }

    /// Check if two intents are of the same type (ignoring inner data)
    fn intents_match(&self, a: &Intent, b: &Intent) -> bool {
        std::mem::discriminant(a) == std::mem::discriminant(b)
    }
    
    /// Use LLM to disambiguate between candidates
    async fn llm_disambiguate(
        &self,
        input: &str,
        candidates: &[(Intent, f32)],
    ) -> Result<IntentResult> {
        // Build a prompt for classification
        let candidate_names: Vec<String> = candidates
            .iter()
            .map(|(intent, _)| self.intent_to_name(intent))
            .collect();
        
        let prompt = format!(
            r#"Classify this user request into one of these categories: {}

User request: "{}"

Respond with JSON only:
{{"intent_type": "<category>", "confidence": <0.0-1.0>}}"#,
            candidate_names.join(", "),
            input
        );
        
        // Query LLM
        let response = self.llm.query(&prompt).await?;
        
        // Try to parse JSON response
        let parsed = self.parse_llm_response(&response, input, candidates)?;
        
        Ok(parsed)
    }
    
    /// Convert intent to name string
    fn intent_to_name(&self, intent: &Intent) -> String {
        match intent {
            Intent::CalendarCreate { .. } => "calendar_create".to_string(),
            Intent::CalendarQuery { .. } => "calendar_query".to_string(),
            Intent::CalendarUpdate { .. } => "calendar_update".to_string(),
            Intent::CalendarDelete { .. } => "calendar_delete".to_string(),
            Intent::EmailCompose { .. } => "email_compose".to_string(),
            Intent::EmailReply { .. } => "email_reply".to_string(),
            Intent::EmailAttendees { .. } => "email_attendees".to_string(),
            Intent::EmailQuery { .. } => "email_query".to_string(),
            Intent::ContactUpdate { .. } => "contact_update".to_string(),
            Intent::KnowledgeQuery { .. } => "knowledge_query".to_string(),
            Intent::Conversation { .. } => "conversation".to_string(),
            Intent::Command(_) => "command".to_string(),
            Intent::Ambiguous { .. } => "ambiguous".to_string(),
        }
    }
    
    /// Parse LLM response and match to candidate intents
    fn parse_llm_response(
        &self,
        response: &str,
        original_input: &str,
        candidates: &[(Intent, f32)],
    ) -> Result<IntentResult> {
        // Extract JSON from response (handle markdown wrapping)
        let json_str = self.extract_json(response);
        
        let parsed: LLMIntentResponse = serde_json::from_str(&json_str)
            .map_err(|e| anyhow!("Failed to parse LLM response: {} - raw: {}", e, response))?;
        
        // Find matching candidate
        let intent_type = parsed.intent_type.to_lowercase();
        for (intent, _) in candidates {
            let name = self.intent_to_name(intent).to_lowercase();
            if name.contains(&intent_type) || intent_type.contains(&name) {
                return Ok(IntentResult::new(
                    intent.clone(),
                    parsed.confidence,
                    original_input,
                ));
            }
        }
        
        // No match - return best keyword candidate
        if let Some((intent, conf)) = candidates.first() {
            Ok(IntentResult::new(intent.clone(), *conf, original_input))
        } else {
            Ok(IntentResult::new(
                Intent::Conversation { input: original_input.to_string() },
                0.5,
                original_input,
            ))
        }
    }
    
    /// Extract JSON from potentially markdown-wrapped response
    fn extract_json(&self, response: &str) -> String {
        let trimmed = response.trim();
        
        // Check for markdown code block
        if trimmed.starts_with("```") {
            let lines: Vec<&str> = trimmed.lines().collect();
            if lines.len() > 2 {
                let json_lines: Vec<&str> = lines[1..lines.len()-1]
                    .iter()
                    .filter(|l| !l.starts_with("```"))
                    .copied()
                    .collect();
                return json_lines.join("\n");
            }
        }
        
        // Try to find JSON object
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                return trimmed[start..=end].to_string();
            }
        }
        
        // Return as-is
        trimmed.to_string()
    }
}

#[async_trait]
impl IntentClassifier for HybridClassifier {
    async fn classify(&self, input: &str, ctx: &IntentContext) -> Result<IntentResult> {
        // Check for explicit commands first
        if input.starts_with('/') {
            return Ok(IntentResult::new(
                Intent::Command(input.to_string()),
                1.0,
                input,
            ));
        }

        // Step 1: Try ONNX classification first (if available and enabled)
        if self.config.use_onnx {
            if let Some(ref onnx) = self.onnx {
                match onnx.classify(input) {
                    Ok(onnx_result) => {
                        // High confidence from ONNX - trust it
                        if onnx_result.confidence >= self.config.onnx_threshold {
                            let intent = onnx.to_intent(&onnx_result, input);
                            let alternatives: Vec<(Intent, f32)> = onnx_result
                                .all_intents
                                .iter()
                                .filter(|(name, conf)| name != &onnx_result.intent && *conf > 0.1)
                                .take(3)
                                .map(|(name, conf)| {
                                    let alt_result = super::onnx::OnnxClassifyResult {
                                        intent: name.clone(),
                                        confidence: *conf,
                                        slots: onnx_result.slots.clone(),
                                        all_intents: vec![],
                                    };
                                    (onnx.to_intent(&alt_result, input), *conf)
                                })
                                .collect();
                            
                            return Ok(IntentResult::new(intent, onnx_result.confidence, input)
                                .with_alternatives(alternatives));
                        }

                        // Medium confidence - cross-check with keyword classifier
                        if onnx_result.confidence >= 0.5 {
                            let keyword_result = self.keyword.classify(input, ctx).await?;
                            
                            // If both agree, boost confidence
                            let onnx_intent = onnx.to_intent(&onnx_result, input);
                            if self.intents_match(&onnx_intent, &keyword_result.intent) {
                                let boosted_conf = (onnx_result.confidence + keyword_result.confidence) / 2.0 + 0.1;
                                return Ok(IntentResult::new(
                                    onnx_intent,
                                    boosted_conf.min(1.0),
                                    input,
                                ));
                            }
                            
                            // Disagreement - prefer ONNX if its confidence is reasonable
                            if onnx_result.confidence > keyword_result.confidence {
                                return Ok(IntentResult::new(onnx_intent, onnx_result.confidence, input));
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("ONNX classification failed: {}, falling back to keyword", e);
                    }
                }
            }
        }

        // Step 2: Fall back to keyword classification
        let keyword_result = self.keyword.classify(input, ctx).await?;
        
        // Step 3: Check if confident enough
        if keyword_result.confidence >= self.config.llm_threshold {
            return Ok(keyword_result);
        }
        
        // Step 4: If LLM disabled or no alternatives, return keyword result
        if !self.config.use_llm || keyword_result.alternatives.is_empty() {
            return Ok(keyword_result);
        }
        
        // Step 5: Use LLM for disambiguation
        let mut candidates = vec![(keyword_result.intent.clone(), keyword_result.confidence)];
        candidates.extend(keyword_result.alternatives.clone());
        
        // Only use LLM if we have real ambiguity
        let top_confidence = candidates[0].1;
        let has_close_second = candidates.len() > 1 && 
            (candidates[0].1 - candidates[1].1).abs() < 0.2;
        
        if has_close_second {
            match self.llm_disambiguate(input, &candidates).await {
                Ok(result) => Ok(result),
                Err(_) => {
                    // LLM failed, fall back to keyword result
                    Ok(IntentResult::new(
                        keyword_result.intent,
                        top_confidence,
                        input,
                    ))
                }
            }
        } else {
            Ok(keyword_result)
        }
    }
    
    fn name(&self) -> &str {
        if self.onnx.is_some() {
            "hybrid-onnx"
        } else {
            "hybrid"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Tests would require mock LLM client
    #[test]
    fn test_config_defaults() {
        let config = HybridConfig::default();
        assert_eq!(config.llm_threshold, 0.8);
        assert!(config.use_llm);
    }
}
