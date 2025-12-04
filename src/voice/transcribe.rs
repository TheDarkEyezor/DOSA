//! Whisper speech-to-text transcription
//! 
//! Uses whisper-rs bindings to whisper.cpp for local STT.

use anyhow::{anyhow, Result};
use std::path::PathBuf;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::voice::{VoiceConfig, WhisperModel};

/// Whisper transcriber for speech-to-text
pub struct WhisperTranscriber {
    ctx: WhisperContext,
}

impl WhisperTranscriber {
    /// Create a new transcriber with the given configuration
    pub fn new(config: &VoiceConfig) -> Result<Self> {
        let model_path = Self::get_model_path(&config.model)?;
        
        if !model_path.exists() {
            return Err(anyhow!(
                "Whisper model not found at {}. Run 'dosa --download-voice-model' to download it.",
                model_path.display()
            ));
        }
        
        let ctx = WhisperContext::new_with_params(
            model_path.to_str().ok_or_else(|| anyhow!("Invalid model path"))?,
            WhisperContextParameters::default(),
        ).map_err(|e| anyhow!("Failed to load Whisper model: {:?}", e))?;
        
        Ok(Self { ctx })
    }
    
    /// Get the path where models should be stored
    pub fn models_dir() -> PathBuf {
        let data_dir = dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("dosa")
            .join("models");
        data_dir
    }
    
    /// Get the path for a specific model
    pub fn get_model_path(model: &WhisperModel) -> Result<PathBuf> {
        let models_dir = Self::models_dir();
        std::fs::create_dir_all(&models_dir)?;
        Ok(models_dir.join(model.filename()))
    }
    
    /// Download a Whisper model
    pub async fn download_model(model: &WhisperModel) -> Result<PathBuf> {
        let model_path = Self::get_model_path(model)?;
        
        if model_path.exists() {
            eprintln!("✓ Model already exists at {}", model_path.display());
            return Ok(model_path);
        }
        
        eprintln!("📥 Downloading {} model...", model.filename());
        eprintln!("   From: {}", model.url());
        
        let response = reqwest::get(model.url()).await?;
        
        if !response.status().is_success() {
            return Err(anyhow!("Failed to download model: HTTP {}", response.status()));
        }
        
        let total_size = response.content_length().unwrap_or(0);
        let bytes = response.bytes().await?;
        
        std::fs::write(&model_path, &bytes)?;
        
        eprintln!("✓ Downloaded {} bytes to {}", total_size, model_path.display());
        
        Ok(model_path)
    }
    
    /// Transcribe audio samples to text
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        if samples.is_empty() {
            return Ok(String::new());
        }
        
        // Create whisper parameters
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        
        // Configure for English speech
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);  // Updated API name
        
        // Create a state for processing
        let mut state = self.ctx.create_state()
            .map_err(|e| anyhow!("Failed to create whisper state: {:?}", e))?;
        
        // Process the audio
        state.full(params, samples)
            .map_err(|e| anyhow!("Whisper transcription failed: {:?}", e))?;
        
        // Get the number of segments (now returns c_int directly)
        let num_segments = state.full_n_segments();
        
        // Collect all text from segments
        let mut text = String::new();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                if let Ok(segment_text) = segment.to_str_lossy() {
                    text.push_str(&segment_text);
                    text.push(' ');
                }
            }
        }
        
        Ok(text.trim().to_string())
    }
    
    /// Check if the model is loaded
    pub fn is_loaded(&self) -> bool {
        true // If we have a context, the model is loaded
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_models_dir() {
        let dir = WhisperTranscriber::models_dir();
        assert!(dir.to_str().is_some());
        assert!(dir.to_str().unwrap().contains("dosa"));
    }
    
    #[test]
    fn test_model_path() {
        let path = WhisperTranscriber::get_model_path(&WhisperModel::Base);
        assert!(path.is_ok());
        assert!(path.unwrap().to_str().unwrap().contains("ggml-base.en.bin"));
    }
}
