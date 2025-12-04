//! Voice interface module for speech-to-text and text-to-speech
//! 
//! This module provides voice input/output capabilities using:
//! - whisper-rs for speech-to-text (using whisper.cpp)
//! - cpal for audio capture
//! - System TTS (macOS `say` command) for responses
//!
//! # Feature Gate
//! This module requires the `voice` feature to be enabled:
//! ```toml
//! [dependencies]
//! dosa = { version = "0.1", features = ["voice"] }
//! ```
//!
//! Building with voice support requires:
//! - cmake (for building whisper.cpp)
//! - A C++ compiler

mod capture;
mod transcribe;
mod tts;

pub use capture::AudioCapture;
pub use transcribe::WhisperTranscriber;
pub use tts::TextToSpeech;

use anyhow::Result;
use std::sync::Arc;

/// Voice interface configuration
#[derive(Clone, Debug)]
pub struct VoiceConfig {
    /// Whisper model to use (tiny.en, base.en, small.en)
    pub model: WhisperModel,
    /// Sample rate for audio capture (default: 16000)
    pub sample_rate: u32,
    /// Channels for audio capture (default: 1 for mono)
    pub channels: u16,
    /// Maximum recording duration in seconds
    pub max_duration_secs: u32,
    /// Silence threshold for auto-stop (0.0-1.0)
    pub silence_threshold: f32,
    /// Silence duration in seconds to trigger stop
    pub silence_duration: f32,
}

impl Default for VoiceConfig {
    fn default() -> Self {
        Self {
            model: WhisperModel::Base,
            sample_rate: 16000,
            channels: 1,
            max_duration_secs: 30,
            silence_threshold: 0.01,
            silence_duration: 1.5,
        }
    }
}

/// Whisper model options
#[derive(Clone, Debug, Copy)]
pub enum WhisperModel {
    /// 39M parameters - fastest, ~10x realtime
    Tiny,
    /// 74M parameters - balanced (recommended)
    Base,
    /// 244M parameters - highest accuracy, ~4x realtime
    Small,
}

impl WhisperModel {
    /// Get the model filename
    pub fn filename(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "ggml-tiny.en.bin",
            WhisperModel::Base => "ggml-base.en.bin",
            WhisperModel::Small => "ggml-small.en.bin",
        }
    }
    
    /// Get the model URL for download
    pub fn url(&self) -> &'static str {
        match self {
            WhisperModel::Tiny => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
            WhisperModel::Base => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            WhisperModel::Small => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
        }
    }
}

/// Main voice interface that combines capture, transcription, and TTS
pub struct VoiceInterface {
    config: VoiceConfig,
    capture: Arc<AudioCapture>,
    transcriber: WhisperTranscriber,
    tts: TextToSpeech,
}

impl VoiceInterface {
    /// Create a new voice interface with the given configuration
    pub async fn new(config: VoiceConfig) -> Result<Self> {
        let capture = Arc::new(AudioCapture::new(config.sample_rate, config.channels)?);
        let transcriber = WhisperTranscriber::new(&config)?;
        let tts = TextToSpeech::new();
        
        Ok(Self {
            config,
            capture,
            transcriber,
            tts,
        })
    }
    
    /// Create with default configuration
    pub async fn default_config() -> Result<Self> {
        Self::new(VoiceConfig::default()).await
    }
    
    /// Listen for voice input and return transcribed text
    pub async fn listen(&self) -> Result<String> {
        // Capture audio from microphone
        let audio = self.capture.record_until_silence(
            self.config.silence_threshold,
            self.config.silence_duration,
            self.config.max_duration_secs,
        ).await?;
        
        // Transcribe the audio
        let text = self.transcriber.transcribe(&audio)?;
        
        Ok(text)
    }
    
    /// Speak the given text using system TTS
    pub async fn speak(&self, text: &str) -> Result<()> {
        self.tts.speak(text).await
    }
    
    /// Get the audio capture component for direct access
    pub fn capture(&self) -> Arc<AudioCapture> {
        Arc::clone(&self.capture)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_voice_config_default() {
        let config = VoiceConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.channels, 1);
    }
    
    #[test]
    fn test_whisper_model_filenames() {
        assert_eq!(WhisperModel::Tiny.filename(), "ggml-tiny.en.bin");
        assert_eq!(WhisperModel::Base.filename(), "ggml-base.en.bin");
        assert_eq!(WhisperModel::Small.filename(), "ggml-small.en.bin");
    }
}
