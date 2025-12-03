//! Text-to-speech using system TTS
//! 
//! Uses platform-specific TTS:
//! - macOS: `say` command
//! - Linux: `espeak` or `festival`
//! - Windows: SAPI (System.Speech)

use anyhow::Result;
use std::process::Command;

/// Text-to-speech output
pub struct TextToSpeech {
    /// Voice to use (platform-specific)
    voice: Option<String>,
    /// Speaking rate (words per minute)
    rate: Option<u32>,
}

impl TextToSpeech {
    /// Create a new TTS instance with default settings
    pub fn new() -> Self {
        Self {
            voice: None,
            rate: None,
        }
    }
    
    /// Create with a specific voice
    pub fn with_voice(voice: &str) -> Self {
        Self {
            voice: Some(voice.to_string()),
            rate: None,
        }
    }
    
    /// Set the speaking rate (words per minute)
    pub fn with_rate(mut self, rate: u32) -> Self {
        self.rate = Some(rate);
        self
    }
    
    /// Speak the given text
    pub async fn speak(&self, text: &str) -> Result<()> {
        // Skip empty text
        if text.trim().is_empty() {
            return Ok(());
        }
        
        #[cfg(target_os = "macos")]
        {
            self.speak_macos(text).await
        }
        
        #[cfg(target_os = "linux")]
        {
            self.speak_linux(text).await
        }
        
        #[cfg(target_os = "windows")]
        {
            self.speak_windows(text).await
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            eprintln!("TTS not supported on this platform");
            Ok(())
        }
    }
    
    /// macOS TTS using `say` command
    #[cfg(target_os = "macos")]
    async fn speak_macos(&self, text: &str) -> Result<()> {
        let mut cmd = Command::new("say");
        
        if let Some(ref voice) = self.voice {
            cmd.args(["-v", voice]);
        }
        
        if let Some(rate) = self.rate {
            cmd.args(["-r", &rate.to_string()]);
        }
        
        cmd.arg(text);
        
        // Run asynchronously
        tokio::task::spawn_blocking(move || {
            cmd.status()
        }).await??;
        
        Ok(())
    }
    
    /// Linux TTS using espeak or festival
    #[cfg(target_os = "linux")]
    async fn speak_linux(&self, text: &str) -> Result<()> {
        // Try espeak first, then festival
        let mut cmd = if Command::new("espeak").arg("--version").output().is_ok() {
            let mut c = Command::new("espeak");
            
            if let Some(ref voice) = self.voice {
                c.args(["-v", voice]);
            }
            
            if let Some(rate) = self.rate {
                // Convert WPM to espeak rate (80-450, default 175)
                let espeak_rate = rate.min(450).max(80);
                c.args(["-s", &espeak_rate.to_string()]);
            }
            
            c.arg(text.to_string());
            c
        } else {
            // Fall back to festival
            let mut c = Command::new("echo");
            c.arg(format!("{} | festival --tts", text));
            c
        };
        
        tokio::task::spawn_blocking(move || {
            cmd.status()
        }).await??;
        
        Ok(())
    }
    
    /// Windows TTS using PowerShell and SAPI
    #[cfg(target_os = "windows")]
    async fn speak_windows(&self, text: &str) -> Result<()> {
        let escaped = text.replace("'", "''");
        
        let script = if let Some(ref voice) = self.voice {
            format!(
                r#"$voice = New-Object -ComObject SAPI.SPVoice; 
                   $voices = $voice.GetVoices(); 
                   foreach ($v in $voices) {{ 
                       if ($v.GetDescription() -like '*{}*') {{ 
                           $voice.Voice = $v; 
                           break 
                       }} 
                   }}; 
                   $voice.Speak('{}')"#,
                voice, escaped
            )
        } else {
            format!(
                r#"Add-Type -AssemblyName System.Speech; 
                   $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; 
                   $synth.Speak('{}')"#,
                escaped
            )
        };
        
        let mut cmd = Command::new("powershell");
        cmd.args(["-Command", &script]);
        
        tokio::task::spawn_blocking(move || {
            cmd.status()
        }).await??;
        
        Ok(())
    }
    
    /// List available voices (macOS only for now)
    #[cfg(target_os = "macos")]
    pub fn list_voices() -> Result<Vec<String>> {
        let output = Command::new("say")
            .args(["-v", "?"])
            .output()?;
        
        let stdout = String::from_utf8_lossy(&output.stdout);
        let voices: Vec<String> = stdout
            .lines()
            .filter_map(|line| {
                line.split_whitespace().next().map(|s| s.to_string())
            })
            .collect();
        
        Ok(voices)
    }
    
    #[cfg(not(target_os = "macos"))]
    pub fn list_voices() -> Result<Vec<String>> {
        Ok(vec!["default".to_string()])
    }
}

impl Default for TextToSpeech {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tts_creation() {
        let tts = TextToSpeech::new();
        assert!(tts.voice.is_none());
        assert!(tts.rate.is_none());
    }
    
    #[test]
    fn test_tts_with_voice() {
        let tts = TextToSpeech::with_voice("Samantha");
        assert_eq!(tts.voice, Some("Samantha".to_string()));
    }
    
    #[test]
    fn test_tts_with_rate() {
        let tts = TextToSpeech::new().with_rate(200);
        assert_eq!(tts.rate, Some(200));
    }
}
