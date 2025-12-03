//! Voice feature test
//! 
//! Run with: cargo run --example voice_test --features voice

#[cfg(feature = "voice")]
use dosa::voice::{TextToSpeech, VoiceConfig, WhisperModel, WhisperTranscriber};

#[cfg(feature = "voice")]
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== DOSA Voice Test ===\n");
    
    // Test 1: TTS
    println!("1. Testing Text-to-Speech...");
    let tts = TextToSpeech::new();
    tts.speak("Hello! I am DOSA, your desktop assistant. Voice test successful.").await?;
    println!("   ✓ TTS works!\n");
    
    // Test 2: List available voices
    println!("2. Available macOS voices:");
    let voices = TextToSpeech::list_voices()?;
    for voice in voices.iter().take(10) {
        println!("   - {}", voice);
    }
    if voices.len() > 10 {
        println!("   ... and {} more", voices.len() - 10);
    }
    println!();
    
    // Test 3: Different voice
    println!("3. Testing with Samantha voice...");
    let tts_samantha = TextToSpeech::with_voice("Samantha");
    tts_samantha.speak("This is Samantha speaking.").await?;
    println!("   ✓ Custom voice works!\n");
    
    // Test 4: Check Whisper model path
    println!("4. Checking Whisper model...");
    let model_path = WhisperTranscriber::get_model_path(&WhisperModel::Base)?;
    println!("   Model path: {}", model_path.display());
    
    if model_path.exists() {
        println!("   ✓ Model already downloaded!");
    } else {
        println!("   Model not found. Downloading...");
        WhisperTranscriber::download_model(&WhisperModel::Base).await?;
        println!("   ✓ Model downloaded!");
    }
    println!();
    
    // Test 5: Initialize transcriber
    println!("5. Initializing Whisper transcriber...");
    let config = VoiceConfig::default();
    match WhisperTranscriber::new(&config) {
        Ok(_) => println!("   ✓ Transcriber initialized successfully!"),
        Err(e) => println!("   ✗ Failed to initialize: {}", e),
    }
    println!();
    
    // Done
    let tts = TextToSpeech::new();
    tts.speak("Voice test complete!").await?;
    
    println!("=== All tests complete! ===");
    
    Ok(())
}

#[cfg(not(feature = "voice"))]
fn main() {
    println!("Voice feature not enabled. Run with: cargo run --example voice_test --features voice");
}
