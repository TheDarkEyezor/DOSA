//! Audio capture using cpal
//! 
//! Provides cross-platform audio input from the system microphone.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, SampleRate, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Audio capture from microphone
pub struct AudioCapture {
    device: Device,
    config: StreamConfig,
}

impl AudioCapture {
    /// Create a new audio capture instance
    pub fn new(sample_rate: u32, channels: u16) -> Result<Self> {
        let host = cpal::default_host();
        
        let device = host.default_input_device()
            .ok_or_else(|| anyhow!("No audio input device available"))?;
        
        let device_name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        eprintln!("🎤 Using input device: {}", device_name);
        
        let config = StreamConfig {
            channels,
            sample_rate: SampleRate(sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };
        
        Ok(Self { device, config })
    }
    
    /// Record audio until silence is detected or max duration is reached
    pub async fn record_until_silence(
        &self,
        silence_threshold: f32,
        silence_duration: f32,
        max_duration_secs: u32,
    ) -> Result<Vec<f32>> {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let samples_clone = Arc::clone(&samples);
        
        let stop_flag = Arc::new(AtomicBool::new(false));
        let stop_flag_clone = Arc::clone(&stop_flag);
        
        let silence_start = Arc::new(Mutex::new(None::<Instant>));
        let silence_start_clone = Arc::clone(&silence_start);
        let silence_duration = Duration::from_secs_f32(silence_duration);
        
        let recording_start = Instant::now();
        let max_duration = Duration::from_secs(max_duration_secs as u64);
        
        // Get supported format
        let supported_config = self.device.supported_input_configs()?
            .find(|c| c.sample_format() == SampleFormat::F32)
            .or_else(|| self.device.supported_input_configs().ok()?.next())
            .ok_or_else(|| anyhow!("No supported audio input format"))?;
        
        let sample_format = supported_config.sample_format();
        
        // Build the stream based on sample format
        let err_fn = |err| eprintln!("Audio stream error: {}", err);
        
        let stream = match sample_format {
            SampleFormat::F32 => {
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        Self::handle_audio_data(
                            data,
                            &samples_clone,
                            &silence_start_clone,
                            &stop_flag_clone,
                            silence_threshold,
                            silence_duration,
                            recording_start,
                            max_duration,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            SampleFormat::I16 => {
                let samples_clone = Arc::clone(&samples);
                let silence_start_clone = Arc::clone(&silence_start);
                let stop_flag_clone = Arc::clone(&stop_flag);
                
                self.device.build_input_stream(
                    &self.config,
                    move |data: &[i16], _: &cpal::InputCallbackInfo| {
                        // Convert i16 to f32
                        let float_data: Vec<f32> = data.iter()
                            .map(|&s| s as f32 / i16::MAX as f32)
                            .collect();
                        
                        Self::handle_audio_data(
                            &float_data,
                            &samples_clone,
                            &silence_start_clone,
                            &stop_flag_clone,
                            silence_threshold,
                            silence_duration,
                            recording_start,
                            max_duration,
                        );
                    },
                    err_fn,
                    None,
                )?
            }
            sample_format => {
                return Err(anyhow!("Unsupported sample format: {:?}", sample_format));
            }
        };
        
        stream.play()?;
        eprintln!("🔴 Recording... (speak now, will stop on silence)");
        
        // Wait for recording to complete
        while !stop_flag.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        
        drop(stream);
        eprintln!("⏹️ Recording stopped");
        
        let result = samples.lock().unwrap().clone();
        Ok(result)
    }
    
    /// Handle incoming audio data
    fn handle_audio_data(
        data: &[f32],
        samples: &Arc<Mutex<Vec<f32>>>,
        silence_start: &Arc<Mutex<Option<Instant>>>,
        stop_flag: &Arc<AtomicBool>,
        silence_threshold: f32,
        silence_duration: Duration,
        recording_start: Instant,
        max_duration: Duration,
    ) {
        // Check if max duration reached
        if recording_start.elapsed() >= max_duration {
            stop_flag.store(true, Ordering::Relaxed);
            return;
        }
        
        // Calculate RMS amplitude
        let rms = (data.iter().map(|&s| s * s).sum::<f32>() / data.len() as f32).sqrt();
        
        // Append samples
        samples.lock().unwrap().extend_from_slice(data);
        
        // Check for silence
        let mut silence_guard = silence_start.lock().unwrap();
        
        if rms < silence_threshold {
            if silence_guard.is_none() {
                *silence_guard = Some(Instant::now());
            } else if silence_guard.unwrap().elapsed() >= silence_duration {
                // Only stop if we've recorded some audio
                let samples_len = samples.lock().unwrap().len();
                if samples_len > 16000 { // At least 1 second of audio
                    stop_flag.store(true, Ordering::Relaxed);
                }
            }
        } else {
            // Sound detected, reset silence timer
            *silence_guard = None;
        }
    }
    
    /// Record for a fixed duration
    pub async fn record_for_duration(&self, duration_secs: u32) -> Result<Vec<f32>> {
        let samples = Arc::new(Mutex::new(Vec::new()));
        let samples_clone = Arc::clone(&samples);
        
        let err_fn = |err| eprintln!("Audio stream error: {}", err);
        
        let stream = self.device.build_input_stream(
            &self.config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                samples_clone.lock().unwrap().extend_from_slice(data);
            },
            err_fn,
            None,
        )?;
        
        stream.play()?;
        eprintln!("🔴 Recording for {} seconds...", duration_secs);
        
        tokio::time::sleep(Duration::from_secs(duration_secs as u64)).await;
        
        drop(stream);
        eprintln!("⏹️ Recording stopped");
        
        let result = samples.lock().unwrap().clone();
        Ok(result)
    }
    
    /// List available input devices
    pub fn list_devices() -> Result<Vec<String>> {
        let host = cpal::default_host();
        let devices: Vec<String> = host.input_devices()?
            .filter_map(|d| d.name().ok())
            .collect();
        Ok(devices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_list_devices() {
        // This test just verifies the API works, not that devices exist
        let result = AudioCapture::list_devices();
        assert!(result.is_ok());
    }
}
