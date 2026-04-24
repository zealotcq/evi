//! Audio capture abstractions and platform implementations.

pub mod cpal_source;

use crate::AudioSource;

/// Creates the platform-appropriate audio source.
/// Uses `cpal` which abstracts over WASAPI (Windows), CoreAudio (macOS), and ALSA/PulseAudio (Linux).
pub fn create_audio_source(sample_rate: u32) -> anyhow::Result<Box<dyn AudioSource>> {
    Ok(Box::new(cpal_source::CpalAudioSource::new(sample_rate)?))
}
