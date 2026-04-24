use crate::{AudioFrame, AudioSource};
use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{Receiver, Sender};
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

pub struct CpalAudioSource {
    target_sample_rate: u32,
    recording: Mutex<bool>,
    buffer: Arc<Mutex<Vec<i16>>>,
    frame_tx: Sender<AudioFrame>,
    frame_rx: Receiver<AudioFrame>,
    stream: Rc<RefCell<Option<cpal::Stream>>>,
}

unsafe impl Send for CpalAudioSource {}
unsafe impl Sync for CpalAudioSource {}

impl CpalAudioSource {
    pub fn new(sample_rate: u32) -> Result<Self> {
        let (tx, rx) = crossbeam_channel::bounded(200);
        Ok(Self {
            target_sample_rate: sample_rate,
            recording: Mutex::new(false),
            buffer: Arc::new(Mutex::new(Vec::new())),
            frame_tx: tx,
            frame_rx: rx,
            stream: Rc::new(RefCell::new(None)),
        })
    }

    fn build_stream(&self) -> Result<cpal::Stream> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No default input device available")?;

        info!("Audio device: {}", device.name().unwrap_or_default());

        let configs: Vec<_> = device
            .supported_input_configs()
            .context("Failed to query supported input configs")?
            .filter(|c| {
                c.channels() >= 1
                    && matches!(
                        c.sample_format(),
                        cpal::SampleFormat::I16 | cpal::SampleFormat::F32
                    )
            })
            .collect();

        if configs.is_empty() {
            bail!("No supported audio input configs found");
        }

        let config = configs
            .iter()
            .find(|c| {
                c.min_sample_rate().0 <= self.target_sample_rate
                    && c.max_sample_rate().0 >= self.target_sample_rate
            })
            .or_else(|| configs.iter().find(|c| c.max_sample_rate().0 >= 16000))
            .or_else(|| configs.first())
            .context("No usable audio config found")?;

        let actual_sample_rate = config
            .max_sample_rate()
            .0
            .min(self.target_sample_rate)
            .max(config.min_sample_rate().0);
        let channels = config.channels();

        let cpal_config = cpal::StreamConfig {
            channels,
            sample_rate: cpal::SampleRate(actual_sample_rate),
            buffer_size: cpal::BufferSize::Default,
        };

        info!(
            "Audio config: {}Hz, {}ch (target {}Hz)",
            actual_sample_rate, channels, self.target_sample_rate
        );

        let sample_format = config.sample_format();
        let tx = self.frame_tx.clone();
        let buffer = self.buffer.clone();
        let target_sr = self.target_sample_rate;
        let actual_sr = actual_sample_rate;
        let num_channels = channels;

        let stream = match sample_format {
            cpal::SampleFormat::I16 => device.build_input_stream(
                &cpal_config,
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    let mono = downmix_i16(data, num_channels);
                    let resampled = resample_if_needed(&mono, actual_sr, target_sr);
                    let frame = AudioFrame {
                        samples: resampled.clone(),
                        timestamp_us: now_us(),
                    };
                    let _ = tx.try_send(frame);
                    buffer.lock().extend_from_slice(&resampled);
                },
                |err| error!("Audio capture error: {err}"),
                None,
            ),
            cpal::SampleFormat::F32 => device.build_input_stream(
                &cpal_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    let mono = downmix_f32(data, num_channels);
                    let i16_data: Vec<i16> = mono
                        .iter()
                        .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
                        .collect();
                    let resampled = resample_if_needed(&i16_data, actual_sr, target_sr);
                    let frame = AudioFrame {
                        samples: resampled.clone(),
                        timestamp_us: now_us(),
                    };
                    let _ = tx.try_send(frame);
                    buffer.lock().extend_from_slice(&resampled);
                },
                |err| error!("Audio capture error: {err}"),
                None,
            ),
            _ => bail!("Unsupported sample format: {sample_format:?}"),
        }
        .context("Failed to build audio input stream")?;

        Ok(stream)
    }
}

fn now_us() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

fn downmix_i16(data: &[i16], channels: u16) -> Vec<i16> {
    if channels <= 1 {
        return data.to_vec();
    }
    let ch = channels as usize;
    data.chunks(ch)
        .map(|frame| {
            let sum: i64 = frame.iter().map(|&s| s as i64).sum();
            (sum / ch as i64) as i16
        })
        .collect()
}

fn downmix_f32(data: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return data.to_vec();
    }
    let ch = channels as usize;
    data.chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

fn resample_if_needed(samples: &[i16], from_rate: u32, to_rate: u32) -> Vec<i16> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio) as usize;
    let mut result = Vec::with_capacity(output_len);
    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let s0 = samples[idx.min(samples.len() - 1)] as f64;
        let s1 = samples[(idx + 1).min(samples.len() - 1)] as f64;
        result.push((s0 * (1.0 - frac) + s1 * frac) as i16);
    }
    result
}

impl AudioSource for CpalAudioSource {
    fn start(&mut self) -> Result<()> {
        if *self.recording.lock() {
            warn!("Already recording, ignoring start");
            return Ok(());
        }

        self.buffer.lock().clear();
        while self.frame_rx.try_recv().is_ok() {}

        let stream = self.build_stream()?;
        stream.play().context("Failed to start audio stream")?;

        *self.stream.borrow_mut() = Some(stream);
        *self.recording.lock() = true;
        debug!("Audio capture started");
        Ok(())
    }

    fn stop(&mut self) -> Result<Vec<i16>> {
        *self.recording.lock() = false;
        *self.stream.borrow_mut() = None;
        let samples: Vec<i16> = self.buffer.lock().drain(..).collect();
        debug!("Audio capture stopped, collected {} samples", samples.len());
        Ok(samples)
    }

    fn is_recording(&self) -> bool {
        *self.recording.lock()
    }

    fn sample_rate(&self) -> u32 {
        self.target_sample_rate
    }

    fn receiver(&self) -> &Receiver<AudioFrame> {
        &self.frame_rx
    }
}
