//! Fbank (log Mel filterbank) feature extraction.
//!
//! Reproduces the same preprocessing FunASR uses:
//!   i16 PCM → f32 [-1,1] → pre-emphasis → framing → Hamming → FFT → mel → log → CMVN
//!
//! Parameters match FunASR defaults:
//!   frame_length = 25 ms, frame_shift = 10 ms, fft_size = 512, num_mel_bins = 80

use anyhow::{bail, Context, Result};
use log::debug;
use ndarray::Array1;
use ndarray::Array2;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use std::path::Path;

// ── Config ─────────────────────────────────────────────────────────────────────

pub struct FbankConfig {
    pub sample_rate: u32,
    pub frame_length_ms: u32,
    pub frame_shift_ms: u32,
    pub fft_size: usize,
    pub num_mel_bins: usize,
    pub pre_emphasis_coeff: f64,
    pub mel_floor: f64,
}

impl Default for FbankConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            frame_length_ms: 25,
            frame_shift_ms: 10,
            fft_size: 512,
            num_mel_bins: 80,
            pre_emphasis_coeff: 0.97,
            mel_floor: 1e-10,
        }
    }
}

impl FbankConfig {
    pub fn frame_length_samples(&self) -> usize {
        (self.sample_rate as usize * self.frame_length_ms as usize) / 1000
    }

    pub fn frame_shift_samples(&self) -> usize {
        (self.sample_rate as usize * self.frame_shift_ms as usize) / 1000
    }
}

// ── CMVN ───────────────────────────────────────────────────────────────────────

/// Cepstral Mean and Variance Normalization (Kaldi Nnet format).
///
/// FunASR `am.mvn` stores `<AddShift>` (means) and `<Rescale>` (vars).
/// Application: `(feat + means) * vars`
pub struct Cmvn {
    pub means: Array1<f64>,
    pub vars: Array1<f64>,
    pub dim: usize,
}

impl Cmvn {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read CMVN file: {}", path.display()))?;
        Self::parse_kaldi_nnet(&text)
    }

    fn parse_kaldi_nnet(text: &str) -> Result<Self> {
        let lines: Vec<&str> = text.lines().collect();
        let mut means_list: Vec<f64> = Vec::new();
        let mut vars_list: Vec<f64> = Vec::new();

        for i in 0..lines.len() {
            let parts: Vec<&str> = lines[i].split_whitespace().collect();
            if parts.first() == Some(&"<AddShift>") {
                if i + 1 < lines.len() {
                    let next_parts: Vec<&str> = lines[i + 1].split_whitespace().collect();
                    if next_parts.first() == Some(&"<LearnRateCoef>") {
                        let bracket_start =
                            next_parts.iter().position(|&p| p == "[").unwrap_or(0) + 1;
                        let bracket_end = next_parts
                            .iter()
                            .position(|&p| p == "]")
                            .unwrap_or(next_parts.len());
                        means_list = next_parts[bracket_start..bracket_end]
                            .iter()
                            .filter_map(|v| v.parse::<f64>().ok())
                            .collect();
                    }
                }
            } else if parts.first() == Some(&"<Rescale>") && i + 1 < lines.len() {
                let next_parts: Vec<&str> = lines[i + 1].split_whitespace().collect();
                if next_parts.first() == Some(&"<LearnRateCoef>") {
                    let bracket_start = next_parts.iter().position(|&p| p == "[").unwrap_or(0) + 1;
                    let bracket_end = next_parts
                        .iter()
                        .position(|&p| p == "]")
                        .unwrap_or(next_parts.len());
                    vars_list = next_parts[bracket_start..bracket_end]
                        .iter()
                        .filter_map(|v| v.parse::<f64>().ok())
                        .collect();
                }
            }
        }

        if means_list.is_empty() || vars_list.is_empty() {
            bail!("CMVN: failed to parse <AddShift> or <Rescale> from am.mvn");
        }
        if means_list.len() != vars_list.len() {
            bail!(
                "CMVN: means len {} != vars len {}",
                means_list.len(),
                vars_list.len()
            );
        }

        let dim = means_list.len();
        Ok(Self {
            means: Array1::from_vec(means_list),
            vars: Array1::from_vec(vars_list),
            dim,
        })
    }

    /// Apply CMVN: `(feat + means) * vars`, matching FunASR's apply_cmvn.
    pub fn apply(&self, features: &mut Array2<f64>) {
        if features.ncols() != self.dim {
            debug!(
                "CMVN dim mismatch: features={}, cmvn={}, skipping",
                features.ncols(),
                self.dim
            );
            return;
        }
        for row in 0..features.nrows() {
            for col in 0..self.dim {
                features[[row, col]] = (features[[row, col]] + self.means[col]) * self.vars[col];
            }
        }
    }
}

// ── Fbank computation ──────────────────────────────────────────────────────────

/// Compute log Mel filterbank features from raw i16 PCM.
///
/// Internally scales to int16 range to match kaldi_native_fbank behavior.
pub fn compute_fbank(pcm_i16: &[i16], cfg: &FbankConfig) -> Array2<f64> {
    let pcm: Vec<f64> = pcm_i16.iter().map(|&s| s as f64).collect();
    compute_fbank_f64(&pcm, cfg)
}

/// Compute fbank from f32 PCM (normalized to [-1, 1]).
///
/// Scales by `1 << 15` to match kaldi_native_fbank which expects int16-range input.
pub fn compute_fbank_f32(pcm_f32: &[f32], cfg: &FbankConfig) -> Array2<f64> {
    let pcm: Vec<f64> = pcm_f32.iter().map(|&s| (s as f64) * 32768.0).collect();
    compute_fbank_f64(&pcm, cfg)
}

fn compute_fbank_f64(pcm: &[f64], cfg: &FbankConfig) -> Array2<f64> {
    let pcm = pre_emphasis(pcm, cfg.pre_emphasis_coeff);
    let frame_len = cfg.frame_length_samples();
    let frame_shift = cfg.frame_shift_samples();
    let num_frames = if pcm.len() > frame_len {
        (pcm.len() - frame_len) / frame_shift + 1
    } else {
        1
    };
    let hamming = make_hamming(frame_len);
    let mel_fb = make_mel_filterbank(cfg.num_mel_bins, cfg.fft_size, cfg.sample_rate);
    let fft_size = cfg.fft_size;

    let mut features = Array2::zeros((num_frames, cfg.num_mel_bins));

    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    for i in 0..num_frames {
        let start = i * frame_shift;
        let _end = (start + frame_len).min(pcm.len());

        let mut frame_buf: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); fft_size];
        for j in 0..frame_len {
            if start + j < pcm.len() {
                frame_buf[j] = Complex::new(pcm[start + j] * hamming[j], 0.0);
            }
        }

        fft.process(&mut frame_buf);

        // Power spectrum
        let power: Vec<f64> = frame_buf[..fft_size / 2 + 1]
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im) / fft_size as f64)
            .collect();

        // Apply mel filterbank
        for mel_idx in 0..cfg.num_mel_bins {
            let mut val = 0.0f64;
            for (k, &p) in power.iter().enumerate() {
                val += mel_fb[[mel_idx, k]] * p;
            }
            features[[i, mel_idx]] = (val + cfg.mel_floor).ln();
        }
    }

    features
}

/// Apply CMVN normalization to fbank features.
pub fn apply_cmvn(features: &mut Array2<f64>, cmvn: &Cmvn) {
    cmvn.apply(features);
}

fn pre_emphasis(samples: &[f64], coeff: f64) -> Vec<f64> {
    if samples.is_empty() {
        return vec![];
    }
    let mut out = Vec::with_capacity(samples.len());
    out.push(samples[0]);
    for i in 1..samples.len() {
        out.push(samples[i] - coeff * samples[i - 1]);
    }
    out
}

fn make_hamming(len: usize) -> Vec<f64> {
    (0..len)
        .map(|n| 0.54 - 0.46 * (2.0 * PI * n as f64 / (len - 1) as f64).cos())
        .collect()
}

/// Create a mel filterbank matrix: `[num_mel_bins, fft_size/2 + 1]`.
fn make_mel_filterbank(num_mel_bins: usize, fft_size: usize, sample_rate: u32) -> Array2<f64> {
    let num_fft_bins = fft_size / 2 + 1;
    let lo_hz = 80.0f64;
    let hi_hz = sample_rate as f64 / 2.0;
    let lo_mel = hz_to_mel(lo_hz);
    let hi_mel = hz_to_mel(hi_hz);

    let center_mels: Vec<f64> = (0..num_mel_bins + 2)
        .map(|i| lo_mel + (hi_mel - lo_mel) * i as f64 / (num_mel_bins + 1) as f64)
        .collect();

    let center_hz: Vec<f64> = center_mels.iter().map(|&m| mel_to_hz(m)).collect();

    let mut fb = Array2::zeros((num_mel_bins, num_fft_bins));
    let fft_bin_width = sample_rate as f64 / fft_size as f64;

    for mel_idx in 0..num_mel_bins {
        let f_left = center_hz[mel_idx];
        let f_center = center_hz[mel_idx + 1];
        let f_right = center_hz[mel_idx + 2];

        for bin in 0..num_fft_bins {
            let freq = bin as f64 * fft_bin_width;
            if freq >= f_left && freq <= f_center && f_center > f_left {
                fb[[mel_idx, bin]] = (freq - f_left) / (f_center - f_left);
            } else if freq > f_center && freq <= f_right && f_right > f_center {
                fb[[mel_idx, bin]] = (f_right - freq) / (f_right - f_center);
            }
        }
    }

    fb
}

#[inline]
fn hz_to_mel(hz: f64) -> f64 {
    1127.0 * (1.0 + hz / 700.0).ln()
}

#[inline]
fn mel_to_hz(mel: f64) -> f64 {
    700.0 * ((mel / 1127.0).exp() - 1.0)
}
