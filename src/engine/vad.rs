//! FSMN-VAD ONNX model wrapper with optional adaptive energy gate.
//!
//! Loads the FunASR VAD model (`iic/speech_fsmn_vad_zh-cn-16k-common-onnx`)
//! and detects speech / non-speech segments in audio.
//!
//! When the adaptive energy gate is enabled, frames with RMS energy below
//! `reference_energy - db_offset` are discarded even if VAD says speech.

use anyhow::{Context, Result};
use log::{debug, warn};
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

use super::features::{apply_cmvn, compute_fbank_f32, Cmvn, FbankConfig};

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub start_ms: u64,
    pub end_ms: u64,
}

// ── Adaptive Energy Gate ───────────────────────────────────────────────────────

struct EnergyGate {
    enabled: bool,
    db_offset: f64,
    reference_energy: Option<f64>,
    init_samples: Vec<f64>,
    min_init_frames: usize,
    frames_below_threshold: usize,
}

impl EnergyGate {
    fn new(enabled: bool, db_offset: f64) -> Self {
        Self {
            enabled,
            db_offset,
            reference_energy: None,
            init_samples: Vec::new(),
            min_init_frames: 20,
            frames_below_threshold: 0,
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.reference_energy = None;
            self.init_samples.clear();
            self.frames_below_threshold = 0;
        }
    }

    fn begin_utterance(&mut self) {
        self.init_samples.clear();
        self.frames_below_threshold = 0;
    }

    fn process_frame(&mut self, energy: f64, speech_prob: f32) -> bool {
        if !self.enabled {
            return true;
        }

        if energy <= 0.0 {
            return false;
        }

        if let Some(ref_e) = self.reference_energy {
            let threshold = ref_e * 10f64.powf(-self.db_offset / 10.0);

            if energy >= threshold {
                if speech_prob >= 0.7 && energy < ref_e * 8.0 {
                    let alpha = if energy > ref_e * 2.0 {
                        0.03
                    } else if energy > ref_e {
                        0.05
                    } else {
                        0.12
                    };
                    self.reference_energy = Some(alpha * energy + (1.0 - alpha) * ref_e);
                }
                self.frames_below_threshold = 0;
                return true;
            } else {
                self.frames_below_threshold += 1;
                if self.frames_below_threshold >= 50 {
                    self.reference_energy = Some(ref_e * 0.92);
                    self.frames_below_threshold = 0;
                    debug!(
                        "EnergyGate: auto-relax reference to {:.6}",
                        ref_e * 0.92
                    );
                }
                return false;
            }
        }

        if speech_prob >= 0.8 {
            self.init_samples.push(energy);
            if self.init_samples.len() >= self.min_init_frames {
                let mut sorted = self.init_samples.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mid = sorted.len() / 2;
                self.reference_energy = Some(sorted[mid]);
                debug!(
                    "EnergyGate: initialized reference_energy={:.6} from {} samples",
                    sorted[mid],
                    self.init_samples.len()
                );
            }
        }

        true
    }
}

// ── VAD Engine ─────────────────────────────────────────────────────────────────

pub struct VadEngine {
    session: Session,
    cmvn: Option<Cmvn>,
    config: FbankConfig,
    energy_gate: EnergyGate,
}

impl VadEngine {
    pub fn new(model_dir: &Path, energy_gate_enabled: bool, energy_gate_db_offset: f64) -> Result<Self> {
        let model_path = if model_dir.join("model_quant.onnx").exists() {
            model_dir.join("model_quant.onnx")
        } else {
            model_dir.join("model.onnx")
        };

        if !model_path.exists() {
            anyhow::bail!("VAD model not found: {}", model_path.display());
        }

        let session = Session::builder()
            .with_context(|| "Failed to create ONNX session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load VAD model: {}", model_path.display()))?;

        log::debug!("VadEngine: loaded from {}", model_path.display());

        for input in session.inputs() {
            debug!("VAD input: {:?}", input);
        }
        for output in session.outputs() {
            debug!("VAD output: {:?}", output);
        }

        let cmvn_path = model_dir.join("am.mvn");
        let cmvn = if cmvn_path.exists() {
            match Cmvn::from_file(&cmvn_path) {
                Ok(c) => {
                    debug!("VAD: loaded CMVN (dim={})", c.dim);
                    Some(c)
                }
                Err(e) => {
                    warn!("VAD: failed to load CMVN: {e}, skipping normalization");
                    None
                }
            }
        } else {
            debug!("VAD: no CMVN file found, skipping normalization");
            None
        };

        if energy_gate_enabled {
            debug!(
                "VadEngine: adaptive energy gate enabled (db_offset={:.1})",
                energy_gate_db_offset
            );
        }

        Ok(Self {
            session,
            cmvn,
            config: FbankConfig::default(),
            energy_gate: EnergyGate::new(energy_gate_enabled, energy_gate_db_offset),
        })
    }

    pub fn set_energy_gate_enabled(&mut self, enabled: bool) {
        self.energy_gate.set_enabled(enabled);
    }

    pub fn detect(&mut self, pcm: &[f32]) -> Result<Vec<SpeechSegment>> {
        if pcm.is_empty() {
            return Ok(vec![]);
        }

        let fbank = compute_fbank_f32(pcm, &self.config);
        let num_frames = fbank.nrows();
        if num_frames == 0 {
            return Ok(vec![]);
        }

        let lfr_m = 5usize;
        let lfr_n = 1usize;
        let feat_dim = self.config.num_mel_bins;
        let lfr_dim = feat_dim * lfr_m;

        let mut lfr_frames = apply_lfr(&fbank, lfr_m, lfr_n);
        if let Some(ref cmvn) = self.cmvn {
            apply_cmvn(&mut lfr_frames, cmvn);
        }
        let num_lfr = lfr_frames.nrows();

        let mut input_data = Vec::with_capacity(num_lfr * lfr_dim);
        for row in lfr_frames.rows() {
            for &val in row {
                input_data.push(val as f32);
            }
        }
        let input_tensor = ndarray::Array3::from_shape_vec((1, num_lfr, lfr_dim), input_data)
            .context("Failed to create VAD input tensor")?;

        let input_value = Tensor::from_array(input_tensor)?;

        let cache0 = ndarray::Array4::<f32>::zeros((1, 128, 19, 1));
        let cache1 = ndarray::Array4::<f32>::zeros((1, 128, 19, 1));
        let cache2 = ndarray::Array4::<f32>::zeros((1, 128, 19, 1));
        let cache3 = ndarray::Array4::<f32>::zeros((1, 128, 19, 1));

        let outputs = self
            .session
            .run(ort::inputs! {
                "speech" => input_value,
                "in_cache0" => Tensor::from_array(cache0)?,
                "in_cache1" => Tensor::from_array(cache1)?,
                "in_cache2" => Tensor::from_array(cache2)?,
                "in_cache3" => Tensor::from_array(cache3)?,
            })
            .with_context(|| {
                format!(
                    "VAD inference failed, input shape: [1, {}, {}]",
                    num_lfr, lfr_dim
                )
            })?;

        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract VAD output")?;

        let frame_shift_ms = self.config.frame_shift_ms as f64;
        let threshold = 0.5f32;
        let num_classes = 248usize;
        let num_time = logits_data.len() / num_classes;

        let frame_energies = compute_vad_frame_energies(pcm, &self.config);
        self.energy_gate.begin_utterance();

        let mut segments = Vec::new();
        let mut in_speech = false;
        let mut seg_start: u64 = 0;

        for i in 0..num_time {
            let sil_prob = logits_data[i * num_classes];
            let speech_prob = 1.0 - sil_prob;
            let time_ms = (i as f64 * frame_shift_ms) as u64;

            let frame_energy = frame_energies.get(i).copied().unwrap_or(0.0);
            let energy_ok = self.energy_gate.process_frame(frame_energy, speech_prob);

            let effective_speech = speech_prob >= threshold && energy_ok;

            if effective_speech && !in_speech {
                in_speech = true;
                seg_start = time_ms;
            } else if !effective_speech && in_speech {
                in_speech = false;
                segments.push(SpeechSegment {
                    start_ms: seg_start,
                    end_ms: time_ms,
                });
            }
        }
        if in_speech {
            let total_ms = (pcm.len() as f64 / self.config.sample_rate as f64 * 1000.0) as u64;
            segments.push(SpeechSegment {
                start_ms: seg_start,
                end_ms: total_ms,
            });
        }

        let segments = merge_close_segments(segments, 300);

        debug!("VAD: detected {} speech segments", segments.len());
        Ok(segments)
    }
}

fn compute_vad_frame_energies(pcm: &[f32], config: &FbankConfig) -> Vec<f64> {
    let frame_shift = config.frame_shift_samples();
    let frame_len = config.frame_length_samples();
    let num_frames = if pcm.len() > frame_len {
        (pcm.len() - frame_len) / frame_shift + 1
    } else {
        1
    };

    let mut energies = Vec::with_capacity(num_frames);
    for i in 0..num_frames {
        let start = i * frame_shift;
        let end = (start + frame_len).min(pcm.len());
        let rms: f64 = pcm[start..end]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / (end - start) as f64;
        energies.push(rms);
    }
    energies
}

fn merge_close_segments(segments: Vec<SpeechSegment>, min_gap_ms: u64) -> Vec<SpeechSegment> {
    if segments.len() <= 1 {
        return segments;
    }
    let mut merged = Vec::new();
    let mut current = segments[0].clone();
    for seg in &segments[1..] {
        if seg.start_ms.saturating_sub(current.end_ms) <= min_gap_ms {
            current.end_ms = seg.end_ms;
        } else {
            merged.push(current);
            current = seg.clone();
        }
    }
    merged.push(current);
    merged
}

fn apply_lfr(fbank: &ndarray::Array2<f64>, lfr_m: usize, lfr_n: usize) -> ndarray::Array2<f64> {
    let num_frames = fbank.nrows();
    let feat_dim = fbank.ncols();
    let lfr_dim = feat_dim * lfr_m;

    let left_pad = (lfr_m - 1) / 2;
    let mut padded: Vec<Vec<f64>> = Vec::with_capacity(num_frames + left_pad);
    let first_row: Vec<f64> = fbank.row(0).to_vec();
    for _ in 0..left_pad {
        padded.push(first_row.clone());
    }
    for i in 0..num_frames {
        padded.push(fbank.row(i).to_vec());
    }

    let padded_len = padded.len();
    let num_lfr = (num_frames as f64 / lfr_n as f64).ceil() as usize;
    let last_row: Vec<f64> = fbank.row(num_frames - 1).to_vec();

    let mut lfr_feats = ndarray::Array2::zeros((num_lfr, lfr_dim));
    for i in 0..num_lfr {
        let start = i * lfr_n;
        for m in 0..lfr_m {
            let idx = start + m;
            if idx < padded_len {
                for d in 0..feat_dim {
                    lfr_feats[[i, m * feat_dim + d]] = padded[idx][d];
                }
            } else {
                for d in 0..feat_dim {
                    lfr_feats[[i, m * feat_dim + d]] = last_row[d];
                }
            }
        }
    }
    lfr_feats
}
