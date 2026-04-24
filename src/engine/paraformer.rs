//! Paraformer ASR ONNX model wrapper.
//!
//! Loads the FunASR Paraformer model
//! (`iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx`)
//! and recognizes speech with per-character timing.

use anyhow::{bail, Context, Result};
use log::{debug, warn};
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;

use super::features::{apply_cmvn, compute_fbank_f32, Cmvn, FbankConfig};
use crate::{CharTiming, TokenScore};

pub struct AsrEngine {
    session: Session,
    vocab: HashMap<u32, String>,
    cmvn: Option<Cmvn>,
    config: FbankConfig,
}

impl AsrEngine {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_path = if model_dir.join("model_quant.onnx").exists() {
            model_dir.join("model_quant.onnx")
        } else {
            model_dir.join("model.onnx")
        };

        if !model_path.exists() {
            bail!("ASR model not found: {}", model_path.display());
        }

        let session = Session::builder()
            .with_context(|| "Failed to create ONNX session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ASR model: {}", model_path.display()))?;

        debug!("AsrEngine: loaded from {}", model_path.display());

        for input in session.inputs() {
            debug!("ASR input: {:?}", input);
        }
        for output in session.outputs() {
            debug!("ASR output: {:?}", output);
        }

        let vocab_path = model_dir.join("tokens.txt");
        let vocab_json_path = model_dir.join("tokens.json");
        let vocab = if vocab_path.exists() {
            load_vocab_text(&vocab_path)?
        } else if vocab_json_path.exists() {
            load_vocab_json(&vocab_json_path)?
        } else {
            bail!(
                "No tokens.txt or tokens.json found in {}",
                model_dir.display()
            );
        };
        debug!("ASR: loaded {} vocabulary tokens", vocab.len());

        let cmvn_path = model_dir.join("am.mvn");
        let cmvn = if cmvn_path.exists() {
            match Cmvn::from_file(&cmvn_path) {
                Ok(c) => {
                    debug!("ASR: loaded CMVN (dim={})", c.dim);
                    Some(c)
                }
                Err(e) => {
                    warn!("ASR: failed to load CMVN: {e}, skipping normalization");
                    None
                }
            }
        } else {
            debug!("ASR: no CMVN file found, skipping normalization");
            None
        };

        Ok(Self {
            session,
            vocab,
            cmvn,
            config: FbankConfig::default(),
        })
    }

    pub fn recognize(
        &mut self,
        pcm: &[f32],
        audio_start_ms: u64,
    ) -> Result<(String, Vec<CharTiming>, Vec<TokenScore>)> {
        if pcm.is_empty() {
            return Ok((String::new(), vec![], vec![]));
        }

        let fbank = compute_fbank_f32(pcm, &self.config);

        let num_frames = fbank.nrows();
        if num_frames == 0 {
            return Ok((String::new(), vec![], vec![]));
        }

        let lfr_m = 7usize;
        let lfr_n = 6usize;
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
            .context("Failed to create ASR input tensor")?;

        let input_value = Tensor::from_array(input_tensor)?;
        let speech_lengths = ndarray::Array1::from_vec(vec![num_lfr as i32]);
        let outputs = self
            .session
            .run(ort::inputs! {
                "speech" => input_value,
                "speech_lengths" => Tensor::from_array(speech_lengths)?,
            })
            .with_context(|| {
                format!(
                    "ASR inference failed, input shape: [1, {}, {}]",
                    num_lfr, lfr_dim
                )
            })?;

        let (logits_shape, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract ASR logits")?;

        let num_tokens = if outputs.len() > 1 {
            if let Ok((_, tn_data)) = outputs[1].try_extract_tensor::<i32>() {
                if !tn_data.is_empty() {
                    tn_data[0] as usize
                } else {
                    logits_shape[1] as usize
                }
            } else {
                logits_shape[1] as usize
            }
        } else {
            logits_shape[1] as usize
        };

        let vocab_size = logits_shape[2] as usize;
        let decoded = argmax_decode(logits_data, num_tokens, vocab_size);
        let token_ids: Vec<u32> = decoded.iter().map(|(id, _)| *id).collect();
        let text = ids_to_text(&token_ids, &self.vocab);

        let tokens: Vec<TokenScore> = decoded
            .iter()
            .filter_map(|(id, conf)| {
                self.vocab.get(id).map(|t| TokenScore {
                    token: t.clone(),
                    confidence: *conf,
                })
            })
            .filter(|t| !t.token.starts_with('<') || !t.token.ends_with('>'))
            .collect();

        let chars = compute_char_timing(
            &text,
            audio_start_ms,
            audio_start_ms + (pcm.len() as f64 / self.config.sample_rate as f64 * 1000.0) as u64,
        );

        debug!(
            "ASR: recognized \"{}\" ({} chars, {} tokens)",
            text,
            text.len(),
            num_tokens
        );
        Ok((text, chars, tokens))
    }
}

fn argmax_decode(logits: &[f32], num_tokens: usize, vocab_size: usize) -> Vec<(u32, f32)> {
    let mut ids: Vec<(u32, f32)> = Vec::new();

    if vocab_size == 0 || num_tokens == 0 {
        return ids;
    }

    for t in 0..num_tokens {
        let offset = t * vocab_size;
        if offset + vocab_size > logits.len() {
            break;
        }
        let mut best_id = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        let mut min_val = f32::INFINITY;
        let mut sum: f32 = 0.0;
        for v in 0..vocab_size {
            let val = logits[offset + v];
            if val > best_val {
                best_val = val;
                best_id = v as u32;
            }
            if val < min_val {
                min_val = val;
            }
            sum += val;
        }
        let shifted_sum = sum - min_val * vocab_size as f32;
        let confidence = if shifted_sum > 0.0 {
            (best_val - min_val) / shifted_sum
        } else {
            1.0
        };
        ids.push((best_id, confidence));
    }

    let mut filtered: Vec<(u32, f32)> = vec![(1, 1.0)];
    filtered.extend_from_slice(&ids);
    filtered.push((2, 1.0));

    filtered
        .into_iter()
        .filter(|(id, _)| *id != 0 && *id != 2)
        .collect()
}

fn ids_to_text(ids: &[u32], vocab: &HashMap<u32, String>) -> String {
    let tokens: Vec<&str> = ids
        .iter()
        .filter_map(|&id| vocab.get(&id).map(|s| s.as_str()))
        .filter(|t| !t.starts_with('<') || !t.ends_with('>'))
        .collect();

    let mut text = String::new();
    let mut i = 0;
    while i < tokens.len() {
        let mut word = String::from(tokens[i]);
        while word.ends_with("@@") && i + 1 < tokens.len() {
            word = word[..word.len() - 2].to_string();
            i += 1;
            word.push_str(tokens[i]);
        }
        text.push_str(&word);
        i += 1;
    }
    text
}

fn load_vocab_text(path: &Path) -> Result<HashMap<u32, String>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read vocab: {}", path.display()))?;
    let mut vocab = HashMap::new();
    for (id, line) in text.lines().enumerate() {
        let token = line.trim().to_string();
        if !token.is_empty() {
            vocab.insert(id as u32, token);
        }
    }
    Ok(vocab)
}

fn load_vocab_json(path: &Path) -> Result<HashMap<u32, String>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read vocab: {}", path.display()))?;
    let tokens: Vec<String> = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse vocab JSON: {}", path.display()))?;
    let mut vocab = HashMap::new();
    for (id, token) in tokens.into_iter().enumerate() {
        vocab.insert(id as u32, token);
    }
    Ok(vocab)
}

fn compute_char_timing(text: &str, audio_start_ms: u64, audio_end_ms: u64) -> Vec<CharTiming> {
    if text.is_empty() || audio_start_ms >= audio_end_ms {
        return vec![];
    }

    let chars: Vec<&str> = text.graphemes(true).collect();
    let n = chars.len();
    if n == 0 {
        return vec![];
    }

    let duration_ms = audio_end_ms - audio_start_ms;
    let char_duration = duration_ms as f64 / n as f64;

    chars
        .iter()
        .enumerate()
        .map(|(i, &ch)| {
            let start = audio_start_ms + (i as f64 * char_duration) as u64;
            let end = audio_start_ms + ((i + 1) as f64 * char_duration) as u64;
            CharTiming {
                text: ch.to_string(),
                start_ms: start,
                end_ms: end,
            }
        })
        .collect()
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
