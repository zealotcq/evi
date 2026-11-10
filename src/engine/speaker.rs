//! CAM++ speaker embedding engine (ONNX, sherpa-onnx export).
//!
//! Model: speech_campplus_sv_zh_en_16k-common_advanced.onnx
//! Input: float32 fbank [N, T, 80] @ 16kHz, 25ms/10ms frames
//! Output: L2-ish speaker embedding [N, 192]; cosine similarity for scoring.

use anyhow::{Context, Result};
use log::debug;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

use super::features::{compute_fbank_f32, FbankConfig};

pub fn default_model_path() -> std::path::PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    std::path::PathBuf::from(home)
        .join(".cache/modelscope/hub/models/echo0hello")
        .join("speech_campplus_sv_zh_en_16k-common_advanced.onnx")
}

pub struct SpeakerEngine {
    session: Session,
    config: FbankConfig,
}

impl SpeakerEngine {
    pub fn new(model_path: &Path) -> Result<Self> {
        if !model_path.exists() {
            anyhow::bail!("Speaker model not found: {}", model_path.display());
        }
        let session = Session::builder()
            .with_context(|| "Failed to create speaker ONNX session builder")?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load speaker model: {}", model_path.display()))?;

        for input in session.inputs() {
            debug!("Speaker input: {:?}", input);
        }
        for output in session.outputs() {
            debug!("Speaker output: {:?}", output);
        }

        Ok(Self {
            session,
            config: FbankConfig::default(),
        })
    }

    pub fn embed(&mut self, pcm: &[f32]) -> Result<Vec<f32>> {
        if pcm.is_empty() {
            anyhow::bail!("empty audio for speaker embedding");
        }
        let fbank = compute_fbank_f32(pcm, &self.config);
        if fbank.nrows() == 0 {
            anyhow::bail!("no fbank frames for speaker embedding");
        }
        let (t, d) = (fbank.nrows(), fbank.ncols());
        let data: Vec<f32> = fbank.iter().map(|&v| v as f32).collect();
        let input = ndarray::Array3::from_shape_vec((1, t, d), data)
            .context("Failed to build speaker input tensor")?;

        let input_name = self.session.inputs()[0].name().to_string();
        let outputs = self
            .session
            .run(ort::inputs! { (input_name) => Tensor::from_array(input)? })
            .context("Speaker inference failed")?;

        let (_, emb) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract speaker embedding")?;
        Ok(emb.to_vec())
    }

    pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len().min(b.len());
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        for i in 0..n {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64).powi(2);
            nb += (b[i] as f64).powi(2);
        }
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }
}
