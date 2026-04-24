//! CT-Transformer Punctuator ONNX model wrapper.
//!
//! Loads the FunASR punctuation model
//! (`iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx`)
//! and restores punctuation in text.

use anyhow::{bail, Context, Result};
use log::debug;
use ort::session::Session;
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use unicode_segmentation::UnicodeSegmentation;

pub struct PuncEngine {
    session: Session,
    token_to_id: HashMap<String, u32>,
    punc_list: Vec<String>,
}

impl PuncEngine {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_path = if model_dir.join("model_quant.onnx").exists() {
            model_dir.join("model_quant.onnx")
        } else {
            model_dir.join("model.onnx")
        };

        if !model_path.exists() {
            bail!("Punc model not found: {}", model_path.display());
        }

        let session = Session::builder()
            .with_context(|| "Failed to create ONNX session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load Punc model: {}", model_path.display()))?;

        debug!("PuncEngine: loaded from {}", model_path.display());

        for input in session.inputs() {
            debug!("Punc input: {:?}", input);
        }
        for output in session.outputs() {
            debug!("Punc output: {:?}", output);
        }

        // Load tokens.json
        let tokens_path = model_dir.join("tokens.json");
        let token_to_id = if tokens_path.exists() {
            load_tokens_json(&tokens_path)?
        } else {
            bail!("tokens.json not found in {}", model_dir.display())
        };
        debug!("Punc: loaded {} tokens", token_to_id.len());

        // Load config.yaml for punc_list
        let config_path = model_dir.join("config.yaml");
        let punc_list = if config_path.exists() {
            load_punc_list(&config_path)?
        } else {
            debug!("Punc: config.yaml not found, using default");
            vec!["，".to_string(), "。".to_string(), "？".to_string()]
        };
        debug!("Punc: loaded {} punctuation marks", punc_list.len());

        Ok(Self {
            session,
            token_to_id,
            punc_list,
        })
    }

    pub fn add_punct(&mut self, text: &str) -> Result<String> {
        if text.is_empty() {
            return Ok(String::new());
        }

        let words = split_words(text);
        if words.is_empty() {
            return Ok(text.to_string());
        }

        let token_ids: Vec<i32> = words
            .iter()
            .map(|w| *self.token_to_id.get(w).unwrap_or(&0) as i32)
            .collect();

        if token_ids.iter().all(|&id| id == 0) {
            debug!("Punc: no tokens matched, returning original");
            return Ok(text.to_string());
        }

        let seq_len = token_ids.len();
        let input_tensor = ndarray::Array2::from_shape_vec((1, seq_len), token_ids.clone())
            .context("Failed to create Punc input tensor")?;
        let lengths_tensor = ndarray::Array1::from_vec(vec![seq_len as i32]);

        let outputs = self
            .session
            .run(ort::inputs! {
                "inputs" => Tensor::from_array(input_tensor)?,
                "text_lengths" => Tensor::from_array(lengths_tensor)?,
            })
            .with_context(|| format!("Punc inference failed, words={}", words.len()))?;

        let (_, logits_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract Punc output")?;

        // Get num_classes from logits shape
        let num_classes = if logits_data.len() >= seq_len {
            logits_data.len() / seq_len
        } else {
            self.punc_list.len()
        };

        // Argmax decode
        let mut punc_indices: Vec<usize> = Vec::with_capacity(seq_len);
        for i in 0..seq_len {
            let offset = i * num_classes;
            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..num_classes {
                if offset + j >= logits_data.len() {
                    break;
                }
                let val = logits_data[offset + j];
                if val > best_val {
                    best_val = val;
                    best_idx = j;
                }
            }
            punc_indices.push(best_idx);
        }

        // Build result with punctuation
        let mut result = String::new();
        for (i, word) in words.iter().enumerate() {
            result.push_str(word);

            if i < punc_indices.len() {
                let punc_idx = punc_indices[i];
                if punc_idx < self.punc_list.len() {
                    let punct = &self.punc_list[punc_idx];
                    if punct != "_" && !punct.is_empty() {
                        result.push_str(punct);
                    }
                }
            }
        }

        debug!("Punc: '{}' -> '{}'", text, result);
        Ok(result)
    }
}

fn split_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut ascii_buf = String::new();

    for g in text.graphemes(true) {
        if g.as_bytes()[0] < 128 {
            ascii_buf.push_str(g);
        } else {
            if !ascii_buf.is_empty() {
                words.push(ascii_buf.clone());
                ascii_buf.clear();
            }
            words.push(g.to_string());
        }
    }
    if !ascii_buf.is_empty() {
        words.push(ascii_buf);
    }

    words
}

fn load_tokens_json(path: &Path) -> Result<HashMap<String, u32>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read tokens: {}", path.display()))?;
    let tokens: Vec<String> = serde_json::from_str(&data)
        .with_context(|| format!("Failed to parse tokens JSON: {}", path.display()))?;

    let mut token_to_id = HashMap::new();

    for (id, token) in tokens.into_iter().enumerate() {
        token_to_id.insert(token, id as u32);
    }

    Ok(token_to_id)
}

fn load_punc_list(path: &Path) -> Result<Vec<String>> {
    let data = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let mut punc_list = Vec::new();
    let mut in_punc_list = false;

    for line in data.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("punc_list:") {
            in_punc_list = true;
            continue;
        }
        if in_punc_list {
            if let Some(stripped) = trimmed.strip_prefix("- ") {
                let item = stripped.trim();
                punc_list.push(item.to_string());
            } else if !trimmed.is_empty() && !trimmed.starts_with('-') {
                in_punc_list = false;
            }
        }
    }

    if punc_list.is_empty() {
        punc_list = vec![
            "<unk>".to_string(),
            "_".to_string(),
            "，".to_string(),
            "。".to_string(),
            "？".to_string(),
            "、".to_string(),
        ];
    }

    Ok(punc_list)
}
