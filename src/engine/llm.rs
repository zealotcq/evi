//! Qwen2.5-0.5B-Instruct ONNX text post-processing engine.
//!
//! Loads the Qwen ONNX model and performs autoregressive generation with KV-cache
//! for text refinement (removing filler words, adding punctuation, fixing typos).

use crate::engine::debug_refine::DebugRefine;
use crate::TokenScore;
use anyhow::{bail, Context, Result};
use log::{debug, info, warn};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;
use parking_lot::Mutex;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

const NUM_LAYERS: usize = 24;
const NUM_KV_HEADS: usize = 2;
const HEAD_DIM: usize = 64;
const VOCAB_SIZE: usize = 151936;

const EOS_TOKEN_ID: u32 = 151645;

struct BpeTokenizer {
    vocab: HashMap<Vec<u8>, u32>,
    id_to_token: HashMap<u32, Vec<u8>>,
    merge_rank: HashMap<(Vec<u8>, Vec<u8>), usize>,
    special_tokens: HashMap<String, u32>,
    added_token_ids: std::collections::HashSet<u32>,
    byte_fallback: bool,
}

fn bytes_to_unicode() -> Vec<Vec<u8>> {
    let bs: Vec<u32> = (33..=126u32).chain(161..=172).chain(174..=255).collect();
    let bs_set: std::collections::HashSet<u32> = bs.iter().copied().collect();
    let mut table = vec![Vec::new(); 256];
    let mut n: u32 = 0;
    for b in 0u32..256 {
        let ch = if bs_set.contains(&b) {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
        let mut buf = [0u8; 4];
        let len = ch.encode_utf8(&mut buf).len();
        table[b as usize] = buf[..len].to_vec();
    }
    table
}

impl BpeTokenizer {
    fn from_tokenizer_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read tokenizer.json: {}", path.display()))?;
        let json: serde_json::Value = serde_json::from_str(&data)
            .with_context(|| format!("Failed to parse tokenizer.json: {}", path.display()))?;

        let model = &json["model"];
        let vocab_map = model["vocab"]
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Missing vocab in tokenizer.json"))?;

        let mut vocab = HashMap::new();
        let mut id_to_token = HashMap::new();
        for (token_str, id_val) in vocab_map {
            let id = id_val
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("Invalid vocab id"))? as u32;
            let token_bytes = token_str.as_bytes().to_vec();
            vocab.insert(token_bytes.clone(), id);
            id_to_token.insert(id, token_bytes);
        }

        let merges_arr = model["merges"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Missing merges in tokenizer.json"))?;
        let mut merge_rank = HashMap::new();
        for (rank, merge_val) in merges_arr.iter().enumerate() {
            let merge_str = merge_val
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("Invalid merge"))?;
            let parts: Vec<&str> = merge_str.splitn(2, ' ').collect();
            if parts.len() != 2 {
                continue;
            }
            let a = parts[0].as_bytes().to_vec();
            let b = parts[1].as_bytes().to_vec();
            merge_rank.insert((a, b), rank);
        }

        let mut special_tokens = HashMap::new();
        let mut added_token_ids = std::collections::HashSet::new();
        if let Some(added) = json["added_tokens"].as_array() {
            for tok in added {
                if let (Some(id), Some(content)) = (tok["id"].as_u64(), tok["content"].as_str()) {
                    let id = id as u32;
                    special_tokens.insert(content.to_string(), id);
                    added_token_ids.insert(id);
                    let token_bytes = content.as_bytes().to_vec();
                    vocab.insert(token_bytes.clone(), id);
                    id_to_token.insert(id, token_bytes);
                }
            }
        }

        let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);

        Ok(Self {
            vocab,
            id_to_token,
            merge_rank,
            special_tokens,
            added_token_ids,
            byte_fallback,
        })
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let greek_tokens = bytes_to_unicode();
        let mut result = Vec::new();
        let words = regex_split(text);

        for word in &words {
            let mut tokens: Vec<Vec<u8>> = word
                .as_bytes()
                .iter()
                .map(|&b| greek_tokens[b as usize].clone())
                .collect();

            loop {
                if tokens.len() < 2 {
                    break;
                }
                let mut best_rank = usize::MAX;
                let mut best_idx = 0;
                for i in 0..tokens.len() - 1 {
                    if let Some(&rank) = self
                        .merge_rank
                        .get(&(tokens[i].clone(), tokens[i + 1].clone()))
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_idx = i;
                        }
                    }
                }
                if best_rank == usize::MAX {
                    break;
                }
                let mut merged = tokens[best_idx].clone();
                merged.extend_from_slice(&tokens[best_idx + 1]);
                tokens.splice(best_idx..best_idx + 2, std::iter::once(merged));
            }

            for token in tokens {
                if let Some(&id) = self.vocab.get(&token) {
                    result.push(id);
                }
            }
        }

        result
    }

    fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        let mut remaining = text;

        while !remaining.is_empty() {
            let mut earliest = None;
            let mut earliest_pos = remaining.len();
            let mut earliest_len = 0;

            for (special_str, &special_id) in &self.special_tokens {
                if let Some(pos) = remaining.find(special_str) {
                    if pos < earliest_pos {
                        earliest_pos = pos;
                        earliest_len = special_str.len();
                        earliest = Some(special_id);
                    }
                }
            }

            if earliest_pos > 0 {
                let before = &remaining[..earliest_pos];
                result.extend(self.encode(before));
            }

            if let Some(id) = earliest {
                result.push(id);
                remaining = &remaining[earliest_pos + earliest_len..];
            } else {
                break;
            }
        }

        result
    }

    fn decode(&self, ids: &[u32]) -> String {
        let greek_tokens = bytes_to_unicode();
        let mut byte_buf = Vec::new();

        for &id in ids {
            if let Some(token_bytes) = self.id_to_token.get(&id) {
                if self.added_token_ids.contains(&id) {
                    byte_buf.extend_from_slice(token_bytes);
                    continue;
                }
                if self.byte_fallback {
                    if let Ok(token_str) = std::str::from_utf8(token_bytes) {
                        if token_str.starts_with("<0x")
                            && token_str.ends_with('>')
                            && token_str.len() == 6
                        {
                            if let Ok(byte_val) = u8::from_str_radix(&token_str[3..5], 16) {
                                byte_buf.push(byte_val);
                                continue;
                            }
                        }
                    }
                }
                byte_buf.extend_from_slice(token_bytes);
            }
        }

        let decoded = decode_byte_tokens(&byte_buf, &greek_tokens);
        String::from_utf8(decoded)
            .unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).to_string())
    }
}

fn decode_byte_tokens(token_bytes: &[u8], greek_tokens: &[Vec<u8>]) -> Vec<u8> {
    let mut inv: HashMap<String, u8> = HashMap::new();
    for (byte_val, gt) in greek_tokens.iter().enumerate() {
        let s = String::from_utf8_lossy(gt).to_string();
        inv.insert(s, byte_val as u8);
    }

    let input = String::from_utf8_lossy(token_bytes);
    let mut result = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut pos = 0;

    while pos < chars.len() {
        let mut best_match: Option<u8> = None;
        let mut best_len = 0;

        for try_len in 1..=4.min(chars.len() - pos) {
            let candidate: String = chars[pos..pos + try_len].iter().collect();
            if let Some(&byte_val) = inv.get(&candidate) {
                if try_len > best_len {
                    best_len = try_len;
                    best_match = Some(byte_val);
                }
            }
        }

        if let Some(byte_val) = best_match {
            result.push(byte_val);
            pos += best_len;
        } else {
            let c = chars[pos];
            let mut buf = [0u8; 4];
            let len = c.encode_utf8(&mut buf).len();
            result.extend_from_slice(&buf[..len]);
            pos += 1;
        }
    }

    result
}

fn regex_split(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut chars = text.char_indices().peekable();

    while let Some((_, c)) = chars.next() {
        if c.is_ascii_alphabetic() || (c as u32) > 127 {
            current.push(c);
            continue;
        }
        if !current.is_empty() {
            result.push(std::mem::take(&mut current));
        }

        if c.is_ascii_digit() {
            let mut num = String::new();
            num.push(c);
            while let Some(&(_, nc)) = chars.peek() {
                if nc.is_ascii_digit() {
                    num.push(nc);
                    chars.next();
                } else {
                    break;
                }
            }
            result.push(num);
        } else {
            result.push(c.to_string());
        }
    }
    if !current.is_empty() {
        result.push(current);
    }
    result
}

#[derive(Clone)]
struct KvCache {
    data: Vec<ndarray::Array4<f32>>,
}

impl KvCache {
    fn new() -> Self {
        let empty = ndarray::Array4::zeros((1, NUM_KV_HEADS, 0, HEAD_DIM));
        Self {
            data: vec![empty; NUM_LAYERS * 2],
        }
    }

    fn seq_len(&self) -> usize {
        self.data[0].shape()[2]
    }
}

pub struct LlmEngine {
    session: Mutex<Session>,
    tokenizer: BpeTokenizer,
    system_cache: KvCache,
    system_prompt_len: usize,
    prefill_template: String,
    result_field: String,
    min_refine_length: usize,
    max_refine_ratio: f64,
}

impl LlmEngine {
    pub fn new(cfg: &crate::Config) -> Result<Self> {
        Self::build(cfg)
    }

    pub fn rebuild_prompt(
        &mut self,
        new_system_prompt: &str,
        new_prefill_template: &str,
    ) -> Result<()> {
        let system_part = format!("<|im_start|>system\n{}<|im_end|>\n", new_system_prompt);
        let system_ids = self.tokenizer.encode_with_special(&system_part);
        info!(
            "LlmEngine: rebuild system prompt -> {} tokens",
            system_ids.len()
        );

        let mut new_cache = KvCache::new();
        {
            let mut session = self.session.lock();
            let ids_i64: Vec<i64> = system_ids.iter().map(|&id| id as i64).collect();
            let seq_len = system_ids.len();
            let input_tensor = ndarray::Array2::from_shape_vec((1, seq_len), ids_i64)
                .context("Failed to create system input_ids tensor")?;
            let attn_mask: Vec<i64> = vec![1; seq_len];
            let attn_tensor = ndarray::Array2::from_shape_vec((1, seq_len), attn_mask)
                .context("Failed to create system attn tensor")?;
            let pos_ids: Vec<i64> = (0..seq_len).map(|x| x as i64).collect();
            let pos_tensor = ndarray::Array2::from_shape_vec((1, seq_len), pos_ids)
                .context("Failed to create system pos tensor")?;

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
                ("input_ids".into(), Tensor::from_array(input_tensor)?.into()),
                (
                    "attention_mask".into(),
                    Tensor::from_array(attn_tensor)?.into(),
                ),
                (
                    "position_ids".into(),
                    Tensor::from_array(pos_tensor)?.into(),
                ),
            ];

            for layer in 0..NUM_LAYERS {
                inputs.push((
                    format!("past_key_values.{}.key", layer).into(),
                    Tensor::from_array(new_cache.data[layer * 2].clone().into_dyn())?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer).into(),
                    Tensor::from_array(new_cache.data[layer * 2 + 1].clone().into_dyn())?.into(),
                ));
            }

            let outputs = session
                .run(inputs)
                .with_context(|| "LLM system prompt rebuild failed")?;
            update_cache_from_outputs(&mut new_cache, &outputs)?;
        }

        let last_colon = new_prefill_template.rfind(':').with_context(|| {
            format!("No ':' found in prefill template: {}", new_prefill_template)
        })?;
        let before_last_colon = &new_prefill_template[..last_colon];
        let last_comma = before_last_colon.rfind(',').unwrap_or(0);
        let field_start = if last_comma > 0 { last_comma + 1 } else { 1 };
        let field_end = last_colon;
        let result_field_raw = &new_prefill_template[field_start..field_end];
        let result_field = result_field_raw
            .trim_matches(|c: char| c == '"' || c == ' ')
            .to_string();

        self.system_cache = new_cache;
        self.system_prompt_len = system_ids.len();
        self.prefill_template = new_prefill_template.to_string();
        self.result_field = result_field;

        info!(
            "LlmEngine: prompt rebuilt - system_prompt={} chars, prefill_template={:?} ({} chars), {} tokens",
            new_system_prompt.len(),
            new_prefill_template,
            new_prefill_template.len(),
            self.system_prompt_len
        );
        Ok(())
    }

    fn build(cfg: &crate::Config) -> Result<Self> {
        let base = crate::models::base_dir(cfg);
        let model_dir = crate::models::llm_model_dir(&base);
        let model_path = if model_dir.join("onnx").join("model_int8.onnx").exists() {
            model_dir.join("onnx").join("model_int8.onnx")
        } else if model_dir.join("onnx").join("model.onnx").exists() {
            model_dir.join("onnx").join("model.onnx")
        } else if model_dir.join("model_int8.onnx").exists() {
            model_dir.join("model_int8.onnx")
        } else {
            model_dir.join("model.onnx")
        };

        if !model_path.exists() {
            bail!("LLM model not found: {}", model_path.display());
        }

        debug!("LlmEngine: loading from {}...", model_path.display());
        let mut session = Session::builder()
            .with_context(|| "Failed to create LLM ONNX session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load LLM model: {}", model_path.display()))?;

        debug!("LlmEngine: model loaded");
        for input in session.inputs() {
            debug!("LLM input: {:?}", input);
        }
        for output in session.outputs() {
            debug!("LLM output: {:?}", output);
        }

        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            bail!("tokenizer.json not found in {}", model_dir.display());
        }
        let tokenizer = BpeTokenizer::from_tokenizer_json(&tokenizer_path)?;
        debug!("LlmEngine: tokenizer loaded");

        let exe_dir = crate::Config::exe_dir()?;

        let prompt_path = exe_dir.join("system_prompt.txt");
        let system_prompt = if prompt_path.exists() {
            std::fs::read_to_string(&prompt_path)
                .with_context(|| format!("Failed to read {}", prompt_path.display()))?
        } else {
            String::new()
        };
        debug!(
            "LlmEngine: system prompt loaded from {} ({} chars)",
            prompt_path.display(),
            system_prompt.len()
        );

        let system_part = format!("<|im_start|>system\n{}<|im_end|>\n", system_prompt);
        let system_ids = tokenizer.encode_with_special(&system_part);
        debug!(
            "LlmEngine: system prompt -> {} tokens, prefilling...",
            system_ids.len()
        );
        info!("LLM: system prefill text: {:?}", system_part);

        let mut system_cache = KvCache::new();
        {
            let ids_i64: Vec<i64> = system_ids.iter().map(|&id| id as i64).collect();
            let seq_len = system_ids.len();
            let input_tensor = ndarray::Array2::from_shape_vec((1, seq_len), ids_i64)
                .context("Failed to create system input_ids tensor")?;
            let attn_mask: Vec<i64> = vec![1; seq_len];
            let attn_tensor = ndarray::Array2::from_shape_vec((1, seq_len), attn_mask)
                .context("Failed to create system attn tensor")?;
            let pos_ids: Vec<i64> = (0..seq_len).map(|x| x as i64).collect();
            let pos_tensor = ndarray::Array2::from_shape_vec((1, seq_len), pos_ids)
                .context("Failed to create system pos tensor")?;

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
                ("input_ids".into(), Tensor::from_array(input_tensor)?.into()),
                (
                    "attention_mask".into(),
                    Tensor::from_array(attn_tensor)?.into(),
                ),
                (
                    "position_ids".into(),
                    Tensor::from_array(pos_tensor)?.into(),
                ),
            ];

            for layer in 0..NUM_LAYERS {
                inputs.push((
                    format!("past_key_values.{}.key", layer).into(),
                    Tensor::from_array(system_cache.data[layer * 2].clone().into_dyn())?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer).into(),
                    Tensor::from_array(system_cache.data[layer * 2 + 1].clone().into_dyn())?.into(),
                ));
            }

            let outputs = session
                .run(inputs)
                .with_context(|| "LLM system prompt prefill failed")?;
            update_cache_from_outputs(&mut system_cache, &outputs)?;
        }

        debug!(
            "LlmEngine: system prompt KV cache ready ({} tokens)",
            system_ids.len()
        );

        let prefill_path = exe_dir.join("prefill_template.txt");
        let prefill_template = if prefill_path.exists() {
            std::fs::read_to_string(&prefill_path)
                .with_context(|| format!("Failed to read {}", prefill_path.display()))?
        } else {
            r#"{"语音识别文本":"{INPUT}","校对输出":""#.to_string()
        };
        debug!(
            "LlmEngine: prefill template loaded ({} chars)",
            prefill_template.len()
        );

        let last_colon = prefill_template
            .rfind(':')
            .with_context(|| format!("No ':' found in prefill template: {}", prefill_template))?;
        let before_last_colon = &prefill_template[..last_colon];
        let last_comma = before_last_colon.rfind(',').unwrap_or(0);
        let field_start = if last_comma > 0 { last_comma + 1 } else { 1 };
        let field_end = last_colon;
        let result_field_raw = &prefill_template[field_start..field_end];
        let result_field = result_field_raw
            .trim_matches(|c: char| c == '"' || c == ' ')
            .to_string();
        if result_field.is_empty() {
            bail!("Failed to extract result field name from prefill template");
        }
        debug!(
            "LlmEngine: result field name from template: '{}'",
            result_field
        );

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            system_cache,
            system_prompt_len: system_ids.len(),
            prefill_template,
            result_field,
            min_refine_length: cfg.min_refine_length,
            max_refine_ratio: cfg.max_refine_ratio,
        })
    }

    pub fn refine(&self, text: &str, dr: &DebugRefine) -> Result<(String, Vec<TokenScore>)> {
        let text = text.trim();
        if text.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        if text.chars().count() <= self.min_refine_length {
            info!(
                "LLM: skip (chars={} <= {}), output: '{}'",
                text.chars().count(),
                self.min_refine_length,
                text
            );
            return Ok((text.to_string(), vec![]));
        }

        let (refined, tokens) = match self.try_refine(text) {
            Ok(r) => r,
            Err(e) => {
                info!("LLM: refine failed: {}, output (original): '{}'", e, text);
                return Ok((text.to_string(), vec![]));
            }
        };

        let refined = refined.trim();

        dr.log_refine(text, refined);

        debug!("LLM: refined output: '{}'", refined);
        Ok((refined.to_string(), tokens))
    }

    fn try_refine(&self, text: &str) -> Result<(String, Vec<TokenScore>)> {
        info!("LLM: current prefill_template: {:?}", self.prefill_template);
        let assistant_prefix = self.prefill_template.replace("{INPUT}", text);
        let user_part = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n{}",
            text, assistant_prefix
        );
        let user_ids = self.tokenizer.encode_with_special(&user_part);
        info!(
            "LLM: {} chars -> {} tokens (cache: {}, prefill: {} chars)",
            text.len(),
            user_ids.len(),
            self.system_prompt_len,
            assistant_prefix.len()
        );
        debug!("LLM: user text: {:?}", text);
        info!("LLM: prefill text: {:?}", user_part);

        let input_token_count = user_ids.len();
        let max_gen_tokens = (input_token_count as f64 * self.max_refine_ratio) as usize + 8;

        let generated = self.generate(
            self.system_cache.clone(),
            self.system_prompt_len,
            &user_ids,
            2048,
            Some(max_gen_tokens),
        )?;

        let gen_ids: Vec<u32> = generated.iter().map(|(id, _)| *id).collect();
        let decoded = self.tokenizer.decode(&gen_ids);
        info!("LLM: raw decoded: {:?}", decoded);
        let full = format!(
            "{}{}",
            assistant_prefix,
            decoded.trim_end_matches("<|im_end|>").trim()
        );

        let val: serde_json::Value = serde_json::from_str(&full)
            .with_context(|| format!("JSON parse failed, raw: {}", &full[..full.len().min(200)]))?;

        let refined = val
            .get(&self.result_field)
            .and_then(|v| v.as_str())
            .with_context(|| {
                format!(
                    "Missing '{}' field, got: {}",
                    self.result_field,
                    &full[..full.len().min(200)]
                )
            })?;

        if refined.trim().is_empty() {
            anyhow::bail!("Empty refined result");
        }

        let tokens: Vec<TokenScore> = generated
            .iter()
            .filter_map(|(id, conf)| {
                let text = self.tokenizer.decode(&[*id]);
                if text.is_empty() || text == "<|im_end|>" {
                    return None;
                }
                Some(TokenScore {
                    token: text,
                    confidence: *conf,
                })
            })
            .collect();

        Ok((refined.to_string(), tokens))
    }

    fn generate(
        &self,
        init_cache: KvCache,
        init_cache_len: usize,
        input_ids: &[u32],
        max_new_tokens: usize,
        max_gen_tokens: Option<usize>,
    ) -> Result<Vec<(u32, f32)>> {
        let mut session = self.session.lock();
        let prompt_len = input_ids.len();
        let mut cache = init_cache;
        let mut generated: Vec<(u32, f32)> = Vec::new();
        let mut all_ids: Vec<u32> = input_ids.to_vec();

        {
            let chunk_ids: Vec<i64> = input_ids.iter().map(|&id| id as i64).collect();
            let input_tensor = ndarray::Array2::from_shape_vec((1, prompt_len), chunk_ids)
                .context("Failed to create input_ids tensor")?;
            let total_len = init_cache_len + prompt_len;
            let attn_mask: Vec<i64> = vec![1; total_len];
            let attn_tensor = ndarray::Array2::from_shape_vec((1, total_len), attn_mask)
                .context("Failed to create attention_mask tensor")?;
            let pos_ids: Vec<i64> = (init_cache_len..total_len).map(|x| x as i64).collect();
            let pos_tensor = ndarray::Array2::from_shape_vec((1, prompt_len), pos_ids)
                .context("Failed to create position_ids tensor")?;

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
                ("input_ids".into(), Tensor::from_array(input_tensor)?.into()),
                (
                    "attention_mask".into(),
                    Tensor::from_array(attn_tensor)?.into(),
                ),
                (
                    "position_ids".into(),
                    Tensor::from_array(pos_tensor)?.into(),
                ),
            ];

            for layer in 0..NUM_LAYERS {
                inputs.push((
                    format!("past_key_values.{}.key", layer).into(),
                    Tensor::from_array(cache.data[layer * 2].clone().into_dyn())?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer).into(),
                    Tensor::from_array(cache.data[layer * 2 + 1].clone().into_dyn())?.into(),
                ));
            }

            let outputs = session
                .run(inputs)
                .with_context(|| "LLM prefill inference failed")?;

            update_cache_from_outputs(&mut cache, &outputs)?;
            let logits_data = extract_last_logits(&outputs, prompt_len)?;

            debug!(
                "LLM: prefill logits size={}, first 5: {:?}",
                logits_data.len(),
                &logits_data[..5.min(logits_data.len())]
            );
            let (next_token, confidence) = argmax_with_conf(&logits_data);
            debug!(
                "LLM: first generated token id={} text={:?}",
                next_token,
                self.tokenizer.decode(&[next_token as u32])
            );
            if next_token == EOS_TOKEN_ID as usize {
                debug!("LLM: first token is EOS, stopping");
                return Ok(generated);
            }
            generated.push((next_token as u32, confidence));
            all_ids.push(next_token as u32);
        }

        for _ in 1..max_new_tokens {
            let past_len = cache.seq_len();
            let last_id = *all_ids.last().unwrap() as i64;
            let input_tensor = ndarray::Array2::from_shape_vec((1, 1), vec![last_id])?;
            let attn_mask: Vec<i64> = vec![1; past_len + 1];
            let attn_tensor = ndarray::Array2::from_shape_vec((1, past_len + 1), attn_mask)?;
            let pos_tensor = ndarray::Array2::from_shape_vec((1, 1), vec![past_len as i64])?;

            let mut inputs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> = vec![
                ("input_ids".into(), Tensor::from_array(input_tensor)?.into()),
                (
                    "attention_mask".into(),
                    Tensor::from_array(attn_tensor)?.into(),
                ),
                (
                    "position_ids".into(),
                    Tensor::from_array(pos_tensor)?.into(),
                ),
            ];

            for layer in 0..NUM_LAYERS {
                inputs.push((
                    format!("past_key_values.{}.key", layer).into(),
                    Tensor::from_array(cache.data[layer * 2].clone().into_dyn())?.into(),
                ));
                inputs.push((
                    format!("past_key_values.{}.value", layer).into(),
                    Tensor::from_array(cache.data[layer * 2 + 1].clone().into_dyn())?.into(),
                ));
            }

            let outputs = session
                .run(inputs)
                .with_context(|| "LLM decode step inference failed")?;

            update_cache_from_outputs(&mut cache, &outputs)?;

            let logits_data = extract_last_logits(&outputs, 1)?;
            let (next_token, confidence) = argmax_with_conf(&logits_data);
            if next_token == EOS_TOKEN_ID as usize {
                let gen_ids: Vec<u32> = generated.iter().map(|(id, _)| *id).collect();
                let gen_text = self.tokenizer.decode(&gen_ids);
                debug!(
                    "LLM: EOS at step {}, generated {} tokens, text: {:?}",
                    generated.len(),
                    generated.len(),
                    gen_text
                );
                break;
            }
            if generated.len() <= 10 {
                debug!(
                    "LLM: decode step {} -> token id={} text={:?}",
                    generated.len(),
                    next_token,
                    self.tokenizer.decode(&[next_token as u32])
                );
            }
            generated.push((next_token as u32, confidence));
            all_ids.push(next_token as u32);

            if let Some(max_tokens) = max_gen_tokens {
                if generated.len() > max_tokens {
                    info!(
                        "LLM: early stop, {} generated tokens > max {}",
                        generated.len(),
                        max_tokens
                    );
                    anyhow::bail!(
                        "output too long: {} tokens > max {}",
                        generated.len(),
                        max_tokens
                    );
                }
            }
        }

        if generated.len() == max_new_tokens {
            let gen_ids: Vec<u32> = generated.iter().map(|(id, _)| *id).collect();
            let gen_text = self.tokenizer.decode(&gen_ids);
            warn!(
                "LLM: hit max_new_tokens={} without EOS, text: {:?}",
                max_new_tokens, gen_text
            );
        }

        Ok(generated)
    }
}

fn extract_last_logits(outputs: &ort::session::SessionOutputs, seq_len: usize) -> Result<Vec<f32>> {
    let (shape, logits_data) = outputs[0]
        .try_extract_tensor::<f32>()
        .context("Failed to extract logits")?;

    if shape.len() == 3 && shape[1] as usize > 1 {
        let offset = (shape[1] as usize - 1) * VOCAB_SIZE;
        Ok(logits_data[offset..offset + VOCAB_SIZE].to_vec())
    } else {
        let offset = (seq_len - 1) * VOCAB_SIZE;
        if offset + VOCAB_SIZE <= logits_data.len() {
            Ok(logits_data[offset..offset + VOCAB_SIZE].to_vec())
        } else {
            Ok(logits_data.to_vec())
        }
    }
}

fn update_cache_from_outputs(
    cache: &mut KvCache,
    outputs: &ort::session::SessionOutputs,
) -> Result<()> {
    for layer in 0..NUM_LAYERS {
        let k_out_idx = 1 + layer * 2;
        let v_out_idx = 2 + layer * 2;

        let (k_shape, k_data) = outputs[k_out_idx]
            .try_extract_tensor::<f32>()
            .with_context(|| format!("Failed to extract present key for layer {}", layer))?;
        let (v_shape, v_data) = outputs[v_out_idx]
            .try_extract_tensor::<f32>()
            .with_context(|| format!("Failed to extract present value for layer {}", layer))?;

        let k_arr = ndarray::Array4::from_shape_vec(
            (
                k_shape[0] as usize,
                k_shape[1] as usize,
                k_shape[2] as usize,
                k_shape[3] as usize,
            ),
            k_data.to_vec(),
        )
        .with_context(|| format!("Failed to reshape key cache for layer {}", layer))?;
        let v_arr = ndarray::Array4::from_shape_vec(
            (
                v_shape[0] as usize,
                v_shape[1] as usize,
                v_shape[2] as usize,
                v_shape[3] as usize,
            ),
            v_data.to_vec(),
        )
        .with_context(|| format!("Failed to reshape value cache for layer {}", layer))?;

        cache.data[layer * 2] = k_arr;
        cache.data[layer * 2 + 1] = v_arr;
    }
    Ok(())
}

fn argmax_with_conf(logits: &[f32]) -> (usize, f32) {
    if logits.is_empty() {
        return (0, 0.0);
    }
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let mut min_val = f32::INFINITY;
    let mut sum: f32 = 0.0;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best = i;
        }
        if v < min_val {
            min_val = v;
        }
        sum += v;
    }
    let shifted_sum = sum - min_val * logits.len() as f32;
    let confidence = if shifted_sum > 0.0 {
        (best_val - min_val) / shifted_sum
    } else {
        1.0
    };
    (best, confidence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_encode_decode_roundtrip() {
        let cfg = match crate::Config::load() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: config.json not found");
                return;
            }
        };
        let base = crate::models::default_base_dir();
        let model_dir = crate::models::llm_model_dir(&base);
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!("Skipping test: tokenizer.json not found");
            return;
        }
        let tok = BpeTokenizer::from_tokenizer_json(&tokenizer_path).unwrap();

        let text = "恩我的那个恩肚子有点疼";
        let ids = tok.encode(text);
        eprintln!("encode('{}') = {:?}", text, ids);
        assert!(!ids.is_empty(), "encode should produce tokens");

        let decoded = tok.decode(&ids);
        eprintln!("decode({:?}) = '{}'", ids, decoded);
        assert_eq!(decoded, text, "encode-decode roundtrip should be lossless");
    }

    #[test]
    fn test_encode_with_special_tokens() {
        let cfg = match crate::Config::load() {
            Ok(c) => c,
            Err(_) => {
                eprintln!("Skipping test: config.json not found");
                return;
            }
        };
        let base = crate::models::default_base_dir();
        let model_dir = crate::models::llm_model_dir(&base);
        let tokenizer_path = model_dir.join("tokenizer.json");
        if !tokenizer_path.exists() {
            eprintln!("Skipping test: tokenizer.json not found");
            return;
        }
        let tok = BpeTokenizer::from_tokenizer_json(&tokenizer_path).unwrap();

        let text = "<|im_start|>system\nhello<|im_end|>";
        let ids = tok.encode_with_special(text);
        eprintln!("encode_with_special('{}') = {:?}", text, ids);
        assert!(
            ids.contains(&151644u32),
            "should contain <|im_start|> id=151644"
        );
        assert!(
            ids.contains(&151645u32),
            "should contain <|im_end|> id=151645"
        );

        let decoded = tok.decode(&ids);
        eprintln!("decoded = '{}'", decoded);
    }

    #[test]
    fn test_refine_with_kv_cache() {
        let _ = env_logger::Builder::from_env("VI_LOG")
            .filter_level(log::LevelFilter::Info)
            .try_init();

        let cfg = match crate::Config::load() {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Skipping test: Config::load() failed: {e}");
                return;
            }
        };
        let base = crate::models::default_base_dir();
        let model_dir = crate::models::llm_model_dir(&base);
        let model_path = model_dir.join("onnx").join("model_int8.onnx");
        if !model_path.exists() {
            eprintln!(
                "Skipping test: LLM model not found at {}",
                model_path.display()
            );
            return;
        }

        let engine = LlmEngine::build(&cfg).unwrap();
        eprintln!(
            "System prompt KV cache: {} tokens",
            engine.system_prompt_len
        );
        assert!(
            engine.system_prompt_len > 0,
            "system prompt cache should not be empty"
        );

        let dr = crate::engine::debug_refine::DebugRefine::open(":memory:").unwrap();
        let cases = vec![
            ("嗯我的那个嗯肚子有点疼", true),
            ("我们现在来测试一下", true),
            ("嗯输出的是什么", true),
            ("再来试一下第三次", true),
        ];

        for (i, (input, expect_nontrivial)) in cases.iter().enumerate() {
            eprintln!("--- Case {}: '{}' ---", i + 1, input);
            let (refined, _tokens) = engine.refine(input, &dr).unwrap();
            eprintln!("    -> '{}'", refined);

            assert!(
                !refined.is_empty(),
                "[case {}] refine should produce non-empty output for '{}'",
                i + 1,
                input
            );
            if *expect_nontrivial {
                assert!(
                    !refined.contains('*')
                        || refined.contains("不改变")
                        || refined.contains("口头语"),
                    "[case {}] output should not contain markdown: '{}'",
                    i + 1,
                    refined
                );
                assert!(
                    !refined.contains("不改变")
                        && !refined.contains("原始用词")
                        && !refined.contains("口头语"),
                    "[case {}] output should not leak system prompt: '{}'",
                    i + 1,
                    refined
                );
            }
        }
    }
}
