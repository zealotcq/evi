//! Voice Input Method — core trait definitions and data types.
//!
//! ```text
//! ┌───────────────────────────────────────────────────────┐
//! │                 Rust Frontend (this crate)             │
//! │                                                        │
//! │  ┌────────────┐  ┌─────────────┐  ┌────────────────┐ │
//! │  │ AudioSource │  │ VadEngine   │  │ AsrEngine      │ │
//! │  │ (cpal)      │→ │ (ONNX VAD)  │  │ (ONNX Paraformer)│
//! │  └────────────┘  └──────┬──────┘  └───────┬────────┘ │
//! │                         │                   │          │
//! │                  ┌──────┴───────────────────┴───┐     │
//! │                  │     Segmenter (≤10s chunks)   │     │
//! │                  └──────────────┬────────────────┘     │
//! │                                 │                      │
//! │  ┌──────────────────────────────┴──────────────────┐  │
//! │  │  TextSession (TSF inject + monitor)              │  │
//! │  │  + CorrectionMapper (text↔audio time mapping)    │  │
//! │  └─────────────────────────────────────────────────┘  │
//! └───────────────────────────────────────────────────────┘
//! ```

pub mod audio;
pub mod download_model;
pub mod engine;
pub mod models;
pub mod refine_mgr;
pub mod secret;
pub mod text;
pub mod ui;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use serde::Deserialize;
use serde_json::Value;
use std::fmt;
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════════
// Audio
// ═══════════════════════════════════════════════════════════════════════════════

pub struct AudioFrame {
    pub samples: Vec<i16>,
    pub timestamp_us: u64,
}

pub trait AudioSource: Send {
    fn start(&mut self) -> Result<()>;
    fn stop(&mut self) -> Result<Vec<i16>>;
    fn is_recording(&self) -> bool;
    fn sample_rate(&self) -> u32;
    fn receiver(&self) -> &Receiver<AudioFrame>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Timing & Segmentation
// ═══════════════════════════════════════════════════════════════════════════════

/// A single recognized character with its audio timing.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CharTiming {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// A single ASR token with its character text and confidence score.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TokenScore {
    pub token: String,
    pub confidence: f32,
}

/// Result for one audio segment after ASR.
#[derive(Debug, Clone)]
pub struct SegmentResult {
    pub segment_index: usize,
    pub text: String,
    pub chars: Vec<CharTiming>,
    pub tokens: Vec<TokenScore>,
    pub audio_start_ms: u64,
    pub audio_end_ms: u64,
    pub audio_samples: Vec<f32>,
    pub sample_rate: u32,
}

impl fmt::Display for SegmentResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[#{}] \"{}\" ({}-{}ms, {} chars)",
            self.segment_index,
            self.text,
            self.audio_start_ms,
            self.audio_end_ms,
            self.chars.len()
        )
    }
}

/// Full recognition result combining all segments.
#[derive(Debug, Clone)]
pub struct Recognized {
    pub full_text: String,
    pub segments: Vec<SegmentResult>,
}

impl fmt::Display for Recognized {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\"{}\" ({} segments)",
            self.full_text,
            self.segments.len()
        )
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Calibration audio archive
// ═══════════════════════════════════════════════════════════════════════════════

pub fn calib_audio_dir() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    let dir = PathBuf::from(home).join(".evi").join("calib_audio");
    std::fs::create_dir_all(&dir).ok();
    dir
}

pub fn save_wav_16k(path: &std::path::Path, samples: &[i16]) {
    use std::io::Write;
    let sr: u32 = 16000;
    let data_size = samples.len() * 2;
    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(b"RIFF");
        let _ = f.write_all(&(36 + data_size as u32).to_le_bytes());
        let _ = f.write_all(b"WAVE");
        let _ = f.write_all(b"fmt ");
        let _ = f.write_all(&16u32.to_le_bytes());
        let _ = f.write_all(&1u16.to_le_bytes());
        let _ = f.write_all(&1u16.to_le_bytes());
        let _ = f.write_all(&sr.to_le_bytes());
        let _ = f.write_all(&(sr * 2).to_le_bytes());
        let _ = f.write_all(&2u16.to_le_bytes());
        let _ = f.write_all(&16u16.to_le_bytes());
        let _ = f.write_all(b"data");
        let _ = f.write_all(&(data_size as u32).to_le_bytes());
        for &x in samples {
            let _ = f.write_all(&x.to_le_bytes());
        }
    }
}

pub fn load_wav_16k(path: &std::path::Path) -> Option<Vec<f32>> {
    let data = std::fs::read(path).ok()?;
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return None;
    }
    Some(
        data[44..]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect(),
    )
}

#[derive(serde::Serialize, serde::Deserialize)]
struct VoiceRefEntry {
    ts: String,
    embedding: Vec<f64>,
}

fn voice_refs_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".evi").join("voice_refs.json")
}

pub fn load_voice_refs() -> Vec<(String, Vec<f32>)> {
    let Ok(raw) = std::fs::read_to_string(voice_refs_path()) else {
        return Vec::new();
    };
    let Ok(entries) = serde_json::from_str::<Vec<VoiceRefEntry>>(&raw) else {
        return Vec::new();
    };
    entries
        .into_iter()
        .map(|e| (e.ts, e.embedding.into_iter().map(|x| x as f32).collect()))
        .collect()
}

pub fn save_voice_refs(refs: &[(String, Vec<f32>)]) {
    let entries: Vec<VoiceRefEntry> = refs
        .iter()
        .map(|(ts, e)| VoiceRefEntry {
            ts: ts.clone(),
            embedding: e.iter().map(|&x| x as f64).collect(),
        })
        .collect();
    if let Ok(out) = serde_json::to_string_pretty(&entries) {
        let _ = std::fs::write(voice_refs_path(), out);
    }
}

pub fn push_voice_ref(ts: &str, emb: &[f32]) {
    let mut refs = load_voice_refs();
    refs.push((ts.to_string(), emb.to_vec()));
    let cut = refs.len().saturating_sub(3);
    refs.drain(..cut);
    let entries: Vec<VoiceRefEntry> = refs
        .into_iter()
        .map(|(ts, e)| VoiceRefEntry {
            ts,
            embedding: e.into_iter().map(|x| x as f64).collect(),
        })
        .collect();
    if let Ok(out) = serde_json::to_string_pretty(&entries) {
        let _ = std::fs::write(voice_refs_path(), out);
    }
}

pub fn calib_timestamp() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let days = d.as_secs() / 86400;
    let tod = d.as_secs() % 86400;
    let (y, mo, dd) = days_to_ymd(days);
    format!(
        "{:04}{:02}{:02}_{:02}{:02}{:02}",
        y,
        mo,
        dd,
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

fn days_to_ymd(d: u64) -> (u64, u64, u64) {
    let mut y = 1970u64;
    let mut r = d;
    loop {
        let dy = if y.is_multiple_of(4) && !y.is_multiple_of(100) || y.is_multiple_of(400) {
            366
        } else {
            365
        };
        if r < dy {
            break;
        }
        r -= dy;
        y += 1;
    }
    let leap = y.is_multiple_of(4) && !y.is_multiple_of(100) || y.is_multiple_of(400);
    let md = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0;
    for &days in &md {
        if r < days {
            break;
        }
        r -= days;
        m += 1;
    }
    (y, m + 1, r + 1)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Text Management
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub enum TextChangeEvent {
    Inserted {
        position: usize,
        inserted: String,
    },
    Replaced {
        range_start: usize,
        range_end: usize,
        old: String,
        new: String,
    },
    Deleted {
        range_start: usize,
        range_end: usize,
        deleted: String,
    },
    FullReplace {
        old: String,
        new: String,
    },
}

#[derive(Debug, Clone)]
pub struct TextSnapshot {
    pub full_text: String,
    pub cursor_position: usize,
    pub selection_start: usize,
    pub selection_end: usize,
}

pub trait TextOutput: Send {
    fn commit_text(&self, text: &str) -> Result<()>;
    fn method_name(&self) -> &str;
    fn as_any(&self) -> &dyn std::any::Any {
        &() as &dyn std::any::Any
    }
}

pub trait TextObserver: Send {
    fn start_monitoring(&mut self) -> Result<()>;
    fn stop_monitoring(&mut self) -> Result<()>;
    fn poll_changes(&self) -> Vec<TextChangeEvent>;
    fn snapshot(&self) -> Result<TextSnapshot>;
}

pub trait TextSession: TextOutput + TextObserver {}

// ═══════════════════════════════════════════════════════════════════════════════
// Correction
// ═══════════════════════════════════════════════════════════════════════════════

/// A correction with audio provenance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Correction {
    pub original: String,
    pub corrected: String,
    pub timestamp_iso: String,
    pub context_before: String,
    pub context_after: String,
    pub audio_start_ms: u64,
    pub audio_end_ms: u64,
    pub audio_path: Option<String>,
}

pub trait CorrectionStore: Send {
    fn record(&self, correction: Correction) -> Result<()>;
    fn recent(&self, limit: usize) -> Vec<Correction>;
    fn load(&mut self) -> Result<()>;
    fn save(&self) -> Result<()>;
}

fn get_exe_dir() -> Result<PathBuf> {
    let mut dir = std::env::current_exe()
        .with_context(|| "Failed to get exe path")?
        .parent()
        .with_context(|| "Failed to get exe directory")?
        .to_path_buf();
    if dir.ends_with("deps") {
        dir.pop();
    }
    Ok(dir)
}

#[derive(Deserialize, Clone)]
pub struct LlmRemoteConfig {
    #[serde(default = "default_llm_remote_url")]
    pub url: String,
    #[serde(default = "default_llm_remote_model")]
    pub model: String,
    #[serde(default = "default_llm_remote_timeout")]
    pub timeout: u64,
    #[serde(default = "default_llm_remote_max_tokens_ratio")]
    pub max_tokens_ratio: f64,
}

fn default_llm_remote_url() -> String {
    "https://api.openai.com/v1/chat/completions".to_string()
}

fn default_llm_remote_model() -> String {
    "gpt-3.5-turbo".to_string()
}

fn default_llm_remote_timeout() -> u64 {
    30
}

fn default_llm_remote_max_tokens_ratio() -> f64 {
    1.2
}

#[derive(Deserialize, Clone)]
pub struct Config {
    #[serde(default)]
    pub model_base_dir: Option<String>,
    #[serde(default)]
    pub vad_model_dir: Option<String>,
    #[serde(default)]
    pub asr_model_dir: Option<String>,
    #[serde(default)]
    pub punc_model_dir: Option<String>,
    #[serde(default)]
    pub llm_model_dir: Option<String>,
    #[serde(default = "default_false")]
    pub llm_refine: bool,
    #[serde(default = "default_refine_fallback")]
    pub refine_fallback: Vec<String>,
    #[serde(default = "default_min_refine_length")]
    pub min_refine_length: usize,
    #[serde(default)]
    pub refine_db_path: Option<String>,
    #[serde(default = "default_max_refine_ratio")]
    pub max_refine_ratio: f64,
    #[serde(default = "default_false")]
    pub debug: bool,
    #[serde(default = "default_active_scheme")]
    pub active_scheme: Option<String>,
    #[serde(default = "default_trailing_punct")]
    pub trailing_punct: String,
    #[serde(default)]
    pub llm_remote: Option<LlmRemoteConfig>,
    #[serde(default)]
    pub llm_remote_enabled: bool,
    #[serde(default = "default_refine_scheme")]
    pub refine_scheme: String,
    #[serde(default = "default_true")]
    pub punc_enabled: bool,
    #[serde(default)]
    pub voice_gate_enabled: bool,
    #[serde(default = "default_voice_gate_threshold")]
    pub voice_gate_threshold: f64,
    #[serde(default)]
    pub voice_gate_embedding: Option<Vec<f64>>,
    #[serde(default)]
    pub server_url: Option<String>,
    #[serde(default = "default_false")]
    pub privacy_acknowledged: bool,
}

fn default_true() -> bool {
    true
}
fn default_false() -> bool {
    false
}
fn default_refine_fallback() -> Vec<String> {
    vec!["嗯".to_string(), "呃".to_string()]
}
fn default_min_refine_length() -> usize {
    5
}
fn default_max_refine_ratio() -> f64 {
    1.2
}
fn default_active_scheme() -> Option<String> {
    None
}
fn default_trailing_punct() -> String {
    ",".to_string()
}
fn default_voice_gate_threshold() -> f64 {
    0.45
}
fn default_refine_scheme() -> String {
    "default".to_string()
}

impl Config {
    fn home_config_path() -> PathBuf {
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .unwrap_or_else(|_| ".".to_string());
        PathBuf::from(home).join(".evi_config.json")
    }

    pub fn load() -> Result<Self> {
        let exe_dir = get_exe_dir()?;
        let home_path = Self::home_config_path();

        let config_path = if home_path.exists() {
            home_path.clone()
        } else {
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                if let Ok(raw) = std::fs::read_to_string(&exe_path) {
                    let _ = std::fs::write(&home_path, &raw);
                }
            }
            exe_path
        };

        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let cfg: Config = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        Ok(cfg)
    }

    pub fn exe_dir() -> Result<PathBuf> {
        get_exe_dir()
    }

    pub fn save_active_scheme(scheme_name: &str) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert(
                "active_scheme".to_string(),
                Value::String(scheme_name.to_string()),
            );
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }

    pub fn save_llm_remote_enabled(enabled: bool) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert("llm_remote_enabled".to_string(), Value::Bool(enabled));
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }

    pub fn save_refine_scheme(scheme: &str) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert(
                "refine_scheme".to_string(),
                Value::String(scheme.to_string()),
            );
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }

    pub fn save_voice_gate_enabled(val: bool) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert("voice_gate_enabled".to_string(), Value::Bool(val));
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }

    pub fn save_voice_gate_threshold(val: f64) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert("voice_gate_threshold".to_string(), Value::from(val));
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }

    pub fn save_voice_gate_embedding(val: Option<&[f32]>) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            match val {
                Some(v) => {
                    let arr: Vec<f64> = v.iter().map(|&x| x as f64).collect();
                    obj.insert(
                        "voice_gate_embedding".to_string(),
                        serde_json::to_value(&arr)?,
                    );
                }
                None => {
                    obj.remove("voice_gate_embedding");
                }
            }
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }
    pub fn save_punc_enabled(enabled: bool) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert("punc_enabled".to_string(), Value::Bool(enabled));
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }

    pub fn save_model_base_dir(path: Option<&str>) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            match path {
                Some(p) => {
                    obj.insert("model_base_dir".to_string(), Value::String(p.to_string()));
                }
                None => {
                    obj.remove("model_base_dir");
                }
            }
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }

    pub fn save_privacy_acknowledged(ack: bool) -> Result<()> {
        let config_path = Self::home_config_path();
        if !config_path.exists() {
            let exe_dir = get_exe_dir()?;
            let exe_path = exe_dir.join("config.json");
            if exe_path.exists() {
                let raw = std::fs::read_to_string(&exe_path)?;
                let _ = std::fs::write(&config_path, &raw);
            }
        }
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut json: Value = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        if let Some(obj) = json.as_object_mut() {
            obj.insert("privacy_acknowledged".to_string(), Value::Bool(ack));
        }
        let out =
            serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
        std::fs::write(&config_path, out)
            .with_context(|| format!("Failed to write {}", config_path.display()))?;
        Ok(())
    }
}
