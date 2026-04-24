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
pub mod engine;
#[cfg(target_os = "windows")]
pub mod overlay;
pub mod text;
pub mod ui;

use anyhow::{Context, Result};
use crossbeam_channel::Receiver;
use serde::Deserialize;
use serde_json::Value;
use std::fmt;
use std::path::{Path, PathBuf};

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

fn resolve_path(base: &Path, path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        base.join(p)
    }
}

#[derive(Deserialize, Clone)]
pub struct Config {
    pub vad_model_dir: String,
    pub asr_model_dir: String,
    pub punc_model_dir: String,
    pub llm_model_dir: String,
    #[serde(default = "default_true")]
    pub llm_refine: bool,
    #[serde(default = "default_refine_fallback")]
    pub refine_fallback: Vec<String>,
    #[serde(default = "default_min_refine_length")]
    pub min_refine_length: usize,
    #[serde(default = "default_refine_db_path")]
    pub refine_db_path: String,
    #[serde(default = "default_max_refine_ratio")]
    pub max_refine_ratio: f64,
    #[serde(default)]
    pub save_log: bool,
    #[serde(default)]
    pub debug: bool,
    #[serde(default = "default_active_scheme")]
    pub active_scheme: Option<String>,
    #[serde(default = "default_trailing_punct")]
    pub trailing_punct: String,
}

fn default_true() -> bool {
    true
}
fn default_refine_fallback() -> Vec<String> {
    vec!["嗯".to_string(), "呃".to_string()]
}
fn default_min_refine_length() -> usize {
    5
}
fn default_refine_db_path() -> String {
    "refine_log.db".to_string()
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

impl Config {
    pub fn load() -> Result<Self> {
        let exe_dir = get_exe_dir()?;
        let config_path = exe_dir.join("config.json");
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?;
        let mut cfg: Config = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", config_path.display()))?;
        cfg.vad_model_dir = resolve_path(&exe_dir, &cfg.vad_model_dir)
            .to_str()
            .unwrap()
            .to_string();
        cfg.asr_model_dir = resolve_path(&exe_dir, &cfg.asr_model_dir)
            .to_str()
            .unwrap()
            .to_string();
        cfg.punc_model_dir = resolve_path(&exe_dir, &cfg.punc_model_dir)
            .to_str()
            .unwrap()
            .to_string();
        cfg.llm_model_dir = resolve_path(&exe_dir, &cfg.llm_model_dir)
            .to_str()
            .unwrap()
            .to_string();
        cfg.refine_db_path = resolve_path(&exe_dir, &cfg.refine_db_path)
            .to_str()
            .unwrap()
            .to_string();
        Ok(cfg)
    }

    pub fn exe_dir() -> Result<PathBuf> {
        get_exe_dir()
    }

    pub fn save_active_scheme(scheme_name: &str) -> Result<()> {
        let exe_dir = get_exe_dir()?;
        let config_path = exe_dir.join("config.json");
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
}
