//! Correction store with audio-time mapping.
//!
//! When the user modifies recognized text, we:
//! 1. Diff the original vs modified text to find changed spans
//! 2. Map changed spans back to audio timestamps using CharTiming
//! 3. Persist corrections as JSONL for future ASR improvement

use crate::{CharTiming, Correction, CorrectionStore};
use anyhow::{Context, Result};
use log::info;
use parking_lot::Mutex;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use unicode_segmentation::UnicodeSegmentation;

pub struct FileCorrectionStore {
    path: PathBuf,
    corrections: Mutex<Vec<Correction>>,
}

impl FileCorrectionStore {
    pub fn new(dir: &str) -> Result<Self> {
        let path = PathBuf::from(dir);
        std::fs::create_dir_all(&path)?;
        let file = path.join("corrections.jsonl");
        Ok(Self {
            path: file,
            corrections: Mutex::new(Vec::new()),
        })
    }

    /// Detect corrections by comparing original recognized text with user-modified text.
    ///
    /// Uses the CharTiming info to map changed text positions back to audio timestamps.
    pub fn detect_and_record(
        &self,
        original: &str,
        modified: &str,
        char_timings: &[CharTiming],
        timestamp_iso: &str,
    ) -> Vec<Correction> {
        let corrections = diff_corrections(original, modified, char_timings, timestamp_iso);
        for c in &corrections {
            if let Err(e) = self.record(c.clone()) {
                log::error!("Failed to record correction: {e}");
            }
        }
        corrections
    }
}

impl CorrectionStore for FileCorrectionStore {
    fn record(&self, correction: Correction) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .context("Failed to open corrections file")?;
        writeln!(file, "{}", serde_json::to_string(&correction)?)?;
        file.flush()?;
        self.corrections.lock().push(correction);
        Ok(())
    }

    fn recent(&self, limit: usize) -> Vec<Correction> {
        self.corrections
            .lock()
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    fn load(&mut self) -> Result<()> {
        if !self.path.exists() {
            return Ok(());
        }
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        let mut loaded = Vec::new();
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(c) = serde_json::from_str::<Correction>(&line) {
                loaded.push(c);
            }
        }
        info!("Loaded {} corrections", loaded.len());
        *self.corrections.lock() = loaded;
        Ok(())
    }

    fn save(&self) -> Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        for c in self.corrections.lock().iter() {
            writeln!(file, "{}", serde_json::to_string(c)?)?;
        }
        file.flush()?;
        Ok(())
    }
}

/// Diff original vs modified text and produce corrections with audio timing.
fn diff_corrections(
    original: &str,
    modified: &str,
    timings: &[CharTiming],
    timestamp_iso: &str,
) -> Vec<Correction> {
    let mut corrections = Vec::new();

    if original == modified || timings.is_empty() {
        return corrections;
    }

    // Simple approach: find common prefix and suffix, the middle part is changed
    let orig_chars: Vec<&str> = original.graphemes(true).collect();
    let mod_chars: Vec<&str> = modified.graphemes(true).collect();

    let prefix_len = orig_chars
        .iter()
        .zip(mod_chars.iter())
        .take_while(|(a, b)| a == b)
        .count();

    let suffix_len = orig_chars[prefix_len..]
        .iter()
        .rev()
        .zip(mod_chars[prefix_len..].iter().rev())
        .take_while(|(a, b)| a == b)
        .count();

    let change_start = prefix_len;
    let change_end_orig = orig_chars.len() - suffix_len;
    let change_end_mod = mod_chars.len() - suffix_len;

    if change_start >= change_end_orig && change_start >= change_end_mod {
        return corrections;
    }

    let old_part: String = orig_chars[change_start..change_end_orig].concat();
    let new_part: String = mod_chars[change_start..change_end_mod].concat();

    if old_part == new_part {
        return corrections;
    }

    // Map to audio timing
    let audio_start = timings
        .get(change_start.min(timings.len() - 1))
        .map(|t| t.start_ms)
        .unwrap_or(0);

    let audio_end = timings
        .get(change_end_orig.saturating_sub(1).min(timings.len() - 1))
        .map(|t| t.end_ms)
        .unwrap_or(audio_start);

    let context_before: String = orig_chars[change_start.saturating_sub(5)..change_start].concat();
    let context_after: String = orig_chars
        [change_end_orig..change_end_orig + 5.min(orig_chars.len() - change_end_orig)]
        .concat();

    corrections.push(Correction {
        original: old_part,
        corrected: new_part,
        timestamp_iso: timestamp_iso.to_string(),
        context_before,
        context_after,
        audio_start_ms: audio_start,
        audio_end_ms: audio_end,
        audio_path: None,
    });

    corrections
}
