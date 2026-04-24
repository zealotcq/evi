//! Audio segmentation: VAD-based splitting, cut at next pause after exceeding 10s.
//!
//! Strategy:
//! 1. Run VAD on the full audio → get speech segments
//! 2. For each speech segment, if ≤ 10s → keep as-is
//! 3. If > 10s → start scanning from the 10s mark onward, wait for the
//!    next energy low point (pause), split there, repeat for the remainder

use crate::engine::vad::VadEngine;
use log::debug;

const MAX_SEGMENT_MS: u64 = 10_000;
const MIN_SEGMENT_MS: u64 = 100;
const ENERGY_FRAME_MS: u64 = 10;
const PAUSE_ENERGY_THRESHOLD: f64 = 0.005;
const MIN_PAUSE_FRAMES: usize = 5;

/// A single audio segment ready for ASR.
#[derive(Debug, Clone)]
pub struct AudioSegment {
    pub index: usize,
    pub start_ms: u64,
    pub end_ms: u64,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl AudioSegment {
    pub fn duration_ms(&self) -> u64 {
        self.end_ms - self.start_ms
    }
}

/// Segment audio using VAD, splitting long segments at the next pause after 10s.
pub fn segment_audio(
    pcm: &[f32],
    sample_rate: u32,
    vad: &mut VadEngine,
) -> anyhow::Result<Vec<AudioSegment>> {
    let total_ms = (pcm.len() as f64 / sample_rate as f64 * 1000.0) as u64;
    debug!(
        "Segmenting {} samples ({:.1}s)...",
        pcm.len(),
        total_ms as f64 / 1000.0
    );

    let speech_segments = vad.detect(pcm)?;
    if speech_segments.is_empty() {
        debug!("No speech detected");
        return Ok(vec![]);
    }

    debug!("VAD found {} speech segments", speech_segments.len());

    // Pre-compute frame-level energy for the entire recording
    let energies = compute_frame_energies(pcm, sample_rate);

    let mut result = Vec::new();
    let mut seg_index = 0;

    for speech in &speech_segments {
        let duration_ms = speech.end_ms - speech.start_ms;

        if duration_ms <= MAX_SEGMENT_MS {
            let samples = extract_samples(pcm, sample_rate, speech.start_ms, speech.end_ms);
            if samples.len() >= ms_to_samples(MIN_SEGMENT_MS, sample_rate) {
                result.push(AudioSegment {
                    index: seg_index,
                    start_ms: speech.start_ms,
                    end_ms: speech.end_ms,
                    samples,
                    sample_rate,
                });
                seg_index += 1;
            }
        } else {
            let sub =
                split_long_segment(pcm, sample_rate, speech.start_ms, speech.end_ms, &energies);
            for s in sub {
                if s.samples.len() >= ms_to_samples(MIN_SEGMENT_MS, sample_rate) {
                    result.push(AudioSegment {
                        index: seg_index,
                        start_ms: s.start_ms,
                        end_ms: s.end_ms,
                        samples: s.samples,
                        sample_rate,
                    });
                    seg_index += 1;
                }
            }
        }
    }

    debug!("Segmentation complete: {} segments", result.len());
    Ok(result)
}

/// Split a long segment by waiting for the next energy low point after 10s.
///
/// For each chunk: accumulate up to 10s, then keep scanning forward until a
/// genuine pause (consecutive low-energy frames) is found, and cut there.
fn split_long_segment(
    pcm: &[f32],
    sample_rate: u32,
    start_ms: u64,
    end_ms: u64,
    energies: &[FrameEnergy],
) -> Vec<AudioSegment> {
    let mut segments = Vec::new();
    let mut seg_start_ms = start_ms;

    while seg_start_ms < end_ms {
        let remaining_ms = end_ms - seg_start_ms;

        if remaining_ms <= MAX_SEGMENT_MS {
            // Remainder fits in one segment
            segments.push(AudioSegment {
                index: 0,
                start_ms: seg_start_ms,
                end_ms,
                samples: extract_samples(pcm, sample_rate, seg_start_ms, end_ms),
                sample_rate,
            });
            break;
        }

        // Exceeded 10s → scan forward from the 10s mark to find the next pause
        let threshold_ms = seg_start_ms + MAX_SEGMENT_MS;
        let split_ms = find_next_pause(energies, threshold_ms, end_ms);

        debug!(
            "Split at {}ms (started {}, threshold 10s at {}ms, next pause at {}ms)",
            split_ms, seg_start_ms, threshold_ms, split_ms
        );

        segments.push(AudioSegment {
            index: 0,
            start_ms: seg_start_ms,
            end_ms: split_ms,
            samples: extract_samples(pcm, sample_rate, seg_start_ms, split_ms),
            sample_rate,
        });

        seg_start_ms = split_ms;
    }

    segments
}

/// Starting from `threshold_ms`, scan forward until we find a genuine pause:
/// a run of ≥ MIN_PAUSE_FRAMES consecutive frames with energy below threshold.
///
/// Falls back to `threshold_ms` if we reach `end_ms` without finding a pause
/// (shouldn't happen with VAD-gated input, but defensive).
fn find_next_pause(energies: &[FrameEnergy], threshold_ms: u64, end_ms: u64) -> u64 {
    let mut consecutive_low = 0;
    let mut pause_start = threshold_ms;
    let mut global_min_energy = f64::MAX;
    let mut global_min_pos = threshold_ms;

    for fe in energies {
        if fe.time_ms < threshold_ms {
            continue;
        }
        if fe.time_ms > end_ms {
            break;
        }

        if fe.energy < global_min_energy {
            global_min_energy = fe.energy;
            global_min_pos = fe.time_ms;
        }

        if fe.energy < PAUSE_ENERGY_THRESHOLD {
            if consecutive_low == 0 {
                pause_start = fe.time_ms;
            }
            consecutive_low += 1;

            if consecutive_low >= MIN_PAUSE_FRAMES {
                debug!(
                    "Found pause at {}ms ({} consecutive low frames, energy={:.6})",
                    pause_start, consecutive_low, fe.energy
                );
                return pause_start;
            }
        } else {
            consecutive_low = 0;
        }
    }

    // No clear pause found → split at the global minimum energy point
    if global_min_pos > threshold_ms {
        debug!(
            "No clear pause after {}ms, splitting at energy minimum {}ms ({:.6})",
            threshold_ms, global_min_pos, global_min_energy
        );
        return global_min_pos;
    }

    threshold_ms
}

// ── Energy computation ─────────────────────────────────────────────────────────

struct FrameEnergy {
    time_ms: u64,
    energy: f64,
}

/// Compute per-frame RMS energy for the entire recording.
fn compute_frame_energies(pcm: &[f32], sample_rate: u32) -> Vec<FrameEnergy> {
    let frame_samples = (sample_rate as usize * ENERGY_FRAME_MS as usize) / 1000;
    let hop = frame_samples;
    let mut out = Vec::new();

    let mut pos = 0;
    while pos + frame_samples <= pcm.len() {
        let rms: f64 = pcm[pos..pos + frame_samples]
            .iter()
            .map(|&s| (s as f64) * (s as f64))
            .sum::<f64>()
            / frame_samples as f64;
        let time_ms = (pos as f64 / sample_rate as f64 * 1000.0) as u64;
        out.push(FrameEnergy {
            time_ms,
            energy: rms,
        });
        pos += hop;
    }

    out
}

// ── Helpers ────────────────────────────────────────────────────────────────────

fn extract_samples(pcm: &[f32], sample_rate: u32, start_ms: u64, end_ms: u64) -> Vec<f32> {
    let start = ms_to_samples(start_ms, sample_rate);
    let end = ms_to_samples(end_ms, sample_rate).min(pcm.len());
    if start >= end {
        vec![]
    } else {
        pcm[start..end].to_vec()
    }
}

fn ms_to_samples(ms: u64, sample_rate: u32) -> usize {
    (ms as f64 / 1000.0 * sample_rate as f64) as usize
}
