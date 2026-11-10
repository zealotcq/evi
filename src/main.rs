//! Voice Input Method — main entry point.
//!
//! Interaction: Hold Right Ctrl → record → VAD segments in real-time → ASR each
//! segment → release Ctrl → inject combined text → detect corrections.
//!
//! Also supports toggling recording via the TSF language bar button.
//!
//! Overlay states: Hidden → Recording (while held) → Processing (after release) → Hidden

#![cfg_attr(target_os = "windows", windows_subsystem = "windows")]

use anyhow::Result;
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(not(target_os = "macos"))]
use std::time::Duration;
use vi::audio::cpal_source::CpalAudioSource;
use vi::engine::correction::FileCorrectionStore;
use vi::engine::debug_refine::DebugRefine;
use vi::engine::paraformer::AsrEngine;
use vi::engine::punc::PuncEngine;
use vi::engine::segmenter::segment_audio;
use vi::engine::speaker::{default_model_path, SpeakerEngine};
use vi::engine::vad::VadEngine;
use vi::refine_mgr::RefineManager;
use vi::ui::log_capture::CaptureLogger;
#[cfg(not(target_os = "macos"))]
use vi::ui::PromptState;
use vi::*;

#[cfg(target_os = "macos")]
use vi::ui::macos_tray::MacTray;

fn log_event(event: &str, detail: &str) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let ts = now.as_millis();
    let msg = if detail.is_empty() {
        format!("{} [{}]\n", ts, event)
    } else {
        format!("{} [{}] {}\n", ts, event, detail)
    };
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("log/vi_events.log")
    {
        let _ = f.write_all(msg.as_bytes());
    }
    info!("{}", msg.trim());
}

const DEFAULT_SAMPLE_RATE: u32 = 16000;
struct Session {
    audio_source: Mutex<Box<dyn AudioSource>>,
    text_session: Mutex<vi::text::PlatformTextSession>,
    vad: Mutex<VadEngine>,
    asr: Mutex<AsrEngine>,
    dr: Mutex<DebugRefine>,
    refine_mgr: Mutex<RefineManager>,
    speaker: Mutex<Option<SpeakerEngine>>,
    voice_refs: Mutex<Vec<Vec<f32>>>,
    correction: FileCorrectionStore,
    recording: AtomicBool,
    calibrating: AtomicBool,
    results: Mutex<Vec<SegmentResult>>,
}

impl Session {
    #[cfg(target_os = "windows")]
    fn new(cfg: &Config) -> Result<Self> {
        let base = crate::models::base_dir(cfg);
        let vad_dir = crate::models::vad_model_dir(&base);
        let asr_dir = crate::models::asr_model_dir(&base);
        let punc_dir = crate::models::punc_model_dir(&base);

        debug!("Loading VAD model from {}...", vad_dir.display());
        let vad = VadEngine::new(&vad_dir)?;
        debug!("Loading ASR model from {}...", asr_dir.display());
        let asr = AsrEngine::new(&asr_dir)?;
        debug!("Loading Punc model from {}...", punc_dir.display());
        let punc = PuncEngine::new(&punc_dir)?;

        let dr = DebugRefine::open(crate::models::refine_db_path().to_str().unwrap())?;
        let refine_mgr = RefineManager::new(cfg, punc)?;

        let audio_source: Box<dyn AudioSource> =
            Box::new(CpalAudioSource::new(DEFAULT_SAMPLE_RATE)?);
        let text_session = vi::text::tsf::create_platform_session()?;
        let correction = FileCorrectionStore::new("refine_log")?;

        let sp_path = default_model_path();
        let speaker = if sp_path.exists() {
            match SpeakerEngine::new(&sp_path) {
                Ok(sp) => {
                    info!("Speaker model loaded.");
                    Some(sp)
                }
                Err(e) => {
                    warn!("Speaker model load failed: {e}");
                    None
                }
            }
        } else {
            info!("Speaker model not found, voice gate inactive.");
            None
        };
        let voice_refs: Vec<Vec<f32>> = vi::load_voice_refs().into_iter().map(|(_, e)| e).collect();
        info!("Voice gate references: {}", voice_refs.len());

        Ok(Self {
            speaker: Mutex::new(speaker),
            voice_refs: Mutex::new(voice_refs),
            audio_source: Mutex::new(audio_source),
            text_session: Mutex::new(text_session),
            vad: Mutex::new(vad),
            asr: Mutex::new(asr),
            dr: Mutex::new(dr),
            refine_mgr: Mutex::new(refine_mgr),
            correction,
            recording: AtomicBool::new(false),
            calibrating: AtomicBool::new(false),
            results: Mutex::new(Vec::new()),
        })
    }

    #[cfg(not(target_os = "windows"))]
    fn new(_cfg: &Config) -> Result<Self> {
        let base = crate::models::base_dir(_cfg);
        let vad_dir = crate::models::vad_model_dir(&base);
        let asr_dir = crate::models::asr_model_dir(&base);
        let punc_dir = crate::models::punc_model_dir(&base);

        debug!("Loading VAD model from {}...", vad_dir.display());
        let vad = VadEngine::new(&vad_dir)?;
        debug!("Loading ASR model from {}...", asr_dir.display());
        let asr = AsrEngine::new(&asr_dir)?;
        debug!("Loading Punc model from {}...", punc_dir.display());
        let punc = PuncEngine::new(&punc_dir)?;

        let dr = DebugRefine::open(crate::models::refine_db_path().to_str().unwrap())?;
        let refine_mgr = RefineManager::new(_cfg, punc)?;

        let audio_source: Box<dyn AudioSource> =
            Box::new(CpalAudioSource::new(DEFAULT_SAMPLE_RATE)?);

        #[cfg(target_os = "macos")]
        let text_session = vi::text::macos::create_platform_session()?;

        #[cfg(not(target_os = "macos"))]
        let text_session = {
            use vi::text::PlatformTextSession;
            struct SOut;
            impl TextOutput for SOut {
                fn commit_text(&self, t: &str) -> Result<()> {
                    print!("{}", t);
                    Ok(())
                }
                fn method_name(&self) -> &str {
                    "stdio"
                }
            }
            struct SObs;
            impl TextObserver for SObs {
                fn start_monitoring(&mut self) -> Result<()> {
                    Ok(())
                }
                fn stop_monitoring(&mut self) -> Result<()> {
                    Ok(())
                }
                fn poll_changes(&self) -> Vec<TextChangeEvent> {
                    Vec::new()
                }
                fn snapshot(&self) -> Result<TextSnapshot> {
                    Ok(TextSnapshot {
                        full_text: String::new(),
                        cursor_position: 0,
                        selection_start: 0,
                        selection_end: 0,
                    })
                }
            }
            PlatformTextSession::new(Box::new(SOut), Box::new(SObs))
        };

        let correction = FileCorrectionStore::new("refine_log")?;

        let sp_path = default_model_path();
        let speaker = if sp_path.exists() {
            match SpeakerEngine::new(&sp_path) {
                Ok(sp) => {
                    info!("Speaker model loaded.");
                    Some(sp)
                }
                Err(e) => {
                    warn!("Speaker model load failed: {e}");
                    None
                }
            }
        } else {
            info!("Speaker model not found, voice gate inactive.");
            None
        };
        let voice_refs: Vec<Vec<f32>> = vi::load_voice_refs().into_iter().map(|(_, e)| e).collect();
        info!("Voice gate references: {}", voice_refs.len());

        Ok(Self {
            speaker: Mutex::new(speaker),
            voice_refs: Mutex::new(voice_refs),
            audio_source: Mutex::new(audio_source),
            text_session: Mutex::new(text_session),
            vad: Mutex::new(vad),
            asr: Mutex::new(asr),
            dr: Mutex::new(dr),
            refine_mgr: Mutex::new(refine_mgr),
            correction,
            recording: AtomicBool::new(false),
            calibrating: AtomicBool::new(false),
            results: Mutex::new(Vec::new()),
        })
    }

    fn start_recording(&self, calibrate: bool) {
        if self.recording.load(Ordering::SeqCst) {
            return;
        }
        self.calibrating.store(calibrate, Ordering::SeqCst);
        if calibrate {
            debug!("Calibration recording started (Right Shift held)");
            log_event("CALIBRATION_START", "");
        } else {
            debug!("Recording started (Right Ctrl held)");
            log_event("RECORDING_START", "");
            if self.voice_refs.lock().is_empty() {
                info!("Voice gate: no references yet; hold Right Shift and speak to calibrate.");
            }
        }
        self.results.lock().clear();
        if let Some(mut audio) = self.audio_source.try_lock() {
            if let Err(e) = audio.start() {
                error!("Failed to start recording: {e}");
                return;
            }
        }
        self.recording.store(true, Ordering::SeqCst);
        if let Some(mut ts) = self.text_session.try_lock() {
            ts.start_monitoring().ok();
        }
    }

    fn stop_recording_and_process(&self) {
        if !self.recording.load(Ordering::SeqCst) {
            return;
        }
        self.recording.store(false, Ordering::SeqCst);
        let calibrating = self.calibrating.swap(false, Ordering::SeqCst);
        if calibrating {
            debug!("Calibration recording stopped (Right Shift released)");
        } else {
            debug!("Recording stopped (Right Ctrl released)");
        }
        log_event("RECORDING_STOP", "");

        let i16_samples = {
            let mut audio = self.audio_source.lock();
            match audio.stop() {
                Ok(s) => s,
                Err(e) => {
                    error!("Failed to stop recording: {e}");
                    return;
                }
            }
        };

        if i16_samples.is_empty() {
            if calibrating {
                warn!("Calibration failed: no audio captured");
                log_event("CALIBRATION_FAILED", "no audio");
            } else {
                warn!("No audio captured");
            }
            return;
        }

        let pcm: Vec<f32> = i16_samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let total_s = pcm.len() as f64 / DEFAULT_SAMPLE_RATE as f64;
        debug!("Captured {:.1}s of audio", total_s);

        if calibrating {
            let analysis = self.vad.lock().analyze(&pcm).ok().flatten();
            if let Some(segments) = analysis {
                let mut speech = Vec::new();
                for seg in &segments {
                    let s0 = (seg.start_ms as f64 / 1000.0 * 16000.0) as usize;
                    let s1 = ((seg.end_ms as f64 / 1000.0 * 16000.0) as usize).min(pcm.len());
                    if s1 > s0 {
                        speech.extend_from_slice(&pcm[s0..s1]);
                    }
                }
                if !speech.is_empty() {
                    let emb = {
                        let mut guard = self.speaker.lock();
                        match guard.as_mut() {
                            Some(sp) => sp.embed(&speech).ok(),
                            None => None,
                        }
                    };
                    if let Some(emb) = emb {
                        vi::push_voice_ref(&vi::calib_timestamp(), &emb);
                        *self.voice_refs.lock() =
                            vi::load_voice_refs().into_iter().map(|(_, e)| e).collect();
                        info!(
                            "Voice gate: reference saved ({} refs)",
                            self.voice_refs.lock().len()
                        );
                        log_event("VOICE_REF_SAVED", "");
                    }
                }
            }
        }

        let segments = match segment_audio(&pcm, DEFAULT_SAMPLE_RATE, &mut self.vad.lock()) {
            Ok(s) => s,
            Err(e) => {
                error!("Segmentation failed: {e}");
                return;
            }
        };

        if segments.is_empty() {
            warn!("No speech detected");
            return;
        }

        if vi::ui::get_voice_gate_enabled() {
            let refs = self.voice_refs.lock().clone();
            if refs.is_empty() {
                info!("Voice gate enabled but no references; hold Right Shift to calibrate");
            } else {
                let mut speech = Vec::new();
                for seg in &segments {
                    speech.extend_from_slice(&seg.samples);
                }
                let allowed = {
                    let mut sp_guard = self.speaker.lock();
                    match sp_guard.as_mut() {
                        Some(sp) => match sp.embed(&speech) {
                            Ok(emb) => {
                                let thr = vi::ui::get_voice_gate_threshold();
                                let sims: Vec<f32> = refs
                                    .iter()
                                    .map(|r| SpeakerEngine::cosine(r, &emb))
                                    .collect();
                                let pass = sims.iter().any(|&s| s as f64 >= thr);
                                info!("Voice gate: sims {:?} threshold {} -> {}", sims, thr, pass);
                                pass
                            }
                            Err(_) => true,
                        },
                        None => true,
                    }
                };
                if !allowed {
                    info!("Voice gate: recording rejected (no reference matched)");
                    log_event("VOICE_GATE_REJECTED", "");
                    return;
                }
            }
        }

        for seg in &segments {
            match self.asr.lock().recognize(&seg.samples, seg.start_ms) {
                Ok((text, chars, tokens)) => {
                    if !text.is_empty() {
                        self.results.lock().push(SegmentResult {
                            segment_index: seg.index,
                            text,
                            chars,
                            tokens,
                            audio_start_ms: seg.start_ms,
                            audio_end_ms: seg.end_ms,
                            audio_samples: seg.samples.clone(),
                            sample_rate: seg.sample_rate,
                        });
                    }
                }
                Err(e) => error!("ASR failed for segment {}: {e}", seg.index),
            }
        }

        let full_text: String = self
            .results
            .lock()
            .iter()
            .map(|r| r.text.as_str())
            .collect();
        if full_text.is_empty() {
            warn!("ASR produced no text");
            return;
        }

        debug!("Recognized: {}", full_text);
        log_event("ASR_COMPLETE", &full_text);

        let dr = self.dr.lock();
        let (final_text, _) = self.refine_mgr.lock().refine(&full_text, &dr);

        debug!("Refined: {}", final_text);
        log_event("REFINE_COMPLETE", &final_text);

        if let Err(e) = self.text_session.lock().commit_text(&final_text) {
            error!("Failed to inject text: {e}");
        }
    }

    fn check_corrections(&self) {
        let mut changes = self.text_session.lock().poll_changes();
        if changes.is_empty() {
            let full_text: String = {
                let results = self.results.lock();
                results.iter().map(|r| r.text.as_str()).collect()
            };
            if !full_text.is_empty() {
                if let Ok(snapshot) = self.text_session.lock().snapshot() {
                    if !snapshot.full_text.is_empty() && snapshot.full_text != full_text {
                        changes.push(TextChangeEvent::FullReplace {
                            old: full_text.clone(),
                            new: snapshot.full_text.clone(),
                        });
                    }
                }
            }
        }
        if changes.is_empty() {
            return;
        }

        let results = self.results.lock();
        let full_text: String = results.iter().map(|r| r.text.as_str()).collect();
        let all_timings: Vec<CharTiming> = results.iter().flat_map(|r| r.chars.clone()).collect();

        for change in &changes {
            match change {
                TextChangeEvent::FullReplace { old, new } if old == &full_text => {
                    self.correction
                        .detect_and_record(old, new, &all_timings, &now_iso());
                }
                TextChangeEvent::Replaced { old, new, .. } if old != new && !old.is_empty() => {
                    self.correction
                        .detect_and_record(old, new, &all_timings, &now_iso());
                }
                _ => {}
            }
        }
    }
}

fn now_iso() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = secs / 86400;
    let tod = secs % 86400;
    let (y, mo, d) = days_to_date(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        mo,
        d,
        tod / 3600,
        (tod % 3600) / 60,
        tod % 60
    )
}

fn days_to_date(d: u64) -> (u64, u64, u64) {
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
// Windows: low-level keyboard hook for Right Ctrl hold detection
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(target_os = "windows")]
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--refine-editor") {
        return crate::ui::refine_editor::run_refine_editor();
    }
    if args.iter().any(|a| a == "--privacy-dialog") {
        return crate::ui::privacy_dialog::run_privacy_dialog();
    }

    let log_buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    vi::ui::log_capture::init_log_capture(log_buffer.clone());

    log::set_boxed_logger(Box::new(CaptureLogger))
        .map(|()| log::set_max_level(log::LevelFilter::Info))
        .unwrap();

    info!("EVI Voice Input Method starting...");

    vi::secret::load_key();

    if crate::models::find_model_base_dir().is_none() {
        info!("Models not found in any ModelScope cache, downloading...");
        if let Err(e) = vi::download_model::run_download_window() {
            show_messagebox("EVI 输入法", &format!("模型下载失败: {}", e));
            return Err(e);
        }
        show_messagebox("EVI 输入法", "模型下载完成，请重新启动程序。");
        return Ok(());
    }

    if let Ok(cfg) = Config::load() {
        if !cfg.privacy_acknowledged {
            let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("evi"));
            let _ = std::process::Command::new(exe)
                .arg("--privacy-dialog")
                .status();
        }
        let scheme = if cfg.refine_scheme != "default" {
            cfg.refine_scheme.clone()
        } else if cfg.llm_remote_enabled {
            "llm_remote".to_string()
        } else {
            cfg.refine_scheme.clone()
        };
        vi::ui::set_refine_scheme(&scheme);
        info!("Restored refine_scheme from config: {}", scheme);
        vi::ui::set_voice_gate_enabled(cfg.voice_gate_enabled);
        info!(
            "Restored voice_gate_enabled from config: {}",
            cfg.voice_gate_enabled
        );
        vi::ui::set_voice_gate_threshold(cfg.voice_gate_threshold);
        info!(
            "Restored voice_gate_threshold from config: {}",
            cfg.voice_gate_threshold
        );
        vi::ui::set_punc_enabled(cfg.punc_enabled);
        info!("Restored punc_enabled from config: {}", cfg.punc_enabled);
    }

    {
        use windows::core::PCWSTR;
        use windows::Win32::System::Threading::CreateMutexW;
        let name: Vec<u16> = "Global\\vi_input_single_instance\0"
            .encode_utf16()
            .collect();
        let mutex = unsafe { CreateMutexW(None, false, PCWSTR(name.as_ptr())) };
        let already_exists = unsafe {
            windows::Win32::Foundation::GetLastError()
                == windows::Win32::Foundation::ERROR_ALREADY_EXISTS
        };
        if already_exists {
            info!("Another instance is already running, exiting.");
            return Ok(());
        }
        std::mem::forget(mutex);
    }

    let cfg = Config::load()?;

    let running = Arc::new(AtomicBool::new(true));
    let needs_rebuild = Arc::new(AtomicBool::new(false));
    let session_holder: Arc<Mutex<Option<Arc<Session>>>> = Arc::new(Mutex::new(None));

    let session_hook = session_holder.clone();
    let running_hook = running.clone();

    std::thread::spawn(move || {
        use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
        use windows::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE, WM_QUIT,
        };

        const VK_RCONTROL: i32 = 0xA3;
        const VK_RSHIFT: i32 = 0xA1;
        let mut ctrl_held = false;
        let mut shift_held = false;
        let mut active_mode: Option<bool> = None;

        let mut msg = MSG::default();
        while running_hook.load(Ordering::SeqCst) {
            unsafe {
                while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                    if msg.message == WM_QUIT {
                        break;
                    }
                    TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }

            let ctrl_down = unsafe { GetAsyncKeyState(VK_RCONTROL) } as u16 & 0x8000 != 0;
            let shift_down = unsafe { GetAsyncKeyState(VK_RSHIFT) } as u16 & 0x8000 != 0;

            if shift_down != shift_held {
                shift_held = shift_down;
                let sess = session_hook.lock().clone();
                if let Some(sess) = sess {
                    if shift_down {
                        if active_mode.is_none() {
                            active_mode = Some(true);
                            sess.start_recording(true);
                        }
                    } else if active_mode == Some(true) {
                        active_mode = None;
                        sess.stop_recording_and_process();
                    }
                }
            }

            if ctrl_down != ctrl_held {
                ctrl_held = ctrl_down;
                let sess = session_hook.lock().clone();
                if let Some(sess) = sess {
                    if ctrl_down {
                        if active_mode.is_none() {
                            active_mode = Some(false);
                            sess.start_recording(false);
                        }
                    } else if active_mode == Some(false) {
                        active_mode = None;
                        sess.stop_recording_and_process();
                    }
                }
            }

            std::thread::sleep(Duration::from_millis(30));
        }
    });

    let session_rebuild_holder = session_holder.clone();
    let rebuild_running = running.clone();
    let needs_rebuild_hook = needs_rebuild.clone();
    let prompt_state: Arc<Mutex<Option<PromptState>>> = Arc::new(Mutex::new(None));
    let prompt_state_hook = prompt_state.clone();
    std::thread::spawn(move || {
        while rebuild_running.load(Ordering::SeqCst) {
            if needs_rebuild_hook.load(Ordering::SeqCst) {
                needs_rebuild_hook.store(false, Ordering::SeqCst);
                let state = prompt_state_hook.lock().take();
                if let Some(ps) = state {
                    let sess = session_rebuild_holder.lock().clone();
                    if let Some(sess) = sess {
                        match sess
                            .refine_mgr
                            .lock()
                            .rebuild_llm_prompt(&ps.system_prompt, &ps.prefill_template)
                        {
                            Ok(()) => info!("LLM prompt rebuilt successfully"),
                            Err(e) => error!("LLM prompt rebuild failed: {}", e),
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    });

    let session_load = session_holder.clone();
    let mut cfg_load = cfg.clone();
    std::thread::spawn(move || {
        info!("Loading models...");
        crate::models::ensure_model_dir(&mut cfg_load);
        match Session::new(&cfg_load) {
            Ok(session) => {
                *session_load.lock() = Some(Arc::new(session));
                info!("All models loaded.");
                info!("EVI 输入法已经上线! 按下右Ctrl键开始说话,说完后,松开. 按住右Shift键说一句话可校准音量门控.");
            }
            Err(e) => {
                error!("Failed to load models: {}", e);
                std::process::exit(1);
            }
        }
    });

    vi::ui::run_gui(
        log_buffer.clone(),
        needs_rebuild.clone(),
        prompt_state,
        cfg.debug,
    )?;

    running.store(false, Ordering::SeqCst);
    info!("Shutting down...");
    {
        let sess = session_holder.lock().clone();
        if let Some(sess) = sess {
            if sess.recording.load(Ordering::SeqCst) {
                sess.audio_source.lock().stop().ok();
            }
            sess.check_corrections();
        }
    }
    std::thread::sleep(Duration::from_secs(3));
    info!("Bye!");
    Ok(())
}

#[cfg(target_os = "windows")]
fn show_messagebox(title: &str, message: &str) {
    use windows::core::PCWSTR;
    let title_w: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    let msg_w: Vec<u16> = message.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        let _ = windows::Win32::UI::WindowsAndMessaging::MessageBoxW(
            None,
            PCWSTR(msg_w.as_ptr()),
            PCWSTR(title_w.as_ptr()),
            windows::Win32::UI::WindowsAndMessaging::MB_OK,
        );
    };
}

#[cfg(target_os = "macos")]
fn show_messagebox(title: &str, message: &str) {
    let esc = |s: &str| s.replace('\\', "\\\\").replace('"', "\\\"");
    let _ = std::process::Command::new("osascript")
        .arg("-e")
        .arg(format!(
            "display dialog \"{}\" with title \"{}\" buttons {{\"确定\"}} default button \"确定\" with icon note",
            esc(message),
            esc(title)
        ))
        .status();
}

#[cfg(target_os = "macos")]
fn request_all_permissions() -> Result<()> {
    use std::ffi::c_void;
    use std::ptr;

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn AXIsProcessTrustedWithOptions(options: *const c_void) -> bool;
        static kAXTrustedCheckOptionPrompt: *const c_void;
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGPreflightListenEventAccess() -> bool;
        fn CGRequestListenEventAccess() -> bool;
    }

    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFDictionaryCreate(
            allocator: *const c_void,
            keys: *const *const c_void,
            values: *const *const c_void,
            num_values: i64,
            key_callbacks: *const c_void,
            value_callbacks: *const c_void,
        ) -> *const c_void;
        static kCFTypeDictionaryKeyCallBacks: *const c_void;
        static kCFTypeDictionaryValueCallBacks: *const c_void;
        static kCFBooleanTrue: *const c_void;
    }

    let mut need_exit = false;

    // 1. Accessibility
    let ax_trusted = unsafe { AXIsProcessTrustedWithOptions(ptr::null()) };
    if !ax_trusted {
        eprintln!("需要「辅助功能」权限，正在请求...");
        unsafe {
            let key = kAXTrustedCheckOptionPrompt;
            let value = kCFBooleanTrue;
            let dict = CFDictionaryCreate(
                ptr::null(),
                &key,
                &value,
                1,
                kCFTypeDictionaryKeyCallBacks,
                kCFTypeDictionaryValueCallBacks,
            );
            AXIsProcessTrustedWithOptions(dict);
        }
        need_exit = true;
    }

    // 2. Input Monitoring
    let listen_ok = unsafe { CGPreflightListenEventAccess() };
    if !listen_ok {
        eprintln!("需要「输入监控」权限，正在请求...");
        unsafe {
            CGRequestListenEventAccess();
        }
        need_exit = true;
    }

    if need_exit {
        eprintln!("\n请在「系统设置 > 隐私与安全性」中启用以下权限：");
        if !ax_trusted {
            eprintln!("  - 辅助功能");
        }
        if !listen_ok {
            eprintln!("  - 输入监控");
        }
        eprintln!("  - 麦克风（首次录音时系统会自动弹出）");
        eprintln!("\n授权后请重新启动应用。");
        std::process::exit(1);
    }

    Ok(())
}

#[cfg(target_os = "macos")]
mod cg_key {
    use super::MacEvent;
    use core_graphics::event::{
        CGEventFlags, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType,
        EventField,
    };
    use parking_lot::Mutex;
    use std::cell::Cell;
    use std::sync::Arc;

    const KVK_RIGHT_COMMAND: i64 = 0x36;
    const KVK_RIGHT_SHIFT: i64 = 0x7C;
    const KVK_RIGHT_SHIFT_ADB: i64 = 0x3C;

    pub fn start(proxy: Arc<Mutex<Option<tao::event_loop::EventLoopProxy<MacEvent>>>>) {
        let events = vec![CGEventType::FlagsChanged];
        let prev_cmd_down = Cell::new(false);
        let prev_shift_down = Cell::new(false);

        let tap = core_graphics::event::CGEventTap::new(
            CGEventTapLocation::HID,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::ListenOnly,
            events,
            move |_proxy, _event_type, event| {
                let keycode = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
                let flags = event.get_flags();
                log::debug!("CGEvent: keycode={}, flags={:?}", keycode, flags);
                match keycode {
                    KVK_RIGHT_COMMAND => {
                        let now_down = flags.contains(CGEventFlags::CGEventFlagCommand);
                        let was_down = prev_cmd_down.replace(now_down);
                        if now_down != was_down {
                            if let Some(ref p) = *proxy.lock() {
                                if now_down {
                                    let _ = p.send_event(MacEvent::KeyDown);
                                } else {
                                    let _ = p.send_event(MacEvent::KeyUp);
                                }
                            }
                        }
                    }
                    KVK_RIGHT_SHIFT | KVK_RIGHT_SHIFT_ADB => {
                        let now_down = flags.contains(CGEventFlags::CGEventFlagShift);
                        let was_down = prev_shift_down.replace(now_down);
                        if now_down != was_down {
                            if let Some(ref p) = *proxy.lock() {
                                if now_down {
                                    let _ = p.send_event(MacEvent::CalKeyDown);
                                } else {
                                    let _ = p.send_event(MacEvent::CalKeyUp);
                                }
                            }
                        }
                    }
                    _ => {}
                }
                None
            },
        );

        match tap {
            Ok(t) => {
                t.enable();
                if let Ok(source) = t.mach_port.create_runloop_source(0) {
                    let rl = core_foundation::runloop::CFRunLoop::get_current();
                    unsafe {
                        rl.add_source(&source, core_foundation::runloop::kCFRunLoopCommonModes);
                    }
                }
                std::mem::forget(t);
                log::info!("CGEventTap created, listening for Right Command...");
                core_foundation::runloop::CFRunLoop::run_current();
            }
            Err(_) => {
                log::error!(
                    "CGEventTap failed. Grant Accessibility + Input Monitoring in System Settings."
                );
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
enum MacEvent {
    KeyDown,
    KeyUp,
    CalKeyDown,
    CalKeyUp,
    ProcessingDone,
    UpdateTray,
    Quit,
}

#[cfg(target_os = "macos")]
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--refine-editor") {
        return crate::ui::refine_editor::run_refine_editor();
    }
    if args.iter().any(|a| a == "--privacy-dialog") {
        return crate::ui::privacy_dialog::run_privacy_dialog();
    }

    request_all_permissions()?;

    #[cfg(target_arch = "x86_64")]
    let dylib_rel = "ort-dylib/onnxruntime-osx-x86_64-1.24.2/lib/libonnxruntime.1.24.2.dylib";
    #[cfg(target_arch = "aarch64")]
    let dylib_rel = "ort-dylib/onnxruntime-osx-arm64-1.24.2/lib/libonnxruntime.1.24.2.dylib";

    {
        let exe_dir = Config::exe_dir()?;
        let dylib_path = exe_dir.join(dylib_rel);
        if !dylib_path.exists() {
            anyhow::bail!("ORT dylib not found at {}", dylib_path.display());
        }
        if !ort::init_from(dylib_path)?.commit() {
            anyhow::bail!("Failed to initialize ONNX Runtime");
        }
    }

    use tao::event::{Event, StartCause};
    use tao::event_loop::{ControlFlow, EventLoopBuilder};
    use tao::platform::macos::{ActivationPolicy, EventLoopExtMacOS};
    use tray_icon::menu::{MenuEvent, MenuItem};

    let log_buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    vi::ui::log_capture::init_log_capture(log_buffer.clone());

    log::set_boxed_logger(Box::new(CaptureLogger))
        .map(|()| log::set_max_level(log::LevelFilter::Info))
        .unwrap();

    info!("EVI Voice Input Method starting (macOS)...");

    vi::secret::load_key();

    if let Ok(cfg) = Config::load() {
        let scheme = if cfg.refine_scheme != "default" {
            cfg.refine_scheme.clone()
        } else if cfg.llm_remote_enabled {
            "llm_remote".to_string()
        } else {
            cfg.refine_scheme.clone()
        };
        vi::ui::set_refine_scheme(&scheme);
        info!("Restored refine_scheme from config: {}", scheme);
        vi::ui::set_voice_gate_enabled(cfg.voice_gate_enabled);
        info!(
            "Restored voice_gate_enabled from config: {}",
            cfg.voice_gate_enabled
        );
        vi::ui::set_voice_gate_threshold(cfg.voice_gate_threshold);
        info!(
            "Restored voice_gate_threshold from config: {}",
            cfg.voice_gate_threshold
        );
        vi::ui::set_punc_enabled(cfg.punc_enabled);
        info!("Restored punc_enabled from config: {}", cfg.punc_enabled);
        if !cfg.privacy_acknowledged {
            let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("evi"));
            let _ = std::process::Command::new(exe)
                .arg("--privacy-dialog")
                .status();
        }
    }

    if crate::models::find_model_base_dir().is_none() {
        info!("Models not found in any ModelScope cache, downloading...");
        vi::download_model::run_download_window().map_err(|e| {
            show_messagebox("EVI 输入法", &format!("模型下载失败: {}", e));
            e
        })?;
        show_messagebox("EVI 输入法", "模型下载完成，请重新启动程序。");
        return Ok(());
    }

    let mut cfg = Config::load()?;
    if let Some(base) = crate::models::find_model_base_dir() {
        cfg.model_base_dir = Some(base.to_string_lossy().into_owned());
        let _ = crate::Config::save_model_base_dir(cfg.model_base_dir.as_deref());
    }

    info!("Loading models...");

    let session = Arc::new(Session::new(&cfg)?);
    info!("All models loaded.");

    let mut event_loop = EventLoopBuilder::<MacEvent>::with_user_event().build();
    event_loop.set_activation_policy(ActivationPolicy::Accessory);
    let proxy = event_loop.create_proxy();

    let quit_item = MenuItem::new("退出", true, None);
    let quit_id = quit_item.id().clone();
    let refine_default_item = MenuItem::new("默认润色", true, None);
    let refine_default_id = refine_default_item.id().clone();
    let coze_refine_item = MenuItem::new("网络大模型润色", true, None);
    let coze_id = coze_refine_item.id().clone();
    let voice_gate_item = MenuItem::new("声纹门控", true, None);
    let voice_gate_id = voice_gate_item.id().clone();
    let punc_item = MenuItem::new("智能标点", true, None);
    let punc_id = punc_item.id().clone();
    let set_key_item = MenuItem::new("设置 API Key", true, None);
    let set_key_id = set_key_item.id().clone();
    let edit_db_item = MenuItem::new("帮助我们改进", true, None);
    let edit_db_id = edit_db_item.id().clone();
    let privacy_item = MenuItem::new("隐私说明...", true, None);
    let privacy_id = privacy_item.id().clone();

    let tray: Arc<MacTray> = Arc::new(
        MacTray::new(
            quit_item,
            refine_default_item,
            coze_refine_item,
            voice_gate_item,
            punc_item,
            edit_db_item,
            set_key_item,
            privacy_item,
            cfg.debug,
        )
        .map_err(|e| anyhow::anyhow!("Failed to create tray: {}", e))?,
    );

    {
        let has_key = vi::secret::get_api_key().is_some();
        tray.update_refine_route(&vi::ui::get_refine_scheme(), has_key);
        tray.update_voice_gate(vi::ui::get_voice_gate_enabled());
        tray.update_punc(vi::ui::get_punc_enabled());
    }

    let menu_proxy = proxy.clone();
    MenuEvent::set_event_handler(Some(move |event: tray_icon::menu::MenuEvent| {
        if event.id == quit_id {
            let _ = menu_proxy.send_event(MacEvent::Quit);
        } else if event.id == refine_default_id {
            vi::ui::set_refine_scheme("default");
            let _ = menu_proxy.send_event(MacEvent::UpdateTray);
        } else if event.id == coze_id {
            if vi::secret::get_api_key().is_none() {
                vi::ui::api_key_dialog::request_api_key_dialog();
            } else {
                vi::ui::set_refine_scheme("llm_remote");
            }
            let _ = menu_proxy.send_event(MacEvent::UpdateTray);
        } else if event.id == voice_gate_id {
            let current = vi::ui::get_voice_gate_enabled();
            vi::ui::set_voice_gate_enabled(!current);
            let _ = menu_proxy.send_event(MacEvent::UpdateTray);
        } else if event.id == punc_id {
            let current = vi::ui::get_punc_enabled();
            vi::ui::set_punc_enabled(!current);
            let _ = menu_proxy.send_event(MacEvent::UpdateTray);
        } else if event.id == set_key_id {
            vi::ui::api_key_dialog::request_api_key_dialog();
            let _ = menu_proxy.send_event(MacEvent::UpdateTray);
        } else if event.id == edit_db_id {
            if let Ok(exe) = std::env::current_exe() {
                let _ = std::process::Command::new(exe)
                    .arg("--refine-editor")
                    .spawn();
            }
        } else if event.id == privacy_id {
            if let Ok(exe) = std::env::current_exe() {
                let _ = std::process::Command::new(exe)
                    .arg("--privacy-dialog")
                    .spawn();
            }
        }
    }));

    let cg_proxy: Arc<Mutex<Option<tao::event_loop::EventLoopProxy<MacEvent>>>> =
        Arc::new(Mutex::new(Some(proxy.clone())));

    std::thread::Builder::new()
        .name("cg-event-tap".into())
        .spawn(move || {
            cg_key::start(cg_proxy);
        })
        .expect("Failed to spawn key listener thread");

    let mut recording = false;
    let mut calibrating = false;
    let mut processing = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::NewEvents(StartCause::Init) => {
                info!("EVI 输入法已上线! 按住右Command键录音，松开识别。按住右Shift键说一句话可校准音量门控。");
            }

            Event::UserEvent(MacEvent::KeyDown) => {
                if !recording && !calibrating && !processing {
                    recording = true;
                    info!("Starting recording (Right Command)...");
                    session.start_recording(false);
                }
            }

            Event::UserEvent(MacEvent::CalKeyDown) => {
                if !recording && !calibrating && !processing {
                    calibrating = true;
                    info!("Starting calibration recording (Right Shift)...");
                    session.start_recording(true);
                }
            }

            Event::UserEvent(MacEvent::KeyUp) => {
                if recording && !processing {
                    recording = false;
                    processing = true;
                    info!("Stopping recording (Right Command)...");
                    let sess = session.clone();
                    let done_proxy = proxy.clone();
                    std::thread::Builder::new()
                        .name("asr-process".into())
                        .spawn(move || {
                            sess.stop_recording_and_process();
                            let _ = done_proxy.send_event(MacEvent::ProcessingDone);
                        })
                        .expect("Failed to spawn ASR thread");
                }
            }

            Event::UserEvent(MacEvent::CalKeyUp) => {
                if calibrating && !processing {
                    calibrating = false;
                    processing = true;
                    info!("Stopping calibration recording (Right Shift)...");
                    let sess = session.clone();
                    let done_proxy = proxy.clone();
                    std::thread::Builder::new()
                        .name("calibration-process".into())
                        .spawn(move || {
                            sess.stop_recording_and_process();
                            let _ = done_proxy.send_event(MacEvent::ProcessingDone);
                        })
                        .expect("Failed to spawn calibration thread");
                }
            }

            Event::UserEvent(MacEvent::ProcessingDone) | Event::UserEvent(MacEvent::UpdateTray) => {
                if let Event::UserEvent(MacEvent::ProcessingDone) = event {
                    processing = false;
                }
                let has_key = vi::secret::get_api_key().is_some();
                let scheme = vi::ui::get_refine_scheme();
                tray.update_refine_route(&scheme, has_key);
                tray.update_voice_gate(vi::ui::get_voice_gate_enabled());
                let punc_enabled = vi::ui::get_punc_enabled();
                tray.update_punc(punc_enabled);
            }

            Event::UserEvent(MacEvent::Quit) => {
                info!("Shutting down...");
                if session.recording.load(Ordering::SeqCst) {
                    session.audio_source.lock().stop().ok();
                }
                session.check_corrections();
                *control_flow = ControlFlow::Exit;
            }

            _ => {}
        }
    });

    #[allow(unreachable_code)]
    Ok(())
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--refine-editor") {
        return crate::ui::refine_editor::run_refine_editor();
    }
    if args.iter().any(|a| a == "--privacy-dialog") {
        return crate::ui::privacy_dialog::run_privacy_dialog();
    }

    let log_buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    vi::ui::log_capture::init_log_capture(log_buffer.clone());

    log::set_boxed_logger(Box::new(CaptureLogger))
        .map(|()| log::set_max_level(log::LevelFilter::Info))
        .unwrap();

    info!("EVI Voice Input Method starting...");

    if let Ok(cfg) = Config::load() {
        if !cfg.privacy_acknowledged {
            let exe = std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("evi"));
            let _ = std::process::Command::new(exe)
                .arg("--privacy-dialog")
                .status();
        }
    }

    let cfg = Config::load()?;
    let session = Arc::new(Session::new(&cfg)?);
    info!("All models loaded.");

    let running = Arc::new(AtomicBool::new(true));
    let needs_rebuild = Arc::new(AtomicBool::new(false));
    let prompt_state: Arc<Mutex<Option<PromptState>>> = Arc::new(Mutex::new(None));

    let session_hook = session.clone();
    let running_hook = running.clone();
    let needs_rebuild_hook = needs_rebuild.clone();
    let prompt_state_hook = prompt_state.clone();
    std::thread::spawn(move || {
        while running_hook.load(Ordering::SeqCst) {
            if needs_rebuild_hook.load(Ordering::SeqCst) {
                needs_rebuild_hook.store(false, Ordering::SeqCst);
                let state = prompt_state_hook.lock().take();
                if let Some(ps) = state {
                    match session_hook
                        .refine_mgr
                        .lock()
                        .rebuild_llm_prompt(&ps.system_prompt, &ps.prefill_template)
                    {
                        Ok(()) => info!("LLM prompt rebuilt successfully"),
                        Err(e) => error!("LLM prompt rebuild failed: {}", e),
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    });

    info!("VI ready!");
    vi::ui::run_gui(
        log_buffer.clone(),
        needs_rebuild.clone(),
        prompt_state,
        cfg.debug,
    )?;

    running.store(false, Ordering::SeqCst);
    info!("Shutting down...");
    if session.recording.load(Ordering::SeqCst) {
        session.audio_source.lock().stop().ok();
    }
    std::thread::sleep(Duration::from_secs(3));
    session.check_corrections();
    info!("Bye!");
    Ok(())
}
