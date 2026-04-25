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
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
#[cfg(not(target_os = "macos"))]
use std::time::Duration;
use vi::audio::cpal_source::CpalAudioSource;
use vi::engine::correction::FileCorrectionStore;
use vi::engine::fallback::FallbackRefineEngine;
use vi::engine::paraformer::AsrEngine;
use vi::engine::punc::PuncEngine;
use vi::engine::debug_refine::DebugRefine;
use vi::TokenScore;
use vi::engine::llm::LlmEngine;
use vi::engine::segmenter::segment_audio;
use vi::engine::vad::VadEngine;
use vi::ui::log_capture::CaptureLogger;
#[cfg(not(target_os = "macos"))]
use vi::ui::PromptState;
use vi::*;

#[cfg(target_os = "windows")]
use vi::overlay::Overlay;

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
    punc: Mutex<PuncEngine>,
    db: Mutex<DebugRefine>,
    refine: RefineEngine,
    correction: FileCorrectionStore,
    recording: AtomicBool,
    #[cfg(target_os = "windows")]
    overlay: Overlay,
    results: Mutex<Vec<SegmentResult>>,
    audio_seq: AtomicU64,
    save_log: bool,
    trailing_punct: String,
}

enum RefineEngine {
    Llm(Box<Mutex<LlmEngine>>),
    Fallback(FallbackRefineEngine),
}

impl Session {
    #[cfg(target_os = "windows")]
    fn new(cfg: &Config) -> Result<Self> {
        let vad_dir = PathBuf::from(&cfg.vad_model_dir);
        let asr_dir = PathBuf::from(&cfg.asr_model_dir);
        let punc_dir = PathBuf::from(&cfg.punc_model_dir);

        debug!("Loading VAD model from {}...", vad_dir.display());
        let vad = VadEngine::new(&vad_dir)?;

        debug!("Loading ASR model from {}...", asr_dir.display());
        let asr = AsrEngine::new(&asr_dir)?;

        debug!("Loading Punc model from {}...", punc_dir.display());
        let punc = PuncEngine::new(&punc_dir)?;

        let db = DebugRefine::open(&cfg.refine_db_path)?;
        let refine = if cfg.llm_refine {
            debug!("Loading LLM model...");
            let engine = LlmEngine::new(cfg)?;
            RefineEngine::Llm(Box::new(Mutex::new(engine)))
        } else {
            RefineEngine::Fallback(FallbackRefineEngine::new(cfg.refine_fallback.clone()))
        };

        if cfg.save_log {
            let log_dir = PathBuf::from("log");
            std::fs::create_dir_all(&log_dir).ok();
        }

        let audio_source: Box<dyn AudioSource> =
            Box::new(CpalAudioSource::new(DEFAULT_SAMPLE_RATE)?);
        let text_session = vi::text::tsf::create_platform_session()?;
        let correction = FileCorrectionStore::new("refine_log")?;
        let overlay = Overlay::new();

        Ok(Self {
            audio_source: Mutex::new(audio_source),
            text_session: Mutex::new(text_session),
            vad: Mutex::new(vad),
            asr: Mutex::new(asr),
            punc: Mutex::new(punc),
            db: Mutex::new(db),
            refine,
            correction,
            recording: AtomicBool::new(false),
            overlay,
            results: Mutex::new(Vec::new()),
            audio_seq: AtomicU64::new(1),
            save_log: cfg.save_log,
            trailing_punct: cfg.trailing_punct.clone(),
        })
    }

    #[cfg(not(target_os = "windows"))]
    fn new(cfg: &Config) -> Result<Self> {
        let vad_dir = PathBuf::from(&cfg.vad_model_dir);
        let asr_dir = PathBuf::from(&cfg.asr_model_dir);

        debug!("Loading VAD model from {}...", vad_dir.display());
        let vad = VadEngine::new(&vad_dir)?;
        debug!("Loading ASR model from {}...", asr_dir.display());
        let asr = AsrEngine::new(&asr_dir)?;

        let db = DebugRefine::open(&cfg.refine_db_path)?;
        let refine = if cfg.llm_refine {
            debug!("Loading LLM model...");
            let engine = LlmEngine::new(cfg)?;
            RefineEngine::Llm(Box::new(Mutex::new(engine)))
        } else {
            RefineEngine::Fallback(FallbackRefineEngine::new(cfg.refine_fallback.clone()))
        };
        let refine = RefineEngine::Fallback(FallbackRefineEngine::new(cfg.refine_fallback.clone()));

        if cfg.save_log {
            let log_dir = PathBuf::from("log");
            std::fs::create_dir_all(&log_dir).ok();
        }

        let audio_source: Box<dyn AudioSource> =
            Box::new(CpalAudioSource::new(DEFAULT_SAMPLE_RATE)?);
        let correction = FileCorrectionStore::new("refine_log")?;

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

        Ok(Self {
            audio_source: Mutex::new(audio_source),
            text_session: Mutex::new(text_session),
            vad: Mutex::new(vad),
            asr: Mutex::new(asr),
            punc: Mutex::new(PuncEngine::new(&PathBuf::from(&cfg.punc_model_dir))?),
            db: Mutex::new(db),
            refine,
            correction,
            recording: AtomicBool::new(false),
            results: Mutex::new(Vec::new()),
            audio_seq: AtomicU64::new(1),
            save_log: cfg.save_log,
            trailing_punct: cfg.trailing_punct.clone(),
        })
    }

    fn start_recording(&self) {
        if self.recording.load(Ordering::SeqCst) {
            return;
        }
        debug!("Recording started (Right Ctrl held)");
        log_event("RECORDING_START", "");
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
        debug!("Recording stopped (Right Ctrl released)");
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
            warn!("No audio captured");
            return;
        }

        let pcm: Vec<f32> = i16_samples.iter().map(|&s| s as f32 / 32768.0).collect();
        let total_s = pcm.len() as f64 / DEFAULT_SAMPLE_RATE as f64;
        debug!("Captured {:.1}s of audio", total_s);

        let seq = self.audio_seq.fetch_add(1, Ordering::Relaxed);
        if self.save_log {
            if let Err(e) = save_wav(&i16_samples, DEFAULT_SAMPLE_RATE, seq) {
                warn!("Failed to save audio: {e}");
            }
        }

        let asr_result_path = PathBuf::from(format!("log/{}.json", seq));

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

        #[cfg(target_os = "windows")]
        self.overlay.show();

        let (refined_text, llm_tokens) = match &self.refine {
            RefineEngine::Llm(engine) => {
                let db = self.db.lock();
                match engine.lock().refine(&full_text, &db) {
                    Ok((refined, tokens)) => {
                        debug!("LLM refined: '{}' -> '{}'", full_text, refined);
                        log_event("LLM_REFINED", &format!("{} | {}", full_text, refined));
                        (refined, tokens)
                    }
                    Err(e) => {
                        warn!("LLM refinement failed: {e}, using raw ASR output");
                        (full_text, vec![])
                    }
                }
            }
            RefineEngine::Fallback(engine) => {
                let db = self.db.lock();
                let filtered = engine.refine(&full_text, &db);
                if filtered != full_text {
                    log_event("FALLBACK_FILTER", &format!("{} | {}", full_text, filtered));
                }
                (filtered, Vec::<TokenScore>::new())
            }
        };

        let final_text = match self.punc.lock().add_punct(&refined_text) {
            Ok(punct_text) => {
                debug!("Punc added: '{}' -> '{}'", refined_text, punct_text);
                log_event("PUNC_ADDED", &format!("{} | {}", refined_text, punct_text));
                punct_text
            }
            Err(e) => {
                warn!("Punc failed: {e}, using raw text");
                refined_text
            }
        };

        let final_text = self.ensure_trailing_punct(&final_text);

        {
            use serde_json::json;
            let segments_json: Vec<_> = self
                .results
                .lock()
                .iter()
                .map(|r| {
                    json!({
                        "index": r.segment_index,
                        "text": r.text,
                        "start_ms": r.audio_start_ms,
                        "end_ms": r.audio_end_ms,
                        "chars": r.chars.iter().map(|c| json!({
                            "text": c.text,
                            "start_ms": c.start_ms,
                            "end_ms": c.end_ms,
                        })).collect::<Vec<_>>(),
                        "tokens": r.tokens.iter().map(|t| json!({
                            "token": t.token,
                            "confidence": t.confidence,
                        })).collect::<Vec<_>>(),
                    })
                })
                .collect();
            let llm_tokens_json: Vec<_> = llm_tokens
                .iter()
                .map(|t| {
                    json!({
                        "token": t.token,
                        "confidence": t.confidence,
                    })
                })
                .collect();
            let result_json = json!({
                "text": final_text,
                "segments": segments_json,
                "llm_tokens": llm_tokens_json,
            });
            if self.save_log {
                if let Err(e) = std::fs::write(
                    &asr_result_path,
                    serde_json::to_string_pretty(&result_json).unwrap(),
                ) {
                    warn!("Failed to save ASR result: {e}");
                }
            }
        }

        #[cfg(target_os = "windows")]
        self.overlay.hide();

        if let Err(e) = self.text_session.lock().commit_text(&final_text) {
            error!("Failed to inject text: {e}");
        }
    }

    fn ensure_trailing_punct(&self, text: &str) -> String {
        let punct = ",.:?!;，。？！：";
        if let Some(ch) = text.chars().last() {
            if punct.contains(ch) {
                return text.to_string();
            }
        }
        format!("{}{}", text, self.trailing_punct)
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
    let log_buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    vi::ui::log_capture::init_log_capture(log_buffer.clone());

    log::set_boxed_logger(Box::new(CaptureLogger))
        .map(|()| log::set_max_level(log::LevelFilter::Info))
        .unwrap();

    info!("EVI Voice Input Method starting...");

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
    let ctrl_held = Arc::new(AtomicBool::new(false));
    let needs_rebuild = Arc::new(AtomicBool::new(false));
    let session_holder: Arc<Mutex<Option<Arc<Session>>>> = Arc::new(Mutex::new(None));

    let (ctrl_tx, ctrl_rx) = crossbeam_channel::unbounded::<bool>();

    let session_hook = session_holder.clone();
    let ctrl_held_hook = ctrl_held.clone();
    let running_hook = running.clone();

    vi::ui::HOOK_CHANNEL.lock().replace(ctrl_tx);

    std::thread::spawn(move || {
        use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
        use windows::Win32::UI::WindowsAndMessaging::{
            CallNextHookEx, DispatchMessageW, PeekMessageW, SetWindowsHookExW, TranslateMessage,
            UnhookWindowsHookEx, KBDLLHOOKSTRUCT, MSG, PM_REMOVE, WH_KEYBOARD_LL, WM_KEYDOWN,
            WM_KEYUP, WM_QUIT, WM_SYSKEYDOWN, WM_SYSKEYUP,
        };

        unsafe extern "system" fn ll_keyboard_proc(
            code: i32,
            wparam: WPARAM,
            lparam: LPARAM,
        ) -> LRESULT {
            if code >= 0 {
                let kb = *(lparam.0 as *const KBDLLHOOKSTRUCT);
                let vk = kb.vkCode;

                if vk == 0xA3 {
                    let msg = wparam.0 as u32;
                    let is_down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
                    let is_up = msg == WM_KEYUP || msg == WM_SYSKEYUP;

                    if is_down && kb.flags.0 & 0x10 == 0 {
                        if let Some(tx) = vi::ui::HOOK_CHANNEL.lock().as_ref() {
                            let _ = tx.try_send(true);
                        }
                    }
                    if is_up {
                        if let Some(tx) = vi::ui::HOOK_CHANNEL.lock().as_ref() {
                            let _ = tx.try_send(false);
                        }
                    }
                }
            }

            CallNextHookEx(None, code, wparam, lparam)
        }

        let hook = unsafe { SetWindowsHookExW(WH_KEYBOARD_LL, Some(ll_keyboard_proc), None, 0) }
            .expect("Failed to install keyboard hook");

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

            while let Ok(pressed) = ctrl_rx.try_recv() {
                let sess = session_hook.lock().clone();
                let Some(sess) = sess else { continue };
                let was_held = ctrl_held_hook.load(Ordering::SeqCst);
                if pressed && !was_held {
                    ctrl_held_hook.store(true, Ordering::SeqCst);
                    sess.start_recording();
                    sess.overlay.show();
                } else if !pressed && was_held {
                    ctrl_held_hook.store(false, Ordering::SeqCst);
                    sess.overlay.hide();
                    sess.stop_recording_and_process();
                }
            }

            std::thread::sleep(Duration::from_millis(10));
        }

        let _ = unsafe { UnhookWindowsHookEx(hook) };
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
                        if let RefineEngine::Llm(engine) = &sess.refine {
                            match engine
                                .lock()
                                .rebuild_prompt(&ps.system_prompt, &ps.prefill_template)
                            {
                                Ok(()) => info!("LLM prompt rebuilt successfully"),
                                Err(e) => error!("LLM prompt rebuild failed: {}", e),
                            }
                        }
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    });

    let session_load = session_holder.clone();
    let cfg_load = cfg.clone();
    std::thread::spawn(move || {
        info!("Loading models...");
        match Session::new(&cfg_load) {
            Ok(session) => {
                *session_load.lock() = Some(Arc::new(session));
                info!("All models loaded.");
                info!("EVI 输入法已经上线! 按下右Ctrl键开始说话,说完后,松开.");
            }
            Err(e) => {
                error!("Failed to load models: {}", e);
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
                ptr::null(), &key, &value, 1,
                kCFTypeDictionaryKeyCallBacks, kCFTypeDictionaryValueCallBacks,
            );
            AXIsProcessTrustedWithOptions(dict);
        }
        need_exit = true;
    }

    // 2. Input Monitoring
    let listen_ok = unsafe { CGPreflightListenEventAccess() };
    if !listen_ok {
        eprintln!("需要「输入监控」权限，正在请求...");
        unsafe { CGRequestListenEventAccess(); }
        need_exit = true;
    }

    if need_exit {
        eprintln!("\n请在「系统设置 > 隐私与安全性」中启用以下权限：");
        if !ax_trusted { eprintln!("  - 辅助功能"); }
        if !listen_ok { eprintln!("  - 输入监控"); }
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
        CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType, EventField,
        CGEventFlags,
    };
    use parking_lot::Mutex;
    use std::cell::Cell;
    use std::sync::Arc;

    const KVK_RIGHT_COMMAND: i64 = 0x36;

    pub fn start(proxy: Arc<Mutex<Option<tao::event_loop::EventLoopProxy<MacEvent>>>>) {
        let events = vec![CGEventType::FlagsChanged];
        let prev_down = Cell::new(false);

        let tap = core_graphics::event::CGEventTap::new(
            CGEventTapLocation::HID,
            CGEventTapPlacement::HeadInsertEventTap,
            CGEventTapOptions::ListenOnly,
            events,
            move |_proxy, _event_type, event| {
                let keycode = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
                let flags = event.get_flags();
                log::debug!("CGEvent: keycode={}, flags={:?}", keycode, flags);
                if keycode != KVK_RIGHT_COMMAND {
                    return None;
                }
                let now_down = event.get_flags().contains(CGEventFlags::CGEventFlagCommand);
                let was_down = prev_down.replace(now_down);
                if now_down != was_down {
                    if let Some(ref p) = *proxy.lock() {
                        if now_down {
                            let _ = p.send_event(MacEvent::KeyDown);
                        } else {
                            let _ = p.send_event(MacEvent::KeyUp);
                        }
                    }
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
                log::error!("CGEventTap failed. Grant Accessibility + Input Monitoring in System Settings.");
            }
        }
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug, Clone)]
enum MacEvent {
    KeyDown,
    KeyUp,
    ProcessingDone,
    Quit,
}

#[cfg(target_os = "macos")]
fn main() -> Result<()> {
    request_all_permissions()?;

    {
        let exe_dir = Config::exe_dir()?;
        let dylib_path = exe_dir.join("ort-dylib/onnxruntime-osx-x86_64-1.24.2/lib/libonnxruntime.1.24.2.dylib");
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
    use tray_icon::TrayIconEvent;

    let log_buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    vi::ui::log_capture::init_log_capture(log_buffer.clone());

    log::set_boxed_logger(Box::new(CaptureLogger))
        .map(|()| log::set_max_level(log::LevelFilter::Info))
        .unwrap();

    info!("EVI Voice Input Method starting (macOS)...");

    let cfg = Config::load()?;

    info!("Loading models...");
    let session = Arc::new(Session::new(&cfg)?);
    info!("All models loaded.");

    let mut event_loop = EventLoopBuilder::<MacEvent>::with_user_event()
        .build();
    event_loop.set_activation_policy(ActivationPolicy::Accessory);
    let proxy = event_loop.create_proxy();

    let quit_item = MenuItem::new("退出", true, None);
    let quit_id = quit_item.id().clone();

    let menu_proxy = proxy.clone();
    MenuEvent::set_event_handler(Some(move |event: tray_icon::menu::MenuEvent| {
        if event.id == quit_id {
            let _ = menu_proxy.send_event(MacEvent::Quit);
        }
    }));

    TrayIconEvent::set_event_handler(Some(move |_event| {}));

    let _tray = MacTray::new(quit_item)
        .map_err(|e| anyhow::anyhow!("Failed to create tray: {}", e))?;

    let cg_proxy: Arc<Mutex<Option<tao::event_loop::EventLoopProxy<MacEvent>>>> =
        Arc::new(Mutex::new(Some(proxy.clone())));

    std::thread::Builder::new()
        .name("cg-event-tap".into())
        .spawn(move || {
            cg_key::start(cg_proxy);
        })
        .expect("Failed to spawn key listener thread");

    let mut recording = false;
    let mut processing = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Wait;

        match event {
            Event::NewEvents(StartCause::Init) => {
                info!("EVI 输入法已上线! 按住右Command键录音，松开识别。");
            }

            Event::UserEvent(MacEvent::KeyDown) => {
                if !recording && !processing {
                    recording = true;
                    info!("Starting recording (Right Command)...");
                    session.start_recording();
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

            Event::UserEvent(MacEvent::ProcessingDone) => {
                processing = false;
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
    let log_buffer: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    vi::ui::log_capture::init_log_capture(log_buffer.clone());

    log::set_boxed_logger(Box::new(CaptureLogger))
        .map(|()| log::set_max_level(log::LevelFilter::Info))
        .unwrap();

    info!("EVI Voice Input Method starting...");

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
                    if let RefineEngine::Llm(engine) = &session_hook.refine {
                        match engine.lock().rebuild_prompt(&ps.system_prompt, &ps.prefill_template) {
                            Ok(()) => info!("LLM prompt rebuilt successfully"),
                            Err(e) => error!("LLM prompt rebuild failed: {}", e),
                        }
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

fn save_wav(samples: &[i16], sample_rate: u32, seq: u64) -> Result<()> {
    use std::io::Write;
    let path = PathBuf::from(format!("log/{}.wav", seq));
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * num_channels as u32 * bits_per_sample as u32 / 8;
    let block_align = num_channels * bits_per_sample / 8;
    let data_size = samples.len() * 2;
    let file_size = 36 + data_size;

    let mut f = std::fs::File::create(&path)?;
    f.write_all(b"RIFF")?;
    f.write_all(&(file_size as u32).to_le_bytes())?;
    f.write_all(b"WAVE")?;
    f.write_all(b"fmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?;
    f.write_all(&num_channels.to_le_bytes())?;
    f.write_all(&sample_rate.to_le_bytes())?;
    f.write_all(&byte_rate.to_le_bytes())?;
    f.write_all(&block_align.to_le_bytes())?;
    f.write_all(&bits_per_sample.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&(data_size as u32).to_le_bytes())?;
    for &s in samples {
        f.write_all(&s.to_le_bytes())?;
    }
    info!("Saved audio to {}", path.display());
    Ok(())
}
