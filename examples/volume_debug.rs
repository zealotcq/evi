//! EVI 声纹门控调试工具（独立 app，仅 macOS 本地使用）
//!
//! 用法: ./run_volume_debug.sh 或 cargo run --release --bin volume_debug
//!
//! - 最多保留 3 条校准音色（特征持久化于 ~/.evi/voice_refs.json，音频仅存内存可播放）
//! - 右Shift：校准。选中某条时替换该条，否则新增（满3条时淘汰最旧）
//! - 右Command：测试。显示与每条参考的相似度及判定（任一条 ≥ 阈值即通过）
//! - 阈值/开关与 EVI 共享配置

#![cfg(target_os = "macos")]

use crossbeam_channel::{Receiver, Sender};
use eframe::egui;
use egui::Color32;
use parking_lot::Mutex;
use rodio::OutputStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use vi::audio::cpal_source::CpalAudioSource;
use vi::engine::speaker::SpeakerEngine;
use vi::engine::vad::SpeechSegment;
use vi::engine::vad::VadEngine;
use vi::AudioSource;

const KVK_RIGHT_COMMAND: i64 = 0x36;
const KVK_RIGHT_SHIFT: i64 = 0x7C;
const KVK_RIGHT_SHIFT_ADB: i64 = 0x3C;
const SAMPLE_RATE: u32 = 16000;
const MAX_REFS: usize = 3;

#[derive(Clone, Copy, PartialEq)]
enum KeyEvent {
    CmdDown,
    CmdUp,
    CalDown,
    CalUp,
}

#[derive(Clone, Copy, PartialEq)]
enum Phase {
    Idle,
    Recording { calibrate: bool },
}

struct RefSlot {
    ts: String,
    emb: Vec<f32>,
    audio: Option<Vec<i16>>,
}

struct VoiceResult {
    sims: Vec<(String, f32)>,
    verdict: bool,
}

struct VoiceDebugApp {
    event_rx: Receiver<KeyEvent>,
    tap_error: Arc<Mutex<Option<String>>>,
    vad: Option<VadEngine>,
    vad_status: String,
    speaker: Option<SpeakerEngine>,
    speaker_status: String,
    audio: Option<Box<dyn AudioSource>>,
    recording_flag: Arc<AtomicBool>,
    live_level_db: Arc<AtomicU64>,
    phase: Phase,
    refs: Vec<RefSlot>,
    selected: Option<usize>,
    voice_gate_enabled: bool,
    voice_threshold: f64,
    last_voice: Option<VoiceResult>,
    player: Option<(OutputStream, rodio::Sink)>,
    status: String,
}

impl VoiceDebugApp {
    fn now_hms() -> String {
        let d = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let tod = d.as_secs() % 86400;
        format!("{:02}:{:02}:{:02}", tod / 3600, (tod % 3600) / 60, tod % 60)
    }

    fn persist_refs(&self) {
        let list: Vec<(String, Vec<f32>)> = self
            .refs
            .iter()
            .map(|r| (r.ts.clone(), r.emb.clone()))
            .collect();
        vi::save_voice_refs(&list);
    }

    fn handle_key(&mut self, ev: KeyEvent) {
        match ev {
            KeyEvent::CalDown | KeyEvent::CmdDown => {
                let calibrate = ev == KeyEvent::CalDown;
                eprintln!("key down: calibrate={calibrate}");
                if self.phase != Phase::Idle || self.vad.is_none() {
                    return;
                }
                if let Some(audio) = self.audio.as_mut() {
                    match audio.start() {
                        Ok(()) => {
                            self.recording_flag.store(true, Ordering::SeqCst);
                            self.phase = Phase::Recording { calibrate };
                            self.status = if calibrate {
                                if let Some(i) = self.selected {
                                    format!("录音中（校准，将替换参考 #{}）… 松开右Shift", i + 1)
                                } else {
                                    "录音中（校准，新增参考）… 说满5秒以上，松开右Shift".to_string()
                                }
                            } else {
                                "录音中（测试）… 松开右Command".to_string()
                            };
                        }
                        Err(e) => self.status = format!("麦克风启动失败: {e}"),
                    }
                }
            }
            KeyEvent::CalUp | KeyEvent::CmdUp => {
                let calibrate = ev == KeyEvent::CalUp;
                if let Phase::Recording { calibrate: active } = self.phase {
                    if calibrate != active {
                        return;
                    }
                } else {
                    return;
                }
                self.recording_flag.store(false, Ordering::SeqCst);

                let samples = self
                    .audio
                    .as_mut()
                    .and_then(|a| a.stop().ok())
                    .unwrap_or_default();
                self.phase = Phase::Idle;
                if samples.is_empty() {
                    self.status = "没有采集到音频".into();
                    return;
                }

                let pcm: Vec<f32> = samples.iter().map(|&s| s as f32 / 32768.0).collect();
                let dur = pcm.len() as f64 / SAMPLE_RATE as f64;

                let segments = self
                    .vad
                    .as_mut()
                    .and_then(|v| v.analyze(&pcm).ok().flatten());
                let Some(segments) = segments else {
                    self.status = format!("分析失败（{dur:.1}s 音频）: no speech detected");
                    return;
                };

                let emb = {
                    let sp = self.speaker.as_mut();
                    let vad = self.vad.as_mut();
                    match (sp, vad) {
                        (Some(sp), Some(_)) => speech_embedding(sp, &pcm, &segments),
                        _ => None,
                    }
                };

                if calibrate {
                    match emb {
                        Some(e) => {
                            let ts = vi::calib_timestamp();
                            let slot = RefSlot {
                                ts,
                                emb: e,
                                audio: Some(samples.clone()),
                            };
                            match self.selected {
                                Some(i) if i < self.refs.len() => {
                                    self.refs[i] = slot;
                                    self.status =
                                        format!("参考 #{} 已替换（{dur:.1}s 音频）", i + 1);
                                }
                                _ => {
                                    self.refs.push(slot);
                                    if self.refs.len() > MAX_REFS {
                                        self.refs.remove(0);
                                    }
                                    self.status = format!(
                                        "新增参考（共 {} 条，{dur:.1}s 音频）",
                                        self.refs.len()
                                    );
                                }
                            }
                            self.persist_refs();
                            self.last_voice = None;
                        }
                        None => {
                            self.status =
                                format!("校准失败（{dur:.1}s）: 未检出语音或声纹计算失败");
                        }
                    }
                } else {
                    match emb {
                        Some(e) if !self.refs.is_empty() => {
                            let sims: Vec<(String, f32)> = self
                                .refs
                                .iter()
                                .map(|r| (r.ts.clone(), SpeakerEngine::cosine(&r.emb, &e)))
                                .collect();
                            let verdict =
                                sims.iter().any(|(_, s)| *s as f64 >= self.voice_threshold);
                            self.status = if verdict {
                                format!("测试通过（{dur:.1}s）")
                            } else {
                                format!("测试拒绝（{dur:.1}s）")
                            };
                            self.last_voice = Some(VoiceResult { sims, verdict });
                        }
                        Some(_) => {
                            self.status = "无参考数据，先按右Shift校准".into();
                        }
                        None => {
                            self.status = format!("测试失败（{dur:.1}s）: 未检出语音");
                        }
                    }
                }
            }
        }
    }
}

fn speech_embedding(
    sp: &mut SpeakerEngine,
    pcm: &[f32],
    segments: &[SpeechSegment],
) -> Option<Vec<f32>> {
    let mut speech = Vec::new();
    for seg in segments {
        let s0 = (seg.start_ms as f64 / 1000.0 * SAMPLE_RATE as f64) as usize;
        let s1 = ((seg.end_ms as f64 / 1000.0 * SAMPLE_RATE as f64) as usize).min(pcm.len());
        if s1 > s0 {
            speech.extend_from_slice(&pcm[s0..s1]);
        }
    }
    if speech.is_empty() {
        return None;
    }
    sp.embed(&speech).ok()
}

impl eframe::App for VoiceDebugApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(ev) = self.event_rx.try_recv() {
            self.handle_key(ev);
        }
        if let Some(err) = self.tap_error.lock().take() {
            eprintln!("tap error: {err}");
            self.status = format!("快捷键监听失败: {err}（需要输入监控权限）");
        }
        if let Some((_, sink)) = &self.player {
            if sink.empty() {
                self.player = None;
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.heading("EVI 声纹门控调试");
                ui.separator();
                let phase_text = match self.phase {
                    Phase::Idle => "空闲",
                    Phase::Recording { calibrate } => {
                        if calibrate {
                            "● 校准录音中"
                        } else {
                            "● 测试录音中"
                        }
                    }
                };
                ui.label(
                    egui::RichText::new(phase_text)
                        .color(match self.phase {
                            Phase::Idle => Color32::GRAY,
                            _ => Color32::from_rgb(200, 60, 60),
                        })
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "VAD: {} | 声纹: {}",
                            self.vad_status, self.speaker_status
                        ))
                        .small()
                        .color(Color32::GRAY),
                    );
                });
            });
            ui.separator();
            ui.label(
                egui::RichText::new("右Shift=校准（选中某条则替换）  |  右Command=测试打分")
                    .color(Color32::from_rgb(70, 70, 70)),
            );
            ui.add_space(6.0);

            let live_db = f64::from_bits(self.live_level_db.load(Ordering::Relaxed));
            let live_db = if live_db > 0.0 {
                10.0 * live_db.log10()
            } else {
                -100.0
            };
            ui.horizontal(|ui| {
                ui.label("麦克风实时:");
                let meter =
                    egui::ProgressBar::new(((live_db + 80.0) / 80.0).clamp(0.0, 1.0) as f32)
                        .text(format!("{live_db:+.0} dB"));
                ui.add_sized([ui.available_width() - 90.0, 16.0], meter);
            });
            ui.add_space(8.0);

            ui.heading("参考音色（最多3条）");
            if self.refs.is_empty() {
                ui.label(
                    egui::RichText::new("无参考——按住右Shift说一句话（5秒以上）校准")
                        .color(Color32::from_rgb(180, 90, 20)),
                );
            }
            let mut delete_idx = None;
            for i in 0..self.refs.len() {
                egui::Frame::group(ui.style()).show(ui, |ui| {
                    ui.horizontal(|ui| {
                        let label = format!(" #{} ", i + 1);
                        ui.radio_value(&mut self.selected, Some(i), label);
                        ui.monospace(&self.refs[i].ts);
                        match self.refs[i].audio.as_ref() {
                            Some(a) => {
                                ui.label(format!("{:.1}s", a.len() as f64 / SAMPLE_RATE as f64));
                                if ui.button("▶ 播放").clicked() {
                                    match rodio::OutputStream::try_default() {
                                        Ok((stream, handle)) => {
                                            match rodio::Sink::try_new(&handle) {
                                                Ok(sink) => {
                                                    sink.append(rodio::buffer::SamplesBuffer::new(
                                                        1,
                                                        SAMPLE_RATE,
                                                        a.clone(),
                                                    ));
                                                    self.player = Some((stream, sink));
                                                }
                                                Err(e) => {
                                                    self.status = format!("播放失败: {e}");
                                                }
                                            }
                                        }
                                        Err(e) => self.status = format!("音频输出失败: {e}"),
                                    }
                                }
                            }
                            None => {
                                ui.label(
                                    egui::RichText::new("（历史参考，本次会话无音频）")
                                        .small()
                                        .color(Color32::GRAY),
                                );
                            }
                        }
                        if ui.button("🗑 删除").clicked() {
                            delete_idx = Some(i);
                        }
                    });
                });
            }
            if let Some(i) = delete_idx {
                self.refs.remove(i);
                if self.selected == Some(i) {
                    self.selected = None;
                } else if let Some(s) = self.selected {
                    if s > i {
                        self.selected = Some(s - 1);
                    }
                }
                self.persist_refs();
                self.status = format!("参考 #{} 已删除，剩余 {} 条", i + 1, self.refs.len());
            }
            ui.horizontal(|ui| {
                if ui.button("清除选择").clicked() {
                    self.selected = None;
                }
                ui.label(
                    egui::RichText::new(if self.selected.is_some() {
                        "右Shift将替换选中参考"
                    } else {
                        "右Shift将新增参考（满3条淘汰最旧）"
                    })
                    .small()
                    .color(Color32::GRAY),
                );
            });
            ui.add_space(8.0);

            ui.heading("门控设置");
            ui.horizontal(|ui| {
                let mut changed = false;
                if ui
                    .checkbox(&mut self.voice_gate_enabled, "启用声纹门控（与 EVI 共享）")
                    .changed()
                {
                    changed = true;
                }
                if ui
                    .add(
                        egui::Slider::new(&mut self.voice_threshold, 0.0..=0.8)
                            .step_by(0.01)
                            .text("相似度阈值"),
                    )
                    .changed()
                {
                    changed = true;
                }
                if changed {
                    let _ = vi::Config::save_voice_gate_enabled(self.voice_gate_enabled);
                    let _ = vi::Config::save_voice_gate_threshold(self.voice_threshold);
                }
            });
            ui.add_space(8.0);

            ui.heading("本次判定");
            match &self.last_voice {
                Some(v) => {
                    ui.horizontal(|ui| {
                        ui.strong("结果:");
                        if v.verdict {
                            ui.colored_label(
                                Color32::from_rgb(30, 130, 60),
                                format!("通过（任一参考 ≥ 阈值 {:.2}）", self.voice_threshold),
                            );
                        } else {
                            ui.colored_label(
                                Color32::from_rgb(190, 40, 40),
                                format!("拒绝（无参考 ≥ 阈值 {:.2}）", self.voice_threshold),
                            );
                        }
                    });
                    for (label, sim) in &v.sims {
                        let ok = *sim as f64 >= self.voice_threshold;
                        ui.label(
                            egui::RichText::new(format!(
                                "  参考 {}: 相似度 {:.3}  {}",
                                label,
                                sim,
                                if ok { "✓" } else { "✗" }
                            ))
                            .color(if ok {
                                Color32::from_rgb(30, 130, 60)
                            } else {
                                Color32::from_rgb(190, 40, 40)
                            }),
                        );
                    }
                }
                None => {
                    ui.label("按住右Command说一句话测试…");
                }
            }
            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                ui.strong("状态:");
                ui.label(&self.status);
            });
        });

        ctx.request_repaint_after(std::time::Duration::from_millis(50));
    }
}

fn start_key_tap(tx: Sender<KeyEvent>, err: Arc<Mutex<Option<String>>>) {
    use core_graphics::event::{
        CGEventFlags, CGEventTapLocation, CGEventTapOptions, CGEventTapPlacement, CGEventType,
        EventField,
    };
    use std::cell::Cell;

    let prev_cmd = Cell::new(false);
    let prev_shift = Cell::new(false);

    let tap = core_graphics::event::CGEventTap::new(
        CGEventTapLocation::HID,
        CGEventTapPlacement::HeadInsertEventTap,
        CGEventTapOptions::ListenOnly,
        vec![CGEventType::FlagsChanged],
        move |_proxy, _ty, event| {
            let keycode = event.get_integer_value_field(EventField::KEYBOARD_EVENT_KEYCODE);
            let flags = event.get_flags();
            match keycode {
                KVK_RIGHT_COMMAND => {
                    let down = flags.contains(CGEventFlags::CGEventFlagCommand);
                    if down != prev_cmd.replace(down) {
                        let _ = tx.send(if down {
                            KeyEvent::CmdDown
                        } else {
                            KeyEvent::CmdUp
                        });
                    }
                }
                KVK_RIGHT_SHIFT | KVK_RIGHT_SHIFT_ADB => {
                    let down = flags.contains(CGEventFlags::CGEventFlagShift);
                    if down != prev_shift.replace(down) {
                        let _ = tx.send(if down {
                            KeyEvent::CalDown
                        } else {
                            KeyEvent::CalUp
                        });
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
            core_foundation::runloop::CFRunLoop::run_current();
        }
        Err(_) => {
            *err.lock() = Some("CGEventTap 创建失败".into());
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(target_arch = "x86_64")]
    let rel = "ort-dylib/onnxruntime-osx-x86_64-1.24.2/lib/libonnxruntime.1.24.2.dylib";
    #[cfg(target_arch = "aarch64")]
    let rel = "ort-dylib/onnxruntime-osx-arm64-1.24.2/lib/libonnxruntime.1.24.2.dylib";
    let dylib = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    if !ort::init_from(dylib)?.commit() {
        eprintln!("ONNX Runtime 初始化失败");
        return Ok(());
    }

    let cfg = vi::Config::load()?;

    let vad_dir = vi::models::vad_model_dir(&vi::models::base_dir(&cfg));
    eprintln!("加载 VAD 模型: {}", vad_dir.display());
    let vad = match VadEngine::new(&vad_dir) {
        Ok(v) => {
            eprintln!("VAD 模型加载完成");
            v
        }
        Err(e) => {
            eprintln!("VAD 模型加载失败: {e}");
            std::process::exit(1);
        }
    };

    let sp_path = vi::engine::speaker::default_model_path();
    let speaker = if sp_path.exists() {
        match SpeakerEngine::new(&sp_path) {
            Ok(sp) => {
                eprintln!("声纹模型加载完成");
                Some(sp)
            }
            Err(e) => {
                eprintln!("声纹模型加载失败: {e}");
                None
            }
        }
    } else {
        eprintln!("声纹模型未找到: {}", sp_path.display());
        None
    };
    let speaker_status = if speaker.is_some() {
        "就绪"
    } else {
        "不可用"
    }
    .to_string();

    let refs: Vec<RefSlot> = vi::load_voice_refs()
        .into_iter()
        .map(|(ts, emb)| RefSlot {
            ts,
            emb,
            audio: None,
        })
        .collect();
    eprintln!("声纹参考: {} 条", refs.len());

    let audio: Box<dyn AudioSource> = Box::new(CpalAudioSource::new(SAMPLE_RATE)?);
    let recording_flag = Arc::new(AtomicBool::new(false));
    let live_level = Arc::new(AtomicU64::new(0));

    {
        let rx = audio.receiver().clone();
        let flag = recording_flag.clone();
        let level = live_level.clone();
        std::thread::Builder::new()
            .name("level-meter".into())
            .spawn(move || {
                let mut recent: Vec<i16> = Vec::new();
                loop {
                    match rx.recv_timeout(std::time::Duration::from_millis(50)) {
                        Ok(frame) => {
                            if !flag.load(Ordering::Relaxed) {
                                recent.clear();
                                level.store(0, Ordering::Relaxed);
                                continue;
                            }
                            recent.extend_from_slice(&frame.samples);
                            let win = SAMPLE_RATE as usize / 10;
                            if recent.len() > win {
                                let cut = recent.len() - win;
                                recent.drain(..cut);
                            }
                            let ms = recent
                                .iter()
                                .map(|&s| {
                                    let v = s as f64 / 32768.0;
                                    v * v
                                })
                                .sum::<f64>()
                                / recent.len().max(1) as f64;
                            level.store(ms.to_bits(), Ordering::Relaxed);
                        }
                        Err(_) => {
                            if !flag.load(Ordering::Relaxed) {
                                recent.clear();
                                level.store(0, Ordering::Relaxed);
                            }
                        }
                    }
                }
            })
            .expect("spawn level meter");
    }

    let (event_tx, event_rx) = crossbeam_channel::unbounded::<KeyEvent>();
    let tap_error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let tap_err_clone = tap_error.clone();
    std::thread::Builder::new()
        .name("cg-event-tap".into())
        .spawn(move || start_key_tap(event_tx, tap_err_clone))
        .expect("spawn key tap");

    let native_options = eframe::NativeOptions {
        centered: true,
        viewport: egui::ViewportBuilder::default()
            .with_title("EVI 声纹门控调试")
            .with_inner_size([860.0, 680.0])
            .with_min_inner_size([700.0, 560.0]),
        ..Default::default()
    };

    eframe::run_native(
        "EVI 声纹门控调试",
        native_options,
        Box::new(move |cc| {
            load_fonts(&cc.egui_ctx);
            Ok(Box::new(VoiceDebugApp {
                event_rx,
                tap_error,
                vad: Some(vad),
                vad_status: "就绪".into(),
                speaker,
                speaker_status,
                audio: Some(audio),
                recording_flag,
                live_level_db: live_level,
                phase: Phase::Idle,
                refs,
                selected: None,
                voice_gate_enabled: cfg.voice_gate_enabled,
                voice_threshold: cfg.voice_gate_threshold,
                last_voice: None,
                player: None,
                status: "就绪。右Shift校准（选中则替换），右Command测试".into(),
            }))
        }),
    )
    .map_err(|e| -> Box<dyn std::error::Error> { Box::new(e) })?;
    Ok(())
}

fn load_fonts(ctx: &egui::Context) {
    use std::sync::Arc;
    let paths: &[&str] = &[
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ];
    for path in paths {
        if let Ok(data) = std::fs::read(path) {
            let mut fonts = egui::FontDefinitions::default();
            fonts
                .font_data
                .insert("chinese".into(), Arc::new(egui::FontData::from_owned(data)));
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, "chinese".into());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .insert(0, "chinese".into());
            ctx.set_fonts(fonts);
            return;
        }
    }
}
