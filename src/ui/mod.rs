pub mod api_key_dialog;
pub mod log_capture;
#[cfg(target_os = "windows")]
pub mod win32;

#[cfg(target_os = "macos")]
pub mod macos_tray;

use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use egui::ViewportCommand;

#[cfg(target_os = "windows")]
pub static HOOK_CHANNEL: parking_lot::Mutex<Option<crossbeam_channel::Sender<bool>>> =
    parking_lot::Mutex::new(None);

#[cfg(not(target_os = "windows"))]
pub static HOOK_CHANNEL: std::sync::Mutex<Option<crossbeam_channel::Sender<bool>>> =
    std::sync::Mutex::new(None);

pub static USE_LLM_REMOTE: AtomicBool = AtomicBool::new(false);

pub fn get_llm_remote_enabled() -> bool {
    USE_LLM_REMOTE.load(Ordering::SeqCst)
}

pub fn set_llm_remote(enabled: bool) {
    USE_LLM_REMOTE.store(enabled, Ordering::SeqCst);
    if let Err(e) = crate::Config::save_llm_remote_enabled(enabled) {
        log::warn!("Failed to save llm_remote_enabled: {}", e);
    }
}

pub static ENERGY_GATE_ENABLED: AtomicBool = AtomicBool::new(false);

pub fn get_energy_gate_enabled() -> bool {
    ENERGY_GATE_ENABLED.load(Ordering::SeqCst)
}

pub fn set_energy_gate_enabled(enabled: bool) {
    ENERGY_GATE_ENABLED.store(enabled, Ordering::SeqCst);
    if let Err(e) = crate::Config::save_energy_gate_enabled(enabled) {
        log::warn!("Failed to save energy_gate_enabled: {}", e);
    }
}

pub static CLIPBOARD_RESTORE_BEHAVIOR: std::sync::LazyLock<parking_lot::RwLock<String>> =
    std::sync::LazyLock::new(|| parking_lot::RwLock::new("100ms".to_string()));

pub fn get_clipboard_restore_behavior() -> String {
    CLIPBOARD_RESTORE_BEHAVIOR.read().clone()
}

pub fn set_clipboard_restore_behavior(behavior: &str) {
    *CLIPBOARD_RESTORE_BEHAVIOR.write() = behavior.to_string();
    if let Err(e) = crate::Config::save_clipboard_restore_behavior(behavior) {
        log::warn!("Failed to save clipboard_restore_behavior: {}", e);
    }
}

pub struct Scheme {
    pub name: String,
    pub system_prompt: String,
    pub prefill_template: String,
}

pub struct PromptState {
    pub system_prompt: String,
    pub prefill_template: String,
}

pub struct ViApp {
    log_buffer: Arc<Mutex<Vec<String>>>,
    system_prompt_edit: String,
    prefill_template_edit: String,
    schemes: BTreeMap<String, Scheme>,
    current_scheme: String,
    needs_rebuild: Arc<AtomicBool>,
    prompt_state: Arc<Mutex<Option<PromptState>>>,
    new_scheme_name: String,
    show_save_as_popup: bool,
    test_input: String,
    first_frame: bool,
    debug: bool,
    #[cfg(target_os = "windows")]
    tray_initialized: bool,
    #[cfg(target_os = "windows")]
    ctrl_held: bool,
}

impl ViApp {
    #[cfg(target_os = "windows")]
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        log_buffer: Arc<Mutex<Vec<String>>>,
        needs_rebuild: Arc<AtomicBool>,
        prompt_state: Arc<Mutex<Option<PromptState>>>,
        debug: bool,
    ) -> Self {
        load_chinese_fonts(&cc.egui_ctx);

        let exe_dir = crate::Config::exe_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let default_sp =
            std::fs::read_to_string(exe_dir.join("system_prompt.txt")).unwrap_or_default();
        let default_pt = std::fs::read_to_string(exe_dir.join("prefill_template.txt"))
            .unwrap_or_else(|_| r#"{"校对和轻度润色结果":""#.to_string());
        let (schemes, init_scheme, init_sp, init_pt, need_rebuild) =
            load_schemes(&exe_dir, default_sp, default_pt);

        Self {
            log_buffer,
            system_prompt_edit: init_sp,
            prefill_template_edit: init_pt,
            schemes,
            current_scheme: init_scheme,
            needs_rebuild,
            prompt_state,
            new_scheme_name: String::new(),
            show_save_as_popup: false,
            test_input: String::new(),
            first_frame: need_rebuild,
            debug,
            tray_initialized: false,
            ctrl_held: false,
        }
    }

    #[cfg(not(target_os = "windows"))]
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        log_buffer: Arc<Mutex<Vec<String>>>,
        needs_rebuild: Arc<AtomicBool>,
        prompt_state: Arc<Mutex<Option<PromptState>>>,
        debug: bool,
    ) -> Self {
        load_chinese_fonts(&cc.egui_ctx);

        let exe_dir = crate::Config::exe_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
        let default_sp =
            std::fs::read_to_string(exe_dir.join("system_prompt.txt")).unwrap_or_default();
        let default_pt = std::fs::read_to_string(exe_dir.join("prefill_template.txt"))
            .unwrap_or_else(|_| r#"{"校对和轻度润色结果":""#.to_string());
        let (schemes, init_scheme, init_sp, init_pt, need_rebuild) =
            load_schemes(&exe_dir, default_sp, default_pt);

        Self {
            log_buffer,
            system_prompt_edit: init_sp,
            prefill_template_edit: init_pt,
            schemes,
            current_scheme: init_scheme,
            needs_rebuild,
            prompt_state,
            new_scheme_name: String::new(),
            show_save_as_popup: false,
            test_input: String::new(),
            first_frame: need_rebuild,
            debug,
        }
    }
}

fn load_schemes(
    exe_dir: &Path,
    default_sp: String,
    default_pt: String,
) -> (BTreeMap<String, Scheme>, String, String, String, bool) {
    let mut schemes = BTreeMap::new();
    schemes.insert(
        "默认".to_string(),
        Scheme {
            name: "默认".to_string(),
            system_prompt: default_sp.clone(),
            prefill_template: default_pt.clone(),
        },
    );

    let schemes_dir = exe_dir.join("schemes");
    if schemes_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&schemes_dir) {
            for entry in entries.flatten() {
                let dir = entry.path();
                if dir.is_dir() {
                    if let Some(name) = dir.file_name().and_then(|n| n.to_str()) {
                        let sp = std::fs::read_to_string(dir.join("system_prompt.txt"))
                            .unwrap_or_default();
                        let pt = std::fs::read_to_string(dir.join("prefill_template.txt"))
                            .unwrap_or_default();
                        schemes.insert(
                            name.to_string(),
                            Scheme {
                                name: name.to_string(),
                                system_prompt: sp,
                                prefill_template: pt,
                            },
                        );
                    }
                }
            }
        }
    }

    let saved_scheme = crate::Config::load().ok().and_then(|c| c.active_scheme);

    let (init_scheme, init_sp, init_pt, need_rebuild) = if let Some(ref name) = saved_scheme {
        if let Some(s) = schemes.get(name) {
            (
                name.clone(),
                s.system_prompt.clone(),
                s.prefill_template.clone(),
                name != "默认",
            )
        } else {
            ("默认".to_string(), default_sp, default_pt, false)
        }
    } else {
        ("默认".to_string(), default_sp, default_pt, false)
    };

    (schemes, init_scheme, init_sp, init_pt, need_rebuild)
}

impl eframe::App for ViApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        #[cfg(target_os = "windows")]
        {
            if !self.tray_initialized {
                self.tray_initialized = true;
                if let Some(hwnd) = unsafe { win32::find_main_window() } {
                    unsafe {
                        win32::setup_tray(hwnd, self.debug);
                    }
                }
            }

            use windows::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
            const VK_RCONTROL: i32 = 0xA3;
            let rctrl_down = unsafe { GetAsyncKeyState(VK_RCONTROL) } as u16 & 0x8000 != 0;

            if rctrl_down != self.ctrl_held {
                self.ctrl_held = rctrl_down;
                if let Some(tx) = HOOK_CHANNEL.lock().as_ref() {
                    let _ = tx.try_send(rctrl_down);
                }
            }
        }

        if self.first_frame {
            self.first_frame = false;
            self.trigger_rebuild();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_height();
            let log_h = (avail * 0.33).max(100.0);
            let cfg_h = (avail * 0.34).max(120.0);

            ui.allocate_ui(egui::vec2(ui.available_width(), log_h), |ui| {
                ui.heading("日志");
                ui.separator();
                egui::ScrollArea::vertical()
                    .stick_to_bottom(true)
                    .max_height(log_h - 30.0)
                    .show(ui, |ui| {
                        let logs = self.log_buffer.lock();
                        let text = logs.join("\n");
                        ui.monospace(&text);
                    });
            });

            ui.separator();

            ui.allocate_ui(egui::vec2(ui.available_width(), cfg_h), |ui| {
                ui.horizontal(|ui| {
                    ui.heading("配置");
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("另存").clicked() {
                            self.show_save_as_popup = true;
                            self.new_scheme_name.clear();
                        }
                        if ui.button("保存").clicked() {
                            if self.current_scheme == "默认" {
                                self.show_save_as_popup = true;
                                self.new_scheme_name.clear();
                            } else if let Some(scheme) = self.schemes.get_mut(&self.current_scheme)
                            {
                                scheme.system_prompt = self.system_prompt_edit.clone();
                                scheme.prefill_template = self.prefill_template_edit.clone();
                                self.save_scheme_to_disk(&self.current_scheme);
                            }
                        }
                        if ui.button("应用").clicked() {
                            self.trigger_rebuild();
                            let _ = crate::Config::save_active_scheme(&self.current_scheme);
                        }
                        let scheme_names: Vec<String> = self.schemes.keys().cloned().collect();
                        let current = self.current_scheme.clone();
                        egui::ComboBox::from_id_salt("scheme_selector")
                            .selected_text(&current)
                            .show_ui(ui, |ui| {
                                for name in &scheme_names {
                                    if ui.selectable_label(name == &current, name).clicked() {
                                        if let Some(scheme) = self.schemes.get(name) {
                                            self.system_prompt_edit = scheme.system_prompt.clone();
                                            self.prefill_template_edit =
                                                scheme.prefill_template.clone();
                                            self.current_scheme = name.clone();
                                            self.trigger_rebuild();
                                        }
                                    }
                                }
                            });
                    });
                });

                if self.show_save_as_popup {
                    ui.horizontal(|ui| {
                        ui.label("新方案名称:");
                        let resp = ui.text_edit_singleline(&mut self.new_scheme_name);
                        if ui.button("确定").clicked() {
                            let name = self.new_scheme_name.trim().to_string();
                            if !name.is_empty() {
                                self.schemes.insert(
                                    name.clone(),
                                    Scheme {
                                        name: name.clone(),
                                        system_prompt: self.system_prompt_edit.clone(),
                                        prefill_template: self.prefill_template_edit.clone(),
                                    },
                                );
                                self.current_scheme = name.clone();
                                self.save_scheme_to_disk(&self.current_scheme);
                                self.trigger_rebuild();
                                self.show_save_as_popup = false;
                            }
                        }
                        if ui.button("取消").clicked() {
                            self.show_save_as_popup = false;
                        }
                        resp.request_focus();
                    });
                }

                let text_edit_rows = ((ui.available_height() - 20.0) / 18.0).max(3.0) as usize;
                ui.horizontal(|ui| {
                    let col_w = (ui.available_width() - 8.0) / 2.0;
                    ui.vertical(|ui| {
                        ui.label("System Prompt:");
                        ui.add(
                            egui::TextEdit::multiline(&mut self.system_prompt_edit)
                                .desired_rows(text_edit_rows)
                                .desired_width(col_w),
                        );
                    });
                    ui.vertical(|ui| {
                        ui.label("Prefill Template:");
                        ui.add(
                            egui::TextEdit::multiline(&mut self.prefill_template_edit)
                                .desired_rows(text_edit_rows)
                                .desired_width(col_w),
                        );
                    });
                });
            });

            ui.separator();

            ui.vertical(|ui| {
                ui.heading("测试输入");
                let test_rows = ((ui.available_height() - 20.0) / 18.0).max(2.0) as usize;
                ui.add(
                    egui::TextEdit::multiline(&mut self.test_input)
                        .desired_rows(test_rows)
                        .desired_width(f32::INFINITY)
                        .hint_text("在此输入文本测试输入法..."),
                );
            });
        });

        if self.debug {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }
        else {
            ctx.send_viewport_cmd(ViewportCommand::Visible(false)); // 隐藏窗口
            ctx.send_viewport_cmd(ViewportCommand::Minimized(true)); // 最小化窗口
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        let _ = crate::Config::save_active_scheme(&self.current_scheme);
    }
}

impl ViApp {
    fn save_scheme_to_disk(&self, name: &str) {
        if let Ok(exe_dir) = crate::Config::exe_dir() {
            let dir = exe_dir.join("schemes").join(name);
            let _ = std::fs::create_dir_all(&dir);
            if let Some(scheme) = self.schemes.get(name) {
                let _ = std::fs::write(dir.join("system_prompt.txt"), &scheme.system_prompt);
                let _ = std::fs::write(dir.join("prefill_template.txt"), &scheme.prefill_template);
            }
        }
    }

    fn trigger_rebuild(&self) {
        *self.prompt_state.lock() = Some(PromptState {
            system_prompt: self.system_prompt_edit.clone(),
            prefill_template: self.prefill_template_edit.clone(),
        });
        self.needs_rebuild.store(true, Ordering::SeqCst);
    }
}

fn load_chinese_fonts(ctx: &egui::Context) {
    let font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\msyhbd.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
    ];
    for path in &font_paths {
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

pub fn run_gui(
    log_buffer: Arc<Mutex<Vec<String>>>,
    needs_rebuild: Arc<AtomicBool>,
    prompt_state: Arc<Mutex<Option<PromptState>>>,
    debug: bool,
) -> anyhow::Result<()> {
    let icon_data = load_ico_from_exe_dir();

    let mut viewport = egui::ViewportBuilder::default()
        .with_title("EVI 语音输入法")
        .with_inner_size([800.0, 600.0]);

    if let Some(ic) = icon_data {
        viewport = viewport.with_icon(Arc::new(ic));
    }

    let native_options = eframe::NativeOptions {
        viewport,
        ..Default::default()
    };

    eframe::run_native(
        "EVI 语音输入法",
        native_options,
        Box::new(move |cc| {
            Ok(Box::new(ViApp::new(
                cc,
                log_buffer,
                needs_rebuild,
                prompt_state,
                debug,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))
}

fn load_ico_from_exe_dir() -> Option<egui::viewport::IconData> {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            let icon_path = dir.join("evi.ico");
            if icon_path.exists() {
                if let Ok(bytes) = std::fs::read(&icon_path) {
                    if let Ok(img) = image::load_from_memory(&bytes) {
                        let rgba = img.to_rgba8();
                        let (w, h) = rgba.dimensions();
                        return Some(egui::viewport::IconData {
                            rgba: rgba.into_raw(),
                            width: w,
                            height: h,
                        });
                    }
                }
            }
        }
    }
    None
}
