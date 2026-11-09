use anyhow::Result;
use std::sync::Arc;

use crate::ui::refine_api::{AuditItem, AuditSubmit, RefineApi, RefineSubmit};

const THANK_YOU_MESSAGES: &[&str] = &[
    "太棒了！你刚刚帮助提升了语音识别的准确度，千万人受益！",
    "你的审阅让更多用户享受到了更精准的文字输入体验！",
    "很厉害！又一条高质量的优化建议，系统因你而进步！",
    "感谢你的专业眼光！你的每一次审阅都在推动技术向前！",
    "你的贡献正在让语音输入变得更自然、更智能！",
    "又一个值得骄傲的贡献！你是最棒的体验优化官！",
    "你的细致审核是产品不断进化的动力，为你点赞！",
    "优秀！你正在帮助构建更好的语音技术生态系统！",
    "这份认真令人感动！你的贡献让每个使用语音输入的人都受益！",
    "完美！你的专业判断让 AI 变得更加可靠和智能！",
];

const PRIVACY_NOTICE: &str = "\
• 对话记录仅保存在本地，不会自动上传
• 你可以随时查看、删除任意一条记录
• 上传由你手动选择，仅提交你确认的对话条目
• 通过匿名随机序列标识用户身份，不收集任何个人信息
• 数据传输全程加密，服务器仅存储提交的内容";

const USAGE_DESCRIPTION: &str = "\
【我的对话】标签页
选择一条识别记录，修改为更准确的文本后提交。你的校正将帮助系统学习正确的表达方式。
【审核助手】标签页
审核其他用户提交的校正结果，点击「通过」或「拒绝」帮助筛选高质量数据。
每完成一次提交或审核，你的贡献值都会增加。";

const LEVEL_TITLES: &[(i32, &str, &str)] = &[
    (0, "新手体验官", "🌱"),
    (5, "初级体验官", "🌿"),
    (20, "中级体验官", "🌳"),
    (50, "高级体验官", "⭐"),
    (100, "资深体验官", "🌟"),
    (200, "首席体验官", "💎"),
    (500, "传奇体验官", "🏆"),
];

fn get_level(contribution: i32) -> (usize, &'static str, &'static str) {
    let mut idx = 0;
    for (i, &(threshold, _, _)) in LEVEL_TITLES.iter().enumerate() {
        if contribution >= threshold {
            idx = i;
        }
    }
    (idx, LEVEL_TITLES[idx].1, LEVEL_TITLES[idx].2)
}

fn get_next_level_progress(contribution: i32) -> (f32, i32) {
    let (idx, _, _) = get_level(contribution);
    let current_threshold = LEVEL_TITLES[idx].0;
    let next_threshold = if idx + 1 < LEVEL_TITLES.len() {
        LEVEL_TITLES[idx + 1].0
    } else {
        LEVEL_TITLES[idx].0
    };
    if next_threshold == current_threshold {
        return (1.0, 0);
    }
    let progress =
        (contribution - current_threshold) as f32 / (next_threshold - current_threshold) as f32;
    (progress.min(1.0), next_threshold - contribution)
}

const SUBMIT_THANK_YOU: &[&str] = &[
    "感谢你的校正建议！这将帮助系统识别得更好。",
    "太好了！你刚刚为提升语音识别准确率做出了贡献！",
    "你的修改非常到位！系统会从你的反馈中不断学习进步。",
    "又一条高质量校正！你正在让语音输入变得更智能！",
    "优秀的建议！你的贡献正在帮助千万人获得更好的输入体验！",
    "精准的修改！你的专业反馈是系统进步的宝贵财富！",
];

const MAX_DISPLAY: usize = 5;

#[derive(Debug, Clone)]
struct RefineEntry {
    id: i64,
    original: String,
    refined: String,
    submitted: bool,
}

enum Tab {
    Optimize,
    Audit,
}

enum AuditState {
    Idle,
    Loaded(AuditItem),
    ThankYou(String),
    Error(String),
}

enum SubmitFeedback {
    None,
    ThankYou(String),
}

struct RefineEditorApp {
    api: Box<dyn RefineApi>,
    uuid: String,
    contribution: i32,
    base_height_set: bool,
    focus_requested: bool,
    ctx: Option<egui::Context>,

    tab: Tab,
    error_msg: Option<String>,

    entries: Vec<RefineEntry>,
    selected_id: Option<i64>,
    original_display: String,
    edit_text: String,

    audit_state: AuditState,
    submit_feedback: SubmitFeedback,
}

impl RefineEditorApp {
    fn new(api: Box<dyn RefineApi>, uuid: String) -> Self {
        let contribution = match api.get_user_profile(&uuid) {
            Ok(p) => p.contribution,
            Err(e) => {
                log::warn!("获取用户资料失败: {}", e);
                0
            }
        };
        let mut app = Self {
            api,
            uuid,
            contribution,
            base_height_set: false,
            focus_requested: false,
            ctx: None,
            tab: Tab::Optimize,
            error_msg: None,
            entries: Vec::new(),
            selected_id: None,
            original_display: String::new(),
            edit_text: String::new(),
            audit_state: AuditState::Idle,
            submit_feedback: SubmitFeedback::None,
        };
        app.reload();
        app.load_audit_item();
        app
    }

    fn reload(&mut self) {
        self.entries.clear();
        self.selected_id = None;
        self.original_display.clear();
        self.edit_text.clear();
        self.error_msg = None;

        let path = crate::models::refine_db_path();
        match rusqlite::Connection::open(&path) {
            Ok(conn) => {
                let mut stmt = match conn.prepare(
                    "SELECT id, original, refined, submitted FROM refine_log ORDER BY id DESC",
                ) {
                    Ok(s) => s,
                    Err(e) => {
                        self.error_msg = Some(format!("查询失败: {}", e));
                        return;
                    }
                };
                let rows = stmt.query_map([], |row| {
                    Ok(RefineEntry {
                        id: row.get(0)?,
                        original: row.get(1)?,
                        refined: row.get(2)?,
                        submitted: row.get::<_, i32>(3)? != 0,
                    })
                });
                match rows {
                    Ok(rows) => {
                        for row in rows {
                            match row {
                                Ok(e) => self.entries.push(e),
                                Err(e) => {
                                    self.error_msg = Some(format!("读取记录失败: {}", e));
                                    return;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        self.error_msg = Some(format!("查询失败: {}", e));
                    }
                }
            }
            Err(e) => {
                self.error_msg = Some(format!("打开数据库失败: {}", e));
            }
        }
    }

    fn display_entries(&self) -> Vec<&RefineEntry> {
        let mut pending: Vec<&RefineEntry> = self.entries.iter().filter(|e| !e.submitted).collect();
        let mut submitted: Vec<&RefineEntry> =
            self.entries.iter().filter(|e| e.submitted).collect();

        pending.truncate(MAX_DISPLAY);
        if pending.len() < MAX_DISPLAY {
            let need = MAX_DISPLAY - pending.len();
            submitted.truncate(need);
            pending.extend(submitted);
        }
        pending
    }

    fn delete_entry(&mut self, id: i64) {
        let path = crate::models::refine_db_path();
        if let Ok(conn) = rusqlite::Connection::open(&path) {
            if conn
                .execute(
                    "DELETE FROM refine_log WHERE id = ?1",
                    rusqlite::params![id],
                )
                .is_ok()
            {
                self.entries.retain(|e| e.id != id);
                if self.selected_id == Some(id) {
                    self.selected_id = None;
                    self.original_display.clear();
                    self.edit_text.clear();
                }
            }
        }
    }

    fn submit_refine(&mut self, id: i64, new_refined: &str) {
        let original = match self.entries.iter().find(|e| e.id == id) {
            Some(e) => e.original.clone(),
            None => return,
        };

        let path = crate::models::refine_db_path();
        if let Ok(conn) = rusqlite::Connection::open(&path) {
            if conn
                .execute(
                    "UPDATE refine_log SET refined = ?1, submitted = 1 WHERE id = ?2",
                    rusqlite::params![new_refined, id],
                )
                .is_ok()
            {
                for e in &mut self.entries {
                    if e.id == id {
                        e.refined = new_refined.to_string();
                        e.submitted = true;
                        break;
                    }
                }
                self.selected_id = None;
                self.original_display.clear();
                self.edit_text.clear();
            }
        }

        match self.api.submit_refine(RefineSubmit {
            uuid: self.uuid.clone(),
            original,
            refined: new_refined.to_string(),
        }) {
            Ok(resp) => {
                self.contribution += 1;
                if let Some(ctx) = &self.ctx {
                    ctx.request_repaint();
                }
                if resp.success {
                    let idx = std::time::SystemTime::now()
                        .duration_since(std::time::SystemTime::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis() as usize
                        % SUBMIT_THANK_YOU.len();
                    self.submit_feedback =
                        SubmitFeedback::ThankYou(SUBMIT_THANK_YOU[idx].to_string());
                } else {
                    log::warn!("submit_refine API returned failure: {:?}", resp.message);
                }
            }
            Err(e) => {
                self.contribution += 1;
                if let Some(ctx) = &self.ctx {
                    ctx.request_repaint();
                }
                log::warn!("submit_refine API error: {}", e);
            }
        }
    }

    fn load_audit_item(&mut self) {
        match self.api.get_pending_audit(&self.uuid) {
            Ok(Some(item)) => {
                self.audit_state = AuditState::Loaded(item);
            }
            Ok(None) => {
                self.audit_state = AuditState::Idle;
            }
            Err(e) => {
                self.audit_state = AuditState::Error(format!("获取待审核内容失败: {}", e));
            }
        }
    }

    fn submit_audit(&mut self, task_id: i64, approved: bool) {
        match self.api.submit_audit(AuditSubmit {
            uuid: self.uuid.clone(),
            task_id,
            approved,
        }) {
            Ok(resp) if resp.success => {
                self.contribution += 1;
                if let Some(ctx) = &self.ctx {
                    ctx.request_repaint();
                }
                let idx = std::time::SystemTime::now()
                    .duration_since(std::time::SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as usize
                    % THANK_YOU_MESSAGES.len();
                self.audit_state = AuditState::ThankYou(THANK_YOU_MESSAGES[idx].to_string());
            }
            Ok(resp) => {
                self.audit_state = AuditState::Error(format!(
                    "提交失败: {}",
                    resp.message.unwrap_or_else(|| "未知错误".to_string())
                ));
            }
            Err(e) => {
                self.audit_state = AuditState::Error(format!("提交失败: {}", e));
            }
        }
    }
}

impl eframe::App for RefineEditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.ctx.is_none() {
            self.ctx = Some(ctx.clone());
        }
        if !self.focus_requested {
            self.focus_requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
        if !ctx.input(|i| i.focused) {
            ctx.input_mut(|i| i.focused = true);
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(500));

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.heading("体验优化官");
                    let (_, title, icon) = get_level(self.contribution);
                    ui.label(
                        egui::RichText::new(format!("{} {}", icon, title))
                            .size(14.0)
                            .color(egui::Color32::from_rgb(180, 130, 20)),
                    );
                });

                ui.add_space(2.0);

                let (progress, remaining) = get_next_level_progress(self.contribution);
                let bar_w = ui.available_width().min(320.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!("贡献值: {}", self.contribution))
                            .size(13.0)
                            .color(egui::Color32::from_rgb(100, 100, 100)),
                    );
                    if remaining > 0 {
                        ui.label(
                            egui::RichText::new(format!("(距下一级还差 {} 次)", remaining))
                                .size(12.0)
                                .color(egui::Color32::from_rgb(140, 140, 140)),
                        );
                    }
                });

                let progress_color = egui::Color32::from_rgb(76, 175, 80);
                let bg_color = egui::Color32::from_rgb(230, 230, 230);
                egui::Frame::canvas(ui.style())
                    .inner_margin(egui::Margin::same(0))
                    .show(ui, |ui| {
                        let (rect, _) =
                            ui.allocate_exact_size(egui::vec2(bar_w, 8.0), egui::Sense::hover());
                        if ui.is_rect_visible(rect) {
                            ui.painter().rect_filled(rect, 4.0, bg_color);
                            let filled_w = rect.width() * progress;
                            if filled_w > 0.0 {
                                let filled_rect = egui::Rect::from_min_max(
                                    rect.min,
                                    egui::pos2(rect.min.x + filled_w, rect.max.y),
                                );
                                ui.painter().rect_filled(filled_rect, 4.0, progress_color);
                            }
                        }
                    });

                ui.add_space(4.0);

                ui.horizontal(|ui| {
                    let col_w = (ui.available_width() - 8.0) / 2.0;
                    ui.vertical(|ui| {
                        ui.set_width(col_w);
                        egui::Frame::group(ui.style())
                            .inner_margin(10)
                            .fill(egui::Color32::from_rgb(245, 255, 245))
                            .show(ui, |ui| {
                                egui::Frame::new()
                                    .inner_margin(egui::Margin::symmetric(8, 3))
                                    .corner_radius(egui::CornerRadius::same(4))
                                    .fill(egui::Color32::from_rgb(46, 139, 87))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new("🔒 隐私保护承诺")
                                                .size(13.0)
                                                .color(egui::Color32::WHITE),
                                        );
                                    });
                                ui.add_space(6.0);
                                ui.label(
                                    egui::RichText::new(PRIVACY_NOTICE)
                                        .size(12.0)
                                        .color(egui::Color32::from_rgb(50, 50, 50)),
                                );
                            });
                    });
                    ui.vertical(|ui| {
                        ui.set_width(col_w);
                        egui::Frame::group(ui.style())
                            .inner_margin(10)
                            .fill(egui::Color32::from_rgb(240, 245, 255))
                            .show(ui, |ui| {
                                egui::Frame::new()
                                    .inner_margin(egui::Margin::symmetric(8, 3))
                                    .corner_radius(egui::CornerRadius::same(4))
                                    .fill(egui::Color32::from_rgb(33, 150, 243))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new("📖 使用说明")
                                                .size(13.0)
                                                .color(egui::Color32::WHITE),
                                        );
                                    });
                                ui.add_space(6.0);
                                ui.label(
                                    egui::RichText::new(USAGE_DESCRIPTION)
                                        .size(12.0)
                                        .color(egui::Color32::from_rgb(50, 50, 50)),
                                );
                            });
                    });
                });

                ui.separator();

                ui.horizontal(|ui| {
                    let opt_active = matches!(self.tab, Tab::Optimize);
                    let audit_active = matches!(self.tab, Tab::Audit);
                    let tab_w = (ui.available_width() - 4.0) / 2.0;
                    let opt_btn =
                        egui::Button::new(egui::RichText::new("我的对话").size(14.0).color(
                            if opt_active {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::from_rgb(80, 80, 80)
                            },
                        ))
                        .min_size(egui::vec2(tab_w, 32.0))
                        .fill(if opt_active {
                            egui::Color32::from_rgb(33, 150, 243)
                        } else {
                            egui::Color32::from_rgb(230, 230, 230)
                        })
                        .stroke(egui::Stroke::new(
                            if opt_active { 1.5 } else { 0.5 },
                            if opt_active {
                                egui::Color32::from_rgb(25, 118, 210)
                            } else {
                                egui::Color32::from_rgb(180, 180, 180)
                            },
                        ))
                        .corner_radius(egui::CornerRadius::same(4));
                    if ui.add(opt_btn).clicked() {
                        self.tab = Tab::Optimize;
                    }
                    let audit_btn =
                        egui::Button::new(egui::RichText::new("审核助手").size(14.0).color(
                            if audit_active {
                                egui::Color32::WHITE
                            } else {
                                egui::Color32::from_rgb(80, 80, 80)
                            },
                        ))
                        .min_size(egui::vec2(tab_w, 32.0))
                        .fill(if audit_active {
                            egui::Color32::from_rgb(33, 150, 243)
                        } else {
                            egui::Color32::from_rgb(230, 230, 230)
                        })
                        .stroke(egui::Stroke::new(
                            if audit_active { 1.5 } else { 0.5 },
                            if audit_active {
                                egui::Color32::from_rgb(25, 118, 210)
                            } else {
                                egui::Color32::from_rgb(180, 180, 180)
                            },
                        ))
                        .corner_radius(egui::CornerRadius::same(4));
                    if ui.add(audit_btn).clicked() {
                        self.tab = Tab::Audit;
                    }
                });
                ui.separator();

                if let SubmitFeedback::ThankYou(_) = self.submit_feedback {
                    let msg = match &self.submit_feedback {
                        SubmitFeedback::ThankYou(m) => m.clone(),
                        _ => String::new(),
                    };
                    egui::Frame::group(ui.style())
                        .inner_margin(8)
                        .fill(egui::Color32::from_rgb(230, 255, 230))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                egui::Frame::new()
                                    .inner_margin(egui::Margin::symmetric(8, 3))
                                    .corner_radius(egui::CornerRadius::same(4))
                                    .fill(egui::Color32::from_rgb(33, 150, 243))
                                    .show(ui, |ui| {
                                        ui.label(
                                            egui::RichText::new("📖 提交成功")
                                                .size(13.0)
                                                .color(egui::Color32::WHITE),
                                        );
                                    });
                                ui.label(
                                    egui::RichText::new(&msg)
                                        .size(13.0)
                                        .color(egui::Color32::from_rgb(30, 100, 50)),
                                );
                            });
                        });
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui.small_button("关闭").clicked() {
                            self.submit_feedback = SubmitFeedback::None;
                        }
                    });
                    ui.add_space(2.0);
                }

                match self.tab {
                    Tab::Optimize => self.show_optimize_page(ui, ctx),
                    Tab::Audit => self.show_audit_page(ui),
                }

                if let Some(ref err) = self.error_msg {
                    ui.colored_label(egui::Color32::RED, err);
                }
            });
        });
    }
}

impl RefineEditorApp {
    fn show_optimize_page(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let has_selection = self.selected_id.is_some();
        if has_selection && !self.base_height_set {
            self.base_height_set = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(720.0, 780.0)));
        } else if !has_selection && self.base_height_set {
            self.base_height_set = false;
            ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(720.0, 640.0)));
        }

        let list_h = if has_selection {
            180.0
        } else {
            (ui.available_height() * 0.50).max(100.0)
        };

        let display: Vec<(i64, String, String, bool)> = self
            .display_entries()
            .into_iter()
            .map(|e| (e.id, e.original.clone(), e.refined.clone(), e.submitted))
            .collect();

        egui::Frame::group(ui.style())
            .inner_margin(8)
            .outer_margin(egui::Margin::same(0))
            .show(ui, |ui| {
                ui.set_min_height(list_h);
                if display.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.label(
                            egui::RichText::new("暂无记录")
                                .color(egui::Color32::GRAY)
                                .size(14.0),
                        );
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("启用「记录对话」后，你的识别记录会出现在这里")
                                .color(egui::Color32::from_rgb(150, 150, 150))
                                .size(12.0),
                        );
                        ui.label(
                            egui::RichText::new("所有数据仅保存在本地，你随时可以查看和删除")
                                .color(egui::Color32::from_rgb(150, 150, 150))
                                .size(12.0),
                        );
                    });
                } else {
                    egui::ScrollArea::vertical()
                        .max_height(list_h)
                        .show(ui, |ui| {
                            for (id, original, refined, is_submitted) in &display {
                                let is_selected = self.selected_id == Some(*id);
                                let row_height = 28.0f32;
                                let (rect, _) = ui.allocate_exact_size(
                                    egui::vec2(ui.available_width(), row_height),
                                    egui::Sense::hover(),
                                );
                                if is_selected {
                                    ui.painter().rect_filled(
                                        rect,
                                        4.0,
                                        egui::Color32::from_rgb(220, 235, 255),
                                    );
                                }
                                let btn_w = 40.0f32;
                                let text_max_x = rect.max.x - btn_w - 4.0;
                                if *is_submitted {
                                    let badge_text = "已提交";
                                    let badge_font = egui::FontId::proportional(12.0);
                                    let badge_w = ui
                                        .painter()
                                        .layout_no_wrap(
                                            badge_text.to_string(),
                                            badge_font.clone(),
                                            egui::Color32::from_rgb(46, 139, 87),
                                        )
                                        .size()
                                        .x
                                        + 10.0;
                                    let badge_x = text_max_x - badge_w - 4.0;
                                    let badge_rect = egui::Rect::from_min_max(
                                        egui::pos2(badge_x, rect.min.y + 4.0),
                                        egui::pos2(badge_x + badge_w, rect.max.y - 4.0),
                                    );
                                    ui.painter().rect_filled(
                                        badge_rect,
                                        4.0,
                                        egui::Color32::from_rgb(220, 245, 220),
                                    );
                                    ui.painter().text(
                                        badge_rect.center(),
                                        egui::Align2::CENTER_CENTER,
                                        badge_text,
                                        badge_font,
                                        egui::Color32::from_rgb(46, 139, 87),
                                    );
                                }
                                let display_text = if refined.is_empty() {
                                    original
                                } else {
                                    refined
                                };
                                ui.painter().text(
                                    egui::pos2(rect.min.x + 6.0, rect.center().y),
                                    egui::Align2::LEFT_CENTER,
                                    display_text,
                                    egui::FontId::proportional(14.0),
                                    ui.visuals().text_color(),
                                );
                                let btn_rect = egui::Rect::from_min_max(
                                    egui::pos2(rect.max.x - btn_w, rect.min.y),
                                    rect.max,
                                );
                                let btn_resp = ui.put(btn_rect, egui::Button::new("删除").small());
                                if btn_resp.clicked() {
                                    self.delete_entry(*id);
                                } else {
                                    let left_rect = egui::Rect::from_min_max(
                                        rect.min,
                                        egui::pos2(rect.max.x - btn_w, rect.max.y),
                                    );
                                    let click_resp = ui.interact(
                                        left_rect,
                                        ui.id().with("select").with(*id),
                                        egui::Sense::click(),
                                    );
                                    if click_resp.clicked() && !is_submitted {
                                        self.selected_id = Some(*id);
                                        self.original_display = if refined.is_empty() {
                                            original.clone()
                                        } else {
                                            refined.clone()
                                        };
                                        self.edit_text = self.original_display.clone();
                                    }
                                }
                            }
                        });
                }
            });

        ui.add_space(8.0);

        egui::Frame::group(ui.style())
            .inner_margin(10)
            .outer_margin(egui::Margin::same(0))
            .show(ui, |ui| {
                if let Some(id) = self.selected_id {
                    ui.label(egui::RichText::new("识别原文:").strong());
                    ui.add(
                        egui::TextEdit::multiline(&mut self.original_display.as_str())
                            .desired_width(f32::INFINITY)
                            .desired_rows(2)
                            .interactive(false),
                    );

                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("校正为:").strong());
                    let edit_resp = ui.add(
                        egui::TextEdit::multiline(&mut self.edit_text)
                            .desired_width(f32::INFINITY)
                            .desired_rows(2),
                    );

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            egui::RichText::new("提示：仅会上传你主动提交的校正条目")
                                .size(11.0)
                                .color(egui::Color32::from_rgb(120, 120, 120)),
                        );
                    });

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("提交校正").color(egui::Color32::WHITE),
                                )
                                .fill(egui::Color32::from_rgb(46, 139, 87)),
                            )
                            .clicked()
                        {
                            let text = self.edit_text.trim().to_string();
                            if !text.is_empty() {
                                self.submit_refine(id, &text);
                            }
                        }
                        if ui.button("取消").clicked() {
                            self.selected_id = None;
                            self.original_display.clear();
                            self.edit_text.clear();
                        }
                        edit_resp.request_focus();
                    });
                } else {
                    ui.vertical_centered(|ui| {
                        ui.add_space(6.0);
                        ui.label(
                            egui::RichText::new("你的修改帮助系统学习，每次提交都会增加贡献值")
                                .size(12.0)
                                .color(egui::Color32::from_rgb(150, 150, 150)),
                        );
                    });
                }
            });
    }

    fn show_audit_page(&mut self, ui: &mut egui::Ui) {
        match &self.audit_state {
            AuditState::Idle => {
                ui.vertical_centered(|ui| {
                    ui.add_space(30.0);
                    ui.label(
                        egui::RichText::new("审核其他用户的校对结果，帮助筛选高质量数据")
                            .size(14.0)
                            .color(egui::Color32::from_rgb(80, 80, 80)),
                    );
                    ui.add_space(6.0);
                    ui.label(
                        egui::RichText::new("每次审核都会增加你的贡献值和等级")
                            .size(12.0)
                            .color(egui::Color32::GRAY),
                    );
                    ui.add_space(16.0);
                    if ui
                        .add(egui::Button::new("获取待审核内容").min_size(egui::vec2(160.0, 36.0)))
                        .clicked()
                    {
                        self.load_audit_item();
                    }
                });
            }
            AuditState::Loaded(item) => {
                let item = item.clone();
                ui.add_space(8.0);
                egui::Frame::group(ui.style())
                    .inner_margin(10)
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new("校正前:").strong());
                        ui.add(
                            egui::TextEdit::multiline(&mut item.original.as_str())
                                .desired_width(f32::INFINITY)
                                .desired_rows(3)
                                .interactive(false),
                        );
                        ui.add_space(8.0);
                        ui.label(egui::RichText::new("校正后:").strong());
                        ui.add(
                            egui::TextEdit::multiline(&mut item.refined.as_str())
                                .desired_width(f32::INFINITY)
                                .desired_rows(3)
                                .interactive(false),
                        );
                    });

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new("你的审核帮助系统学习什么样的识别结果是好的")
                            .size(11.0)
                            .color(egui::Color32::from_rgb(120, 120, 120)),
                    );
                });

                ui.add_space(12.0);
                ui.vertical_centered(|ui| {
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("✓ 通过").color(egui::Color32::WHITE),
                                )
                                .fill(egui::Color32::from_rgb(46, 139, 87))
                                .min_size(egui::vec2(120.0, 40.0)),
                            )
                            .clicked()
                        {
                            self.submit_audit(item.task_id, true);
                        }
                        if ui
                            .add(
                                egui::Button::new(
                                    egui::RichText::new("✗ 拒绝").color(egui::Color32::WHITE),
                                )
                                .fill(egui::Color32::from_rgb(200, 60, 60))
                                .min_size(egui::vec2(120.0, 40.0)),
                            )
                            .clicked()
                        {
                            self.submit_audit(item.task_id, false);
                        }
                    });
                });
            }
            AuditState::ThankYou(msg) => {
                let msg = msg.clone();
                let (_, title, icon) = get_level(self.contribution);
                ui.vertical_centered(|ui| {
                    ui.add_space(30.0);
                    ui.label(
                        egui::RichText::new("审核已提交")
                            .size(20.0)
                            .color(egui::Color32::from_rgb(46, 139, 87)),
                    );
                    ui.add_space(8.0);
                    ui.label(
                        egui::RichText::new(format!(
                            "当前贡献: {} | {} {}",
                            self.contribution, icon, title
                        ))
                        .size(14.0)
                        .color(egui::Color32::from_rgb(180, 130, 20)),
                    );
                    ui.add_space(12.0);
                    ui.label(
                        egui::RichText::new(&msg)
                            .size(14.0)
                            .color(egui::Color32::from_rgb(80, 80, 80)),
                    );
                    ui.add_space(20.0);
                    if ui
                        .add(egui::Button::new("下一条").min_size(egui::vec2(120.0, 32.0)))
                        .clicked()
                    {
                        self.load_audit_item();
                    }
                });
            }
            AuditState::Error(msg) => {
                let msg = msg.clone();
                ui.vertical_centered(|ui| {
                    ui.add_space(30.0);
                    ui.colored_label(egui::Color32::RED, &msg);
                    ui.add_space(12.0);
                    if ui.button("重试").clicked() {
                        self.load_audit_item();
                    }
                });
            }
        }
    }
}

fn load_fonts(ctx: &egui::Context) {
    let paths: &[&str] = &[
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\msyhbd.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
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

pub fn run_refine_editor() -> Result<()> {
    use std::sync::Arc;
    let log_buffer: Arc<parking_lot::Mutex<Vec<String>>> =
        Arc::new(parking_lot::Mutex::new(Vec::new()));
    crate::ui::log_capture::init_log_capture(log_buffer);
    let _ = log::set_boxed_logger(Box::new(crate::ui::log_capture::CaptureLogger))
        .map(|()| log::set_max_level(log::LevelFilter::Info));
    log::info!("体验优化官 starting...");

    let uuid = crate::ui::refine_api::get_user_uuid().unwrap_or_default();
    let api = crate::ui::refine_api::create_api();

    let native_options = eframe::NativeOptions {
        centered: true,
        viewport: egui::ViewportBuilder::default()
            .with_title("体验优化官")
            .with_inner_size([720.0, 640.0])
            .with_min_inner_size([520.0, 460.0]),
        ..Default::default()
    };

    let result = eframe::run_native(
        "体验优化官",
        native_options,
        Box::new(move |cc| {
            load_fonts(&cc.egui_ctx);
            Ok(Box::new(RefineEditorApp::new(api, uuid)))
        }),
    );
    match &result {
        Ok(()) => log::info!("体验优化官 closed normally"),
        Err(e) => log::error!("体验优化官 error: {}", e),
    }
    result.map_err(|e| anyhow::anyhow!("GUI error: {}", e))
}
