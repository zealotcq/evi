use anyhow::{bail, Context, Result};
use parking_lot::Mutex;
use serde::Deserialize;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

const VAD_MODEL: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-onnx";
const ASR_MODEL: &str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
const PUNC_MODEL: &str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx";
const REQUIRED_MODELS: &[&str] = &[VAD_MODEL, ASR_MODEL, PUNC_MODEL];

const FILES_API: &str = "https://modelscope.cn/api/v1/models";
const DOWNLOAD_BASE: &str = "https://modelscope.cn/models";

#[derive(Debug, Deserialize)]
struct ApiResponse {
    #[serde(rename = "Success")]
    success: bool,
    #[serde(rename = "Message")]
    message: Option<String>,
    #[serde(rename = "Data")]
    data: Option<ApiData>,
}

#[derive(Debug, Deserialize)]
struct ApiData {
    #[serde(rename = "Files")]
    files: Vec<FileEntry>,
}

#[derive(Debug, Deserialize)]
struct FileEntry {
    #[serde(rename = "Name")]
    name: String,
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Type")]
    file_type: String,
}

fn models_download_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(&home)
        .join(".cache")
        .join("modelscope")
        .join("hub")
        .join("models")
}

fn model_onnx_exists(dir: &Path) -> bool {
    dir.join("model.onnx").exists() || dir.join("model_quant.onnx").exists()
}

fn is_model_ready(model_id: &str) -> bool {
    model_onnx_exists(&models_download_dir().join(model_id))
}

fn missing_models() -> Vec<&'static str> {
    REQUIRED_MODELS
        .iter()
        .filter(|id| !is_model_ready(id))
        .copied()
        .collect()
}

fn format_bytes(bytes: u64) -> String {
    const MB: u64 = 1024 * 1024;
    const GB: u64 = 1024 * MB;
    if bytes >= GB {
        format!("{:.1} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.1} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} KB", bytes / 1024)
    }
}

fn fetch_file_list(model_id: &str) -> Result<Vec<FileEntry>> {
    let url = format!("{}/{}/repo/files?Recursive=true", FILES_API, model_id);
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(30)))
        .build()
        .into();
    let mut resp = agent
        .get(&url)
        .header("User-Agent", "evi/2.1")
        .call()
        .context("Failed to fetch file list")?;
    let body: String = resp.body_mut().read_to_string()?;
    let parsed: ApiResponse = serde_json::from_str(&body)
        .with_context(|| format!("Failed to parse file list response for {}", model_id))?;
    if !parsed.success {
        bail!(
            "API error for {}: {}",
            model_id,
            parsed.message.unwrap_or_default()
        );
    }
    let data = parsed.data.context("No data in API response")?;
    Ok(data
        .files
        .into_iter()
        .filter(|f| f.file_type == "blob")
        .collect())
}

fn download_file(
    model_id: &str,
    file: &FileEntry,
    dest_dir: &Path,
    logs: &Arc<Mutex<Vec<String>>>,
) -> Result<()> {
    let file_path = dest_dir.join(&file.path);
    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    if file_path.exists() {
        if let Ok(meta) = std::fs::metadata(&file_path) {
            if meta.len() == file.size {
                logs.lock()
                    .push(format!("  ✓ {} (already exists)", file.name));
                return Ok(());
            }
        }
        let _ = std::fs::remove_file(&file_path);
    }

    let tmp_dir = std::env::temp_dir().join("evi-download").join(model_id);
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp_path = tmp_dir.join(&file.path);

    let existing_size = if tmp_path.exists() {
        match std::fs::metadata(&tmp_path) {
            Ok(meta) if meta.len() == file.size => meta.len(),
            Ok(meta) if meta.len() < file.size => meta.len(),
            _ => {
                let _ = std::fs::remove_file(&tmp_path);
                0
            }
        }
    } else {
        0
    };

    let url = format!(
        "{}/{}/resolve/master/{}",
        DOWNLOAD_BASE, model_id, file.path
    );
    if existing_size > 0 {
        logs.lock().push(format!(
            "  ↓ {} ({}, resuming from {})",
            file.name,
            format_bytes(file.size),
            format_bytes(existing_size)
        ));
    } else {
        logs.lock()
            .push(format!("  ↓ {} ({})", file.name, format_bytes(file.size)));
    }

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(std::time::Duration::from_secs(600)))
        .build()
        .into();
    let mut request = agent.get(&url).header("User-Agent", "evi/2.1");
    if existing_size > 0 {
        request = request.header("Range", format!("bytes={}-", existing_size));
    }
    let resp = request
        .call()
        .with_context(|| format!("Failed to download {}", file.name))?;

    let status = resp.status();
    let (start_offset, total_size) = if status == 206 {
        let content_range = resp
            .headers()
            .get("content-range")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let total = content_range
            .split('/')
            .last()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(file.size);
        (existing_size, total)
    } else {
        (0, file.size)
    };

    let reader = resp.into_body().into_reader();
    let mut reader = ProgressReader::new(reader, total_size, logs.clone(), file.name.clone());
    reader.read = start_offset;

    if let Some(parent) = tmp_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = if start_offset > 0 && tmp_path.exists() {
        std::fs::OpenOptions::new()
            .append(true)
            .open(&tmp_path)
            .with_context(|| format!("Failed to open {}", tmp_path.display()))?
    } else {
        std::fs::File::create(&tmp_path)
            .with_context(|| format!("Failed to create {}", tmp_path.display()))?
    };
    std::io::copy(&mut reader, &mut out)
        .with_context(|| format!("Failed to write {}", file.name))?;
    drop(out);

    std::fs::rename(&tmp_path, &file_path)
        .with_context(|| format!("Failed to move {} to final location", file.name))?;

    logs.lock()
        .push(format!("  ✓ {} ({})", file.name, format_bytes(file.size)));
    Ok(())
}

struct ProgressReader<R: Read> {
    inner: R,
    total: u64,
    read: u64,
    last_pct: u64,
    logs: Arc<Mutex<Vec<String>>>,
    name: String,
}

impl<R: Read> ProgressReader<R> {
    fn new(inner: R, total: u64, logs: Arc<Mutex<Vec<String>>>, name: String) -> Self {
        Self {
            inner,
            total,
            read: 0,
            last_pct: 0,
            logs,
            name,
        }
    }
}

impl<R: Read> Read for ProgressReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.read += n as u64;
            if self.total > 0 {
                let pct = self.read * 100 / self.total;
                if pct >= self.last_pct + 10 {
                    self.last_pct = pct;
                    self.logs
                        .lock()
                        .push(format!("    ... {}% ({})", pct, self.name));
                }
            }
        }
        Ok(n)
    }
}

pub fn download_required_models() -> Result<()> {
    for (idx, model_id) in REQUIRED_MODELS.iter().enumerate() {
        println!("[{}/{}] {} ...", idx + 1, REQUIRED_MODELS.len(), model_id);
        let dest_dir = models_download_dir().join(model_id);
        std::fs::create_dir_all(&dest_dir)?;

        let files = fetch_file_list(model_id)?;
        println!("  {} files to download", files.len());
        for file in &files {
            let fake_logs = Arc::new(Mutex::new(Vec::new()));
            download_file(model_id, file, &dest_dir, &fake_logs)?;
        }
        println!("[done] {}", model_id);
    }
    println!("All models downloaded!");
    Ok(())
}

fn download_to_channel(logs: Arc<Mutex<Vec<String>>>) -> Result<()> {
    let missing = missing_models();
    if missing.is_empty() {
        logs.lock().push("所有模型已存在，无需下载。".to_string());
        return Ok(());
    }

    for (idx, model_id) in missing.iter().enumerate() {
        logs.lock().push(format!(
            "[{}/{}] ===== {} =====",
            idx + 1,
            missing.len(),
            model_id
        ));

        let dest_dir = models_download_dir().join(model_id);
        std::fs::create_dir_all(&dest_dir)?;

        let files = fetch_file_list(model_id).map_err(|e| {
            logs.lock().push(format!("获取文件列表失败: {:?}", e));
            e
        })?;

        for file in &files {
            download_file(model_id, file, &dest_dir, &logs).map_err(|e| {
                logs.lock().push(format!("下载失败: {:?}", e));
                e
            })?;
        }

        logs.lock().push(String::new());
    }
    logs.lock().push("所有模型下载完成！".to_string());
    Ok(())
}

struct DownloadApp {
    logs: Arc<Mutex<Vec<String>>>,
    done: Arc<AtomicBool>,
    done_time: Option<std::time::Instant>,
}

impl eframe::App for DownloadApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let logs = self.logs.lock();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(8.0);
                if self.done.load(Ordering::SeqCst) {
                    ui.heading("模型下载完成");
                } else {
                    ui.heading("正在下载语音识别模型...");
                }
                ui.add_space(4.0);
            });
            ui.separator();
            egui::ScrollArea::vertical()
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    ui.add_space(4.0);
                    for line in logs.iter() {
                        ui.monospace(line);
                    }
                });
        });

        drop(logs);

        if self.done.load(Ordering::SeqCst) {
            if self.done_time.is_none() {
                self.done_time = Some(std::time::Instant::now());
            }
            if self.done_time.unwrap().elapsed() > std::time::Duration::from_secs(2) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        }

        ctx.request_repaint_after(std::time::Duration::from_millis(200));
    }
}

fn load_fonts(ctx: &egui::Context) {
    let paths: &[&str] = &[
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
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

pub fn run_download_window() -> Result<()> {
    let logs: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let done = Arc::new(AtomicBool::new(false));
    let result: Arc<Mutex<Option<Result<()>>>> = Arc::new(Mutex::new(None));

    logs.lock()
        .push("EVI 输入法 - 自动下载语音识别模型".to_string());
    logs.lock().push(String::new());

    {
        let dl_logs = logs.clone();
        let dl_done = done.clone();
        let dl_result = result.clone();
        std::thread::spawn(move || {
            let r = download_to_channel(dl_logs);
            *dl_result.lock() = Some(r);
            dl_done.store(true, Ordering::SeqCst);
        });
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("EVI - 下载模型")
            .with_inner_size([640.0, 480.0])
            .with_min_inner_size([400.0, 300.0]),
        ..Default::default()
    };

    eframe::run_native(
        "EVI - 下载模型",
        native_options,
        Box::new(move |cc| {
            load_fonts(&cc.egui_ctx);
            Ok(Box::new(DownloadApp {
                logs,
                done,
                done_time: None,
            }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI error: {}", e))?;

    let res = result.lock().take();
    match res {
        Some(Ok(())) => Ok(()),
        Some(Err(e)) => Err(e),
        None => anyhow::bail!("下载未完成"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_models() {
        download_required_models().unwrap();
    }
}
