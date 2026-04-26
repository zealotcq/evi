use anyhow::{bail, Context, Result};
use log::{error, info, warn};
use std::path::{Path, PathBuf};
use std::process::Command;

const VAD_MODEL_ID: &str = "iic/speech_fsmn_vad_zh-cn-16k-common-onnx";
const ASR_MODEL_ID: &str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx";
const PUNC_MODEL_ID: &str = "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx";
const LLM_MODEL_ID: &str = "onnx-community/Qwen2___5-0___5B-Instruct";

#[derive(Clone)]
pub struct RequiredModels {
    pub vad: ModelInfo,
    pub asr: ModelInfo,
    pub punc: ModelInfo,
    pub llm: Option<ModelInfo>,
}

#[derive(Clone)]
pub struct ModelInfo {
    pub model_id: String,
    pub dir_name: String,
    pub key_file: String,
    pub alt_key_files: Vec<String>,
}

impl ModelInfo {
    fn check(&self, base_dir: &Path) -> Option<PathBuf> {
        let candidates = vec![
            base_dir.join(&self.dir_name),
            base_dir.join("iic").join(&self.dir_name),
        ];

        for model_dir in candidates {
            if !model_dir.exists() {
                continue;
            }

            if self.key_file.is_empty() {
                if model_dir.is_dir() {
                    return Some(model_dir);
                }
            }

            for key in std::iter::once(&self.key_file).chain(self.alt_key_files.iter()) {
                if model_dir.join(key).exists() {
                    return Some(model_dir);
                }
            }

            if model_dir.is_dir() && model_dir.read_dir().map(|e| e.count()).unwrap_or(0) > 0 {
                return Some(model_dir);
            }
        }
        None
    }
}

pub fn required_models(llm_refine: bool) -> RequiredModels {
    RequiredModels {
        vad: ModelInfo {
            model_id: VAD_MODEL_ID.to_string(),
            dir_name: "iic/speech_fsmn_vad_zh-cn-16k-common-onnx".to_string(),
            key_file: "model.onnx".to_string(),
            alt_key_files: vec!["model_quant.onnx".to_string()],
        },
        asr: ModelInfo {
            model_id: ASR_MODEL_ID.to_string(),
            dir_name: "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx"
                .to_string(),
            key_file: "model.onnx".to_string(),
            alt_key_files: vec!["model_quant.onnx".to_string()],
        },
        punc: ModelInfo {
            model_id: PUNC_MODEL_ID.to_string(),
            dir_name: "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx".to_string(),
            key_file: "model.onnx".to_string(),
            alt_key_files: vec!["model_quant.onnx".to_string()],
        },
        llm: if llm_refine {
            Some(ModelInfo {
                model_id: LLM_MODEL_ID.to_string(),
                dir_name: "onnx-community/Qwen2___5-0___5B-Instruct".to_string(),
                key_file: "model.onnx".to_string(),
                alt_key_files: vec![
                    "model_int8.onnx".to_string(),
                    "onnx/model.onnx".to_string(),
                    "onnx/model_int8.onnx".to_string(),
                ],
            })
        } else {
            None
        },
    }
}

fn get_model_cache_base() -> Option<PathBuf> {
    if let Ok(cache) = std::env::var("MODELSCOPE_CACHE") {
        let p = PathBuf::from(&cache);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(userprofile) = std::env::var("USERPROFILE") {
        let candidates = vec![
            PathBuf::from(&userprofile)
                .join(".cache")
                .join("modelscope")
                .join("models"),
            PathBuf::from(&userprofile)
                .join(".cache")
                .join("model_scope")
                .join("models"),
            PathBuf::from(&userprofile)
                .join(".modelscope")
                .join("cache")
                .join("models"),
            PathBuf::from(&userprofile)
                .join(".cache")
                .join("modelscope-cn"),
            PathBuf::from(&userprofile)
                .join("AppData")
                .join("Local")
                .join("modelscope")
                .join("models"),
            PathBuf::from(&userprofile)
                .join("AppData")
                .join("Roaming")
                .join("modelscope")
                .join("models"),
        ];

        for p in candidates {
            if p.exists() || p.parent().map(|p| p.exists()).unwrap_or(false) {
                return Some(p);
            }
        }
    }

    if let Ok(home) = std::env::var("HOME") {
        let candidates = vec![
            PathBuf::from(&home)
                .join(".cache")
                .join("modelscope")
                .join("models"),
            PathBuf::from(&home)
                .join(".modelscope")
                .join("cache")
                .join("models"),
            PathBuf::from(&home)
                .join(".cache")
                .join("model_scope")
                .join("models"),
        ];
        for p in candidates {
            if p.exists() || p.parent().map(|p| p.exists()).unwrap_or(false) {
                return Some(p);
            }
        }
    }

    None
}

fn find_model_dir_in_cache(cache_base: &Path, model_info: &ModelInfo) -> Option<PathBuf> {
    if !cache_base.exists() {
        return None;
    }

    if let Some(found) = model_info.check(cache_base) {
        return Some(found);
    }

    if let Ok(entries) = std::fs::read_dir(cache_base) {
        for entry in entries.flatten() {
            let sub = entry.path();
            if sub.is_dir() {
                if let Some(found) = model_info.check(&sub) {
                    return Some(found);
                }

                if let Ok(sub_entries) = std::fs::read_dir(&sub) {
                    for sub_entry in sub_entries.flatten() {
                        if sub_entry.path().is_dir() {
                            if let Some(found) = model_info.check(&sub_entry.path()) {
                                return Some(found);
                            }
                        }
                    }
                }
            }
        }
    }

    None
}

pub fn detect_modelscope_base_dir() -> Option<PathBuf> {
    let cache = get_model_cache_base()?;

    if let Ok(entries) = std::fs::read_dir(&cache) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                let name = path.file_name()?.to_str()?;
                if name.contains("speech") || name.contains("punc") || name.contains("onnx") {
                    if let Some(parent) = path.parent() {
                        return Some(parent.to_path_buf());
                    }
                }
            }
        }
    }

    Some(cache)
}

pub fn check_models(
    llm_refine: bool,
    explicit_base_dir: Option<&Path>,
) -> Result<(PathBuf, Vec<ModelInfo>)> {
    let explicit_dir = explicit_base_dir.map(|p| p.to_path_buf());
    let cache_base = detect_modelscope_base_dir();

    let search_dirs: Vec<PathBuf> = {
        let mut dirs = Vec::new();
        if let Some(ref d) = explicit_dir {
            dirs.push(d.clone());
        }
        if let Some(ref d) = cache_base {
            if !dirs.contains(d) {
                dirs.push(d.clone());
            }
        }
        dirs
    };

    let required = required_models(llm_refine);
    let mut missing: Vec<ModelInfo> = Vec::new();

    for model_info in [
        required.vad.clone(),
        required.asr.clone(),
        required.punc.clone(),
    ] {
        let mut found = false;
        for dir in &search_dirs {
            if model_info.check(dir).is_some() {
                found = true;
                break;
            }
        }
        if !found {
            missing.push(model_info);
        }
    }

    if let Some(llm_info) = required.llm {
        let mut found = false;
        for dir in &search_dirs {
            if llm_info.check(dir).is_some() {
                found = true;
                break;
            }
        }
        if !found {
            missing.push(llm_info);
        }
    }

    if missing.is_empty() {
        let base = explicit_dir
            .or(cache_base)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
        info!(
            "All required models found. Using base dir: {}",
            base.display()
        );
        return Ok((base, vec![]));
    }

    Ok((
        explicit_dir
            .or(cache_base)
            .unwrap_or_else(|| PathBuf::from(".")),
        missing,
    ))
}

#[cfg(target_os = "windows")]
fn download_models_with_window(model_ids: &[String]) -> Result<()> {
    if model_ids.is_empty() {
        return Ok(());
    }

    let models_list = model_ids
        .iter()
        .map(|s| format!("\"{}\"", s))
        .collect::<Vec<_>>()
        .join(", ");
    let models_array = format!("[{}]", models_list);

    let script = format!(
        r#"
import sys
from modelscope import snapshot_download

models = {}

for model_id in models:
    print('=== Downloading {{model_id}} ===')
    try:
        cache_dir = snapshot_download(model_id, revision='v2.0.5')
        print('[OK] {{model_id}} -> {{cache_dir}}')
    except Exception as e:
        print('[ERROR] Failed to download {{model_id}}: {{}}'.format(e))
        sys.exit(1)

print('\n=== All downloads complete ===')
input('Press Enter to close...')
"#,
        models_array
    );

    let temp_dir = std::env::temp_dir();
    let script_path = temp_dir.join("evi_model_download.py");
    std::fs::write(&script_path, &script).context("Failed to write download script")?;

    info!(
        "Launching download window for {} model(s)...",
        model_ids.len()
    );

    let output = Command::new("cmd")
        .args(["/k", "python", &script_path.to_string_lossy()])
        .spawn();

    match output {
        Ok(_child) => {
            info!("Download window opened. Please wait for it to complete.");
        }
        Err(e) => {
            error!("Failed to open download window: {}", e);
            bail!("Cannot spawn download window: {}", e);
        }
    }

    Ok(())
}

#[cfg(not(target_os = "windows"))]
fn download_models_with_window(model_ids: &[String]) -> Result<()> {
    if model_ids.is_empty() {
        return Ok(());
    }

    let models_list = model_ids
        .iter()
        .map(|s| format!("\"{}\"", s))
        .collect::<Vec<_>>()
        .join(", ");
    let models_array = format!("[{}]", models_list);

    let script = format!(
        r#"
import sys
from modelscope import snapshot_download

models = {}

for model_id in models:
    print(f'=== Downloading {model_id} ===')
    try:
        cache_dir = snapshot_download(model_id, revision='v2.0.5')
        print(f'[OK] {model_id} -> {cache_dir}')
    except Exception as e:
        print(f'[ERROR] Failed to download {model_id}: {e}')
        sys.exit(1)

print('\n=== All downloads complete ===')
"#,
        models_array
    );

    let temp_dir = std::env::temp_dir();
    let script_path = temp_dir.join("evi_model_download.py");
    std::fs::write(&script_path, &script).context("Failed to write download script")?;

    info!("Launching download for {} model(s)...", model_ids.len());

    let status = Command::new("python")
        .arg(&script_path)
        .status()
        .context("Failed to run download script")?;

    if !status.success() {
        bail!("Download script exited with error");
    }

    Ok(())
}

pub fn download_missing_models(model_ids: &[String]) -> Result<()> {
    if model_ids.is_empty() {
        return Ok(());
    }

    download_models_with_window(model_ids)
}

pub fn wait_for_download_completion(check_fn: impl Fn() -> bool, timeout_secs: u64) -> Result<()> {
    let start = std::time::Instant::now();
    let interval = std::time::Duration::from_secs(2);

    loop {
        if check_fn() {
            return Ok(());
        }

        if start.elapsed().as_secs() >= timeout_secs {
            bail!("Timeout waiting for model download ({}s)", timeout_secs);
        }

        std::thread::sleep(interval);
    }
}

pub fn update_config_model_base_dir(new_base: &Path) -> Result<()> {
    let config_path = crate::Config::home_config_path();

    let raw = if config_path.exists() {
        std::fs::read_to_string(&config_path)
            .with_context(|| format!("Failed to read {}", config_path.display()))?
    } else {
        let exe_dir = crate::Config::exe_dir()?;
        let exe_config = exe_dir.join("config.json");
        std::fs::read_to_string(&exe_config)
            .with_context(|| format!("Failed to read {}", exe_config.display()))?
    };

    let mut json: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("Failed to parse {}", config_path.display()))?;

    let base_str = new_base.to_string_lossy().to_string();

    let obj = json.as_object_mut().unwrap();

    if let Some(existing) = obj.get("model_base_dir") {
        if existing.as_str() == Some(&base_str) {
            info!(
                "Config model_base_dir already set to {}",
                new_base.display()
            );
            return Ok(());
        }
    }

    obj.insert(
        "model_base_dir".to_string(),
        serde_json::Value::String(base_str.clone()),
    );

    let out = serde_json::to_string_pretty(&json).with_context(|| "Failed to serialize config")?;
    std::fs::write(&config_path, &out)
        .with_context(|| format!("Failed to write {}", config_path.display()))?;

    info!("Updated ~/.evi_config.json model_base_dir to: {}", base_str);
    Ok(())
}

pub fn ensure_models_on_startup(llm_refine: bool, explicit_base: Option<&Path>) -> Result<PathBuf> {
    let (base_dir, missing) = check_models(llm_refine, explicit_base)?;

    if missing.is_empty() {
        return Ok(base_dir);
    }

    let missing_ids: Vec<String> = missing.iter().map(|m| m.model_id.clone()).collect();
    warn!(
        "Missing {} model(s): {}",
        missing.len(),
        missing_ids.join(", ")
    );

    download_missing_models(&missing_ids)?;

    let (new_base, still_missing) = check_models(llm_refine, explicit_base)?;

    if still_missing.is_empty() {
        if let Err(e) = update_config_model_base_dir(&new_base) {
            warn!("Failed to update config: {}", e);
        }
        return Ok(new_base);
    }

    let still_ids: Vec<String> = still_missing.iter().map(|m| m.model_id.clone()).collect();
    error!(
        "Models still missing after download attempt: {}",
        still_ids.join(", ")
    );
    bail!("Download failed for: {}", still_ids.join(", "));
}
