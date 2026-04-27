pub fn vad_model_dir(base_dir: &std::path::Path) -> std::path::PathBuf {
    base_dir.join("iic/speech_fsmn_vad_zh-cn-16k-common-onnx")
}

pub fn asr_model_dir(base_dir: &std::path::Path) -> std::path::PathBuf {
    base_dir.join("iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx")
}

pub fn punc_model_dir(base_dir: &std::path::Path) -> std::path::PathBuf {
    base_dir.join("iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx")
}

pub fn llm_model_dir(base_dir: &std::path::Path) -> std::path::PathBuf {
    base_dir.join("onnx-community/Qwen2___5-0___5B-Instruct")
}

pub fn base_dir(cfg: &crate::Config) -> std::path::PathBuf {
    if let Some(ref base) = cfg.model_base_dir {
        expand_home(base)
    } else {
        default_base_dir()
    }
}

pub fn default_base_dir() -> std::path::PathBuf {
    if let Ok(val) = std::env::var("MODELSCOPE_CACHE") {
        let p = std::path::PathBuf::from(&val);
        if REQUIRED_MODEL_SUBDIRS
            .iter()
            .all(|sub| has_onnx_model(&p.join(sub)))
        {
            return p;
        }
    }

    for candidate in modelscope_cache_candidates() {
        if REQUIRED_MODEL_SUBDIRS
            .iter()
            .all(|sub| has_onnx_model(&candidate.join(sub)))
        {
            return candidate;
        }
    }

    crate::Config::exe_dir().unwrap_or_else(|_| std::path::PathBuf::from("."))
}

pub fn refine_db_path() -> std::path::PathBuf {
    crate::Config::exe_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("refine_log.db")
}

fn expand_home(path: &str) -> std::path::PathBuf {
    if path.starts_with('~') {
        let home = std::env::var("USERPROFILE")
            .or_else(|_| std::env::var("HOME"))
            .unwrap_or_else(|_| ".".to_string());
        std::path::PathBuf::from(home).join(&path[1..])
    } else {
        std::path::PathBuf::from(path)
    }
}

fn home_dir() -> std::path::PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string())
        .into()
}

const REQUIRED_MODEL_SUBDIRS: &[&str] = &[
    "iic/speech_fsmn_vad_zh-cn-16k-common-onnx",
    "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
    "iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx",
];

fn has_onnx_model(dir: &std::path::Path) -> bool {
    dir.join("model.onnx").exists() || dir.join("model_quant.onnx").exists()
}

pub fn ensure_model_dir(cfg: &mut crate::Config) {
    let base = base_dir(cfg);
    let all_found = REQUIRED_MODEL_SUBDIRS
        .iter()
        .all(|sub| has_onnx_model(&base.join(sub)));

    if all_found {
        log::debug!("All required models found under {}", base.display());
        return;
    }

    log::info!(
        "Models not found under {}, searching ModelScope cache...",
        base.display()
    );

    let candidates = modelscope_cache_candidates();
    for candidate in &candidates {
        if REQUIRED_MODEL_SUBDIRS
            .iter()
            .all(|sub| has_onnx_model(&candidate.join(sub)))
        {
            log::info!("Found models at {}, updating config", candidate.display());
            cfg.model_base_dir = Some(candidate.to_string_lossy().into_owned());
            if let Err(e) = crate::Config::save_model_base_dir(cfg.model_base_dir.as_deref()) {
                log::warn!("Failed to save model_base_dir to config: {}", e);
            }
            return;
        }
    }

    log::warn!(
        "No ModelScope cache found with required models. Searched: {:?}",
        candidates
    );
}

fn modelscope_cache_candidates() -> Vec<std::path::PathBuf> {
    let home = home_dir();
    let mut candidates = Vec::new();

    if let Ok(val) = std::env::var("MODELSCOPE_CACHE") {
        candidates.push(std::path::PathBuf::from(val).join("models"));
    }

    candidates.push(home.join(".modelscope").join("hub").join("models"));
    candidates.push(
        home.join(".cache")
            .join("modelscope")
            .join("hub")
            .join("models"),
    );

    #[cfg(target_os = "windows")]
    {
        let local_app_data = std::env::var("LOCALAPPDATA").unwrap_or_default();
        if !local_app_data.is_empty() {
            candidates.push(
                std::path::PathBuf::from(local_app_data)
                    .join("modelscope")
                    .join("hub")
                    .join("models"),
            );
        }
    }

    candidates
}

#[cfg(test)]
pub fn test_base_dir() -> std::path::PathBuf {
    default_base_dir()
}
