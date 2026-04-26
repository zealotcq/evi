use crate::engine::debug_refine::DebugRefine;
use crate::engine::fallback::FallbackRefineEngine;
use crate::engine::llm::LlmEngine;
use crate::engine::punc::PuncEngine;
use crate::ui;
use crate::Config;
use crate::TokenScore;
use anyhow::Result;
use log::{debug, info, warn};
use parking_lot::Mutex;

pub struct RefineManager {
    fallback: FallbackRefineEngine,
    punc: Mutex<PuncEngine>,
    llm: Option<LlmEngine>,
    llm_remote: Option<crate::LlmRemoteConfig>,
    min_refine_length: usize,
    max_refine_ratio: f64,
    trailing_punct: String,
}

impl RefineManager {
    pub fn new(cfg: &Config, punc: PuncEngine) -> Result<Self> {
        let fallback = FallbackRefineEngine::new(cfg.refine_fallback.clone());
        let llm = if cfg.llm_refine {
            info!("Loading LLM model...");
            Some(LlmEngine::new(cfg)?)
        } else {
            None
        };

        Ok(Self {
            fallback,
            punc: Mutex::new(punc),
            llm,
            llm_remote: cfg.llm_remote.clone(),
            min_refine_length: cfg.min_refine_length,
            max_refine_ratio: cfg.max_refine_ratio,
            trailing_punct: cfg.trailing_punct.clone(),
        })
    }

    pub fn rebuild_llm_prompt(
        &mut self,
        system_prompt: &str,
        prefill_template: &str,
    ) -> Result<()> {
        if let Some(ref mut llm) = self.llm {
            llm.rebuild_prompt(system_prompt, prefill_template)
        } else {
            Ok(())
        }
    }

    pub fn refine(&self, text: &str, db: &DebugRefine) -> (String, Vec<TokenScore>) {
        let text = text.trim();
        if text.is_empty() {
            return (text.to_string(), vec![]);
        }

        let original_chars = text.chars().count();
        let refined = self.do_refine(text, db);
        let refined_chars = refined.chars().count();

        let refined = if (refined_chars as f64) > (original_chars as f64) * self.max_refine_ratio {
            warn!(
                "RefineMgr: reject (refined chars={} > {} * {}), using original",
                refined_chars, original_chars, self.max_refine_ratio
            );
            text.to_string()
        } else {
            refined
        };

        let refined = self.ensure_trailing_punct(&refined);
        (refined, vec![])
    }

    fn do_refine(&self, text: &str, db: &DebugRefine) -> String {
        let text_chars = text.chars().count();

        if text_chars <= self.min_refine_length {
            return self.fallback_refine_with_punc(text, db);
        }

        if ui::get_llm_remote_enabled() {
            if let Some(ref cfg) = self.llm_remote {
                let refined = crate::engine::llm_remote::llm_remote_refine_with_fallback(
                    text,
                    &cfg.url,
                    &cfg.model,
                    cfg.timeout,
                    cfg.max_tokens_ratio,
                );
                if refined != text {
                    db.log_refine(text, &refined);
                    info!("RefineMgr: LlmRemote refined '{}' -> '{}'", text, refined);
                }
                return refined;
            }
        }

        if let Some(ref llm) = self.llm {
            match llm.refine(text, db) {
                Ok((refined, _tokens)) => {
                    if refined != text {
                        info!("RefineMgr: LLM refined '{}' -> '{}'", text, refined);
                    }
                    return refined;
                }
                Err(e) => {
                    warn!(
                        "RefineMgr: LLM failed: {}, falling back to fallback+punc",
                        e
                    );
                    return self.fallback_refine_with_punc(text, db);
                }
            }
        }

        self.fallback_refine_with_punc(text, db)
    }

    fn fallback_refine_with_punc(&self, text: &str, db: &DebugRefine) -> String {
        let filtered = self.fallback.refine(text, db);
        match self.punc.lock().add_punct(&filtered) {
            Ok(punct_text) => {
                debug!("RefineMgr: punc added: '{}' -> '{}'", filtered, punct_text);
                punct_text
            }
            Err(e) => {
                warn!("RefineMgr: punc failed: {}, using filtered text", e);
                filtered
            }
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
}
