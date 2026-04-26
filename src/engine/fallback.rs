//! Fallback refine engine using keyword replacement.

use crate::engine::debug_refine::DebugRefine;
use log::info;

pub struct FallbackRefineEngine {
    remove_words: Vec<String>,
}

impl FallbackRefineEngine {
    pub fn new(remove_words: Vec<String>) -> Self {
        Self { remove_words }
    }

    pub fn refine(&self, text: &str, dr: &DebugRefine) -> String {
        let mut result = text.to_string();
        for word in &self.remove_words {
            result = result.replace(word, "");
        }
        if result != text {
            info!("FallbackRefine: '{}' -> '{}'", text, result);
        }
        dr.log_refine(text, &result);
        result
    }
}
