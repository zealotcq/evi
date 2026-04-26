use anyhow::{bail, Context, Result};
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

#[derive(Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    content: String,
    #[serde(default)]
    reasoning_content: String,
}

#[derive(Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    temperature: f32,
    max_tokens: usize,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    extra_body: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
}

#[derive(Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

pub struct LlmRemoteEngine {
    api_url: String,
    model: String,
    timeout: Duration,
    max_tokens_ratio: f64,
}

impl LlmRemoteEngine {
    pub fn new(api_url: String, model: String, timeout_secs: u64, max_tokens_ratio: f64) -> Self {
        Self {
            api_url,
            model,
            timeout: Duration::from_secs(timeout_secs),
            max_tokens_ratio,
        }
    }

    pub fn refine(&self, text: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(text.to_string());
        }

        let api_key = crate::secret::get_api_key()
            .ok_or_else(|| anyhow::anyhow!("API key not configured"))?;

        debug!(
            "LlmRemote: sending '{}' (timeout={}s, model={})",
            text,
            self.timeout.as_secs(),
            self.model
        );

        let system_prompt = self.load_system_prompt()?;

        let mut thinking: Option<serde_json::Value> = None;
        let model_lower = self.model.to_lowercase();
        if model_lower.contains("glm-") || model_lower.contains("glm") {
            thinking = Some(serde_json::json!({"type": "disabled"}));
            info!("LlmRemote: disabled thinking for model {}", self.model);
        }

        let input_tokens = estimate_tokens(text);
        let max_tokens = (input_tokens as f64 * self.max_tokens_ratio) as usize + 28;

        let request = OpenAIRequest {
            model: self.model.clone(),
            messages: vec![
                OpenAIMessage {
                    role: "system".to_string(),
                    content: system_prompt,
                    reasoning_content: String::new(),
                },
                OpenAIMessage {
                    role: "user".to_string(),
                    content: format!("请润色以下文本：{}", text),
                    reasoning_content: String::new(),
                },
            ],
            temperature: 0.0,
            max_tokens,
            stream: false,
            thinking,
            extra_body: None,
        };

        let body_json =
            serde_json::to_string(&request).with_context(|| "Failed to serialize request")?;

        let config = ureq::Agent::config_builder()
            .timeout_global(Some(self.timeout))
            .build();
        let agent: ureq::Agent = config.into();

        let full_url = if self.api_url.ends_with("/chat/completions") {
            self.api_url.clone()
        } else {
            format!("{}/chat/completions", self.api_url.trim_end_matches('/'))
        };

        info!(
            "LlmRemote: POST to {} (timeout={}s)",
            full_url,
            self.timeout.as_secs()
        );

        let mut resp = match agent
            .post(&full_url)
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .send(body_json.as_str())
        {
            Ok(r) => r,
            Err(e) => {
                let result_path = std::path::PathBuf::from("result.json");
                let _ = std::fs::write(&result_path, &format!("error: {}", e));
                warn!("LlmRemote: request failed: {}", e);
                bail!("API request failed: {}", e);
            }
        };

        let status = resp.status();
        info!("LlmRemote: response status = {}", status);

        let resp_body: String = match resp.body_mut().read_to_string() {
            Ok(s) => s,
            Err(e) => {
                warn!("LlmRemote: failed to read response body: {}", e);
                bail!("Failed to read response body: {}", e);
            }
        };

        debug!("LlmRemote: raw response ({} bytes)", resp_body.len());

        if resp_body.trim().is_empty() {
            warn!("LlmRemote: empty response body");
            bail!("API returned empty response");
        }

        // let result_path = std::path::PathBuf::from("result.json");
        // if let Err(e) = std::fs::write(&result_path, &resp_body) {
        //     warn!(
        //         "LlmRemote: failed to save result to {}: {}",
        //         result_path.display(),
        //         e
        //     );
        // } else {
        //     info!("LlmRemote: result saved to {}", result_path.display());
        // }

        let response: OpenAIResponse = serde_json::from_str(&resp_body).with_context(|| {
            format!(
                "Failed to parse response: {}",
                &resp_body[..resp_body.len().min(500)]
            )
        })?;

        let raw_output = response
            .choices
            .first()
            .map(|c| {
                let content = c.message.content.trim();
                if content.is_empty() {
                    c.message.reasoning_content.trim().to_string()
                } else {
                    content.to_string()
                }
            })
            .unwrap_or_default();

        if raw_output.is_empty() {
            warn!("LlmRemote: both content and reasoning_content are empty");
            bail!("API returned empty output");
        }

        let output = parse_json_string_field(&raw_output);
        info!("LlmRemote: parsed output: '{}'", output);

        info!("LlmRemote: '{}' -> '{}'", text, output);
        Ok(output)
    }

    fn load_system_prompt(&self) -> Result<String> {
        let exe_dir = crate::Config::exe_dir()?;
        let prompt_path = exe_dir.join("system_prompt.txt");
        if prompt_path.exists() {
            std::fs::read_to_string(&prompt_path)
                .with_context(|| format!("Failed to read {}", prompt_path.display()))
        } else {
            Ok(String::new())
        }
    }
}

fn estimate_tokens(text: &str) -> usize {
    text.chars().count()
}

fn parse_json_string_field(input: &str) -> String {
    let mut cleaned = input.trim().to_string();

    if cleaned.starts_with("```") {
        if let Some(end) = cleaned.find('\n') {
            cleaned = cleaned[end + 1..].to_string();
        }
        if cleaned.ends_with("```") {
            cleaned = cleaned[..cleaned.len() - 3].trim().to_string();
        }
    }

    if !cleaned.starts_with('{') {
        return cleaned;
    }

    match serde_json::from_str::<serde_json::Value>(&cleaned) {
        Ok(serde_json::Value::Object(obj)) => {
            for (_, val) in obj.iter() {
                if let Some(s) = val.as_str() {
                    return s.to_string();
                }
            }
            cleaned
        }
        Ok(_) => cleaned,
        Err(_) => cleaned,
    }
}

pub fn llm_remote_refine_with_fallback(
    text: &str,
    api_url: &str,
    model: &str,
    timeout_secs: u64,
    max_tokens_ratio: f64,
) -> String {
    let engine = LlmRemoteEngine::new(
        api_url.to_string(),
        model.to_string(),
        timeout_secs,
        max_tokens_ratio,
    );
    match engine.refine(text) {
        Ok(refined) => refined,
        Err(e) => {
            warn!("LlmRemote failed: {}, using original text", e);
            text.to_string()
        }
    }
}
