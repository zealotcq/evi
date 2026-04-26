use anyhow::{bail, Context, Result};
use log::{debug, info, warn};
use serde::Serialize;
use std::fs;
use std::time::Duration;

const COZE_API_URL: &str = "https://api.coze.cn/v1/workflow/stream_run";

#[derive(Serialize)]
struct WorkflowRequest {
    workflow_id: String,
    parameters: WorkflowParams,
}

#[derive(Serialize)]
struct WorkflowParams {
    input: String,
}

pub struct CozeRefineEngine {
    timeout: Duration,
}

impl CozeRefineEngine {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
        }
    }

    pub fn refine(&self, text: &str) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(text.to_string());
        }

        let api_key = crate::secret::get_api_key()
            .ok_or_else(|| anyhow::anyhow!("Coze API key not configured"))?;

        let workflow_id = crate::secret::get_workflow_id()
            .ok_or_else(|| anyhow::anyhow!("Coze Workflow ID not configured"))?;

        debug!(
            "CozeRefine: sending '{}' (timeout={}s)",
            text,
            self.timeout.as_secs()
        );

        let body = WorkflowRequest {
            workflow_id,
            parameters: WorkflowParams {
                input: text.to_string(),
            },
        };

        let body_json =
            serde_json::to_string(&body).with_context(|| "Failed to serialize workflow request")?;

        let config = ureq::Agent::config_builder()
            .timeout_global(Some(self.timeout))
            .build();
        let agent: ureq::Agent = config.into();

        let mut resp = agent
            .post(COZE_API_URL)
            .header("Authorization", &format!("Bearer {}", api_key))
            .header("Content-Type", "application/json")
            .send(body_json.as_str())
            .with_context(|| "Coze API request failed")?;

        let resp_body: String = resp
            .body_mut()
            .read_to_string()
            .with_context(|| "Failed to read Coze response body")?;

        debug!("CozeRefine: raw response ({} bytes)", resp_body.len());

        if resp_body.trim().is_empty() {
            bail!("Coze returned empty response");
        }

        //let _ = fs::write("result.json", &resp_body);

        let output = parse_sse_output(&resp_body)?;
        if output.trim().is_empty() {
            bail!("Coze returned empty output");
        }

        info!("CozeRefine: '{}' -> '{}'", text, output);
        Ok(output)
    }
}

pub fn parse_sse_output(body: &str) -> Result<String> {
    let lines: Vec<&str> = body.lines().collect();
    if lines.is_empty() {
        bail!("Empty response");
    }

    let mut event_type = String::new();
    let mut data_json = String::new();
    let mut collecting_data = false;
    let mut outputs = Vec::new();

    for line in lines {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        if line.starts_with("event:") {
            event_type = line.trim_start_matches("event:").trim().to_string();
            continue;
        }

        if line.starts_with("data:") {
            collecting_data = true;
            data_json = line.trim_start_matches("data:").trim().to_string();
            continue;
        }

        if collecting_data && !line.starts_with("id:") && !line.starts_with("event:") {
            data_json.push_str(line);
        }

        if collecting_data
            && (line.starts_with("id:") || line.starts_with("event:") || line.is_empty())
        {
            if event_type == "Message" || event_type == "_interrupt" {
                if let Some(output) = extract_output_from_data(&data_json)? {
                    outputs.push(output);
                }
            }
            event_type.clear();
            data_json.clear();
            collecting_data = false;
        }
    }

    if collecting_data && !data_json.is_empty() {
        if event_type == "Message" || event_type == "_interrupt" {
            if let Some(output) = extract_output_from_data(&data_json)? {
                outputs.push(output);
            }
        }
        if event_type == "Error" {
            if let Err(e) = extract_error_from_data(&data_json) {
                return Err(e);
            }
        }
    }

    if outputs.is_empty() {
        bail!("No Message events found in response");
    }

    Ok(outputs.join(""))
}

fn extract_error_from_data(data_json: &str) -> Result<()> {
    if data_json.is_empty() {
        return Ok(());
    }
    if let Ok(err_obj) = serde_json::from_str::<serde_json::Value>(data_json) {
        let error_code = err_obj
            .get("error_code")
            .and_then(|v| v.as_i64())
            .unwrap_or(-1);
        let msg = err_obj
            .get("error_message")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        bail!("Coze API error {}: {}", error_code, msg);
    }
    Ok(())
}

fn extract_output_from_data(data_json: &str) -> Result<Option<String>> {
    if data_json.is_empty() {
        return Ok(None);
    }

    if let Ok(content_obj) = serde_json::from_str::<serde_json::Value>(data_json) {
        if let Some(content) = content_obj.get("content").and_then(|v| v.as_str()) {
            if let Ok(output_obj) = serde_json::from_str::<serde_json::Value>(content) {
                if let Some(output) = output_obj.get("output").and_then(|v| v.as_str()) {
                    return Ok(Some(output.to_string()));
                }
            }
        }
    }

    Ok(None)
}

pub fn coze_refine_with_fallback(text: &str, timeout_secs: u64) -> String {
    let engine = CozeRefineEngine::new(timeout_secs);
    match engine.refine(text) {
        Ok(refined) => refined,
        Err(e) => {
            warn!("CozeRefine failed: {}, using original text", e);
            text.to_string()
        }
    }
}
