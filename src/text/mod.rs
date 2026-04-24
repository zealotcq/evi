//! Text management abstractions and platform implementations.

#[cfg(target_os = "windows")]
pub mod tsf;

use crate::{TextChangeEvent, TextObserver, TextOutput, TextSnapshot};
use anyhow::Result;

/// A combined text session that provides both injection and monitoring.
pub struct PlatformTextSession {
    output: Box<dyn TextOutput>,
    observer: Box<dyn TextObserver>,
}

impl PlatformTextSession {
    pub fn new(output: Box<dyn TextOutput>, observer: Box<dyn TextObserver>) -> Self {
        Self { output, observer }
    }
}

impl TextOutput for PlatformTextSession {
    fn commit_text(&self, text: &str) -> Result<()> {
        self.output.commit_text(text)
    }

    fn method_name(&self) -> &str {
        self.output.method_name()
    }
}

impl TextObserver for PlatformTextSession {
    fn start_monitoring(&mut self) -> Result<()> {
        self.observer.start_monitoring()
    }

    fn stop_monitoring(&mut self) -> Result<()> {
        self.observer.stop_monitoring()
    }

    fn poll_changes(&self) -> Vec<TextChangeEvent> {
        self.observer.poll_changes()
    }

    fn snapshot(&self) -> Result<TextSnapshot> {
        self.observer.snapshot()
    }
}

impl crate::TextSession for PlatformTextSession {}

pub fn log_event(event: &str, detail: &str) {
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let ts = now.as_millis();
    let msg = if detail.is_empty() {
        format!("{} [{}]\n", ts, event)
    } else {
        format!("{} [{}] {}\n", ts, event, detail)
    };
    if let Ok(mut f) = OpenOptions::new()
        .create(true)
        .append(true)
        .open("log/vi_events.log")
    {
        let _ = f.write_all(msg.as_bytes());
    }
}
