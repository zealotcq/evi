//! Refine log database for recording ASR/LLM results.

use anyhow::{Context, Result};
use log::warn;
use rusqlite::Connection;

pub struct RefineDb {
    conn: Connection,
}

impl RefineDb {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open SQLite DB: {}", path))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA encoding='UTF-8';
             CREATE TABLE IF NOT EXISTS refine_log (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 original TEXT NOT NULL,
                 refined TEXT NOT NULL
             );",
        )
        .with_context(|| "Failed to initialize refine_log table")?;
        Ok(Self { conn })
    }

    pub fn log_refine(&self, original: &str, refined: &str) {
        if original != refined {
            self.insert(original, refined);
        } else if original.chars().count() > 10 {
            self.insert(original, "");
        }
    }

    fn insert(&self, original: &str, refined: &str) {
        if let Err(e) = self.conn.execute(
            "INSERT INTO refine_log (original, refined) VALUES (?1, ?2)",
            rusqlite::params![original, refined],
        ) {
            warn!("RefineDb: failed to insert: {}", e);
        }
    }
}
