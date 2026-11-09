//! Refine log database for recording ASR/LLM results.

use anyhow::{Context, Result};
use log::warn;
use rusqlite::Connection;
use std::sync::atomic::{AtomicBool, Ordering};

static DB_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub struct DebugRefine {
    conn: Connection,
}

impl DebugRefine {
    pub fn open(path: &str) -> Result<Self> {
        let conn = Connection::open(path)
            .with_context(|| format!("Failed to open SQLite DB: {}", path))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA encoding='UTF-8';
             CREATE TABLE IF NOT EXISTS refine_log (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 original TEXT NOT NULL,
                 refined TEXT NOT NULL,
                 submitted INTEGER NOT NULL DEFAULT 0
             );
             CREATE TABLE IF NOT EXISTS user (
                 id INTEGER PRIMARY KEY CHECK (id = 1),
                 uuid TEXT NOT NULL
             );",
        )
        .with_context(|| "Failed to initialize tables")?;

        conn.execute_batch(
            "ALTER TABLE refine_log ADD COLUMN submitted INTEGER NOT NULL DEFAULT 0;",
        )
        .ok();

        if !DB_INITIALIZED.load(Ordering::SeqCst) {
            let count: i64 = conn
                .query_row("SELECT COUNT(*) FROM user", [], |row| row.get(0))
                .unwrap_or(0);
            if count == 0 {
                let uuid = uuid::Uuid::new_v4().to_string();
                if let Err(e) = conn.execute(
                    "INSERT INTO user (id, uuid) VALUES (1, ?1)",
                    rusqlite::params![uuid],
                ) {
                    warn!("DebugRefine: failed to insert user uuid: {}", e);
                }
            }
            DB_INITIALIZED.store(true, Ordering::SeqCst);
        }

        Ok(Self { conn })
    }

    pub fn log_refine(&self, original: &str, refined: &str) {
        if original.is_empty() || original.chars().count() <= 10 {
            return;
        }
        self.insert(original, refined);
    }

    fn insert(&self, original: &str, refined: &str) {
        if let Err(e) = self.conn.execute(
            "INSERT INTO refine_log (original, refined) VALUES (?1, ?2)",
            rusqlite::params![original, refined],
        ) {
            warn!("DebugRefine: failed to insert: {}", e);
        }
    }
}
