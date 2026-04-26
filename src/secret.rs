use log::info;
use parking_lot::Mutex;
use std::path::PathBuf;

static API_KEY: Mutex<Option<String>> = Mutex::new(None);

fn config_file_path() -> PathBuf {
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".evi_key")
}

fn xor_decrypt(data: &[u8], key: &[u8]) -> Vec<u8> {
    data.iter()
        .enumerate()
        .map(|(i, b)| b ^ key[i % key.len()])
        .collect()
}

fn hex_decode(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let mut result = Vec::new();
    for i in (0..s.len()).step_by(2) {
        let byte = u8::from_str_radix(&s[i..i + 2], 16).ok()?;
        result.push(byte);
    }
    Some(result)
}

fn derive_key_from_creation_time(path: &PathBuf) -> Option<Vec<u8>> {
    let meta = std::fs::metadata(path).ok()?;
    let created = meta.created().ok()?;
    let secs = created
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    let password = format!("evi{}", secs);
    Some(password.as_bytes().to_vec())
}

pub fn load_key() {
    let path = config_file_path();
    if path.exists() {
        let cipher_key = derive_key_from_creation_time(&path);
        let hex_content = match std::fs::read_to_string(&path) {
            Ok(s) => s.trim().to_string(),
            Err(_) => return,
        };

        if let Some(key) = cipher_key {
            if let Some(encrypted) = hex_decode(&hex_content) {
                let decrypted = xor_decrypt(&encrypted, &key);
                if let Ok(api_key) = String::from_utf8(decrypted) {
                    if !api_key.is_empty() {
                        *API_KEY.lock() = Some(api_key);
                    }
                }
            }
        }
    }
}

pub fn get_api_key() -> Option<String> {
    API_KEY.lock().clone()
}

pub fn save_key(key: &str) {
    if key.is_empty() {
        return;
    }
    let path = config_file_path();
    let _ = std::fs::write(&path, "");
    let cipher_key = match derive_key_from_creation_time(&path) {
        Some(k) => k,
        None => {
            info!("secret: failed to get creation time");
            return;
        }
    };
    let encrypted = xor_decrypt(key.as_bytes(), &cipher_key);
    let hex: String = encrypted.iter().map(|b| format!("{:02x}", b)).collect();
    let _ = std::fs::write(&path, &hex);
    *API_KEY.lock() = Some(key.to_string());
    info!("secret: API key saved");
}
