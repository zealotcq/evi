use std::sync::atomic::{AtomicBool, Ordering};

static DIALOG_OPEN: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "windows")]
pub fn request_api_key_dialog() {
    if DIALOG_OPEN
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }

    std::thread::spawn(move || {
        let key = crate::ui::win32::show_api_key_dialog();
        if let Some(k) = key {
            if !k.is_empty() {
                crate::secret::save_key(&k);
            }
        }
        DIALOG_OPEN.store(false, Ordering::SeqCst);
    });
}

#[cfg(target_os = "macos")]
pub fn request_api_key_dialog() {
    if DIALOG_OPEN
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }

    std::thread::spawn(move || {
        let key = show_macos_api_key_dialog();
        if let Some(k) = key {
            if !k.is_empty() {
                crate::secret::save_key(&k);
            }
        }
        DIALOG_OPEN.store(false, Ordering::SeqCst);
    });
}

#[cfg(target_os = "macos")]
fn show_macos_api_key_dialog() -> Option<String> {
    use std::process::Command;

    let script = r#"
        set dialogResult to display dialog "请将 API Key 粘贴到输入框中：" default answer "" with title "输入 API Key" buttons {"取消", "确定"} default button 2
        if button returned of dialogResult is "确定" then
            return text returned of dialogResult
        else
            return ""
        end if
    "#;

    let output = Command::new("osascript")
        .arg("-e")
        .arg(script)
        .output()
        .ok()?;

    if output.status.success() {
        let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if text.is_empty() {
            None
        } else {
            Some(text)
        }
    } else {
        None
    }
}
