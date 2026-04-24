use crate::{TextChangeEvent, TextObserver, TextOutput, TextSnapshot};
use anyhow::Result;
use log::info;
use parking_lot::Mutex;
use std::ptr;
use windows::Win32::Foundation::{HANDLE, HGLOBAL};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard,
    SetClipboardData,
};
use windows::Win32::System::Memory::{
    GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock, GMEM_MOVEABLE,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, VIRTUAL_KEY, VK_CONTROL,
};

enum SavedClipboard {
    Text(String),
    Empty,
}

pub struct ClipboardTextOutput {
    last_injected: Mutex<String>,
}

unsafe impl Send for ClipboardTextOutput {}
unsafe impl Sync for ClipboardTextOutput {}

impl ClipboardTextOutput {
    pub fn new() -> Result<Self> {
        Ok(Self {
            last_injected: Mutex::new(String::new()),
        })
    }

    unsafe fn save_clipboard(&self) -> SavedClipboard {
        if OpenClipboard(None).is_err() {
            return SavedClipboard::Empty;
        }
        if IsClipboardFormatAvailable(13u32).is_ok() {
            if let Ok(handle) = GetClipboardData(13u32) {
                let hglobal = HGLOBAL(handle.0 as *mut _);
                let ptr = GlobalLock(hglobal);
                if !ptr.is_null() {
                    let size = GlobalSize(hglobal) as usize / 2;
                    let len = (0..size)
                        .take_while(|&i| *(ptr as *const u16).add(i) != 0)
                        .count();
                    let slice = std::slice::from_raw_parts(ptr as *const u16, len);
                    let text = String::from_utf16_lossy(slice);
                    GlobalUnlock(hglobal).ok();
                    CloseClipboard().ok();
                    return SavedClipboard::Text(text);
                }
            }
        }
        CloseClipboard().ok();
        SavedClipboard::Empty
    }

    unsafe fn restore_clipboard(&self, saved: SavedClipboard) {
        if OpenClipboard(None).is_err() {
            return;
        }
        match saved {
            SavedClipboard::Text(text) => {
                let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0u16)).collect();
                if let Ok(hmem) = GlobalAlloc(GMEM_MOVEABLE, wide.len() * 2) {
                    let ptr = GlobalLock(hmem);
                    if !ptr.is_null() {
                        ptr::copy_nonoverlapping(wide.as_ptr(), ptr as *mut u16, wide.len());
                        GlobalUnlock(hmem).ok();
                        SetClipboardData(13, HANDLE(hmem.0 as isize)).ok();
                    }
                }
            }
            SavedClipboard::Empty => {
                EmptyClipboard().ok();
            }
        }
        CloseClipboard().ok();
    }

    unsafe fn clipboard_inject(&self, text: &str) -> bool {
        let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0u16)).collect();
        if OpenClipboard(None).is_err() {
            return false;
        }
        let hmem = match GlobalAlloc(GMEM_MOVEABLE, wide.len() * 2) {
            Ok(h) => h,
            Err(_) => {
                CloseClipboard().ok();
                return false;
            }
        };
        let ptr = GlobalLock(hmem);
        if ptr.is_null() {
            CloseClipboard().ok();
            return false;
        }
        ptr::copy_nonoverlapping(wide.as_ptr(), ptr as *mut u16, wide.len());
        GlobalUnlock(hmem).ok();
        if SetClipboardData(13, HANDLE(hmem.0 as isize)).is_err() {
            CloseClipboard().ok();
            return false;
        }
        CloseClipboard().ok();
        std::thread::sleep(std::time::Duration::from_millis(30));
        self.emit_ctrl_v()
    }

    unsafe fn emit_ctrl_v(&self) -> bool {
        let inputs = [
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VK_CONTROL,
                        wScan: 0,
                        dwFlags: Default::default(),
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            },
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VIRTUAL_KEY(0x56),
                        wScan: 0,
                        dwFlags: Default::default(),
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            },
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VIRTUAL_KEY(0x56),
                        wScan: 0,
                        dwFlags: KEYEVENTF_KEYUP,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            },
            INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VK_CONTROL,
                        wScan: 0,
                        dwFlags: KEYEVENTF_KEYUP,
                        time: 0,
                        dwExtraInfo: 0,
                    },
                },
            },
        ];
        SendInput(&inputs, std::mem::size_of::<INPUT>() as i32) > 0
    }

    unsafe fn unicode_inject(&self, text: &str) -> bool {
        for ch in text.chars() {
            let inputs = [
                INPUT {
                    r#type: INPUT_KEYBOARD,
                    Anonymous: INPUT_0 {
                        ki: KEYBDINPUT {
                            wVk: VIRTUAL_KEY(0),
                            wScan: ch as u16,
                            dwFlags: windows::Win32::UI::Input::KeyboardAndMouse::KEYEVENTF_UNICODE,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                },
                INPUT {
                    r#type: INPUT_KEYBOARD,
                    Anonymous: INPUT_0 {
                        ki: KEYBDINPUT {
                            wVk: VIRTUAL_KEY(0),
                            wScan: ch as u16,
                            dwFlags: windows::Win32::UI::Input::KeyboardAndMouse::KEYEVENTF_UNICODE
                                | KEYEVENTF_KEYUP,
                            time: 0,
                            dwExtraInfo: 0,
                        },
                    },
                },
            ];
            SendInput(&inputs, std::mem::size_of::<INPUT>() as i32);
        }
        true
    }

    pub fn last_injected_text(&self) -> String {
        self.last_injected.lock().clone()
    }
}

impl TextOutput for ClipboardTextOutput {
    fn commit_text(&self, text: &str) -> Result<()> {
        let saved = unsafe { self.save_clipboard() };
        let result = unsafe {
            if self.clipboard_inject(text) {
                info!("CPTextOutput: {} chars", text.len());
                crate::text::log_event("TEXT_INJECT", &format!("{} chars", text.len()));
                Ok(())
            } else if self.unicode_inject(text) {
                info!("CPTextOutput: {} chars via fallback", text.len());
                crate::text::log_event("TEXT_INJECT_FALLBACK", &format!("{} chars", text.len()));
                Ok(())
            } else {
                anyhow::bail!("All text injection methods failed");
            }
        };
        if result.is_ok() {
            *self.last_injected.lock() = text.to_string();
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        unsafe {
            self.restore_clipboard(saved);
        }
        result
    }
    fn method_name(&self) -> &str {
        "clipboard"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

pub struct NopTextObserver {}

unsafe impl Send for NopTextObserver {}
unsafe impl Sync for NopTextObserver {}

impl NopTextObserver {
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }
}

impl TextObserver for NopTextObserver {
    fn start_monitoring(&mut self) -> Result<()> {
        Ok(())
    }
    fn stop_monitoring(&mut self) -> Result<()> {
        Ok(())
    }
    fn poll_changes(&self) -> Vec<TextChangeEvent> {
        Vec::new()
    }
    fn snapshot(&self) -> Result<TextSnapshot> {
        Ok(TextSnapshot {
            full_text: String::new(),
            cursor_position: 0,
            selection_start: 0,
            selection_end: 0,
        })
    }
}

pub fn create_platform_session() -> Result<super::PlatformTextSession> {
    let output = ClipboardTextOutput::new()?;
    let observer = NopTextObserver::new()?;
    Ok(super::PlatformTextSession::new(
        Box::new(output),
        Box::new(observer),
    ))
}
