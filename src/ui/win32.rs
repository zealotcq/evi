use std::sync::atomic::{AtomicBool, Ordering};
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::UI::Shell::*;
use windows::Win32::UI::WindowsAndMessaging::*;

const WM_TRAYICON: u32 = WM_USER + 1;
const ID_TIMER_HIDE: usize = 2001;
const ID_TRAY_SHOW: usize = 1001;
const ID_TRAY_REFINE_DEFAULT: usize = 1010;
const ID_TRAY_COZE_REFINE: usize = 1003;
const ID_TRAY_ENERGY_GATE: usize = 1005;
const ID_TRAY_GATE_DB3: usize = 1014;
const ID_TRAY_GATE_DB6: usize = 1015;
const ID_TRAY_GATE_DB12: usize = 1016;
const ID_TRAY_GATE_DB18: usize = 1017;
const ID_TRAY_PUNC: usize = 1009;
const ID_TRAY_SET_KEY: usize = 1004;
const ID_TRAY_EXIT: usize = 1002;
const ID_TRAY_EDIT_DB: usize = 1012;
const ID_TRAY_PRIVACY: usize = 1013;

static SHOULD_EXIT: AtomicBool = AtomicBool::new(false);
static TRAY_INSTALLED: AtomicBool = AtomicBool::new(false);
static TRAY_DEBUG: AtomicBool = AtomicBool::new(false);
static ORIGINAL_WNDPROC: parking_lot::Mutex<Option<isize>> = parking_lot::Mutex::new(None);

pub fn should_exit() -> bool {
    SHOULD_EXIT.load(Ordering::SeqCst)
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn setup_tray(hwnd: HWND, debug: bool) {
    if TRAY_INSTALLED.load(Ordering::SeqCst) {
        return;
    }

    TRAY_DEBUG.store(debug, Ordering::SeqCst);

    #[allow(clippy::fn_to_numeric_cast)]
    let proc_ptr: isize = subclass_wndproc
        as unsafe extern "system" fn(HWND, u32, WPARAM, LPARAM) -> LRESULT
        as isize;
    let old = SetWindowLongPtrW(hwnd, GWLP_WNDPROC, proc_ptr);
    ORIGINAL_WNDPROC.lock().replace(old);

    let icon = load_icon_from_resource();

    let mut nid: NOTIFYICONDATAW = std::mem::zeroed();
    nid.cbSize = std::mem::size_of::<NOTIFYICONDATAW>() as u32;
    nid.hWnd = hwnd;
    nid.uID = 1;
    nid.uFlags = NIF_ICON | NIF_MESSAGE | NIF_TIP;
    nid.uCallbackMessage = WM_TRAYICON;
    nid.hIcon = icon;
    let tip: Vec<u16> = "EVI 语音输入法\0".encode_utf16().collect();
    let len = tip.len().min(nid.szTip.len());
    nid.szTip[..len].copy_from_slice(&tip[..len]);

    let _ = Shell_NotifyIconW(NIM_ADD, &nid);

    if !debug {
        let _ = SetTimer(hwnd, ID_TIMER_HIDE, 3000, None);
    }

    TRAY_INSTALLED.store(true, Ordering::SeqCst);
}

unsafe fn load_icon_from_resource() -> HICON {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            let icon_path = dir.join("evi.ico");
            if icon_path.exists() {
                let path_wide: Vec<u16> = icon_path
                    .to_string_lossy()
                    .encode_utf16()
                    .chain(std::iter::once(0))
                    .collect();
                if let Ok(h) = LoadImageW(
                    None,
                    PCWSTR(path_wide.as_ptr()),
                    IMAGE_ICON,
                    GetSystemMetrics(SM_CXSMICON),
                    GetSystemMetrics(SM_CYSMICON),
                    LR_LOADFROMFILE,
                ) {
                    return HICON(h.0);
                }
            }
        }
    }
    LoadIconW(None, IDI_APPLICATION).unwrap_or_default()
}

unsafe extern "system" fn subclass_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_TIMER => {
            if wparam.0 == ID_TIMER_HIDE {
                KillTimer(hwnd, ID_TIMER_HIDE).ok();
                if !TRAY_DEBUG.load(Ordering::SeqCst) {
                    ShowWindow(hwnd, SW_HIDE);
                }
            }
            LRESULT(0)
        }
        WM_CLOSE => {
            if SHOULD_EXIT.load(Ordering::SeqCst) {
                return call_original(hwnd, msg, wparam, lparam);
            }
            ShowWindow(hwnd, SW_HIDE);
            LRESULT(0)
        }
        WM_TRAYICON => {
            let lo = (lparam.0 as u32) & 0xFFFF;
            if lo == WM_RBUTTONUP || lo == WM_CONTEXTMENU {
                show_tray_menu(hwnd);
            } else if lo == WM_LBUTTONDBLCLK {
                ShowWindow(hwnd, SW_SHOW);
                let _ = SetForegroundWindow(hwnd);
            }
            LRESULT(0)
        }
        WM_COMMAND => {
            let id = wparam.0 & 0xFFFF;
            match id {
                ID_TRAY_SHOW => {
                    ShowWindow(hwnd, SW_SHOW);
                    let _ = SetForegroundWindow(hwnd);
                }
                ID_TRAY_REFINE_DEFAULT => {
                    crate::ui::set_refine_scheme("default");
                }
                ID_TRAY_COZE_REFINE => {
                    let has_api_key = crate::secret::get_api_key().is_some();
                    if has_api_key {
                        crate::ui::set_refine_scheme("llm_remote");
                    }
                }
                ID_TRAY_ENERGY_GATE => {
                    let enabled = !crate::ui::get_energy_gate_enabled();
                    crate::ui::set_energy_gate_enabled(enabled);
                }
                ID_TRAY_GATE_DB3 => {
                    crate::ui::set_energy_gate_db(3.0);
                }
                ID_TRAY_GATE_DB6 => {
                    crate::ui::set_energy_gate_db(6.0);
                }
                ID_TRAY_GATE_DB12 => {
                    crate::ui::set_energy_gate_db(12.0);
                }
                ID_TRAY_GATE_DB18 => {
                    crate::ui::set_energy_gate_db(18.0);
                }
                ID_TRAY_PUNC => {
                    let enabled = !crate::ui::get_punc_enabled();
                    crate::ui::set_punc_enabled(enabled);
                }
                ID_TRAY_EDIT_DB => {
                    if let Ok(exe) = std::env::current_exe() {
                        let _ = std::process::Command::new(exe)
                            .arg("--refine-editor")
                            .spawn();
                    }
                }
                ID_TRAY_PRIVACY => {
                    if let Ok(exe) = std::env::current_exe() {
                        let _ = std::process::Command::new(exe)
                            .arg("--privacy-dialog")
                            .spawn();
                    }
                }
                ID_TRAY_SET_KEY => {
                    crate::ui::api_key_dialog::request_api_key_dialog();
                }
                ID_TRAY_EXIT => {
                    SHOULD_EXIT.store(true, Ordering::SeqCst);
                    ShowWindow(hwnd, SW_HIDE);
                    remove_tray_icon(hwnd);
                    let _ = DestroyWindow(hwnd);
                }
                _ => return call_original(hwnd, msg, wparam, lparam),
            }
            LRESULT(0)
        }
        WM_DESTROY => {
            remove_tray_icon(hwnd);
            call_original(hwnd, msg, wparam, lparam)
        }
        _ => call_original(hwnd, msg, wparam, lparam),
    }
}

unsafe fn call_original(hwnd: HWND, msg: u32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    let guard = ORIGINAL_WNDPROC.lock();
    if let Some(proc) = *guard {
        drop(guard);
        let wndproc: WNDPROC = std::mem::transmute::<isize, WNDPROC>(proc);
        CallWindowProcW(wndproc, hwnd, msg, wparam, lparam)
    } else {
        DefWindowProcW(hwnd, msg, wparam, lparam)
    }
}

unsafe fn show_tray_menu(hwnd: HWND) {
    let menu = match CreatePopupMenu() {
        Ok(m) => m,
        Err(_) => return,
    };

    if TRAY_DEBUG.load(Ordering::SeqCst) {
        let show_text: Vec<u16> = "显示主界面\0".encode_utf16().collect();
        let _ = AppendMenuW(menu, MF_STRING, ID_TRAY_SHOW, PCWSTR(show_text.as_ptr()));
        let _ = AppendMenuW(menu, MF_SEPARATOR, 0, PCWSTR::null());
    }

    // 润色路线子菜单
    let refine_text: Vec<u16> = "润色方案\0".encode_utf16().collect();
    let refine_sub = match CreatePopupMenu() {
        Ok(m) => m,
        Err(_) => {
            let _ = DestroyMenu(menu);
            return;
        }
    };
    let default_text: Vec<u16> = "默认润色\0".encode_utf16().collect();
    let refine_remote_text: Vec<u16> = "网络大模型润色\0".encode_utf16().collect();
    let scheme = crate::ui::get_refine_scheme();
    let has_api_key = crate::secret::get_api_key().is_some();

    let f_default = MF_STRING
        | if scheme == "default" {
            MF_CHECKED
        } else {
            MENU_ITEM_FLAGS(0)
        };
    let _ = AppendMenuW(
        refine_sub,
        f_default,
        ID_TRAY_REFINE_DEFAULT,
        PCWSTR(default_text.as_ptr()),
    );

    let f_remote = if has_api_key {
        MF_STRING
            | if scheme == "llm_remote" {
                MF_CHECKED
            } else {
                MENU_ITEM_FLAGS(0)
            }
    } else {
        MF_STRING | MF_GRAYED
    };
    let _ = AppendMenuW(
        refine_sub,
        f_remote,
        ID_TRAY_COZE_REFINE,
        PCWSTR(refine_remote_text.as_ptr()),
    );

    if TRAY_DEBUG.load(Ordering::SeqCst) {
        let _ = AppendMenuW(
            menu,
            MF_STRING | MF_POPUP,
            refine_sub.0 as usize,
            PCWSTR(refine_text.as_ptr()),
        );
    } else {
        let _ = DestroyMenu(refine_sub);
    }

    let gate_text: Vec<u16> = "音量门控\0".encode_utf16().collect();
    let gate_ptr = PCWSTR(gate_text.as_ptr());
    let gate_checked = crate::ui::get_energy_gate_enabled();
    let _ = AppendMenuW(
        menu,
        MF_STRING | if gate_checked { MF_CHECKED } else { MF_STRING },
        ID_TRAY_ENERGY_GATE,
        gate_ptr,
    );

    let db_menu = match CreatePopupMenu() {
        Ok(m) => m,
        Err(_) => {
            let _ = DestroyMenu(menu);
            return;
        }
    };
    let db_val = crate::ui::get_energy_gate_db();
    let db_opts: [(usize, &str, f64); 4] = [
        (ID_TRAY_GATE_DB3, "3dB", 3.0),
        (ID_TRAY_GATE_DB6, "6dB", 6.0),
        (ID_TRAY_GATE_DB12, "12dB", 12.0),
        (ID_TRAY_GATE_DB18, "18dB", 18.0),
    ];
    for (id, text, val) in db_opts {
        let t: Vec<u16> = format!("{}\0", text).encode_utf16().collect();
        let _ = AppendMenuW(
            db_menu,
            MF_STRING | if val == db_val { MF_CHECKED } else { MF_STRING },
            id,
            PCWSTR(t.as_ptr()),
        );
    }
    let range_text: Vec<u16> = "音量范围\0".encode_utf16().collect();
    let _ = AppendMenuW(
        menu,
        MF_STRING | MF_POPUP,
        db_menu.0 as usize,
        PCWSTR(range_text.as_ptr()),
    );

    let punc_text: Vec<u16> = "智能标点\0".encode_utf16().collect();
    let punc_ptr = PCWSTR(punc_text.as_ptr());
    let punc_checked = crate::ui::get_punc_enabled();
    let _ = AppendMenuW(
        menu,
        MF_STRING | if punc_checked { MF_CHECKED } else { MF_STRING },
        ID_TRAY_PUNC,
        punc_ptr,
    );

    if TRAY_DEBUG.load(Ordering::SeqCst) {
        let _ = AppendMenuW(menu, MF_SEPARATOR, 0, PCWSTR::null());

        let set_key_text: Vec<u16> = "设置 API Key\0".encode_utf16().collect();
        let _ = AppendMenuW(
            menu,
            MF_STRING,
            ID_TRAY_SET_KEY,
            PCWSTR(set_key_text.as_ptr()),
        );
    }

    let _ = AppendMenuW(menu, MF_SEPARATOR, 0, PCWSTR::null());

    let edit_db_text: Vec<u16> = "帮助我们改进\0".encode_utf16().collect();
    let _ = AppendMenuW(
        menu,
        MF_STRING,
        ID_TRAY_EDIT_DB,
        PCWSTR(edit_db_text.as_ptr()),
    );

    let privacy_text: Vec<u16> = "隐私说明...\0".encode_utf16().collect();
    let _ = AppendMenuW(
        menu,
        MF_STRING,
        ID_TRAY_PRIVACY,
        PCWSTR(privacy_text.as_ptr()),
    );

    let _ = AppendMenuW(menu, MF_SEPARATOR, 0, PCWSTR::null());

    let exit_text: Vec<u16> = "退出\0".encode_utf16().collect();
    let _ = AppendMenuW(menu, MF_STRING, ID_TRAY_EXIT, PCWSTR(exit_text.as_ptr()));

    let _ = SetForegroundWindow(hwnd);

    let mut pt = POINT { x: 0, y: 0 };
    let _ = GetCursorPos(&mut pt);

    let _ = TrackPopupMenu(
        menu,
        TPM_RIGHTBUTTON | TPM_NONOTIFY,
        pt.x,
        pt.y,
        0,
        hwnd,
        None,
    );

    let _ = DestroyMenu(menu);
}

unsafe fn remove_tray_icon(hwnd: HWND) {
    let mut nid: NOTIFYICONDATAW = std::mem::zeroed();
    nid.cbSize = std::mem::size_of::<NOTIFYICONDATAW>() as u32;
    nid.hWnd = hwnd;
    nid.uID = 1;
    let _ = Shell_NotifyIconW(NIM_DELETE, &nid);
}

#[allow(clippy::missing_safety_doc)]
pub unsafe fn find_main_window() -> Option<HWND> {
    let title: Vec<u16> = "EVI 语音输入法\0".encode_utf16().collect();
    let hwnd = FindWindowW(PCWSTR(std::ptr::null()), PCWSTR(title.as_ptr()));
    if hwnd.0 != 0 {
        Some(hwnd)
    } else {
        None
    }
}

pub fn show_api_key_dialog() -> Option<String> {
    let title: Vec<u16> = "输入 API Key\0".encode_utf16().collect();
    let prompt: Vec<u16> = "请将 API Key 复制到剪贴板，然后点确定\0"
        .encode_utf16()
        .collect();

    unsafe {
        use windows::Win32::UI::WindowsAndMessaging::*;
        let _ = MessageBoxW(
            None,
            windows::core::PCWSTR(prompt.as_ptr()),
            windows::core::PCWSTR(title.as_ptr()),
            MB_OKCANCEL | MB_ICONQUESTION,
        );
    }

    unsafe fn get_clipboard_text() -> Option<String> {
        use windows::Win32::Foundation::HGLOBAL;
        use windows::Win32::System::DataExchange::{
            CloseClipboard, GetClipboardData, IsClipboardFormatAvailable, OpenClipboard,
        };
        use windows::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};

        if OpenClipboard(None).is_err() {
            return None;
        }
        if IsClipboardFormatAvailable(13u32).is_err() {
            let _ = CloseClipboard();
            return None;
        }
        let handle = match GetClipboardData(13u32) {
            Ok(h) => h,
            Err(_) => {
                let _ = CloseClipboard();
                return None;
            }
        };
        let hglobal = HGLOBAL(handle.0 as *mut _);
        let ptr = GlobalLock(hglobal);
        if ptr.is_null() {
            let _ = GlobalUnlock(hglobal);
            let _ = CloseClipboard();
            return None;
        }
        let size = GlobalSize(hglobal) as usize / 2;
        let len = (0..size)
            .take_while(|&i| *(ptr as *const u16).add(i) != 0)
            .count();
        let slice = std::slice::from_raw_parts(ptr as *const u16, len);
        let text = String::from_utf16_lossy(slice);
        let _ = GlobalUnlock(hglobal);
        let _ = CloseClipboard();
        Some(text.trim().to_string())
    }

    unsafe { get_clipboard_text() }
}
