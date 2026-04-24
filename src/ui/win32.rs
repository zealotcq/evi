use std::sync::atomic::{AtomicBool, Ordering};
use windows::core::*;
use windows::Win32::Foundation::*;
use windows::Win32::UI::Shell::*;
use windows::Win32::UI::WindowsAndMessaging::*;

const WM_TRAYICON: u32 = WM_USER + 1;
const ID_TIMER_HIDE: usize = 2001;
const ID_TRAY_SHOW: usize = 1001;
const ID_TRAY_EXIT: usize = 1002;

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
            if wparam.0 as usize == ID_TIMER_HIDE {
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
