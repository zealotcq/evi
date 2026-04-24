//! Transparent overlay window for recording indicator.
//!
//! Uses UpdateLayeredWindow for per-pixel alpha blending (anti-aliased edges).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, RECT, SIZE, WPARAM};
use windows::Win32::Graphics::Gdi::{
    BeginPaint, CreateCompatibleDC, CreateDIBSection, CreatePen, CreateSolidBrush, DeleteDC,
    DeleteObject, EndPaint, GetDC, ReleaseDC, SelectObject, SetBkMode, SetDIBitsToDevice,
    SetTextColor, TextOutW, BITMAPINFO, BITMAPINFOHEADER, BI_RGB, BLENDFUNCTION, DIB_RGB_COLORS,
    PAINTSTRUCT, PS_SOLID, TRANSPARENT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::*;

const OVERLAY_WIDTH: i32 = 80;
const OVERLAY_HEIGHT: i32 = 80;
const BOTTOM_MARGIN: i32 = 200;

struct OverlayBitmap {
    pixels: Vec<u32>,
    width: i32,
    height: i32,
}

unsafe fn load_png_rgba(filename: &str) -> Option<OverlayBitmap> {
    let exe_path = std::env::current_exe().ok()?;
    let dir = exe_path.parent()?;
    let path = dir.join(filename);
    if !path.exists() {
        log::warn!("overlay: {} not found at {:?}", filename, path);
        return None;
    }
    let img = image::open(&path).ok()?;
    let rgba = img.to_rgba8();
    let (w, h) = rgba.dimensions();

    let mut pixels = vec![0u32; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let p = rgba.get_pixel(x, y);
            let a = p[3] as u32;
            let r = (p[0] as u32 * a + 127) / 255;
            let g = (p[1] as u32 * a + 127) / 255;
            let b = (p[2] as u32 * a + 127) / 255;
            pixels[(y * w + x) as usize] = (a << 24) | (r << 16) | (g << 8) | b;
        }
    }

    log::debug!("overlay: loaded {} ({}x{})", filename, w, h);

    Some(OverlayBitmap {
        pixels,
        width: w as i32,
        height: h as i32,
    })
}

pub struct Overlay {
    visible: AtomicBool,
    thread_handle: Option<std::thread::JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    show_tx: crossbeam_channel::Sender<bool>,
}

impl Default for Overlay {
    fn default() -> Self {
        Self::new()
    }
}

impl Overlay {
    pub fn new() -> Self {
        let (show_tx, show_rx) = crossbeam_channel::unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_clone = shutdown.clone();

        let handle = std::thread::Builder::new()
            .name("overlay".into())
            .spawn(move || {
                overlay_window_loop(show_rx, shutdown_clone);
            })
            .expect("failed to spawn overlay thread");

        Self {
            visible: AtomicBool::new(false),
            thread_handle: Some(handle),
            shutdown,
            show_tx,
        }
    }

    pub fn show(&self) {
        self.visible.store(true, Ordering::SeqCst);
        let _ = self.show_tx.send(true);
    }

    pub fn hide(&self) {
        self.visible.store(false, Ordering::SeqCst);
        let _ = self.show_tx.send(false);
    }
}

impl Drop for Overlay {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        let _ = self.show_tx.send(false);
        if let Some(h) = self.thread_handle.take() {
            let _ = h.join();
        }
    }
}

unsafe fn update_window_with_bitmap(hwnd: HWND, bmp: &OverlayBitmap) {
    let screen_dc = GetDC(None);
    let mem_dc = CreateCompatibleDC(screen_dc);

    let bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: bmp.width,
            biHeight: -bmp.height,
            biPlanes: 1,
            biBitCount: 32,
            biCompression: BI_RGB.0,
            ..Default::default()
        },
        ..Default::default()
    };

    let hbitmap = CreateDIBSection(
        screen_dc,
        &bmi,
        DIB_RGB_COLORS,
        std::ptr::null_mut(),
        None,
        0,
    )
    .unwrap_or_default();

    if hbitmap.is_invalid() {
        DeleteDC(mem_dc);
        ReleaseDC(None, screen_dc);
        return;
    }

    let old_bmp = SelectObject(mem_dc, hbitmap);

    let _ = SetDIBitsToDevice(
        mem_dc,
        0,
        0,
        bmp.width as u32,
        bmp.height as u32,
        0,
        0,
        0,
        bmp.height as u32,
        bmp.pixels.as_ptr() as *const _,
        &bmi,
        DIB_RGB_COLORS,
    );

    let size = SIZE {
        cx: bmp.width,
        cy: bmp.height,
    };
    let pt_src = POINT { x: 0, y: 0 };

    let blend = BLENDFUNCTION {
        BlendOp: 0,
        BlendFlags: 0,
        SourceConstantAlpha: 255,
        AlphaFormat: 1,
    };

    let _ = UpdateLayeredWindow(
        hwnd,
        screen_dc,
        None,
        Some(&size),
        mem_dc,
        Some(&pt_src),
        COLORREF(0),
        Some(&blend),
        UPDATE_LAYERED_WINDOW_FLAGS(2),
    );

    SelectObject(mem_dc, old_bmp);
    DeleteObject(hbitmap);
    DeleteDC(mem_dc);
    ReleaseDC(None, screen_dc);
}

unsafe fn draw_fallback(hwnd: HWND) {
    let hdc = GetDC(hwnd);
    let rect = {
        let mut r = RECT::default();
        let _ = GetClientRect(hwnd, &mut r);
        r
    };

    let bg_brush = CreateSolidBrush(COLORREF(0x000000));
    windows::Win32::Graphics::Gdi::FillRect(hdc, &rect, bg_brush);
    DeleteObject(bg_brush);

    let cx = rect.right / 2;
    let cy = rect.bottom / 2;
    let radius = 30;

    let brush = CreateSolidBrush(COLORREF(0x003333CC));
    let pen = CreatePen(PS_SOLID, 2, COLORREF(0x00FFFFFF));
    SelectObject(hdc, brush);
    SelectObject(hdc, pen);
    windows::Win32::Graphics::Gdi::Ellipse(hdc, cx - radius, cy - radius, cx + radius, cy + radius);
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, COLORREF(0x00FFFFFF));
    let text: Vec<u16> = "REC".encode_utf16().collect();
    TextOutW(hdc, cx - 10, cy - 8, &text);
    DeleteObject(brush);
    DeleteObject(pen);

    ReleaseDC(hwnd, hdc);
}

fn overlay_window_loop(show_rx: crossbeam_channel::Receiver<bool>, shutdown: Arc<AtomicBool>) {
    unsafe {
        let bitmap = load_png_rgba("evi_recording.png");

        let instance = GetModuleHandleW(windows::core::PCWSTR::null())
            .unwrap_or_default()
            .into();

        let class_name = windows::core::w!("VIOverlayClass");
        let wc = WNDCLASSW {
            hInstance: instance,
            lpszClassName: class_name,
            lpfnWndProc: Some(overlay_wndproc),
            hbrBackground: CreateSolidBrush(COLORREF(0x000000)),
            hCursor: LoadCursorW(None, IDC_ARROW).unwrap_or_default(),
            ..Default::default()
        };

        RegisterClassW(&wc);

        let screen_w = GetSystemMetrics(SM_CXSCREEN);
        let screen_h = GetSystemMetrics(SM_CYSCREEN);
        let x = (screen_w - OVERLAY_WIDTH) / 2;
        let y = screen_h - BOTTOM_MARGIN - OVERLAY_HEIGHT;

        let hwnd = CreateWindowExW(
            WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_NOACTIVATE,
            class_name,
            windows::core::w!("VI"),
            WS_POPUP,
            x,
            y,
            OVERLAY_WIDTH,
            OVERLAY_HEIGHT,
            None,
            None,
            instance,
            None,
        );

        if hwnd == HWND::default() {
            log::error!("Failed to create overlay window");
            return;
        }

        let mut shown = false;

        while !shutdown.load(Ordering::SeqCst) {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, hwnd, 0, 0, PM_REMOVE).as_bool() {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }

            while let Ok(show) = show_rx.try_recv() {
                if show && !shown {
                    shown = true;
                    ShowWindow(hwnd, SW_SHOWNOACTIVATE);
                    if let Some(bmp) = &bitmap {
                        update_window_with_bitmap(hwnd, bmp);
                    } else {
                        draw_fallback(hwnd);
                    }
                } else if !show && shown {
                    shown = false;
                    ShowWindow(hwnd, SW_HIDE);
                }
            }

            std::thread::sleep(std::time::Duration::from_millis(30));
        }

        let _ = DestroyWindow(hwnd);
    }
}

unsafe extern "system" fn overlay_wndproc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    match msg {
        WM_PAINT => {
            let mut ps = PAINTSTRUCT::default();
            let hdc = BeginPaint(hwnd, &mut ps);
            EndPaint(hwnd, &ps);
            let _ = hdc;
            LRESULT(0)
        }
        WM_ERASEBKGND => LRESULT(1),
        _ => DefWindowProcW(hwnd, msg, wparam, lparam),
    }
}
