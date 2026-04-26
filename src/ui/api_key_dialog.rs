use std::sync::atomic::{AtomicBool, Ordering};

static DIALOG_OPEN: AtomicBool = AtomicBool::new(false);

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
