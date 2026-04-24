use std::fs;
use std::path::Path;

fn main() {
    let profile = std::env::var("PROFILE").unwrap();
    let out_dir = format!("target/{}", profile);
    for name in &["config.json", "system_prompt.txt", "prefill_template.txt"] {
        let src = Path::new(name);
        if src.exists() {
            let dst = Path::new(&out_dir).join(name);
            let _ = fs::copy(src, dst);
        }
    }

    if cfg!(target_os = "windows") && Path::new("evi.ico").exists() {
        let mut res = winres::WindowsResource::new();
        res.set_icon("evi.ico");
        res.compile().expect("winres compile failed");
    }
}
