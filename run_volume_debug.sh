#!/bin/bash
set -e
cd "$(dirname "$0")"
cargo build --release --bin volume_debug
echo "启动音量门控调试工具..."
exec ./target/release/volume_debug
