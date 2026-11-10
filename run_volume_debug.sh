#!/bin/bash
set -e
cd "$(dirname "$0")"
cargo build --release --example volume_debug
echo "启动音量门控调试工具..."
exec ./target/release/examples/volume_debug
