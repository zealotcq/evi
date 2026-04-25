#!/bin/bash
set -e
cd "$(dirname "$0")"
cargo build --release 2>&1 | tee build.log