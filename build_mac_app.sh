#!/bin/bash
set -e
cd "$(dirname "$0")"

ORT_VERSION="1.24.2"
ORT_DIR="ort-dylib/onnxruntime-osx-arm64-${ORT_VERSION}"

if [ ! -d "$ORT_DIR" ]; then
    echo "Downloading ONNX Runtime for ARM64..."
    mkdir -p ort-dylib
    curl -sL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-arm64-${ORT_VERSION}.tgz" | tar xz -C ort-dylib
fi

cargo build --release --features "mac" --target aarch64-apple-darwin 2>&1 | tee build.log
