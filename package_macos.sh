#!/bin/bash
set -e

VERSION="2.0.0"
PKG_NAME="evi-${VERSION}-macos-x86_64"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RELEASE_DIR="${SCRIPT_DIR}/target/release"
STAGING="${SCRIPT_DIR}/dist/staging"
APP_DIR="${STAGING}/EVI.app"

if [ ! -f "${RELEASE_DIR}/vi" ]; then
    echo "Error: vi binary not found. Run 'cargo build --release' first."
    exit 1
fi

echo "Building .app bundle..."
rm -rf "${STAGING}"
mkdir -p "${APP_DIR}/Contents/MacOS"
mkdir -p "${APP_DIR}/Contents/Resources"
mkdir -p "${APP_DIR}/Contents/Frameworks"

# binary
cp "${RELEASE_DIR}/vi" "${APP_DIR}/Contents/MacOS/"
chmod +x "${APP_DIR}/Contents/MacOS/vi"

# config + icon -> MacOS/ (vi looks for these relative to exe)
cp "${SCRIPT_DIR}/config.json" "${APP_DIR}/Contents/MacOS/"
cp "${SCRIPT_DIR}/evi.ico" "${APP_DIR}/Contents/MacOS/"
cp "${SCRIPT_DIR}/prefill_template.txt" "${APP_DIR}/Contents/MacOS/"
cp "${SCRIPT_DIR}/system_prompt.txt" "${APP_DIR}/Contents/MacOS/"

# ORT dylib -> MacOS/ort-dylib/ (same layout as dev)
mkdir -p "${APP_DIR}/Contents/MacOS/ort-dylib"
cp -R "${SCRIPT_DIR}/ort-dylib/onnxruntime-osx-x86_64-1.24.2" \
    "${APP_DIR}/Contents/MacOS/ort-dylib/"

# data directory
mkdir -p "${APP_DIR}/Contents/MacOS/refine_log"

# convert ico to icns for Finder
ICONSET="${STAGING}/icon.iconset"
mkdir -p "${ICONSET}"
# try to extract largest icon from .ico using sips
if sips -s format png "${SCRIPT_DIR}/evi.ico" --out "${ICONSET}/icon_256x256.png" -z 256 256 2>/dev/null; then
    cp "${ICONSET}/icon_256x256.png" "${ICONSET}/icon_128x128.png"
    sips -z 128 128 "${ICONSET}/icon_128x128.png" -o "${ICONSET}/icon_128x128.png" 2>/dev/null || true
    cp "${ICONSET}/icon_128x128.png" "${ICONSET}/icon_32x32.png"
    sips -z 32 32 "${ICONSET}/icon_32x32.png" -o "${ICONSET}/icon_32x32.png" 2>/dev/null || true
    cp "${ICONSET}/icon_32x32.png" "${ICONSET}/icon_16x16.png"
    sips -z 16 16 "${ICONSET}/icon_16x16.png" -o "${ICONSET}/icon_16x16.png" 2>/dev/null || true
    cp "${ICONSET}/icon_256x256.png" "${ICONSET}/icon_256x256@2x.png" 2>/dev/null || true
    iconutil -c icns "${ICONSET}" -o "${APP_DIR}/Contents/Resources/AppIcon.icns" 2>/dev/null || true
fi

# Info.plist
cat > "${APP_DIR}/Contents/Info.plist" << 'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>vi</string>
    <key>CFBundleIdentifier</key>
    <string>com.evi.voice-input</string>
    <key>CFBundleName</key>
    <string>EVI</string>
    <key>CFBundleDisplayName</key>
    <string>EVI 语音输入法</string>
    <key>CFBundleVersion</key>
    <string>VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleIconFile</key>
    <string>AppIcon</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>EVI needs microphone access for voice input.</string>
</dict>
</plist>
PLIST
sed -i '' "s/VERSION/${VERSION}/g" "${APP_DIR}/Contents/Info.plist"

# Applications symlink for DMG
ln -s /Applications "${STAGING}/Applications"

# create DMG
DMG_PATH="${SCRIPT_DIR}/dist/${PKG_NAME}.dmg"
rm -f "${DMG_PATH}"

echo "Creating DMG..."
rm -rf "${ICONSET}"
hdiutil create -volname "EVI" \
    -srcfolder "${STAGING}" \
    -ov -format UDZO \
    "${DMG_PATH}"

rm -rf "${STAGING}"
echo ""
echo "Done: ${DMG_PATH}"
echo "用户双击 DMG → 拖拽 EVI.app 到 Applications 即可安装。"
