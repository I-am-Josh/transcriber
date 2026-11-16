#!/bin/bash
# Builds Transcriber.app, a small launcher for dictate.py in this folder.
# Usage: ./build_app.sh [where]   (default: /Applications/Transcriber.app)
#
# The app runs the code in this folder, so edits to dictate.py or words.txt
# need no rebuild. Rebuilding makes macOS forget its permissions, so only do it
# when this script changes.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
APP="${1:-/Applications/Transcriber.app}"
UV="$(command -v uv)"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Transcriber</string>
  <key>CFBundleDisplayName</key><string>Transcriber</string>
  <key>CFBundleIdentifier</key><string>local.transcriber</string>
  <key>CFBundleExecutable</key><string>Transcriber</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Transcriber listens while you hold Right Option and turns your speech into text on this Mac.</string>
</dict>
</plist>
EOF

cat > "$APP/Contents/MacOS/Transcriber" <<EOF
#!/bin/bash
cd "$HERE"
exec "$UV" run --quiet dictate.py >> "\$HOME/Library/Logs/Transcriber.log" 2>&1
EOF
chmod +x "$APP/Contents/MacOS/Transcriber"

"$UV" run --quiet --project "$HERE" "$HERE/make_icon.py" "$APP/Contents/Resources/AppIcon.icns"
codesign --force --sign - "$APP"
echo "Built $APP"
