#!/bin/bash
set -euo pipefail

#
# make_dmg.sh
#
# Creates a macOS DMG installer containing:
#   - palinstrophy.app  (from dist/)
#   - Applications symlink
#   - README.txt
#
# Usage:
#   1. Build the app:
#        uv run pyinstaller palinstrophy.spec
#   2. Create DMG:
#        ./make_dmg.sh
#

APP_NAME="palinstrophy"
APP_BUNDLE="dist/${APP_NAME}.app"
DMG_DIR="dmg_root"
DMG_NAME="${APP_NAME}.dmg"
VOLNAME="palinstrophy"

# ----------------------------------------------------------
# 0. Safety checks
# ----------------------------------------------------------
if [ ! -d "${APP_BUNDLE}" ]; then
    echo "ERROR: ${APP_BUNDLE} does not exist."
    echo "Run:  uv run pyinstaller palinstrophy.spec"
    exit 1
fi

echo "==> Creating DMG for ${APP_BUNDLE}"

# ----------------------------------------------------------
# 1. Prepare DMG root folder
# ----------------------------------------------------------
rm -rf "${DMG_DIR}"
mkdir -p "${DMG_DIR}"

# Copy the .app bundle
cp -R "${APP_BUNDLE}" "${DMG_DIR}/"

# Add the standard Applications shortcut
ln -s /Applications "${DMG_DIR}/Applications" || true

# Optional README for DMG users
cat > "${DMG_DIR}/README.txt" <<EOF
palinstrophy – 2D DNS simulation viewer

Installation:
  1. Drag palinstrophy.app to the /Applications folder
  2. $ xattr -dr com.apple.quarantine /Applications/palinstrophy.app
  3. Start the app normally from Applications
EOF

# ----------------------------------------------------------
# 2. Remove old DMG
# ----------------------------------------------------------
rm -f "${DMG_NAME}"

# ----------------------------------------------------------
# 3. Create the DMG image
#    - UDZO = compressed read-only disk image
# ----------------------------------------------------------
hdiutil create \
  -volname "${VOLNAME}" \
  -srcfolder "${DMG_DIR}" \
  -ov \
  -format UDZO \
  "${DMG_NAME}"

rm -rf dmg_root
echo "==> DMG created successfully:"
echo "    ${DMG_NAME}"