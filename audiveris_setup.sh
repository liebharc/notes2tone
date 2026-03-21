#!/bin/bash
set -euo pipefail

APP_NAME="Audiveris"
VERSION="5.9.0"
DEB_NAME="Audiveris-5.9.0-ubuntu24.04-x86_64.deb"
URL="https://github.com/Audiveris/audiveris/releases/download/${VERSION}/${DEB_NAME}"

INSTALL_DIR="$HOME/Audiveris"
BIN_PATH="$INSTALL_DIR/opt/audiveris/bin/Audiveris"

echo "Installing $APP_NAME $VERSION..."

# Create install directory
mkdir -p "$INSTALL_DIR"

# Download if not already present
if [ ! -f "$DEB_NAME" ]; then
  echo "Downloading package..."
  if command -v wget >/dev/null; then
    wget "$URL"
  else
    curl -LO "$URL"
  fi
else
  echo "Package already downloaded."
fi

# Extract deb
echo "Extracting package..."
dpkg-deb -x "$DEB_NAME" "$INSTALL_DIR"

# Verify binary
if [ ! -x "$BIN_PATH" ]; then
  echo "Error: Audiveris binary not found!"
  exit 1
fi

echo "Installed to: $INSTALL_DIR"
echo
echo "To run Audiveris:"
echo "$BIN_PATH"
echo

echo "Installation done!"
