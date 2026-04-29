#!/usr/bin/env bash
# Vendor D3.js for offline use in the automation web viewer.
#
# Usage:
#   cd src/python/laboneq/automation/web_viewer
#   bash vendor_d3.sh

set -euo pipefail

cd "$(dirname "$0")"

TARGET_DIR="app/static/js"
OUTPUT="$TARGET_DIR/d3.latest.min.js"

# Fetch latest version from npm registry
VERSION=$(curl -fSs https://registry.npmjs.org/d3/latest | grep -o '"version":"[^"]*"' | head -1 | cut -d'"' -f4)

echo "Latest D3.js version is: ${VERSION}"
echo "The script will download d3.min.js and prepend the D3 license to the top of the file."
echo ""
read -rp "Do you want to download D3.js v${VERSION}? [y/N] " answer

if [[ ! "$answer" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo "Downloading..."

curl -fSL -o /tmp/d3_license.txt \
    "https://raw.githubusercontent.com/d3/d3/main/LICENSE"

curl -fSL -o /tmp/d3.min.js \
    "https://cdnjs.cloudflare.com/ajax/libs/d3/${VERSION}/d3.min.js"

{
    echo "/*!"
    cat /tmp/d3_license.txt
    echo "*/"
    cat /tmp/d3.min.js
} > "$OUTPUT"

rm /tmp/d3_license.txt /tmp/d3.min.js

echo "Written to $OUTPUT ($(wc -c < "$OUTPUT") bytes)"
