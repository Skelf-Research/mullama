#!/bin/bash
# bump-version.sh — Atomically update all version locations in the Mullama project
# Usage: ./scripts/bump-version.sh <new_version>
# Example: ./scripts/bump-version.sh 0.3.0

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <new_version>"
    echo "Example: $0 0.3.0"
    exit 1
fi

NEW_VERSION="$1"

# Validate version format (semver)
if ! echo "$NEW_VERSION" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    echo "Error: Version must be in semver format (e.g., 0.3.0)"
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$NEW_VERSION"

echo "Bumping all versions to $NEW_VERSION..."

# 1. Main Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" Cargo.toml
echo "  Updated Cargo.toml"

# 2. FFI binding Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" bindings/ffi/Cargo.toml
echo "  Updated bindings/ffi/Cargo.toml"

# 3. FFI version constants
sed -i.bak \
    -e "s/MULLAMA_VERSION_MAJOR: u32 = [0-9]*/MULLAMA_VERSION_MAJOR: u32 = $MAJOR/" \
    -e "s/MULLAMA_VERSION_MINOR: u32 = [0-9]*/MULLAMA_VERSION_MINOR: u32 = $MINOR/" \
    -e "s/MULLAMA_VERSION_PATCH: u32 = [0-9]*/MULLAMA_VERSION_PATCH: u32 = $PATCH/" \
    bindings/ffi/src/lib.rs
echo "  Updated bindings/ffi/src/lib.rs"

# 4. Python binding Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" bindings/python/Cargo.toml
echo "  Updated bindings/python/Cargo.toml"

# 5. Node.js binding Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"$NEW_VERSION\"/" bindings/node/Cargo.toml
echo "  Updated bindings/node/Cargo.toml"

# 6. Node.js package.json
sed -i.bak "s/\"version\": \".*\"/\"version\": \"$NEW_VERSION\"/" bindings/node/package.json
echo "  Updated bindings/node/package.json"

# Clean up backup files
find . -name "*.bak" -delete

# Verify build
echo ""
echo "Running cargo check..."
cargo check --no-default-features
echo "  cargo check passed!"

echo ""
echo "All versions bumped to $NEW_VERSION"
echo "Run 'git diff' to review changes, then commit."
