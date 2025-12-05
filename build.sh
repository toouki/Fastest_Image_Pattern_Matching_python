#!/bin/bash

# Fastest Image Pattern Matching - Quick Build Script

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "🔨 Fastest Image Pattern Matching - Quick Build"
echo "Project: $PROJECT_ROOT"
echo ""

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check for OpenCV
echo "🔍 Checking for OpenCV..."
if ! pkg-config --exists opencv4 && ! pkg-config --exists opencv; then
    echo "❌ OpenCV not found!"
    echo "Run this first to install dependencies:"
    echo "  ./install_dependencies.sh"
    echo ""
    echo "Or install manually:"
    echo "  Linux: sudo apt install libopencv-dev"
    echo "  macOS: brew install opencv"
    exit 1
fi

echo "✅ OpenCV found!"

# Configure
echo "⚙️  Configuring project..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo "🏗️  Building project..."
make -j$(nproc)

echo ""
echo "🎉 Build completed successfully!"
echo ""
echo "Binaries created:"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "  Windows: MatchTool.exe"
else
    echo "  Library: libMatchTool.so"
    echo "  Test:    MatchTool_test"
fi
echo ""
echo "Run test: cd build && ./MatchTool_test"