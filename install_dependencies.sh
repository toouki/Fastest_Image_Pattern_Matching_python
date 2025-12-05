#!/bin/bash

# Fastest Image Pattern Matching - Dependency Installer
# Supports Linux, macOS, and Windows (via vcpkg)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo "🚀 Fastest Image Pattern Matching - Dependency Installer"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

OS=$(detect_os)
echo "Detected OS: $OS"

# Function to install on Linux
install_linux() {
    echo "📦 Installing dependencies on Linux..."
    
    # Check for package manager
    if command -v apt-get &> /dev/null; then
        echo "Using apt-get..."
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            cmake \
            git \
            pkg-config \
            libopencv-dev \
            libopencv-contrib-dev \
            libgtk2.0-dev \
            libavcodec-dev \
            libavformat-dev \
            libswscale-dev
    elif command -v yum &> /dev/null; then
        echo "Using yum..."
        sudo yum groupinstall -y "Development Tools"
        sudo yum install -y \
            cmake \
            git \
            opencv-devel \
            opencv-contrib-devel \
            gtk2-devel \
            ffmpeg-devel
    elif command -v pacman &> /dev/null; then
        echo "Using pacman..."
        sudo pacman -S --needed \
            base-devel \
            cmake \
            git \
            opencv \
            gtk2 \
            ffmpeg
    else
        echo "❌ No supported package manager found"
        exit 1
    fi
}

# Function to install on macOS
install_macos() {
    echo "📦 Installing dependencies on macOS..."
    
    if command -v brew &> /dev/null; then
        echo "Using Homebrew..."
        brew install cmake opencv pkg-config
    else
        echo "❌ Homebrew not found. Please install Homebrew first:"
        echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
}

# Function to install on Windows
install_windows() {
    echo "📦 Setting up dependencies on Windows..."
    
    # Check for vcpkg
    if [[ ! -d "$PROJECT_ROOT/vcpkg" ]]; then
        echo "Cloning vcpkg..."
        git clone https://github.com/Microsoft/vcpkg.git "$PROJECT_ROOT/vcpkg"
        
        echo "Bootstrapping vcpkg..."
        cd "$PROJECT_ROOT/vcpkg"
        ./bootstrap-vcpkg.bat
        cd "$PROJECT_ROOT"
    fi
    
    echo "Installing OpenCV via vcpkg..."
    "$PROJECT_ROOT/vcpkg/vcpkg.exe" install opencv4[contrib]:x64-windows
    
    # Set environment variables
    echo "Setting VCPKG_ROOT environment variable..."
    export VCPKG_ROOT="$PROJECT_ROOT/vcpkg"
    echo "export VCPKG_ROOT=\"$PROJECT_ROOT/vcpkg\"" >> ~/.bashrc
    
    echo "Windows dependencies installed via vcpkg!"
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying OpenCV installation..."
    
    # Create a temporary test
    cat > "$BUILD_DIR/test_opencv.cpp" << 'EOF'
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    std::cout << "OpenCV Version: " << CV_VERSION << std::endl;
    cv::Mat test = cv::Mat::zeros(100, 100, CV_8UC3);
    std::cout << "OpenCV is working correctly!" << std::endl;
    return 0;
}
EOF

    cd "$BUILD_DIR"
    
    # Try to compile and run
    if command -v g++ &> /dev/null; then
        g++ -std=c++11 test_opencv.cpp $(pkg-config --cflags --libs opencv4) -o test_opencv 2>/dev/null || \
        g++ -std=c++11 test_opencv.cpp $(pkg-config --cflags --libs opencv) -o test_opencv 2>/dev/null || {
            echo "❌ Could not compile OpenCV test"
            return 1
        }
        
        if ./test_opencv; then
            echo "✅ OpenCV verification successful!"
            rm -f test_opencv test_opencv.cpp
            return 0
        fi
    fi
    
    echo "❌ OpenCV verification failed"
    return 1
}

# Main installation logic
main() {
    # Create build directory
    mkdir -p "$BUILD_DIR"
    
    case $OS in
        "linux")
            install_linux
            ;;
        "macos")
            install_macos
            ;;
        "windows")
            install_windows
            ;;
        *)
            echo "❌ Unsupported OS: $OS"
            exit 1
            ;;
    esac
    
    # Verify installation
    verify_installation
    
    echo ""
    echo "🎉 Installation completed!"
    echo ""
    echo "Next steps:"
    echo "1. cd $PROJECT_ROOT"
    echo "2. mkdir -p build && cd build"
    echo "3. cmake .."
    echo "4. cmake --build . --config Release"
    echo ""
}

# Run main function
main "$@"