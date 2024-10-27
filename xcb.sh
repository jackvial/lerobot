#!/bin/bash

# Script to diagnose and fix Qt XCB display issues
echo "Diagnosing Qt XCB display issues..."

# Check if DISPLAY is set
if [ -z "$DISPLAY" ]; then
    echo "ERROR: DISPLAY environment variable is not set"
    echo "Setting DISPLAY to :0"
    export DISPLAY=:0
fi

# Check for required packages
echo "Checking for required packages..."
REQUIRED_PACKAGES=(
    "libqt5x11extras5"
    "libxcb-xinerama0"
    "libxcb-icccm4"
    "libxcb-image0"
    "libxcb-keysyms1"
    "libxcb-randr0"
    "libxcb-render-util0"
    "libxcb-xkb1"
)

MISSING_PACKAGES=()
for package in "${REQUIRED_PACKAGES[@]}"; do
    if ! dpkg -l | grep -q "^ii.*$package"; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; theny
    echo "Missing required packages. Installing:"
    echo "${MISSING_PACKAGES[@]}"
    sudo apt-get update
    sudo apt-get install -y "${MISSING_PACKAGES[@]}"
fi

# Check Python virtual environment
if [ -d "venv" ]; then
    echo "Checking Python virtual environment..."
    source venv/bin/activate
    
    # Reinstall PyQt5 and opencv-python
    echo "Reinstalling PyQt5 and opencv-python..."
    pip uninstall -y PyQt5 opencv-python
    pip install PyQt5 opencv-python-headless
fi

# Check if running in Docker/container
if [ -f "/.dockerenv" ]; then
    echo "Running in Docker container. Ensuring X11 forwarding is properly set up..."
    xhost +local:root || echo "Warning: Could not set xhost permissions"
fi

# Set QT platform plugin path
export QT_DEBUG_PLUGINS=1
export QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins

echo "Fix attempt complete. Try running your application again."
echo "If the issue persists, try running with:"
echo "QT_DEBUG_PLUGINS=1 python your_script.py"
echo "to get more detailed debugging information."
