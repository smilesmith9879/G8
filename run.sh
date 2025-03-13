#!/bin/bash

# AI Smart Four-Wheel Drive Car Project Runner
# This script sets up and runs the project

echo "Starting AI Smart Four-Wheel Drive Car..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "pip3 is not installed. Please install pip3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create required directories
echo "Creating required directories..."
mkdir -p static/css
mkdir -p static/js
mkdir -p templates
mkdir -p slam

# Check if simulation mode is requested
SIMULATION_FLAG=""
if [ "$1" == "--simulation" ]; then
    SIMULATION_FLAG="--simulation"
    echo "Running in simulation mode (without hardware)"
else
    echo "Running in normal mode (with hardware if available)"
fi

# Check if running on Raspberry Pi
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "raspbian" || "$ID_LIKE" == *"debian"* ]]; then
        echo "Running on Raspberry Pi. Checking GPIO access..."
        # Check if user has GPIO access
        if ! groups | grep -q "gpio"; then
            echo "Warning: Current user may not have GPIO access. You might need to run with sudo."
            echo "Alternatively, you can run with --simulation flag to use simulation mode."
        fi
    fi
fi

# Run the application
echo "Starting the application..."
python app.py $SIMULATION_FLAG

# Deactivate virtual environment on exit
deactivate 