#!/bin/bash

# AI Smart Four-Wheel Drive Car Project - Simulation Mode Runner
# This script runs the project in simulation mode without requiring hardware

echo "Starting AI Smart Four-Wheel Drive Car in Simulation Mode..."

# Make the main run script executable if it's not already
chmod +x run.sh

# Call the main run script with the simulation flag
./run.sh --simulation

echo "Simulation ended." 