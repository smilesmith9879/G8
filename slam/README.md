# RTAB-Map SLAM Integration

This module provides integration with RTAB-Map for real-time 3D mapping using a Raspberry Pi 5 with a camera and MPU6050 IMU.

## Overview

The SLAM system uses:
- RTAB-Map for visual SLAM
- ROS 2 for communication
- MPU6050 for IMU data (optional)
- WebSockets for real-time data streaming
- Three.js for 3D visualization in the browser

## Prerequisites

1. Raspberry Pi 5 with Raspberry Pi OS
2. USB or Raspberry Pi Camera
3. MPU6050 IMU (optional but recommended)
4. ROS 2 Humble installed on Raspberry Pi

## Installation

### 1. Install ROS 2 Humble

Follow the official ROS 2 installation instructions for Raspberry Pi OS:
https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html

### 2. Install RTAB-Map and dependencies

```bash
sudo apt update
sudo apt install -y ros-humble-rtabmap ros-humble-rtabmap-ros
sudo apt install -y ros-humble-cv-bridge ros-humble-image-transport
```

### 3. Install Python dependencies

```bash
pip install mpu6050-raspberrypi
```

### 4. Configure MPU6050 (if using)

Connect the MPU6050 to the Raspberry Pi's I2C pins:
- VCC to 3.3V
- GND to GND
- SCL to SCL (GPIO 3)
- SDA to SDA (GPIO 2)

Enable I2C in Raspberry Pi configuration:
```bash
sudo raspi-config
# Navigate to Interface Options > I2C > Enable
```

## Usage

### 1. Start the web interface

```bash
./run.sh
```

### 2. Start SLAM

In the web interface:
1. Click "Start SLAM" button
2. The 2D map will be displayed by default
3. Click "Toggle 3D View" to switch to 3D visualization

### 3. Control the robot

Use the joystick controls to move the robot around and build the map.

### 4. MPU6050 Calibration

The system automatically calibrates the MPU6050 sensor on startup to remove initial bias:

1. The robot should be stationary during startup for accurate calibration
2. Calibration collects 50 samples to calculate average bias values
3. Bias is removed from all subsequent readings:
   - X and Y accelerometer axes are calibrated to 0g
   - Z accelerometer axis is calibrated to 1g (gravity)
   - All gyroscope axes are calibrated to 0 degrees/second

Example calibration values:
```
Accel (g) - X: -0.003, Y: -0.013, Z: 0.074
Gyro (°/s) - X: -0.152, Y: 0.126, Z: -0.077
```

These calibrated values significantly improve SLAM performance by providing more accurate motion tracking.

## Architecture

### Components

1. **ROS2RTABMapBridge**: Connects to ROS 2 and subscribes to RTAB-Map topics
2. **SLAMWrapper**: Provides a high-level interface to the SLAM system
3. **Web Interface**: Displays the 2D/3D map and provides controls

### Data Flow

1. Camera captures frames
2. Frames are processed by RTAB-Map via ROS 2
3. MPU6050 provides IMU data for improved mapping
4. Map data is sent to the web interface via WebSockets
5. Three.js renders the 3D map in the browser

## Troubleshooting

### RTAB-Map not starting

Check ROS 2 installation:
```bash
ros2 topic list
```

Verify RTAB-Map launch file:
```bash
ros2 launch rtabmap_launch rtabmap.launch.py --show-args
```

### MPU6050 not detected

Check I2C connection:
```bash
sudo i2cdetect -y 1
```

The MPU6050 should appear at address 0x68 or 0x69.

### Poor mapping quality

- Ensure good lighting conditions
- Move the robot slowly and smoothly
- Add more visual features to the environment

## License

This project is licensed under the MIT License - see the LICENSE file for details. 