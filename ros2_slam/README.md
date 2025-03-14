# ROS 2 Distributed SLAM System

This package implements a distributed SLAM (Simultaneous Localization and Mapping) system using ROS 2, designed for a Raspberry Pi 5 + PC architecture. The system enables real-time mapping and localization by offloading computationally intensive SLAM processing to a more powerful PC.

## Architecture

The system is divided into two main components:

### 1. Raspberry Pi 5 (Edge Device)
- **Responsibilities**:
  - Capturing images from a USB camera
  - Collecting IMU data from an MPU6050 sensor
  - Preprocessing data (image compression, IMU filtering)
  - Transmitting processed data to the PC over the network via ROS 2

### 2. PC (Computation Server)
- **Responsibilities**:
  - Running the full ORB-SLAM3 algorithm
  - Processing received camera and IMU data
  - Generating pose and map information
  - Publishing results back to the network

## Prerequisites

### Raspberry Pi 5:
- ROS 2 (Humble or later)
- Python 3.8+
- OpenCV
- smbus2 (for MPU6050)
- RPi.GPIO
- cv_bridge

### PC:
- ROS 2 (Humble or later)
- Python 3.8+
- OpenCV
- NumPy
- ORB-SLAM3 (optional)

## Installation

1. **Clone the repository**:
   ```bash
   mkdir -p ~/ros2_ws/src
   cd ~/ros2_ws/src
   git clone <repository-url>
   ```

2. **Install dependencies**:
   ```bash
   # On both Raspberry Pi and PC
   sudo apt update
   sudo apt install -y python3-pip python3-opencv
   pip3 install numpy

   # Additional for Raspberry Pi
   sudo apt install -y python3-smbus python3-rpi.gpio
   pip3 install smbus2
   ```

3. **Build the package**:
   ```bash
   cd ~/ros2_ws
   colcon build --packages-select ros2_slam
   source install/setup.bash
   ```

## Usage

### Network Setup
1. Ensure both the Raspberry Pi and PC are on the same network
2. Set the ROS domain ID to be the same on both devices:
   ```bash
   # On both devices
   export ROS_DOMAIN_ID=<same-number>
   ```

### Running on Raspberry Pi 5
```bash
# Source your ROS 2 workspace
source ~/ros2_ws/install/setup.bash

# Launch the Raspberry Pi node
ros2 launch ros2_slam raspberry_node.py camera_id:=0 compression_quality:=80
```

### Running on PC
```bash
# Source your ROS 2 workspace
source ~/ros2_ws/install/setup.bash

# Launch the PC node
ros2 launch ros2_slam pc_node.py slam_config_file:=/path/to/orbslam3/config.yaml
```

### Configuration Parameters

#### Raspberry Pi Node:
- `camera_id` (default: 0): Camera device ID
- `camera_width` (default: 640): Camera frame width
- `camera_height` (default: 480): Camera frame height
- `camera_fps` (default: 30): Camera frames per second
- `imu_enabled` (default: true): Enable IMU data collection
- `compression_quality` (default: 80): JPEG compression quality (0-100)
- `image_publish_rate` (default: 15.0): Image publishing rate in Hz
- `imu_publish_rate` (default: 100.0): IMU data publishing rate in Hz

#### PC Node:
- `slam_config_file` (default: ""): Path to ORB-SLAM3 configuration file
- `map_publish_rate` (default: 1.0): Map publishing rate in Hz
- `map_resolution` (default: 0.05): Map resolution in meters per pixel
- `map_size` (default: 10.0): Map size in meters

## ROS 2 Topics

### Published by Raspberry Pi Node:
- `/camera/image` (sensor_msgs/Image): Camera images
- `/imu/data` (sensor_msgs/Imu): IMU data

### Published by PC Node:
- `/slam/pose` (geometry_msgs/PoseStamped): Current camera pose
- `/slam/map` (nav_msgs/OccupancyGrid): 2D occupancy grid map
- `/tf` (tf2_msgs/TFMessage): Transform from map to camera frame

## Integration with ORB-SLAM3

This package is designed to work with ORB-SLAM3. To integrate with ORB-SLAM3:

1. Install ORB-SLAM3 following the official documentation
2. Create Python bindings for ORB-SLAM3 or implement IPC communication
3. Modify the `slam_wrapper.py` file to use the actual ORB-SLAM3 implementation

## Simulation Mode

The package includes a simulation mode that generates random poses and map points when ORB-SLAM3 is not available. This is useful for testing the ROS 2 communication infrastructure without requiring the full SLAM implementation.

## Troubleshooting

- **Camera not found**: Check if the camera ID is correct and the camera is properly connected.
- **IMU not detected**: Verify the I2C connection and address of the MPU6050.
- **Communication issues**: Ensure both devices are on the same network and using the same ROS_DOMAIN_ID.
- **High latency**: Reduce the image resolution or compression quality to decrease network bandwidth usage.

## License

This package is distributed under the MIT license. See the LICENSE file for details. 