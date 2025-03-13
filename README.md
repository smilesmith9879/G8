# AI Smart Four-Wheel Drive Car

A Raspberry Pi 5 based smart car project with real-time mapping, web control, and autonomous navigation capabilities.

## Features

- **Web Control Interface**: Control the car and camera gimbal remotely using virtual joysticks
- **Real-time Video Streaming**: Low-latency HD video transmission using WebRTC
- **Four-Motor Control**: Precise movement control using the LOBOROBOT library
- **AI Autonomous Navigation**: Real-time environment mapping and path planning with ORB-SLAM3
- **Real-time Map Construction**: 2D/3D environment mapping
- **Camera Gimbal Control**: Horizontal and vertical control for camera orientation

## Hardware Requirements

- Raspberry Pi 5
- Four-Wheel Drive Chassis
- USB Camera (320x240, JPEG, 15FPS)
- MG996R Servo Gimbal + PCA9685 PWM Controller
- MPU6050 IMU Sensor (optional)

## Software Requirements

- Ubuntu Server 24.04 LTS
- Python 3.9+
- Flask for web server
- WebSocket for real-time communication
- ORB-SLAM3 for mapping and navigation

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ai-smart-car.git
   cd ai-smart-car
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the web server:
   ```
   python app.py
   ```

4. Access the web interface at `http://<raspberry-pi-ip>:5000`

## Project Structure

- `app.py`: Main Flask application
- `LOBOROBOT.py`: Robot control library
- `static/`: Static files for web interface
- `templates/`: HTML templates
- `slam/`: SLAM implementation files

## Usage

1. Connect to the car's WiFi network or ensure it's on your local network
2. Open the web interface in a browser
3. Use the left joystick to control car movement
4. Use the right joystick to control camera gimbal
5. View real-time video and map on the interface

## License

MIT
