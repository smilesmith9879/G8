#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import threading
import logging
import base64
import argparse
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='AI Smart Four-Wheel Drive Car')
parser.add_argument('--simulation', action='store_true', help='Run in simulation mode without hardware')
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'smartcar2023'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Initialize robot
robot_available = False
if not args.simulation:
    try:
        from LOBOROBOT import LOBOROBOT
        robot = LOBOROBOT()
        robot_available = True
        logger.info("Robot initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize robot: {e}")
        logger.info("Falling back to simulation mode")
else:
    logger.info("Running in simulation mode - robot hardware not initialized")

# Simulation class for robot when hardware is not available
class SimulatedRobot:
    def __init__(self):
        logger.info("Simulated robot initialized")
        
    def t_up(self, speed, t_time):
        logger.info(f"Simulated: Moving forward at speed {speed}")
        
    def t_down(self, speed, t_time):
        logger.info(f"Simulated: Moving backward at speed {speed}")
        
    def moveLeft(self, speed, t_time):
        logger.info(f"Simulated: Moving left at speed {speed}")
        
    def moveRight(self, speed, t_time):
        logger.info(f"Simulated: Moving right at speed {speed}")
        
    def turnLeft(self, speed, t_time):
        logger.info(f"Simulated: Turning left at speed {speed}")
        
    def turnRight(self, speed, t_time):
        logger.info(f"Simulated: Turning right at speed {speed}")
        
    def forward_Left(self, speed, t_time):
        logger.info(f"Simulated: Moving forward-left at speed {speed}")
        
    def forward_Right(self, speed, t_time):
        logger.info(f"Simulated: Moving forward-right at speed {speed}")
        
    def backward_Left(self, speed, t_time):
        logger.info(f"Simulated: Moving backward-left at speed {speed}")
        
    def backward_Right(self, speed, t_time):
        logger.info(f"Simulated: Moving backward-right at speed {speed}")
        
    def t_stop(self, t_time):
        logger.info("Simulated: Stopping")
        
    def set_servo_angle(self, channel, angle):
        logger.info(f"Simulated: Setting servo channel {channel} to angle {angle}°")

# Use simulated robot if hardware is not available
if not robot_available:
    robot = SimulatedRobot()
    # Mark as available since we have a simulation
    robot_available = True

# Initialize camera
camera_available = False
if not args.simulation:
    try:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        camera.set(cv2.CAP_PROP_FPS, 15)
        ret, frame = camera.read()
        if ret:
            camera_available = True
            logger.info("Camera initialized successfully")
        else:
            logger.error("Camera connected but failed to capture frame")
            logger.info("Falling back to simulation mode for camera")
    except Exception as e:
        logger.error(f"Failed to initialize camera: {e}")
        logger.info("Falling back to simulation mode for camera")
else:
    logger.info("Running in simulation mode - camera not initialized")

# MPU6050 data and thread
mpu_data = {
    'accel_x': 0.0,
    'accel_y': 0.0,
    'accel_z': 0.0,
    'gyro_x': 0.0,
    'gyro_y': 0.0,
    'gyro_z': 0.0,
    'temperature': 0.0
}
mpu_thread = None
mpu_running = False

# Function to read MPU6050 data
def read_mpu6050_data():
    global mpu_running, mpu_data
    
    logger.info("MPU6050 data thread started")
    
    while mpu_running and mpu6050_available:
        try:
            if not args.simulation:
                # 这里调用适当的MPU6050读取函数获取真实数据
                try:
                    # WHO_AM_I寄存器地址
                    WHO_AM_I = 0x75
                    # 电源管理寄存器
                    PWR_MGMT_1 = 0x6B
                    # 加速度数据寄存器起始地址
                    ACCEL_XOUT_H = 0x3B
                    # 温度数据寄存器起始地址
                    TEMP_OUT_H = 0x41
                    # 陀螺仪数据寄存器起始地址
                    GYRO_XOUT_H = 0x43
                    
                    # 确保设备唤醒
                    bus.write_byte_data(mpu_addr, PWR_MGMT_1, 0)
                    
                    # 读取加速度数据
                    accel_data = bus.read_i2c_block_data(mpu_addr, ACCEL_XOUT_H, 6)
                    accel_x = (accel_data[0] << 8 | accel_data[1]) / 16384.0  # 转换为g
                    accel_y = (accel_data[2] << 8 | accel_data[3]) / 16384.0  # 转换为g
                    accel_z = (accel_data[4] << 8 | accel_data[5]) / 16384.0  # 转换为g
                    
                    # 读取温度数据
                    temp_data = bus.read_i2c_block_data(mpu_addr, TEMP_OUT_H, 2)
                    temperature = ((temp_data[0] << 8 | temp_data[1]) / 340.0) + 36.53  # 转换为摄氏度
                    
                    # 读取陀螺仪数据
                    gyro_data = bus.read_i2c_block_data(mpu_addr, GYRO_XOUT_H, 6)
                    gyro_x = (gyro_data[0] << 8 | gyro_data[1]) / 131.0  # 转换为度/秒
                    gyro_y = (gyro_data[2] << 8 | gyro_data[3]) / 131.0  # 转换为度/秒
                    gyro_z = (gyro_data[4] << 8 | gyro_data[5]) / 131.0  # 转换为度/秒
                    
                    # 更新全局数据
                    mpu_data = {
                        'accel_x': round(accel_x, 2),
                        'accel_y': round(accel_y, 2),
                        'accel_z': round(accel_z, 2),
                        'gyro_x': round(gyro_x, 2),
                        'gyro_y': round(gyro_y, 2),
                        'gyro_z': round(gyro_z, 2),
                        'temperature': round(temperature, 2)
                    }
                except Exception as e:
                    logger.error(f"Error reading MPU6050 data: {e}")
            else:
                # 在模拟模式下生成随机数据
                mpu_data = {
                    'accel_x': round(np.random.uniform(-1, 1), 2),
                    'accel_y': round(np.random.uniform(-1, 1), 2),
                    'accel_z': round(np.random.uniform(0, 2), 2),  # Z轴通常有重力加速度
                    'gyro_x': round(np.random.uniform(-10, 10), 2),
                    'gyro_y': round(np.random.uniform(-10, 10), 2),
                    'gyro_z': round(np.random.uniform(-10, 10), 2),
                    'temperature': round(np.random.uniform(20, 30), 2)
                }
            
            # 发送数据给客户端
            socketio.emit('mpu6050_data', mpu_data)
            
            # 限制更新频率
            time.sleep(0.5)  # 2Hz更新率
        except Exception as e:
            logger.error(f"Error in MPU6050 data thread: {e}")
            time.sleep(1)
    
    logger.info("MPU6050 data thread stopped")

# Simulated camera frame generator
class SimulatedCamera:
    def __init__(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_position = (50, 120)
        self.font_scale = 0.7
        self.font_color = (255, 255, 255)
        self.line_type = 2
        cv2.putText(self.frame, 'Simulated Camera', 
                    self.text_position, self.font, self.font_scale,
                    self.font_color, self.line_type)
        logger.info("Simulated camera initialized")
        
    def read(self):
        # Create a copy of the base frame
        frame = self.frame.copy()
        
        # Add timestamp
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, timestamp, (10, 30), self.font, 0.5, (0, 255, 0), 1)
        
        # Add some movement to simulate a real camera
        noise = np.random.randint(0, 10, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        return True, frame
        
    def release(self):
        logger.info("Simulated camera released")

# Use simulated camera if hardware camera is not available
if not camera_available:
    camera = SimulatedCamera()
    # Mark as available since we have a simulation
    camera_available = True

# Import SLAM wrapper
try:
    from slam.slam_wrapper import SLAMWrapper
    slam_available = True
except ImportError:
    slam_available = False

# Initialize SLAM
if slam_available and camera_available and not args.simulation:
    try:
        slam = SLAMWrapper()
        slam_initialized = True
        logger.info("SLAM initialized successfully")
    except Exception as e:
        slam_initialized = False
        logger.error(f"Failed to initialize SLAM: {e}")
        logger.info("Falling back to simulation mode for SLAM")
else:
    if args.simulation:
        logger.info("Running in simulation mode - SLAM not initialized")
    slam_initialized = slam_available  # In simulation mode, if the module is available, we can use it

# Global variables
current_speed = 0
is_streaming = False
streaming_thread = None
slam_active = False
slam_thread = None
gimbal_h_angle = 80  # Initial horizontal angle (PWM9)
gimbal_v_angle = 40  # Initial vertical angle (PWM10)

# Check if MPU6050 is available
mpu6050_available = False
if not args.simulation:
    try:
        # 尝试实际初始化MPU6050
        # 这里只是一个示例，实际代码取决于您的MPU6050库
        import smbus2 as smbus
        bus = smbus.SMBus(1)
        # MPU6050的I2C地址通常是0x68或0x69
        mpu_addr = 0x68
        # 尝试读取WHO_AM_I寄存器 (寄存器地址通常是0x75)
        # 如果能读到数据，说明MPU6050连接正常
        try:
            bus.write_byte_data(mpu_addr, 0x6B, 0)  # 唤醒MPU6050
            whoami = bus.read_byte_data(mpu_addr, 0x75)
            if whoami:
                mpu6050_available = True
                logger.info(f"MPU6050 initialized successfully, WHO_AM_I: {whoami}")
        except Exception as inner_e:
            logger.info(f"MPU6050 not responding on address 0x68: {inner_e}")
            # 尝试备用地址0x69
            try:
                mpu_addr = 0x69
                bus.write_byte_data(mpu_addr, 0x6B, 0)  # 唤醒MPU6050
                whoami = bus.read_byte_data(mpu_addr, 0x75)
                if whoami:
                    mpu6050_available = True
                    logger.info(f"MPU6050 initialized successfully on alternate address, WHO_AM_I: {whoami}")
            except Exception as alt_e:
                logger.info(f"MPU6050 not responding on alternate address 0x69: {alt_e}")
    except Exception as e:
        logger.error(f"MPU6050 initialization failed: {e}")
else:
    logger.info("Running in simulation mode - MPU6050 not initialized")

# Routes
@app.route('/')
def index():
    simulation_mode = args.simulation
    return render_template('index.html', 
                          robot_available=robot_available, 
                          camera_available=camera_available,
                          mpu6050_available=mpu6050_available,
                          slam_available=slam_initialized,
                          simulation_mode=simulation_mode)

@app.route('/status')
def status():
    return jsonify({
        'robot_available': robot_available,
        'camera_available': camera_available,
        'mpu6050_available': mpu6050_available,
        'slam_available': slam_initialized,
        'slam_active': slam_active,
        'current_speed': current_speed,
        'gimbal_h_angle': gimbal_h_angle,
        'gimbal_v_angle': gimbal_v_angle
    })

# SLAM processing thread
def slam_processing_thread():
    global slam_active
    
    logger.info("SLAM processing thread started")
    
    while slam_active and camera_available and slam_initialized:
        try:
            # Get a frame from the camera
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame for SLAM processing")
                time.sleep(0.1)
                continue
            
            # Process frame with SLAM
            slam.process_frame(frame)
            
            # Generate 2D map
            map_2d = slam.get_2d_map()
            
            # Convert map to base64 for transmission
            _, buffer = cv2.imencode('.png', map_2d)
            map_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Emit map to clients
            socketio.emit('slam_map', {'map': map_base64})
            
            # Get current pose
            pose = slam.get_current_pose()
            position = pose[0:3, 3].tolist()
            
            # Emit position to clients
            socketio.emit('slam_position', {'position': position})
            
            # Limit update rate
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"Error in SLAM processing: {e}")
            time.sleep(0.5)
    
    logger.info("SLAM processing thread stopped")

# Video streaming function
def generate_frames():
    global is_streaming, camera_available
    
    while is_streaming and camera_available:
        try:
            success, frame = camera.read()
            if not success:
                logger.error("Failed to read frame from camera")
                time.sleep(0.1)
                continue
                
            # Process frame (resize, add overlay, etc.)
            frame = cv2.resize(frame, (320, 240))
            
            # Add HUD overlay
            cv2.putText(frame, f"Speed: {current_speed}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"H: {gimbal_h_angle}°, V: {gimbal_v_angle}°", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # 确保图像颜色空间正确 (Ensure correct color space)
            if len(frame.shape) < 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # 使用高质量的JPEG编码 (Use high quality JPEG encoding)
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Emit frame to clients using base64 encoding
                socketio.emit('video_frame', {'frame': frame_base64})
                time.sleep(1/15)  # Limit to 15 FPS
            except Exception as encode_error:
                logger.error(f"Frame encoding error: {encode_error}")
                time.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error in video streaming: {e}")
            time.sleep(0.1)
    
    logger.info("Video streaming stopped")

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    global mpu_running, mpu_thread
    
    logger.info(f"Client connected: {request.sid}")
    emit('status_update', {
        'robot_available': robot_available,
        'camera_available': camera_available,
        'mpu6050_available': mpu6050_available,
        'slam_available': slam_initialized,
        'slam_active': slam_active
    })
    
    # Start MPU6050 data thread if it's available and not already running
    if mpu6050_available and not mpu_running:
        mpu_running = True
        mpu_thread = threading.Thread(target=read_mpu6050_data)
        mpu_thread.daemon = True
        mpu_thread.start()
        logger.info("MPU6050 data thread started for new client")

@socketio.on('disconnect')
def handle_disconnect(sid=None):
    global mpu_running
    
    logger.info(f"Client disconnected: {request.sid}")
    if robot_available:
        robot.t_stop(0)
        
    # Check if there are any remaining clients before stopping MPU thread
    # Use the Socket.IO server's active connections instead of request.namespace.rooms
    if len(socketio.server.eio.sockets) <= 1 and mpu_running:  # Only one or no connections left
        mpu_running = False
        logger.info("Stopping MPU6050 data thread - no clients left")

@socketio.on('start_stream')
def handle_start_stream():
    global is_streaming, streaming_thread
    
    logger.info(f"Start stream request from client: {request.sid}")
    
    if not is_streaming and camera_available:
        is_streaming = True
        streaming_thread = threading.Thread(target=generate_frames)
        streaming_thread.daemon = True
        streaming_thread.start()
        emit('stream_status', {'status': 'started'})
        logger.info("Video streaming started")
    else:
        if not camera_available:
            logger.error("Camera not available for streaming")
            emit('stream_status', {'status': 'error', 'message': 'Camera not available'})
        elif is_streaming:
            logger.info("Stream already active, sending status update")
            emit('stream_status', {'status': 'started'})
        else:
            logger.error("Unknown error starting stream")
            emit('stream_status', {'status': 'error', 'message': 'Unknown error starting stream'})

@socketio.on('stop_stream')
def handle_stop_stream():
    global is_streaming
    
    logger.info(f"Stop stream request from client: {request.sid}")
    
    if is_streaming:
        is_streaming = False
        emit('stream_status', {'status': 'stopped'})
        logger.info("Video streaming stopped")
    else:
        logger.info("No active stream to stop")
        emit('stream_status', {'status': 'stopped'})

@socketio.on('start_slam')
def handle_start_slam():
    global slam_active, slam_thread
    
    if not slam_active and slam_initialized and camera_available:
        try:
            # Start SLAM
            if slam.start():
                slam_active = True
                slam_thread = threading.Thread(target=slam_processing_thread)
                slam_thread.daemon = True
                slam_thread.start()
                emit('slam_status', {'status': 'started'})
                logger.info("SLAM started")
            else:
                emit('slam_status', {'status': 'error', 'message': 'Failed to start SLAM'})
        except Exception as e:
            emit('slam_status', {'status': 'error', 'message': str(e)})
            logger.error(f"Error starting SLAM: {e}")
    else:
        emit('slam_status', {
            'status': 'error', 
            'message': 'SLAM not available, camera not available, or already active'
        })

@socketio.on('stop_slam')
def handle_stop_slam():
    global slam_active
    
    if slam_active and slam_initialized:
        try:
            # Stop SLAM
            slam_active = False
            if slam.stop():
                emit('slam_status', {'status': 'stopped'})
                logger.info("SLAM stopped")
            else:
                emit('slam_status', {'status': 'error', 'message': 'Failed to stop SLAM'})
        except Exception as e:
            emit('slam_status', {'status': 'error', 'message': str(e)})
            logger.error(f"Error stopping SLAM: {e}")
    else:
        emit('slam_status', {'status': 'error', 'message': 'SLAM not active or not available'})

@socketio.on('car_control')
def handle_car_control(data):
    global current_speed, robot_available
    
    if not robot_available:
        emit('control_response', {'status': 'error', 'message': 'Robot not available'})
        return
    
    try:
        x = data.get('x', 0)  # -1 (left) to 1 (right)
        y = data.get('y', 0)  # -1 (backward) to 1 (forward)
        
        # Calculate speed (0-60 as specified in updated requirements)
        speed = min(60, int(abs(max(x, y, key=abs)) * 60))
        current_speed = speed
        
        # Determine movement direction
        if abs(y) > abs(x):  # Forward/backward movement dominates
            if y > 0.1:  # Forward
                robot.t_up(speed, 0.1)
                logger.info(f"Moving forward at speed {speed}")
            elif y < -0.1:  # Backward
                robot.t_down(speed, 0.1)
                logger.info(f"Moving backward at speed {speed}")
            else:  # Stop
                robot.t_stop(0.1)
                logger.info("Stopped")
        elif abs(x) > abs(y):  # Left/right movement dominates
            # Reduce speed to 70% when turning as per requirements
            turn_speed = int(speed * 0.7)
            if x > 0.1:  # Right (修正为左转)
                robot.turnLeft(turn_speed, 0.1)
                logger.info(f"Turning left at speed {turn_speed}")
            elif x < -0.1:  # Left (修正为右转)
                robot.turnRight(turn_speed, 0.1)
                logger.info(f"Turning right at speed {turn_speed}")
            else:  # Stop
                robot.t_stop(0.1)
                logger.info("Stopped")
        else:  # Joystick is centered
            robot.t_stop(0.1)
            logger.info("Stopped (joystick centered)")
            
        emit('control_response', {'status': 'success', 'speed': speed})
    except Exception as e:
        logger.error(f"Error in car control: {e}")
        emit('control_response', {'status': 'error', 'message': str(e)})

@socketio.on('gimbal_control')
def handle_gimbal_control(data):
    global gimbal_h_angle, gimbal_v_angle, robot_available
    
    if not robot_available:
        emit('control_response', {'status': 'error', 'message': 'Robot not available'})
        return
    
    try:
        x = data.get('x', 0)  # -1 (left) to 1 (right)
        y = data.get('y', 0)  # -1 (down) to 1 (up)
        
        # 反转Y轴值以纠正上下方向
        y = -y
        
        # Calculate new angles
        # Horizontal: 80° ± 45° (35° to 125°)
        # Vertical: 40° ± 45° (0° to 85°)
        
        # If joystick is centered (auto-centering), reset to initial angles
        if abs(x) < 0.1 and abs(y) < 0.1:
            gimbal_h_angle = 80
            gimbal_v_angle = 40
        else:
            # Update angles based on joystick position
            # Map joystick x (-1 to 1) to angle change (-45 to 45)
            h_change = int(x * 45)
            gimbal_h_angle = max(35, min(125, 80 + h_change))
            
            # Map joystick y (-1 to 1) to angle change (-45 to 45)
            v_change = int(y * 45)
            gimbal_v_angle = max(0, min(85, 40 + v_change))
        
        # Set servo angles
        robot.set_servo_angle(9, gimbal_h_angle)  # PWM9 for horizontal
        robot.set_servo_angle(10, gimbal_v_angle)  # PWM10 for vertical
        
        logger.info(f"Gimbal angles set to H:{gimbal_h_angle}°, V:{gimbal_v_angle}°")
        emit('gimbal_response', {
            'status': 'success', 
            'h_angle': gimbal_h_angle, 
            'v_angle': gimbal_v_angle
        })
    except Exception as e:
        logger.error(f"Error in gimbal control: {e}")
        emit('gimbal_response', {'status': 'error', 'message': str(e)})

# Initialize gimbal to default position on startup
if robot_available:
    try:
        robot.set_servo_angle(9, gimbal_h_angle)  # PWM9 for horizontal
        robot.set_servo_angle(10, gimbal_v_angle)  # PWM10 for vertical
        logger.info(f"Gimbal initialized to H:{gimbal_h_angle}°, V:{gimbal_v_angle}°")
    except Exception as e:
        logger.error(f"Failed to initialize gimbal: {e}")

if __name__ == '__main__':
    try:
        # Create required directories if they don't exist
        os.makedirs('static', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('slam', exist_ok=True)
        
        # Start the Flask app with SocketIO
        logger.info("Starting server on 0.0.0.0:5000")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    finally:
        # Clean up resources
        if camera_available:
            camera.release()
        if robot_available:
            robot.t_stop(0)
        if slam_active and slam_initialized:
            slam.stop()
        # Stop MPU6050 thread if running
        if mpu_running:
            mpu_running = False
            logger.info("Stopped MPU6050 data thread during shutdown")
        logger.info("Server shutdown complete") 