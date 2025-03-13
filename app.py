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
        # This is a placeholder for MPU6050 initialization
        # In a real implementation, you would import and initialize the MPU6050 library
        logger.info("MPU6050 check completed")
    except Exception as e:
        logger.error(f"MPU6050 not available: {e}")
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
                break
                
            # Process frame (resize, add overlay, etc.)
            frame = cv2.resize(frame, (320, 240))
            
            # Add HUD overlay
            cv2.putText(frame, f"Speed: {current_speed}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"H: {gimbal_h_angle}°, V: {gimbal_v_angle}°", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Emit frame to clients
            socketio.emit('video_frame', {'frame': frame_bytes})
            time.sleep(1/15)  # Limit to 15 FPS
            
        except Exception as e:
            logger.error(f"Error in video streaming: {e}")
            break
    
    logger.info("Video streaming stopped")

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('status_update', {
        'robot_available': robot_available,
        'camera_available': camera_available,
        'mpu6050_available': mpu6050_available,
        'slam_available': slam_initialized,
        'slam_active': slam_active
    })

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")
    if robot_available:
        robot.t_stop(0)

@socketio.on('start_stream')
def handle_start_stream():
    global is_streaming, streaming_thread
    
    if not is_streaming and camera_available:
        is_streaming = True
        streaming_thread = threading.Thread(target=generate_frames)
        streaming_thread.daemon = True
        streaming_thread.start()
        emit('stream_status', {'status': 'started'})
        logger.info("Video streaming started")
    else:
        emit('stream_status', {'status': 'error', 'message': 'Camera not available or already streaming'})

@socketio.on('stop_stream')
def handle_stop_stream():
    global is_streaming
    
    if is_streaming:
        is_streaming = False
        emit('stream_status', {'status': 'stopped'})
        logger.info("Video streaming stopped")

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
        
        # Calculate speed (0-30 as specified in requirements)
        speed = min(30, int(abs(max(x, y, key=abs)) * 30))
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
            if x > 0.1:  # Right
                robot.turnRight(turn_speed, 0.1)
                logger.info(f"Turning right at speed {turn_speed}")
            elif x < -0.1:  # Left
                robot.turnLeft(turn_speed, 0.1)
                logger.info(f"Turning left at speed {turn_speed}")
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
            
            # Map joystick y (-1 to 1) to angle change (-40 to 45)
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
        socketio.run(app, host='0.0.0.0', port=5000, debug=True)
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
        logger.info("Server shutdown complete") 