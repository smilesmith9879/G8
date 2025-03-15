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
from collections import deque
from threading import Lock, Event, Thread
import subprocess
import socket

# 尝试导入psutil，但允许失败
try:
    import psutil
    psutil_available = True
except ImportError:
    psutil_available = False
    print("psutil库未安装，资源监控功能将被禁用")
    # 定义一个空的monitor_resources函数
    def monitor_resources():
        pass

# Parse command line arguments
parser = argparse.ArgumentParser(description='AI Smart Four-Wheel Drive Car')
parser.add_argument('--simulation', action='store_true', help='Run in simulation mode without hardware')
args = parser.parse_args()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['PROPAGATE_EXCEPTIONS'] = True  # 增加异常传播，使错误更易于追踪
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB最大内容长度

# 创建线程管理器
thread_manager = ThreadManager()

# 优化Socket.IO配置
socketio = SocketIO(
    app, 
    async_mode='threading',  # 使用threading模式
    cors_allowed_origins="*",  # 允许任何来源的跨域请求
    ping_interval=25,  # 25秒ping一次客户端
    ping_timeout=60,   # 60秒超时
    max_http_buffer_size=10 * 1024 * 1024,  # 10MB最大HTTP缓冲区大小
    manage_session=False,  # 禁用会话管理以减少开销
    engineio_logger=False,  # 关闭engineio日志，减少日志噪音
    logger=False  # 关闭socketio日志，减少日志噪音
)

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
        # 尝试设置为MJPG格式提高效率
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)  # 增加内部FPS以提高实际得到的帧率
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 增加OpenCV内部缓冲区大小
        
        # 获取实际相机设置
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        actual_format = camera.get(cv2.CAP_PROP_FOURCC)
        
        logger.info(f"相机初始化 - 宽度: {actual_width}, 高度: {actual_height}, FPS: {actual_fps}, 格式: {chr(int(actual_format) & 0xFF)}{chr((int(actual_format) >> 8) & 0xFF)}{chr((int(actual_format) >> 16) & 0xFF)}{chr((int(actual_format) >> 24) & 0xFF)}")
        
        # 初始化帧缓冲区并启动它
        frame_buffer = FrameBuffer(camera, buffer_size=10, name="main_camera_buffer")
        if frame_buffer.start():
            logger.info("相机帧缓冲区启动成功")
            # 将帧缓冲区线程注册到线程管理器
            thread_manager.register_thread("frame_buffer", frame_buffer.thread, frame_buffer.stop_event)
        else:
            logger.error("相机帧缓冲区启动失败，将使用直接读取模式")
            frame_buffer = None
        
        # 检查相机是否正常
        ret, frame = camera.read() if frame_buffer is None else frame_buffer.get_frame()
        if not ret:
            logger.error("无法从相机读取帧，相机功能将被禁用")
            camera_available = False
        else:
            logger.info(f"相机测试成功，获取到帧大小: {frame.shape}")
            camera_available = True
    except Exception as e:
        logger.error(f"相机初始化错误: {e}")
        camera_available = False
        camera = None
        frame_buffer = None
else:
    camera = None
    frame_buffer = None
    camera_available = False
    logger.info("相机功能已禁用（模拟模式）")

# Add frame buffer for camera performance optimization
class FrameBuffer:
    def __init__(self, camera, buffer_size=5):
        self.buffer = threading.Queue(maxsize=buffer_size)
        self.lock = threading.Lock()
        self.last_frame = None
        self.running = False
        self.capture_thread = None
        self.camera = camera
        self.frame_count = 0
        self.drop_count = 0
        
    def start(self):
        """Start the frame capturing thread"""
        if self.running:
            return False
            
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_frames)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info("Camera frame buffer started")
        return True
        
    def _capture_frames(self):
        """Continuously capture frames in a dedicated thread"""
        while self.running:
            read_start = time.time()
            success, frame = self.camera.read()
            read_time = time.time() - read_start
            
            if read_time > 0.05:  # Reasonable threshold for logging
                logger.debug(f"Camera read took {read_time:.3f}s in buffer thread")
                
            if not success:
                # Failed to read frame, small wait before retry
                time.sleep(0.01)
                continue
                
            self.frame_count += 1
            
            # If buffer is full, remove oldest frame
            if self.buffer.full():
                try:
                    self.buffer.get_nowait()
                    self.drop_count += 1
                    # Log frame drops periodically
                    if self.drop_count % 100 == 0:
                        logger.warning(f"Camera buffer overflow: dropped {self.drop_count} frames")
                except:
                    pass
            
            # Add new frame to buffer
            try:
                self.buffer.put_nowait(frame)
                with self.lock:
                    self.last_frame = frame
            except Exception as e:
                logger.error(f"Error adding frame to buffer: {e}")
                
            # Adaptive sleep to match camera FPS
            target_interval = 1.0 / 30.0  # Target 30 FPS internally for buffer
            elapsed = time.time() - read_start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_frame(self):
        """Get the latest frame from the buffer"""
        if not self.running:
            return False, None
            
        try:
            # Try to get from buffer with short timeout
            frame = self.buffer.get(timeout=0.05)
            with self.lock:
                self.last_frame = frame
            return True, frame
        except:
            # If buffer is empty, return the last known frame
            with self.lock:
                if self.last_frame is not None:
                    return True, self.last_frame.copy()
            # If no frames available, try direct camera read as fallback
            return self.camera.read()
    
    def stop(self):
        """Stop the capture thread"""
        if not self.running:
            return
            
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            if self.capture_thread.is_alive():
                logger.warning("Camera frame buffer thread did not terminate within timeout")
            else:
                logger.info(f"Camera frame buffer stopped after processing {self.frame_count} frames")

# Initialize the frame buffer if camera is available
frame_buffer = None
if camera_available:
    frame_buffer = FrameBuffer(camera, buffer_size=10)
    frame_buffer.start()

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

# MPU6050数据发送线程
def mpu6050_data_thread():
    """周期性读取MPU6050数据并发送到客户端"""
    global mpu_running
    
    logger.info("MPU6050 data thread started")
    
    while mpu_running and mpu6050_available:
        try:
            # 读取校准后的MPU6050数据
            data = read_mpu6050_data()
            if data:
                # 发送数据给客户端
                socketio.emit('mpu6050_data', {
                    'accel_x': data['accel_x'],
                    'accel_y': data['accel_y'],
                    'accel_z': data['accel_z'],
                    'gyro_x': data['gyro_x'],
                    'gyro_y': data['gyro_y'],
                    'gyro_z': data['gyro_z'],
                    'temperature': data['temp']
                })
            
            # 限制更新频率
            time.sleep(0.2)  # 5Hz更新率
        except Exception as e:
            logger.error(f"Error in MPU6050 data thread: {e}")
            time.sleep(1)
    
    logger.info("MPU6050 data thread stopped")

# Function to read MPU6050 data
def read_mpu6050_data():
    """读取MPU6050传感器数据并应用校准"""
    if not mpu6050_available or mpu is None:
        return None
    
    try:
        accel_data = mpu.get_accel_data()
        gyro_data = mpu.get_gyro_data()
        temp = mpu.get_temp()
        
        # 应用校准
        if mpu_calibration['calibrated']:
            accel_data['x'] -= mpu_calibration['accel_bias']['x']
            accel_data['y'] -= mpu_calibration['accel_bias']['y']
            accel_data['z'] -= mpu_calibration['accel_bias']['z']
            gyro_data['x'] -= mpu_calibration['gyro_bias']['x']
            gyro_data['y'] -= mpu_calibration['gyro_bias']['y']
            gyro_data['z'] -= mpu_calibration['gyro_bias']['z']
        
        return {
            'accel_x': f"{accel_data['x']:.3f}",
            'accel_y': f"{accel_data['y']:.3f}",
            'accel_z': f"{accel_data['z']:.3f}",
            'gyro_x': f"{gyro_data['x']:.3f}",
            'gyro_y': f"{gyro_data['y']:.3f}",
            'gyro_z': f"{gyro_data['z']:.3f}",
            'temp': f"{temp:.2f}",
            'raw_accel': accel_data,
            'raw_gyro': gyro_data
        }
    except Exception as e:
        logger.error(f"Error reading MPU6050 data: {e}")
        return None

# Simulated camera frame generator
class SimulatedCamera:
    def __init__(self):
        self.frame_count = 0
        self.base_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.text_position = (50, 120)
        self.font_scale = 0.7
        self.font_color = (255, 255, 255)
        self.line_type = 2
        
        # 创建一个更有视觉吸引力的模拟画面
        # 绘制一个渐变背景
        for y in range(240):
            blue = int(255 * (y / 240))
            green = int(100 + 100 * (1 - y / 240))
            for x in range(320):
                red = int(100 + 155 * (x / 320))
                self.base_frame[y, x] = [blue, green, red]
                
        # 添加一些界面元素
        cv2.rectangle(self.base_frame, (10, 10), (310, 230), (255, 255, 255), 2)
        cv2.putText(self.base_frame, 'Simulated Camera', 
                    self.text_position, self.font, self.font_scale,
                    (0, 0, 0), self.line_type + 1)  # 黑色描边
        cv2.putText(self.base_frame, 'Simulated Camera', 
                    self.text_position, self.font, self.font_scale,
                    self.font_color, self.line_type)
                    
        cv2.putText(self.base_frame, 'AI Smart Car', 
                    (90, 180), self.font, 0.6,
                    (0, 0, 0), 2)  # 黑色描边
        cv2.putText(self.base_frame, 'AI Smart Car', 
                    (90, 180), self.font, 0.6,
                    (255, 255, 255), 1)
                    
        logger.info("Simulated camera initialized with enhanced visuals")
        
    def read(self):
        # Create a copy of the base frame
        self.frame_count += 1
        frame = self.base_frame.copy()
        
        # Add timestamp and frame counter
        timestamp = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, f"Time: {timestamp}", (10, 30), self.font, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Time: {timestamp}", (10, 30), self.font, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 50), self.font, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, 50), self.font, 0.5, (255, 255, 255), 1)
        
        # 添加一个移动的元素，使画面看起来像在移动
        offset_x = int(30 * np.sin(self.frame_count / 10))
        offset_y = int(20 * np.cos(self.frame_count / 10))
        center = (160 + offset_x, 120 + offset_y)
        cv2.circle(frame, center, 15, (0, 255, 255), -1)
        cv2.circle(frame, center, 15, (0, 0, 0), 2)
        
        # Add some movement to simulate a real camera
        noise = np.random.randint(0, 8, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        return True, frame
        
    def isOpened(self):
        return True
        
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
slam_initialized = False
slam_active = False
slam_thread = None
mpu6050_available = False
mpu = None
mpu_calibration = {
    'accel_bias': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    'gyro_bias': {'x': 0.0, 'y': 0.0, 'z': 0.0},
    'calibrated': False
}

try:
    # 检查是否有MPU6050传感器
    try:
        if not args.simulation:
            # 尝试导入MPU6050库
            import mpu6050
            # 尝试初始化MPU6050
            mpu = mpu6050.mpu6050(0x68)
            
            # 执行MPU6050初始校准
            logger.info("Performing MPU6050 initial calibration...")
            accel_x_samples = []
            accel_y_samples = []
            accel_z_samples = []
            gyro_x_samples = []
            gyro_y_samples = []
            gyro_z_samples = []
            
            # 收集多个样本以计算平均偏差
            num_samples = 50
            for i in range(num_samples):
                try:
                    accel_data = mpu.get_accel_data()
                    gyro_data = mpu.get_gyro_data()
                    
                    accel_x_samples.append(accel_data['x'])
                    accel_y_samples.append(accel_data['y'])
                    accel_z_samples.append(accel_data['z'])
                    gyro_x_samples.append(gyro_data['x'])
                    gyro_y_samples.append(gyro_data['y'])
                    gyro_z_samples.append(gyro_data['z'])
                    
                    time.sleep(0.01)  # 短暂延迟以获取不同的读数
                except Exception as e:
                    logger.warning(f"Error during MPU6050 calibration sample {i}: {e}")
            
            # 计算平均偏差
            if len(accel_x_samples) > 0:
                # 加速度计偏差 - 假设Z轴应该是1g (9.81 m/s²)，X和Y轴应该是0g
                mpu_calibration['accel_bias']['x'] = sum(accel_x_samples) / len(accel_x_samples)
                mpu_calibration['accel_bias']['y'] = sum(accel_y_samples) / len(accel_y_samples)
                mpu_calibration['accel_bias']['z'] = sum(accel_z_samples) / len(accel_z_samples) - 1.0  # 减去1g
                
                # 陀螺仪偏差 - 静止时应该是0度/秒
                mpu_calibration['gyro_bias']['x'] = sum(gyro_x_samples) / len(gyro_x_samples)
                mpu_calibration['gyro_bias']['y'] = sum(gyro_y_samples) / len(gyro_y_samples)
                mpu_calibration['gyro_bias']['z'] = sum(gyro_z_samples) / len(gyro_z_samples)
                
                mpu_calibration['calibrated'] = True
                
                logger.info(f"MPU6050 calibration completed. Biases: "
                           f"Accel (g) - X: {mpu_calibration['accel_bias']['x']:.3f}, "
                           f"Y: {mpu_calibration['accel_bias']['y']:.3f}, "
                           f"Z: {mpu_calibration['accel_bias']['z']:.3f}, "
                           f"Gyro (°/s) - X: {mpu_calibration['gyro_bias']['x']:.3f}, "
                           f"Y: {mpu_calibration['gyro_bias']['y']:.3f}, "
                           f"Z: {mpu_calibration['gyro_bias']['z']:.3f}")
            
            # 测试读取校准后的数据
            accel_data = mpu.get_accel_data()
            gyro_data = mpu.get_gyro_data()
            temp = mpu.get_temp()
            
            # 应用校准
            if mpu_calibration['calibrated']:
                accel_data['x'] -= mpu_calibration['accel_bias']['x']
                accel_data['y'] -= mpu_calibration['accel_bias']['y']
                accel_data['z'] -= mpu_calibration['accel_bias']['z']
                gyro_data['x'] -= mpu_calibration['gyro_bias']['x']
                gyro_data['y'] -= mpu_calibration['gyro_bias']['y']
                gyro_data['z'] -= mpu_calibration['gyro_bias']['z']
            
            logger.info(f"MPU6050 initialized with calibrated data: "
                       f"Accel (g) - X: {accel_data['x']:.3f}, Y: {accel_data['y']:.3f}, Z: {accel_data['z']:.3f}, "
                       f"Gyro (°/s) - X: {gyro_data['x']:.3f}, Y: {gyro_data['y']:.3f}, Z: {gyro_data['z']:.3f}, "
                       f"Temperature: {temp:.2f}°C")
            
            mpu6050_available = True
    except Exception as e:
        logger.warning(f"MPU6050 not available: {e}")
        mpu6050_available = False
    
    # 初始化SLAM，如果有MPU6050则使用
    slam = SLAMWrapper(use_mpu6050=mpu6050_available)
    slam_initialized = True
    logger.info("SLAM initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SLAM: {e}")
    slam_initialized = False

# Global variables
current_speed = 0
is_streaming = False
streaming_thread = None
gimbal_h_angle = 80  # Initial horizontal angle (PWM9)
gimbal_v_angle = 40  # Initial vertical angle (PWM10)
# camera_lock = threading.RLock()  # 移除相机资源锁以提高视频流的流畅性
last_resource_check = time.time()  # 资源检查时间记录
resource_monitor_thread = None  # 资源监控线程
resource_monitoring_active = False  # 资源监控活动状态

# 初始化状态变量
robot_available = False    # 机器人控制器是否可用
camera_available = False   # 相机是否可用
mpu6050_available = False  # MPU6050传感器是否可用
slam_initialized = False   # SLAM系统是否初始化
slam_active = False        # SLAM系统是否活跃运行中
is_streaming = False       # 视频流是否活跃

# 数据存储
imu_data_history = []      # 存储最近的IMU数据
resource_usage = {}        # 存储最新的资源使用情况

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

@app.route('/mobile')
def mobile():
    simulation_mode = args.simulation
    return render_template('mobile.html', 
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
    frame_count = 0
    error_count = 0
    last_success_time = time.time()
    last_detailed_log = time.time()  # 上次详细日志时间
    processing_times = []  # 存储处理时间
    
    while slam_active and camera_available and slam_initialized:
        frame_start = time.time()
        frame_processed = False
        
        try:
            # 直接读取相机帧，不使用锁
            read_start = time.time()
            success, frame = camera.read()
            read_time = time.time() - read_start
            
            if read_time > 0.1:
                logger.warning(f"SLAM: Camera read took {read_time:.3f}s, which is slow")
            
            if not success:
                error_count += 1
                if error_count > 5:
                    logger.error("SLAM: Multiple camera read failures, stopping SLAM")
                    slam_active = False
                    break
                continue
            
            # 重置错误计数
            error_count = 0
            
            # 确保帧格式正确 - SLAM通常需要灰度或RGB
            preprocess_start = time.time()
            gray_frame = None
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # 彩色帧，转换为灰度用于某些SLAM算法
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif len(frame.shape) == 2:
                # 已经是灰度
                gray_frame = frame
            preprocess_time = time.time() - preprocess_start
            
            # 如果有MPU6050，读取IMU数据
            imu_data = None
            if mpu6050_available:
                try:
                    mpu_data = read_mpu6050_data()
                    if mpu_data:
                        imu_data = {
                            "accel": [
                                float(mpu_data['raw_accel']['x']), 
                                float(mpu_data['raw_accel']['y']), 
                                float(mpu_data['raw_accel']['z'])
                            ],
                            "gyro": [
                                float(mpu_data['raw_gyro']['x']), 
                                float(mpu_data['raw_gyro']['y']), 
                                float(mpu_data['raw_gyro']['z'])
                            ]
                        }
                except Exception as e:
                    logger.warning(f"Failed to read MPU6050 data: {e}")
            
            # 处理帧用于SLAM
            process_start = time.time()
            slam.process_frame(frame, imu_data)  # 使用原始帧和IMU数据
            process_time = time.time() - process_start
            
            if process_time > 0.2:  # 如果处理时间超过200ms，记录警告
                logger.warning(f"SLAM: Frame processing took {process_time:.3f}s, which may be too slow")
            
            # 收集处理时间统计
            processing_times.append(process_time)
            
            # 每10帧记录一次性能信息
            if frame_count % 10 == 0:
                logger.info(f"SLAM processed {frame_count} frames, last frame process time: {process_time:.3f}s")
            
            # 获取2D地图
            map_start = time.time()
            map_2d = slam.get_2d_map()
            map_time = time.time() - map_start
            
            # 转换为base64
            encode_start = time.time()
            _, buffer = cv2.imencode('.png', map_2d)
            map_base64 = base64.b64encode(buffer).decode('utf-8')
            encode_time = time.time() - encode_start
            
            # 发送地图给客户端
            emit_start = time.time()
            socketio.emit('slam_map', {'map': map_base64})
            emit_time = time.time() - emit_start
            
            # 获取当前位姿
            pose_start = time.time()
            pose = slam.get_current_pose()
            position = pose[0:3, 3].tolist()
            pose_time = time.time() - pose_start
            
            # 发送位置给客户端
            socketio.emit('slam_position', {'position': position})
            
            # 获取3D地图数据
            map3d_start = time.time()
            map_3d = slam.get_3d_map_data()
            map3d_time = time.time() - map3d_start
            
            # 发送3D地图数据给客户端
            if map_3d and map_3d['count'] > 0:
                socketio.emit('slam_map_3d', map_3d)
            
            # 每30秒输出一次详细的性能报告
            current_time = time.time()
            if current_time - last_detailed_log >= 30.0 and len(processing_times) > 0:
                avg_process_time = sum(processing_times) / len(processing_times)
                max_process_time = max(processing_times)
                min_process_time = min(processing_times)
                
                logger.info(f"SLAM performance report - Average process time: {avg_process_time*1000:.1f}ms, Min: {min_process_time*1000:.1f}ms, Max: {max_process_time*1000:.1f}ms")
                logger.info(f"Last frame times - Read: {read_time*1000:.1f}ms, Process: {process_time*1000:.1f}ms, Map: {map_time*1000:.1f}ms, Encode: {encode_time*1000:.1f}ms, Emit: {emit_time*1000:.1f}ms, Pose: {pose_time*1000:.1f}ms, 3D Map: {map3d_time*1000:.1f}ms")
                
                # 重置统计数据
                processing_times = []
                last_detailed_log = current_time
            
            # 监控系统资源
            monitor_resources()
            
            # 计算总处理时间
            total_time = time.time() - frame_start
            
            # 限制更新频率 - 目标是5Hz (200ms/帧)
            sleep_time = max(0, 0.2 - total_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            # 增加帧计数
            frame_count += 1
            frame_processed = True
            last_success_time = time.time()
            
        except Exception as e:
            logger.error(f"Error in SLAM processing: {e}")
            time.sleep(0.5)
        
        # 如果帧没有处理成功，添加简短延迟
        if not frame_processed:
            time.sleep(0.1)
    
    logger.info(f"SLAM processing thread stopped after processing {frame_count} frames")

# 添加资源监控函数
def monitor_resources():
    global last_resource_check
    
    # 如果psutil不可用，直接返回
    if not psutil_available:
        return
    
    # 每10秒检查一次系统资源
    current_time = time.time()
    if current_time - last_resource_check > 10:
        try:
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 记录资源使用情况
            logger.info(f"System resources - CPU: {cpu_percent}%, Memory: {memory_percent}%")
            
            # 如果资源使用过高，发出警告
            if cpu_percent > 80:
                logger.warning(f"High CPU usage: {cpu_percent}%")
            if memory_percent > 80:
                logger.warning(f"High memory usage: {memory_percent}%")
                
            last_resource_check = current_time
        except Exception as e:
            logger.error(f"Error monitoring system resources: {e}")

# Video streaming function
def generate_frames():
    global is_streaming, camera_available, camera, frame_buffer
    
    logger.info("视频流线程启动，使用threading模式，ping间隔: 25秒, ping超时: 60秒")
    frame_count = 0
    error_count = 0
    last_log_time = time.time()
    fps_stats = []  # 存储每秒处理的帧数统计
    last_detailed_log = time.time()  # 上次详细日志时间
    frame_processing_times = []  # 存储帧处理时间统计
    
    # 视频帧率和质量设置
    TARGET_FPS = 10  # 降低目标帧率以减轻服务器负担
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    JPEG_QUALITY = 70  # 降低JPEG质量以减小数据大小
    
    # 记录视频流初始化参数
    logger.info(f"Video stream parameters - Target FPS: {TARGET_FPS}, Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}, JPEG Quality: {JPEG_QUALITY}")
    
    while is_streaming and camera_available:
        loop_start = time.time()  # 测量每帧处理时间
        frame_processed = False
        
        try:
            # 使用帧缓冲区读取帧，如果帧缓冲区可用
            read_start = time.time()
            if frame_buffer and frame_buffer.running:
                success, frame = frame_buffer.get_frame()
            else:
                success, frame = camera.read()
            read_time = time.time() - read_start
            
            if read_time > 0.1:  # 如果读取时间过长，记录日志
                logger.warning(f"Video stream: Camera read took {read_time:.3f}s, which is slow")
            
            if not success:
                error_count += 1
                logger.error(f"Failed to read frame from camera (attempt {error_count})")
                if error_count > 5:
                    logger.error("Too many consecutive frame read failures, checking camera...")
                    # 尝试重新初始化帧缓冲区，如果它存在
                    if frame_buffer and not frame_buffer.running:
                        logger.info("Trying to restart frame buffer...")
                        if frame_buffer.start():
                            logger.info("Frame buffer restarted successfully")
                            error_count = 0
                        else:
                            logger.error("Failed to restart frame buffer")
                    
                    # 尝试重新获取一帧，用于诊断
                    if hasattr(camera, 'isOpened') and not camera.isOpened():
                        logger.error("Camera appears to be closed, attempting to reopen")
                        if not args.simulation:
                            try:
                                camera.release()
                                time.sleep(1)  # 给相机更多时间重置
                                
                                # 重新打开相机并设置参数
                                camera = cv2.VideoCapture(0)
                                # 尝试设置为MJPG格式提高效率
                                camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                                camera.set(cv2.CAP_PROP_FPS, 30)  # 增加内部FPS以提高实际得到的帧率
                                camera.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # 增加OpenCV内部缓冲区大小
                                
                                # 获取实际相机设置
                                actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                                actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                actual_fps = camera.get(cv2.CAP_PROP_FPS)
                                actual_format = camera.get(cv2.CAP_PROP_FOURCC)
                                
                                logger.info(f"Reopened camera settings - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}, Format: {chr(int(actual_format) & 0xFF)}{chr((int(actual_format) >> 8) & 0xFF)}{chr((int(actual_format) >> 16) & 0xFF)}{chr((int(actual_format) >> 24) & 0xFF)}")
                                
                                ret_test, _ = camera.read()
                                if ret_test:
                                    logger.info("Camera successfully reopened")
                                    # 重新创建帧缓冲区
                                    if frame_buffer:
                                        frame_buffer.stop()
                                    frame_buffer = FrameBuffer(camera, buffer_size=10)
                                    frame_buffer.start()
                                    error_count = 0
                                else:
                                    logger.error("Failed to reopen camera")
                                    camera_available = False
                                    break
                            except Exception as cam_error:
                                logger.error(f"Error reopening camera: {cam_error}")
                                camera_available = False
                                break
                    error_count = 0
                time.sleep(0.1)
                continue
            
            # 成功读取帧
            frame_processed = True
            error_count = 0
            frame_count += 1
                
            # 处理前记录原始帧大小
            original_shape = frame.shape
            
            # 调整大小降低分辨率以提高性能
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
            # 添加一个简单的HUD
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cv2.putText(frame, current_time, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 确保图像颜色空间正确
            if len(frame.shape) < 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
            # 添加帧基本信息日志
            if frame_count % 30 == 0:  # 每30帧记录一次详细信息
                logger.info(f"视频帧 #{frame_count}: 形状={frame.shape}, 类型={frame.dtype}, 非零像素={np.count_nonzero(frame)}")
            
            # 使用优化的JPEG编码参数
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                encode_start = time.time()
                _, buffer = cv2.imencode('.jpg', frame, encode_param)
                
                # 验证buffer非空
                if buffer is None or len(buffer) == 0:
                    logger.error(f"帧 #{frame_count}: 编码后buffer为空")
                    time.sleep(0.1)
                    continue
                    
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # 验证Base64数据
                if not frame_base64 or len(frame_base64) < 100:  # 基本验证，确保数据不为空且长度合理
                    logger.error(f"帧 #{frame_count}: Base64编码异常，长度={len(frame_base64) if frame_base64 else 0}")
                    time.sleep(0.1)
                    continue
                
                encode_time = time.time() - encode_start
                
                if encode_time > 0.1:  # 如果编码时间过长，记录日志
                    logger.warning(f"视频流: 帧编码耗时 {encode_time:.3f}s")
                
                # 为帧添加序号和时间戳，便于调试
                frame_data = {
                    'frame': frame_base64,
                    'count': frame_count,
                    'time': time.time(),
                    'size': len(frame_base64)
                }
                
                # 每10帧记录一次详细信息
                if frame_count % 10 == 0:
                    logger.debug(f"帧 #{frame_count}: 大小={len(frame_base64)/1024:.1f}KB, 编码时间={encode_time*1000:.1f}ms")
                
                # 发送帧，测量发送时间
                emit_start = time.time()
                try:
                    # 直接广播帧到所有客户端，而不是尝试获取客户端列表
                    socketio.emit('video_frame', frame_data)
                    
                    emit_time = time.time() - emit_start
                    
                    # 详细日志记录
                    if frame_count % 50 == 0:  # 每50帧记录一次
                        logger.info(f"帧 #{frame_count}: 发送成功, 大小={len(frame_base64)/1024:.1f}KB, 编码耗时={encode_time*1000:.1f}ms, 发送耗时={emit_time*1000:.1f}ms")
                    
                    if emit_time > 0.2:  # 如果发送时间过长，记录日志
                        logger.warning(f"视频流: 发送帧耗时 {emit_time:.3f}s")
                
                except Exception as emit_error:
                    logger.error(f"帧 #{frame_count}: 发送失败: {emit_error}")
                    # 尝试重新连接或恢复
                    socketio.sleep(0.5)  # 短暂等待
                
            except Exception as encode_error:
                logger.error(f"帧 #{frame_count}: 编码错误: {encode_error}")
                logger.debug(f"帧 #{frame_count}: 形状={frame.shape}, 类型={frame.dtype}, 均值={np.mean(frame) if hasattr(frame, 'mean') else 'N/A'}")
                time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Error in video streaming: {e}")
            time.sleep(0.1)
        
        # 如果帧没有处理成功，添加简短延迟
        if not frame_processed:
            time.sleep(0.05)
            continue
            
        # 控制帧率 - 计算每帧处理时间和所需等待时间
        frame_time = time.time() - loop_start
        target_frame_time = 1.0 / TARGET_FPS  # 10 FPS = 100ms per frame
        sleep_time = max(0, target_frame_time - frame_time)
        
        if sleep_time > 0:
            time.sleep(sleep_time)
    
    # 停止帧缓冲区线程
    if frame_buffer and frame_buffer.running:
        frame_buffer.stop()
        
    logger.info(f"Video streaming stopped after {frame_count} frames")

# 资源监控线程
def resource_monitoring_thread():
    """监控CPU、内存和磁盘使用情况的线程"""
    global resource_usage
    
    logger.info("资源监控线程已启动")
    stop_event = thread_manager.stop_events.get("resource_monitor")
    
    # 获取stop_event
    if not stop_event:
        logger.error("资源监控线程未在线程管理器中注册")
        stop_event = Event()  # 创建临时事件
    
    # 检查psutil可用性
    if not psutil_available:
        logger.error("psutil不可用，资源监控线程将退出")
        return
    
    # 获取基本系统信息
    try:
        # 获取CPU信息
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        if cpu_freq:
            cpu_freq = f"{cpu_freq.current:.1f}MHz"
        else:
            cpu_freq = "N/A"
        
        # 获取内存信息
        mem = psutil.virtual_memory()
        total_mem = mem.total / (1024 * 1024 * 1024)  # GB
        
        # 获取磁盘信息
        disk = psutil.disk_usage('/')
        total_disk = disk.total / (1024 * 1024 * 1024)  # GB
        
        # 获取网络信息
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "127.0.0.1"
        
        # 获取操作系统信息
        uname = os.uname() if hasattr(os, 'uname') else None
        if uname:
            os_info = f"{uname.sysname} {uname.release}"
        else:
            os_info = "Unknown"
        
        # 更新系统基本信息
        system_info = {
            'hostname': hostname,
            'ip': ip_address,
            'os': os_info,
            'cpu_count': cpu_count,
            'cpu_freq': cpu_freq,
            'total_mem': f"{total_mem:.1f}GB",
            'total_disk': f"{total_disk:.1f}GB"
        }
        
        logger.info(f"系统信息: {system_info}")
        
    except Exception as e:
        logger.error(f"获取系统信息时出错: {e}")
        system_info = {'error': str(e)}
    
    # 更新间隔（秒）
    update_interval = 5.0
    last_update = time.time()
    
    try:
        # 监控循环
        while not stop_event.is_set():
            current_time = time.time()
            
            # 按指定间隔更新资源使用情况
            if current_time - last_update >= update_interval:
                try:
                    # 获取CPU使用率
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    
                    # 获取内存使用情况
                    mem = psutil.virtual_memory()
                    mem_used = mem.used / (1024 * 1024 * 1024)  # GB
                    mem_percent = mem.percent
                    
                    # 获取磁盘使用情况
                    disk = psutil.disk_usage('/')
                    disk_used = disk.used / (1024 * 1024 * 1024)  # GB
                    disk_percent = disk.percent
                    
                    # 获取网络使用情况
                    net_io = psutil.net_io_counters()
                    net_sent = net_io.bytes_sent / (1024 * 1024)  # MB
                    net_recv = net_io.bytes_recv / (1024 * 1024)  # MB
                    
                    # 获取主要进程信息
                    process_info = []
                    try:
                        # 获取当前进程
                        current_process = psutil.Process()
                        # 获取当前进程的CPU和内存使用率
                        curr_proc_cpu = current_process.cpu_percent() / cpu_count
                        curr_proc_mem = current_process.memory_percent()
                        
                        process_info.append({
                            'name': 'SmartCar App',
                            'cpu': curr_proc_cpu,
                            'memory': curr_proc_mem
                        })
                        
                        # 获取最高CPU的其他进程
                        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                            try:
                                pinfo = proc.info
                                if pinfo['pid'] != current_process.pid and pinfo['cpu_percent'] > 1.0:
                                    process_info.append({
                                        'name': pinfo['name'],
                                        'cpu': pinfo['cpu_percent'] / cpu_count,
                                        'memory': pinfo['memory_percent']
                                    })
                                    
                                # 限制进程列表长度
                                if len(process_info) >= 5:
                                    break
                            except:
                                pass
                    except Exception as proc_err:
                        logger.warning(f"获取进程信息失败: {proc_err}")
                    
                    # 构建资源使用数据包
                    resource_update = {
                        'timestamp': current_time,
                        'system': system_info,
                        'cpu': {
                            'percent': cpu_percent
                        },
                        'memory': {
                            'used': f"{mem_used:.1f}GB",
                            'percent': mem_percent
                        },
                        'disk': {
                            'used': f"{disk_used:.1f}GB",
                            'percent': disk_percent
                        },
                        'network': {
                            'sent': f"{net_sent:.1f}MB",
                            'received': f"{net_recv:.1f}MB"
                        },
                        'processes': process_info
                    }
                    
                    # 更新全局资源使用信息
                    resource_usage = resource_update
                    
                    # 广播资源使用情况
                    socketio.emit('resource_usage', resource_update)
                    
                    # 日志记录当前资源使用情况
                    if cpu_percent > 80 or mem_percent > 80:
                        logger.warning(f"资源使用率较高 - CPU: {cpu_percent}%, 内存: {mem_percent}%")
                    else:
                        logger.debug(f"资源使用情况 - CPU: {cpu_percent}%, 内存: {mem_percent}%")
                    
                    # 更新上次更新时间
                    last_update = current_time
                    
                except Exception as e:
                    logger.error(f"更新资源使用信息时出错: {e}")
                    time.sleep(update_interval)  # 出错时等待下次更新
            
            # 短暂睡眠以降低CPU使用率
            time.sleep(0.5)
    
    except Exception as e:
        logger.error(f"资源监控线程异常: {e}")
    
    finally:
        logger.info("资源监控线程已停止")

# 客户端管理
connected_clients = {}  # {sid: {'type': 'web'|'mobile', 'ip': client_ip, 'connect_time': timestamp}}
client_count_lock = Lock()  # 用于同步客户端计数操作的锁

@socketio.on('connect')
def handle_connect():
    """处理客户端连接事件"""
    try:
        client_ip = request.remote_addr if request else 'unknown'
        client_sid = request.sid if request else 'unknown'
        user_agent = request.headers.get('User-Agent', 'unknown') if request else 'unknown'
        
        # 判断客户端类型
        client_type = 'mobile' if 'Mobile' in user_agent or 'Android' in user_agent or 'iOS' in user_agent else 'web'
        
        # 防止并发修改
        with client_count_lock:
            connected_clients[client_sid] = {
                'type': client_type,
                'ip': client_ip,
                'connect_time': time.time(),
                'user_agent': user_agent[:100]  # 限制长度
            }
        
        client_count = len(connected_clients)
        logger.info(f"Client connected - SID: {client_sid}, IP: {client_ip}, Type: {client_type}, Total: {client_count}")
        
        # 广播客户端数量更新
        socketio.emit('client_count', {'count': client_count})
        
        # 向新客户端发送欢迎消息和初始状态
        emit('server_message', {'message': f'Welcome! You are a {client_type} client.', 'type': 'info'})
        emit('resource_usage', resource_usage)
        
        # 发送初始IMU数据，如果有的话
        if len(imu_data_history) > 0:
            emit('imu_data', imu_data_history[-1])
    except Exception as e:
        logger.error(f"Error handling client connect: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """处理客户端断开连接事件"""
    try:
        client_sid = request.sid if request else 'unknown'
        
        # 防止并发修改
        with client_count_lock:
            if client_sid in connected_clients:
                client_info = connected_clients.pop(client_sid)
                client_type = client_info.get('type', 'unknown')
                client_ip = client_info.get('ip', 'unknown')
                connect_duration = time.time() - client_info.get('connect_time', time.time())
                
                logger.info(f"Client disconnected - SID: {client_sid}, IP: {client_ip}, Type: {client_type}, Duration: {connect_duration:.1f}s, Remaining: {len(connected_clients)}")
            else:
                logger.warning(f"Unknown client disconnected - SID: {client_sid}")
        
        # 更新并广播客户端数量
        client_count = len(connected_clients)
        socketio.emit('client_count', {'count': client_count})
        
        # 如果没有客户端连接，重置相机控制
        if client_count == 0:
            logger.info("No clients connected, resetting car and camera controls")
            handle_car_control({'x': 0, 'y': 0, 'client_type': 'system'})
            handle_gimbal_control({'x': 0, 'y': 0, 'client_type': 'system'})
    except Exception as e:
        logger.error(f"Error handling client disconnect: {e}")

@socketio.on_error_default
def default_error_handler(e):
    """Socket.IO默认错误处理器"""
    logger.error(f"Socket.IO error: {e}")
    # 不要断开连接，让客户端决定重连策略

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
    """处理启动SLAM的请求"""
    if not slam_initialized:
        logger.warning("SLAM未初始化，无法启动")
        return {'status': 'error', 'message': 'SLAM未初始化'}
    
    # 调用启动SLAM函数
    result = start_slam()
    
    # 广播SLAM状态
    socketio.emit('slam_status', {
        'active': slam_active,
        'initialized': slam_initialized
    })
    
    return result

@socketio.on('stop_slam')
def handle_stop_slam():
    """处理停止SLAM的请求"""
    if not slam_active:
        logger.warning("SLAM未运行，无法停止")
        return {'status': 'not_running'}
    
    # 调用停止SLAM函数
    result = stop_slam()
    
    # 广播SLAM状态
    socketio.emit('slam_status', {
        'active': slam_active,
        'initialized': slam_initialized
    })
    
    return result

@socketio.on('get_slam_status')
def handle_get_slam_status():
    """获取SLAM状态"""
    return {
        'active': slam_active,
        'initialized': slam_initialized
    }

@socketio.on('car_control')
def handle_car_control(data):
    """处理小车运动控制"""
    try:
        # 提取控制数据
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        client_type = str(data.get('client_type', 'unknown'))
        
        # 限制范围在-1到1之间
        x = max(min(x, 1.0), -1.0)
        y = max(min(y, 1.0), -1.0)
        
        # 为移动客户端翻转X轴
        if client_type == 'mobile':
            x = -x
        
        # 记录控制命令详情
        if (abs(x) > 0.05 or abs(y) > 0.05):  # 只记录有意义的控制命令
            logger.debug(f"小车控制: x={x:.2f}, y={y:.2f}, 来源={client_type}")
        
        # 发送到机器人，如果可用
        if robot_available:
            robot.run_car(x, y)
            
            # 当控制命令很小时停止机器人
            if abs(x) < 0.05 and abs(y) < 0.05:
                robot.t_stop(0)
        else:
            # 模拟模式下记录，但不实际控制
            if (abs(x) > 0.05 or abs(y) > 0.05):
                logger.debug(f"模拟模式小车控制: x={x:.2f}, y={y:.2f}")
            
        # 将原始控制数据广播到其他客户端，用于状态同步
        socketio.emit('car_control_update', {
            'x': x, 
            'y': y, 
            'client_type': client_type
        }, include_self=False)  # 排除发送控制命令的客户端
        
        return {'status': 'ok'}
    except Exception as e:
        logger.error(f"处理小车控制命令时出错: {e}")
        return {'status': 'error', 'message': str(e)}

@socketio.on('gimbal_control')
def handle_gimbal_control(data):
    """处理云台运动控制"""
    try:
        # 提取控制数据
        x = float(data.get('x', 0))
        y = float(data.get('y', 0))
        client_type = str(data.get('client_type', 'unknown'))
        
        # 限制范围在-1到1之间
        x = max(min(x, 1.0), -1.0)
        y = max(min(y, 1.0), -1.0)
        
        # 为移动客户端翻转X轴
        if client_type == 'mobile':
            x = -x
        
        # 记录控制命令详情
        if (abs(x) > 0.05 or abs(y) > 0.05):  # 只记录有意义的控制命令
            logger.debug(f"云台控制: x={x:.2f}, y={y:.2f}, 来源={client_type}")
        
        # 发送到机器人，如果可用
        if robot_available:
            # 根据当前实现调整
            # 这里假设run_gimbal方法会处理云台控制
            if hasattr(robot, 'run_gimbal'):
                robot.run_gimbal(x, y)
        else:
            # 模拟模式下记录，但不实际控制
            if (abs(x) > 0.05 or abs(y) > 0.05):
                logger.debug(f"模拟模式云台控制: x={x:.2f}, y={y:.2f}")
            
        # 将原始控制数据广播到其他客户端，用于状态同步
        socketio.emit('gimbal_control_update', {
            'x': x, 
            'y': y, 
            'client_type': client_type
        }, include_self=False)  # 排除发送控制命令的客户端
        
        return {'status': 'ok'}
    except Exception as e:
        logger.error(f"处理云台控制命令时出错: {e}")
        return {'status': 'error', 'message': str(e)}

# Initialize gimbal to default position on startup
if robot_available:
    try:
        robot.set_servo_angle(9, gimbal_h_angle)  # PWM9 for horizontal
        robot.set_servo_angle(10, gimbal_v_angle)  # PWM10 for vertical
        logger.info(f"Gimbal initialized to H:{gimbal_h_angle}°, V:{gimbal_v_angle}°")
    except Exception as e:
        logger.error(f"Failed to initialize gimbal: {e}")

@socketio.on('ping_request')
def handle_ping_request():
    # 立即响应ping请求，用于测量延迟
    emit('ping_response')

# 应用关闭时的清理
def shutdown_app():
    """在应用关闭时执行清理操作"""
    logger.info("开始执行应用关闭清理操作...")
    
    # 停止所有线程
    logger.info("正在停止所有线程...")
    thread_manager.stop_all_threads(timeout=3.0)
    
    # 特殊处理帧缓冲区
    global frame_buffer
    if frame_buffer and frame_buffer.running:
        logger.info("正在停止帧缓冲区...")
        frame_buffer.stop()
    
    # 释放相机资源
    global camera, camera_available
    if camera is not None:
        logger.info("正在释放相机资源...")
        try:
            camera.release()
            logger.info("相机资源已释放")
        except Exception as e:
            logger.error(f"释放相机资源时出错: {e}")
    camera_available = False
    
    # 停止SLAM系统
    global slam_active, slam_initialized
    if slam_active:
        logger.info("正在停止SLAM系统...")
        try:
            stop_slam()
            logger.info("SLAM系统已停止")
        except Exception as e:
            logger.error(f"停止SLAM系统时出错: {e}")
    slam_active = False
    slam_initialized = False
    
    # 停止机器人
    global robot_available
    if robot_available:
        logger.info("正在停止机器人...")
        try:
            robot.t_stop(0)
            logger.info("机器人已停止")
        except Exception as e:
            logger.error(f"停止机器人时出错: {e}")
    
    logger.info("应用清理完成，准备退出")

# 注册清理函数
import atexit
atexit.register(shutdown_app)

# SLAM相关函数
def start_slam():
    """启动SLAM系统"""
    global slam_process, slam_active, slam_initialized
    
    if slam_active:
        logger.warning("SLAM已经在运行中")
        return {'status': 'already_running'}
    
    try:
        # 检查依赖项
        try:
            import rclpy
            logger.info("ROS2 依赖项 rclpy 可用")
        except ImportError:
            logger.error("未安装ROS2依赖项 rclpy")
            return {'status': 'error', 'message': 'ROS2 依赖项未安装'}
        
        # 启动SLAM进程
        logger.info("正在启动SLAM系统...")
        
        # 使用子进程启动SLAM
        cmd = ["ros2", "launch", "rtabmap_examples", "rtabmap.launch.py", 
               "rgb_topic:=/image", "depth_topic:=/depth", "camera_info_topic:=/camera_info",
               "frame_id:=camera_link", "approx_sync:=false", "args:=--delete_db_on_start"]
        
        slam_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1
        )
        
        # 短暂等待确保进程启动
        time.sleep(3)
        
        # 检查进程是否仍在运行
        if slam_process.poll() is not None:
            # 进程已退出
            stdout, stderr = slam_process.communicate()
            logger.error(f"SLAM进程启动失败: {stderr}")
            return {'status': 'error', 'message': '启动SLAM失败'}
        
        logger.info("SLAM系统启动成功")
        slam_active = True
        slam_initialized = True
        
        return {'status': 'success'}
    
    except Exception as e:
        logger.error(f"启动SLAM时发生错误: {e}")
        return {'status': 'error', 'message': str(e)}

def stop_slam():
    """停止SLAM系统"""
    global slam_process, slam_active
    
    if not slam_active:
        logger.warning("SLAM未运行，无法停止")
        return {'status': 'not_running'}
    
    try:
        # 先尝试使用ROS2命令终止节点
        try:
            # 使用subprocess调用ROS2命令而不是直接导入rclpy
            subprocess.run(["ros2", "node", "kill", "/rtabmap"], timeout=5)
            logger.info("已使用ROS2命令终止SLAM节点")
        except Exception as ros_error:
            logger.warning(f"无法使用ROS2命令终止SLAM节点: {ros_error}")
        
        # 确保进程被终止
        if slam_process and slam_process.poll() is None:
            logger.info("正在终止SLAM进程...")
            slam_process.terminate()
            
            # 等待进程终止
            try:
                slam_process.wait(timeout=5)
                logger.info("SLAM进程已正常终止")
            except subprocess.TimeoutExpired:
                # 如果进程未能在超时内终止，强制结束它
                logger.warning("SLAM进程未响应终止命令，强制结束进程")
                slam_process.kill()
                slam_process.wait()
                logger.info("SLAM进程已强制终止")
        
        slam_process = None
        slam_active = False
        
        # 清理可能残留的ROS进程
        try:
            # 杀死所有rtabmap相关进程
            subprocess.run("pkill -f rtabmap", shell=True, timeout=3)
        except Exception as clean_error:
            logger.warning(f"清理残留进程时出错: {clean_error}")
        
        return {'status': 'success'}
    
    except Exception as e:
        logger.error(f"停止SLAM时发生错误: {e}")
        # 即使出错也设置为非活动状态，这样下次可以重新启动
        slam_active = False
        return {'status': 'error', 'message': str(e)}

if __name__ == '__main__':
    try:
        # 检查必要的依赖库
        missing_packages = []
        
        # 记录相机锁移除信息
        logger.info("注意: 相机资源锁(camera_lock)已被移除，提高视频流流畅性，可能影响SLAM性能")
        
        # 检查psutil
        if not psutil_available:
            missing_packages.append("psutil")
            logger.warning("psutil library not found. Resource monitoring will be disabled.")
        
        # 检查OpenCV
        try:
            cv2_version = cv2.__version__
            logger.info(f"Using OpenCV version: {cv2_version}")
        except:
            missing_packages.append("opencv-python")
            logger.error("OpenCV (cv2) not properly installed!")
        
        # 检查numpy
        try:
            np_version = np.__version__
            logger.info(f"Using NumPy version: {np_version}")
        except:
            missing_packages.append("numpy")
            logger.error("NumPy not properly installed!")
        
        # 如果有缺失的包，显示安装建议
        if missing_packages:
            install_cmd = "pip install " + " ".join(missing_packages)
            logger.warning(f"Missing required packages. Install with: {install_cmd}")
        
        # Create required directories if they don't exist
        os.makedirs('static', exist_ok=True)
        os.makedirs('templates', exist_ok=True)
        os.makedirs('slam', exist_ok=True)
        
        # 记录系统信息
        try:
            import platform
            system_info = platform.uname()
            logger.info(f"Running on: {system_info.system} {system_info.release}, Python {platform.python_version()}")
            
            # 如果psutil可用，获取更多系统信息
            if psutil_available:
                cpu_count = psutil.cpu_count(logical=False)
                cpu_logical = psutil.cpu_count(logical=True)
                memory = psutil.virtual_memory()
                logger.info(f"System resources: {cpu_count} physical CPU cores ({cpu_logical} logical), {memory.total/(1024*1024*1024):.1f}GB RAM")
        except:
            logger.info("Could not retrieve detailed system information")
        
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
        if resource_monitoring_active:
            resource_monitoring_active = False
            logger.info("Stopped resource monitoring during shutdown")
        logger.info("Server shutdown complete") 