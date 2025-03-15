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
app.config['SECRET_KEY'] = 'smartcar2023'

# 优化Socket.IO配置
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',  # 使用threading模式增强性能
    ping_timeout=60,         # 增加ping超时到60秒
    ping_interval=25,        # 增加ping间隔到25秒
    max_http_buffer_size=10 * 1024 * 1024
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
        logger.info("Initializing camera with optimized settings...")
        camera = cv2.VideoCapture(0)
        
        # 尝试设置为MJPG格式提高效率
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 增加分辨率，有助于SLAM特征检测
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        
        # 获取实际相机设置
        actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = camera.get(cv2.CAP_PROP_FPS)
        actual_format = camera.get(cv2.CAP_PROP_FOURCC)
        format_str = chr(int(actual_format) & 0xFF) + chr((int(actual_format) >> 8) & 0xFF) + chr((int(actual_format) >> 16) & 0xFF) + chr((int(actual_format) >> 24) & 0xFF)
        
        logger.info(f"Camera settings - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}, Format: {format_str}")
        
        # 检查相机是否正常工作
        ret, frame = camera.read()
        if ret:
            camera_available = True
            # 检查帧质量
            if frame is None or frame.size == 0:
                logger.error("Camera connected but returned empty frame")
                camera_available = False
            else:
                logger.info(f"Camera initialized successfully: frame shape {frame.shape}")
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
    global is_streaming, camera_available, camera
    
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
            # 直接读取相机帧，不使用锁
            read_start = time.time()
            success, frame = camera.read()
            read_time = time.time() - read_start
            
            if read_time > 0.1:  # 如果读取时间过长，记录日志
                logger.warning(f"Video stream: Camera read took {read_time:.3f}s, which is slow")
            
            if not success:
                error_count += 1
                logger.error(f"Failed to read frame from camera (attempt {error_count})")
                if error_count > 5:
                    logger.error("Too many consecutive frame read failures, checking camera...")
                    # 检查相机状态的代码保持不变...
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
                                camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                                camera.set(cv2.CAP_PROP_FPS, TARGET_FPS)
                                
                                # 获取实际相机设置
                                actual_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
                                actual_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
                                actual_fps = camera.get(cv2.CAP_PROP_FPS)
                                actual_format = camera.get(cv2.CAP_PROP_FOURCC)
                                
                                logger.info(f"Reopened camera settings - Width: {actual_width}, Height: {actual_height}, FPS: {actual_fps}, Format: {chr(int(actual_format) & 0xFF)}{chr((int(actual_format) >> 8) & 0xFF)}{chr((int(actual_format) >> 16) & 0xFF)}{chr((int(actual_format) >> 24) & 0xFF)}")
                                
                                ret_test, _ = camera.read()
                                if ret_test:
                                    logger.info("Camera successfully reopened")
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
            
            # 调整大小保持原始分辨率
            # frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            
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
    
    logger.info(f"Video streaming stopped after {frame_count} frames")

# 添加资源监控线程函数
def resource_monitoring_thread():
    global resource_monitoring_active
    
    logger.info("Resource monitoring thread started")
    
    while resource_monitoring_active and psutil_available:
        try:
            # 获取系统资源使用情况
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            
            # 获取硬盘使用情况
            try:
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_used_gb = disk.used / (1024 * 1024 * 1024)
                disk_total_gb = disk.total / (1024 * 1024 * 1024)
            except:
                disk_percent = 0
                disk_used_gb = 0
                disk_total_gb = 0
            
            # 获取网络IO统计
            try:
                net_io = psutil.net_io_counters()
                net_sent_mb = net_io.bytes_sent / (1024 * 1024)
                net_recv_mb = net_io.bytes_recv / (1024 * 1024)
            except:
                net_sent_mb = 0
                net_recv_mb = 0
            
            # 获取进程信息
            process = psutil.Process(os.getpid())
            process_cpu = process.cpu_percent(interval=0.5)
            process_memory = process.memory_info().rss / (1024 * 1024)  # 转换为MB
            
            # 创建资源数据字典
            resource_data = {
                'cpu': {
                    'percent': round(cpu_percent, 1),
                    'process_percent': round(process_cpu, 1)
                },
                'memory': {
                    'percent': round(memory_percent, 1),
                    'used_mb': round(memory_used_mb, 1),
                    'total_mb': round(memory_total_mb, 1),
                    'process_mb': round(process_memory, 1)
                },
                'disk': {
                    'percent': round(disk_percent, 1),
                    'used_gb': round(disk_used_gb, 1),
                    'total_gb': round(disk_total_gb, 1)
                },
                'network': {
                    'sent_mb': round(net_sent_mb, 1),
                    'recv_mb': round(net_recv_mb, 1)
                },
                'timestamp': time.time()
            }
            
            # 发送到客户端
            socketio.emit('resource_update', resource_data)
            
            # 记录到日志（仅在资源使用较高时记录，避免日志过多）
            if cpu_percent > 70 or memory_percent > 70:
                logger.warning(f"High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Process CPU: {process_cpu}%, Process Memory: {process_memory:.1f}MB")
            else:
                logger.debug(f"Resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%, Process CPU: {process_cpu}%, Process Memory: {process_memory:.1f}MB")
                
            # 间隔3秒发送一次
            time.sleep(3)
            
        except Exception as e:
            logger.error(f"Error in resource monitoring: {e}")
            time.sleep(5)  # 出错时延长等待时间
    
    logger.info("Resource monitoring thread stopped")

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    global mpu_running, mpu_thread, is_streaming, streaming_thread, resource_monitoring_active, resource_monitor_thread
    
    client_id = request.sid
    logger.info(f"Client connected: {client_id}")
    
    # 发送状态更新给新连接的客户端
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
        mpu_thread = threading.Thread(target=mpu6050_data_thread)
        mpu_thread.daemon = True
        mpu_thread.start()
        logger.info("MPU6050 data thread started for new client")
    
    # 启动资源监控线程（如果尚未运行且psutil可用）
    if psutil_available and not resource_monitoring_active:
        resource_monitoring_active = True
        resource_monitor_thread = threading.Thread(target=resource_monitoring_thread)
        resource_monitor_thread.daemon = True
        resource_monitor_thread.start()
        logger.info("Resource monitoring thread started")
    
    # 自动启动视频流，无需客户端手动点击开始按钮
    if camera_available:
        if not is_streaming:
            try:
                # 如果没有活跃的视频流，启动一个新的
                is_streaming = True
                streaming_thread = threading.Thread(target=generate_frames)
                streaming_thread.daemon = True
                streaming_thread.start()
                emit('stream_status', {'status': 'started'})
                logger.info(f"Video streaming automatically started for client: {client_id}")
            except Exception as e:
                logger.error(f"Error starting video stream: {e}")
                emit('stream_status', {'status': 'error', 'message': str(e)})
        else:
            # 如果视频流已经在运行，只需向此客户端发送状态通知
            emit('stream_status', {'status': 'started'})
            logger.info(f"Existing video stream linked to new client: {client_id}")

@socketio.on('disconnect')
def handle_disconnect(sid=None):
    global mpu_running, is_streaming, resource_monitoring_active, mpu_thread
    
    client_id = request.sid
    logger.info(f"Client disconnected: {client_id}")
    
    # 停止机器人运动
    if robot_available:
        robot.t_stop(0)
        logger.info(f"Robot stopped due to client disconnect: {client_id}")
    
    # 获取实际的活跃连接数量
    active_clients = len(socketio.server.eio.sockets)
    logger.info(f"Remaining active clients: {active_clients}")
    
    # 检查是否需要停止MPU6050线程
    if active_clients <= 1 and mpu_running:  # 仅剩服务器自身或无连接
        mpu_running = False
        # 等待MPU6050线程结束
        if mpu_thread and mpu_thread.is_alive():
            logger.info("Waiting for MPU6050 thread to finish...")
            mpu_thread.join(timeout=2.0)  # 最多等待2秒
            if mpu_thread.is_alive():
                logger.warning("MPU6050 thread did not terminate within timeout")
            else:
                logger.info("MPU6050 thread terminated successfully")
        mpu_thread = None
        logger.info("MPU6050 data thread stopped - no clients left")
    
    # 检查是否需要停止资源监控
    if active_clients <= 1 and resource_monitoring_active:
        resource_monitoring_active = False
        logger.info("Stopping resource monitoring - no clients left")
    
    # 如果没有客户端了，同时停止视频流
    if active_clients <= 1 and is_streaming:
        logger.info("Stopping video stream - no clients left")
        is_streaming = False

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
    
    logger.info(f"Start SLAM request from client: {request.sid}")
    
    if not slam_active and slam_initialized and camera_available:
        try:
            # 检查系统资源是否充足
            if psutil_available:
                try:
                    cpu_percent = psutil.cpu_percent()
                    memory = psutil.virtual_memory()
                    memory_percent = memory.percent
                    
                    if cpu_percent > 80:
                        logger.warning(f"Starting SLAM with high CPU usage: {cpu_percent}%")
                    if memory_percent > 80:
                        logger.warning(f"Starting SLAM with high memory usage: {memory_percent}%")
                except Exception as e:
                    logger.info(f"Unable to check system resources before starting SLAM: {e}")
            
            # 尝试启动SLAM
            logger.info("Preparing to start SLAM...")
            if slam.start():
                slam_active = True
                # 创建并启动SLAM线程
                slam_thread = threading.Thread(target=slam_processing_thread)
                slam_thread.daemon = True
                
                # 不再需要延迟，因为已移除相机锁
                # time.sleep(0.5)
                
                slam_thread.start()
                emit('slam_status', {'status': 'started'})
                logger.info("SLAM started")
            else:
                emit('slam_status', {'status': 'error', 'message': 'Failed to start SLAM'})
                logger.error("Failed to start SLAM")
        except Exception as e:
            emit('slam_status', {'status': 'error', 'message': str(e)})
            logger.error(f"Error starting SLAM: {e}")
    else:
        if slam_active:
            logger.info("SLAM already active")
            emit('slam_status', {'status': 'already_active', 'message': 'SLAM is already running'})
        elif not slam_initialized:
            logger.error("SLAM not initialized, cannot start")
            emit('slam_status', {'status': 'error', 'message': 'SLAM not available'})
        elif not camera_available:
            logger.error("Camera not available, cannot start SLAM")
            emit('slam_status', {'status': 'error', 'message': 'Camera not available'})
        else:
            emit('slam_status', {
                'status': 'error', 
                'message': 'Unknown error starting SLAM'
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
        client_type = data.get('client_type', '')  # 'mobile' or 'web'
        
        # 如果是移动端，反转X轴以纠正左右方向
        if client_type == 'mobile':
            x = -x
        
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
        client_type = data.get('client_type', '')  # 'mobile' or 'web'
        
        # 反转Y轴值以纠正上下方向（对所有客户端）
        y = -y
        
        # 如果是移动端，反转X轴以纠正左右方向
        if client_type == 'mobile':
            x = -x
        
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

@socketio.on('ping_request')
def handle_ping_request():
    # 立即响应ping请求，用于测量延迟
    emit('ping_response')

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