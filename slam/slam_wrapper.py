#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLAM Wrapper for RTAB-Map integration with ROS 2
This module provides a wrapper for RTAB-Map SLAM using ROS 2
"""

import numpy as np
import cv2
import logging
import threading
import time
import json
import base64
import subprocess
import os
import signal
import sys

# Import ROS 2 bridge if available
try:
    import rclpy
    from slam.ros2_rtabmap_bridge import ROS2RTABMapBridge
    ros2_available = True
except ImportError:
    ros2_available = False
    print("ROS 2 or RTAB-Map bridge not available, falling back to simulation mode")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SLAMWrapper:
    """
    Wrapper class for RTAB-Map SLAM integration with ROS 2
    """
    
    def __init__(self, config_file=None, use_mpu6050=False):
        """
        Initialize the SLAM system
        
        Args:
            config_file (str): Path to the RTAB-Map configuration file
            use_mpu6050 (bool): Whether to use MPU6050 IMU data
        """
        self.is_running = False
        self.slam_thread = None
        self.config_file = config_file
        self.use_mpu6050 = use_mpu6050
        self.current_pose = np.eye(4)  # Identity matrix as initial pose
        self.map_points = []
        self.map_lock = threading.Lock()
        self.point_cloud_data = None
        self.ros2_process = None
        self.rtabmap_process = None
        self.bridge = None
        
        # Check if ROS 2 is available
        self.ros2_available = ros2_available
        
        logger.info(f"SLAM wrapper initialized (ROS 2 available: {self.ros2_available})")
    
    def start(self):
        """Start the SLAM system"""
        if self.is_running:
            logger.warning("SLAM system is already running")
            return False
        
        try:
            if self.ros2_available:
                # Start ROS 2 and RTAB-Map processes
                self._start_ros2_rtabmap()
                
                # Initialize ROS 2 node and bridge
                if not rclpy.ok():
                    rclpy.init()
                self.bridge = ROS2RTABMapBridge()
            
            # Start SLAM thread
            self.is_running = True
            self.slam_thread = threading.Thread(target=self._slam_thread_func)
            self.slam_thread.daemon = True
            self.slam_thread.start()
            
            logger.info("SLAM system started")
            return True
        except Exception as e:
            logger.error(f"Failed to start SLAM system: {e}")
            self.is_running = False
            self._cleanup_processes()
            return False
    
    def stop(self):
        """Stop the SLAM system"""
        if not self.is_running:
            logger.warning("SLAM system is not running")
            return False
        
        try:
            # Stop SLAM thread
            self.is_running = False
            if self.slam_thread:
                self.slam_thread.join(timeout=2.0)
            
            # Clean up ROS 2 and RTAB-Map processes
            self._cleanup_processes()
            
            logger.info("SLAM system stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop SLAM system: {e}")
            return False
    
    def process_frame(self, frame, imu_data=None):
        """
        Process a new camera frame and optional IMU data
        
        Args:
            frame (numpy.ndarray): Camera frame (BGR format)
            imu_data (dict): Optional IMU data from MPU6050
                Expected format: {
                    "accel": [x, y, z],  # in g (9.81 m/s²)
                    "gyro": [x, y, z]    # in degrees/second
                }
            
        Returns:
            bool: True if frame was successfully processed
        """
        if not self.is_running:
            logger.warning("Cannot process frame: SLAM system is not running")
            return False
        
        try:
            if self.ros2_available and self.bridge:
                # In ROS 2 mode, the frame is processed by RTAB-Map directly
                # We just update our local data from the bridge
                self.current_pose = self.bridge.get_camera_pose()
                
                # Get point cloud data for 3D visualization
                cloud_data = self.bridge.get_point_cloud_data()
                if cloud_data:
                    self.point_cloud_data = cloud_data
                
                # If IMU data is provided, log it for debugging
                if imu_data:
                    logger.debug(f"Using calibrated IMU data: accel={imu_data['accel']}, gyro={imu_data['gyro']}")
                
                return True
            else:
                # Fallback to simulation mode
                # Simulate some processing delay
                time.sleep(0.01)
                
                # If IMU data is provided, use it to improve pose estimation
                if imu_data:
                    # In a real implementation, this would use the IMU data to improve pose estimation
                    # For simulation, we'll just use it to add some realistic noise
                    accel = imu_data["accel"]
                    gyro = imu_data["gyro"]
                    
                    # Use gyro data to update rotation (simplified)
                    # In a real implementation, this would use proper quaternion math
                    gyro_scale = 0.01  # Scale factor for simulation
                    delta_rotation = np.array([
                        gyro[0] * gyro_scale,
                        gyro[1] * gyro_scale,
                        gyro[2] * gyro_scale
                    ])
                    
                    # Use accel data to update position (simplified)
                    # In a real implementation, this would integrate acceleration properly
                    accel_scale = 0.005  # Scale factor for simulation
                    delta_position = np.array([
                        accel[0] * accel_scale,
                        accel[1] * accel_scale,
                        accel[2] * accel_scale
                    ])
                    
                    # Update pose with IMU-based movement
                    self.current_pose[0:3, 3] += delta_position
                    
                    logger.debug(f"Updated pose using IMU data: delta_pos={delta_position}, delta_rot={delta_rotation}")
                else:
                    # Without IMU, just add random movement
                    delta = np.random.normal(0, 0.01, 3)
                    self.current_pose[0:3, 3] += delta
                
                # Generate some random map points (for simulation)
                if len(self.map_points) < 1000 and np.random.random() < 0.1:
                    new_points = np.random.normal(0, 1, (10, 3))
                    with self.map_lock:
                        self.map_points.extend(new_points)
                
                # Simulate point cloud data
                self._simulate_point_cloud()
                
                return True
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return False
    
    def get_current_pose(self):
        """
        Get the current camera pose
        
        Returns:
            numpy.ndarray: 4x4 transformation matrix
        """
        return self.current_pose.copy()
    
    def get_map_points(self):
        """
        Get the current map points
        
        Returns:
            list: List of 3D points (x, y, z)
        """
        if self.ros2_available and self.bridge:
            # In ROS 2 mode, we don't use this method
            # Return empty list for compatibility
            return []
        else:
            with self.map_lock:
                return self.map_points.copy()
    
    def get_2d_map(self, resolution=0.05, size=10.0):
        """
        Generate a 2D occupancy grid map
        
        Args:
            resolution (float): Map resolution in meters per pixel
            size (float): Map size in meters
            
        Returns:
            numpy.ndarray: 2D occupancy grid map
        """
        if self.ros2_available and self.bridge:
            # Get 2D map from ROS 2 bridge
            return self.bridge.get_2d_map()
        else:
            # Fallback to simulation mode
            # Calculate map dimensions
            dim = int(size / resolution)
            grid_map = np.zeros((dim, dim), dtype=np.uint8)
            
            # Get map points
            points = self.get_map_points()
            
            # Project points to 2D and mark on grid
            if points:
                for point in points:
                    # Convert 3D point to grid coordinates
                    x = int((point[0] + size/2) / resolution)
                    y = int((point[1] + size/2) / resolution)
                    
                    # Check if within grid bounds
                    if 0 <= x < dim and 0 <= y < dim:
                        grid_map[y, x] = 255
                
                # Apply dilation to make points more visible
                kernel = np.ones((3, 3), np.uint8)
                grid_map = cv2.dilate(grid_map, kernel, iterations=1)
            
            return grid_map
    
    def get_3d_map_data(self):
        """
        Get 3D map data for visualization
        
        Returns:
            dict: Point cloud data in a format suitable for Three.js
        """
        if self.point_cloud_data:
            return self.point_cloud_data
        else:
            # Return empty point cloud
            return {
                "positions": [],
                "colors": [],
                "count": 0
            }
    
    def _start_ros2_rtabmap(self):
        """Start ROS 2 and RTAB-Map processes"""
        if not self.ros2_available:
            return
        
        try:
            # Start ROS 2 daemon if not already running
            try:
                subprocess.run(["ros2", "daemon", "status"], 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE)
                logger.info("ROS 2 daemon is already running")
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.info("Starting ROS 2 daemon")
                subprocess.Popen(["ros2", "daemon", "start"], 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE)
                time.sleep(2)  # Give daemon time to start
            
            # Start RTAB-Map with ROS 2
            cmd = ["ros2", "launch", "rtabmap_launch", "rtabmap.launch.py"]
            
            # Add parameters
            cmd.extend(["use_sim_time:=false"])
            
            if self.use_mpu6050:
                # 使用MPU6050 IMU数据
                cmd.extend([
                    "imu_topic:=/imu/data",
                    # 设置IMU参数，提高SLAM性能
                    "imu_linear_variance:=0.001",  # 加速度计方差
                    "imu_angular_variance:=0.001",  # 陀螺仪方差
                    "wait_imu_to_init:=true",      # 等待IMU初始化
                    "imu_filter_angular_velocity:=true"  # 过滤角速度
                ])
            
            # 添加RTAB-Map参数，提高SLAM性能
            cmd.extend([
                "rtabmap_args:=\"--delete_db_on_start\"",  # 每次启动时删除数据库
                "depth_topic:=/camera/aligned_depth_to_color/image_raw",  # 深度图像话题
                "rgb_topic:=/camera/color/image_raw",      # RGB图像话题
                "camera_info_topic:=/camera/color/camera_info",  # 相机信息话题
                "approx_sync:=true",                       # 使用近似同步
                "queue_size:=10",                          # 队列大小
                "frame_rate:=15.0",                        # 帧率
                "Vis/MinInliers:=10",                      # 最小内点数
                "Vis/RoiRatios:=\"0.03 0.03 0.03 0.03\"", # ROI比例
                "Vis/MaxDepth:=5.0",                       # 最大深度
                "RGBD/NeighborLinkRefining:=true",         # 邻居链接精炼
                "RGBD/ProximityBySpace:=true",             # 空间邻近性
                "RGBD/AngularUpdate:=0.05",                # 角度更新阈值
                "RGBD/LinearUpdate:=0.05",                 # 线性更新阈值
                "Reg/Strategy:=1",                         # 注册策略
                "Reg/Force3DoF:=false",                    # 强制3DoF
                "Grid/CellSize:=0.05",                     # 栅格单元大小
                "Grid/RangeMax:=5.0"                       # 最大范围
            ])
            
            # Start RTAB-Map process
            logger.info(f"Starting RTAB-Map with command: {' '.join(cmd)}")
            self.rtabmap_process = subprocess.Popen(cmd, 
                                                   stdout=subprocess.PIPE, 
                                                   stderr=subprocess.PIPE)
            
            # Wait for RTAB-Map to initialize
            time.sleep(5)
            
            logger.info("RTAB-Map started successfully")
        except Exception as e:
            logger.error(f"Failed to start ROS 2 and RTAB-Map: {e}")
            self._cleanup_processes()
            raise
    
    def _cleanup_processes(self):
        """Clean up ROS 2 and RTAB-Map processes"""
        # Stop RTAB-Map process
        if self.rtabmap_process:
            logger.info("Stopping RTAB-Map process")
            try:
                self.rtabmap_process.terminate()
                self.rtabmap_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                self.rtabmap_process.kill()
            self.rtabmap_process = None
        
        # Clean up ROS 2 bridge
        if self.bridge:
            logger.info("Cleaning up ROS 2 bridge")
            self.bridge.destroy_node()
            self.bridge = None
        
        # Shutdown ROS 2 if we initialized it
        if rclpy.ok():
            logger.info("Shutting down ROS 2")
            rclpy.shutdown()
    
    def _simulate_point_cloud(self):
        """Simulate point cloud data for testing"""
        # Create a simple point cloud for visualization testing
        positions = []
        colors = []
        
        # Generate random points
        num_points = 1000
        for _ in range(num_points):
            # Random position
            x = np.random.normal(0, 1)
            y = np.random.normal(0, 1)
            z = np.random.normal(0, 1)
            positions.extend([x, y, z])
            
            # Random color
            r = np.random.random()
            g = np.random.random()
            b = np.random.random()
            colors.extend([r, g, b])
        
        self.point_cloud_data = {
            "positions": positions,
            "colors": colors,
            "count": num_points
        }
    
    def _slam_thread_func(self):
        """Background thread function for SLAM processing"""
        logger.info("SLAM processing thread started")
        
        while self.is_running:
            if self.ros2_available and self.bridge:
                # In ROS 2 mode, we need to spin the ROS 2 node
                rclpy.spin_once(self.bridge, timeout_sec=0.1)
            else:
                # In simulation mode, just sleep
                time.sleep(0.1)
        
        logger.info("SLAM processing thread stopped")


# Example usage
if __name__ == "__main__":
    # Create SLAM wrapper
    slam = SLAMWrapper(use_mpu6050=True)
    
    # Start SLAM
    slam.start()
    
    try:
        # Simulate processing frames
        for i in range(100):
            # Create a dummy frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Simulate IMU data
            imu_data = {
                "accel": [0.0, 0.0, 9.81],  # m/s^2
                "gyro": [0.0, 0.0, 0.0]     # rad/s
            }
            
            # Process frame
            slam.process_frame(frame, imu_data)
            
            # Get current pose and map
            pose = slam.get_current_pose()
            map_2d = slam.get_2d_map()
            map_3d = slam.get_3d_map_data()
            
            print(f"Frame {i}: Pose translation = {pose[0:3, 3]}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop SLAM
        slam.stop() 