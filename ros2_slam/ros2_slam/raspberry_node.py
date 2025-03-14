#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 Node for Raspberry Pi 5
Collects camera and IMU data and sends to PC node

Requirements:
- rclpy
- cv_bridge
- OpenCV
- smbus2 (for MPU6050)
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import time
import logging
from threading import Thread
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Header
from geometry_msgs.msg import Vector3

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import IMU module, but allow running without it
try:
    import smbus2 as smbus
except ImportError:
    logger.warning("smbus2 not found. IMU functionality will be disabled.")
    smbus = None


class MPU6050:
    """Interface for MPU6050 IMU sensor"""
    def __init__(self, bus=1, address=0x68):
        """
        Initialize MPU6050 sensor
        
        Args:
            bus (int): I2C bus number
            address (int): I2C address of the MPU6050
        """
        if smbus is None:
            self.available = False
            return

        try:
            self.bus = smbus.SMBus(bus)
            self.address = address
            
            # Wake up the MPU6050
            self.bus.write_byte_data(self.address, 0x6B, 0)
            
            self.available = True
            logger.info("MPU6050 initialized")
        except Exception as e:
            logger.error(f"Failed to initialize MPU6050: {e}")
            self.available = False
    
    def read_raw_data(self, addr):
        """Read raw data from the MPU6050"""
        if not self.available:
            return 0
            
        try:
            high = self.bus.read_byte_data(self.address, addr)
            low = self.bus.read_byte_data(self.address, addr+1)
            
            # Concatenate higher and lower values
            value = ((high << 8) | low)
            
            # Get signed value
            if value > 32767:
                value = value - 65536
            
            return value
        except Exception as e:
            logger.error(f"Error reading MPU6050 data: {e}")
            return 0
    
    def get_data(self):
        """
        Get acceleration and gyroscope data
        
        Returns:
            tuple: (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp)
        """
        if not self.available:
            # Return zeros if IMU not available
            return (0, 0, 0, 0, 0, 0, 0)
            
        try:
            # Read accelerometer data
            accel_x = self.read_raw_data(0x3B) / 16384.0  # Full scale range ±2g
            accel_y = self.read_raw_data(0x3D) / 16384.0
            accel_z = self.read_raw_data(0x3F) / 16384.0
            
            # Read temperature
            temp = self.read_raw_data(0x41) / 340.0 + 36.53
            
            # Read gyroscope data
            gyro_x = self.read_raw_data(0x43) / 131.0  # Full scale range ±250°/s
            gyro_y = self.read_raw_data(0x45) / 131.0
            gyro_z = self.read_raw_data(0x47) / 131.0
            
            return (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, temp)
        except Exception as e:
            logger.error(f"Error getting MPU6050 data: {e}")
            return (0, 0, 0, 0, 0, 0, 0)


class RaspberryNode(Node):
    """
    ROS 2 Node for Raspberry Pi 5
    - Collects camera images and IMU data
    - Preprocesses data
    - Sends to PC node
    """
    
    def __init__(self):
        """Initialize the node"""
        super().__init__('raspberry_slam_node')
        
        # Initialize parameters
        self.declare_parameter('camera_id', 0)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)
        self.declare_parameter('camera_fps', 30)
        self.declare_parameter('imu_enabled', True)
        self.declare_parameter('compression_quality', 80)
        self.declare_parameter('image_publish_rate', 15.0)  # Hz
        self.declare_parameter('imu_publish_rate', 100.0)   # Hz
        
        # Get parameters
        self.camera_id = self.get_parameter('camera_id').value
        self.camera_width = self.get_parameter('camera_width').value
        self.camera_height = self.get_parameter('camera_height').value
        self.camera_fps = self.get_parameter('camera_fps').value
        self.imu_enabled = self.get_parameter('imu_enabled').value
        self.compression_quality = self.get_parameter('compression_quality').value
        self.image_publish_rate = self.get_parameter('image_publish_rate').value
        self.imu_publish_rate = self.get_parameter('imu_publish_rate').value
        
        # Initialize camera
        self.camera = None
        self.bridge = CvBridge()
        
        # Initialize IMU
        self.imu = None
        if self.imu_enabled:
            self.imu = MPU6050()
            
        # Create publishers
        self.image_pub = self.create_publisher(Image, 'camera/image', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/data', 10)
        
        # Create timer for publishing camera images
        self.image_timer = self.create_timer(1.0/self.image_publish_rate, self.publish_image)
        
        # Create timer for publishing IMU data
        if self.imu_enabled:
            self.imu_timer = self.create_timer(1.0/self.imu_publish_rate, self.publish_imu)
        
        # Start camera in a separate thread
        self.camera_thread = Thread(target=self.camera_thread_func)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        
        self.last_frame = None
        self.frame_lock = rclpy.lock.Lock()
        
        self.get_logger().info('Raspberry Pi SLAM node initialized')
    
    def camera_thread_func(self):
        """Thread function for camera capture"""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
            self.camera.set(cv2.CAP_PROP_FPS, self.camera_fps)
            
            if not self.camera.isOpened():
                self.get_logger().error('Failed to open camera')
                return
                
            self.get_logger().info('Camera started')
            
            while rclpy.ok():
                ret, frame = self.camera.read()
                if not ret:
                    self.get_logger().warning('Failed to capture frame')
                    time.sleep(0.1)
                    continue
                
                # Store the frame
                with self.frame_lock:
                    self.last_frame = frame
                
                time.sleep(0.01)  # Small sleep to prevent CPU overuse
                
        except Exception as e:
            self.get_logger().error(f'Camera thread error: {e}')
        finally:
            if self.camera is not None:
                self.camera.release()
                self.get_logger().info('Camera released')
    
    def publish_image(self):
        """Publish the latest camera image"""
        with self.frame_lock:
            if self.last_frame is None:
                return
            
            frame = self.last_frame.copy()
        
        try:
            # Preprocess image (resize, compress, etc.)
            # This is a simple compression using JPEG encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.compression_quality]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            compressed_frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            
            # Convert to ROS message
            msg = self.bridge.cv2_to_imgmsg(compressed_frame, encoding="bgr8")
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "camera_frame"
            
            # Publish
            self.image_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing image: {e}')
    
    def publish_imu(self):
        """Publish the latest IMU data"""
        if self.imu is None or not self.imu.available:
            return
            
        try:
            # Get IMU data
            accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, _ = self.imu.get_data()
            
            # Create IMU message
            msg = Imu()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "imu_frame"
            
            # Set linear acceleration (in m/s^2)
            msg.linear_acceleration.x = accel_x * 9.81
            msg.linear_acceleration.y = accel_y * 9.81
            msg.linear_acceleration.z = accel_z * 9.81
            
            # Set angular velocity (in rad/s)
            msg.angular_velocity.x = gyro_x * (np.pi / 180.0)
            msg.angular_velocity.y = gyro_y * (np.pi / 180.0)
            msg.angular_velocity.z = gyro_z * (np.pi / 180.0)
            
            # Set covariance matrices (example values, should be calibrated properly)
            accel_cov = 0.01
            gyro_cov = 0.001
            
            # Identity covariance with variance on diagonal
            for i in range(3):
                msg.linear_acceleration_covariance[i*3+i] = accel_cov
                msg.angular_velocity_covariance[i*3+i] = gyro_cov
            
            # Publish
            self.imu_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing IMU data: {e}')
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.camera is not None:
            self.camera.release()
        super().destroy_node()


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    node = RaspberryNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 