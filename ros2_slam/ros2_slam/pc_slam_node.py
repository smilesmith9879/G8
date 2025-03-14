#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 Node for PC
Processes camera and IMU data from Raspberry Pi
Runs ORB-SLAM3 algorithm
Publishes pose and map data
"""

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import threading
import os
import time
import logging
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from std_msgs.msg import Header

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import SLAM wrapper
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
try:
    from slam.slam_wrapper import SLAMWrapper
    SLAM_AVAILABLE = True
except ImportError:
    logger.warning("SLAM wrapper not found. Using simulation mode.")
    SLAM_AVAILABLE = False
    
# Import SLAM simulator
from .slam_simulator import SLAMSimulator


class PCNode(Node):
    """
    ROS 2 Node for PC
    - Receives camera images and IMU data from Raspberry Pi
    - Processes data with ORB-SLAM3
    - Publishes pose and map data
    """
    
    def __init__(self):
        """Initialize the node"""
        super().__init__('pc_slam_node')
        
        # Initialize parameters
        self.declare_parameter('slam_config_file', '')
        self.declare_parameter('map_publish_rate', 1.0)  # Hz
        self.declare_parameter('map_resolution', 0.05)   # meters per pixel
        self.declare_parameter('map_size', 10.0)        # meters
        
        # Get parameters
        self.slam_config_file = self.get_parameter('slam_config_file').value
        self.map_publish_rate = self.get_parameter('map_publish_rate').value
        self.map_resolution = self.get_parameter('map_resolution').value
        self.map_size = self.get_parameter('map_size').value
        
        # Initialize SLAM
        if SLAM_AVAILABLE:
            self.slam = SLAMWrapper(config_file=self.slam_config_file)
            self.slam.start()
        else:
            # Create dummy SLAM functionality for testing
            self.slam = SLAMSimulator()
            self.slam.start()
        
        # Locks
        self.slam_lock = threading.Lock()
        self.camera_data_ready = False
        self.imu_data_ready = False
        
        # Initialize bridge for OpenCV<->ROS conversion
        self.bridge = CvBridge()
        
        # Create subscribers
        self.image_sub = self.create_subscription(
            Image,
            'camera/image',
            self.image_callback,
            10
        )
        
        self.imu_sub = self.create_subscription(
            Imu,
            'imu/data',
            self.imu_callback,
            10
        )
        
        # Create publishers
        self.pose_pub = self.create_publisher(
            PoseStamped,
            'slam/pose',
            10
        )
        
        self.map_pub = self.create_publisher(
            OccupancyGrid,
            'slam/map',
            10
        )
        
        # Setup TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Create timer for publishing map
        self.map_timer = self.create_timer(1.0/self.map_publish_rate, self.publish_map)
        
        self.get_logger().info('PC SLAM node initialized')
    
    def image_callback(self, msg):
        """
        Callback for processing camera images
        
        Args:
            msg (Image): ROS Image message
        """
        try:
            # Convert ROS image to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Process image with SLAM
            with self.slam_lock:
                self.slam.process_frame(cv_image)
                self.camera_data_ready = True
            
            # Get and publish pose
            self.publish_pose(msg.header.stamp)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
    
    def imu_callback(self, msg):
        """
        Callback for processing IMU data
        
        Args:
            msg (Imu): ROS IMU message
        """
        try:
            # For now, we're not using IMU data in the SLAM wrapper
            # When ORB-SLAM3 bindings with IMU support are implemented,
            # this would pass the IMU data to the SLAM system
            
            self.imu_data_ready = True
            
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {e}')
    
    def publish_pose(self, timestamp):
        """
        Publish the current camera pose and transform
        
        Args:
            timestamp: Timestamp from the camera image
        """
        try:
            # Get current pose from SLAM
            with self.slam_lock:
                pose_matrix = self.slam.get_current_pose()
            
            # Create pose message
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = timestamp
            pose_msg.header.frame_id = "map"
            
            # Extract position from pose matrix
            position = pose_matrix[0:3, 3]
            pose_msg.pose.position.x = position[0]
            pose_msg.pose.position.y = position[1]
            pose_msg.pose.position.z = position[2]
            
            # Extract orientation (convert rotation matrix to quaternion)
            rotation_matrix = pose_matrix[0:3, 0:3]
            qw, qx, qy, qz = self.rotation_matrix_to_quaternion(rotation_matrix)
            
            pose_msg.pose.orientation.w = qw
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            
            # Publish pose
            self.pose_pub.publish(pose_msg)
            
            # Publish transform
            transform_msg = TransformStamped()
            transform_msg.header = pose_msg.header
            transform_msg.child_frame_id = "camera_frame"
            
            transform_msg.transform.translation.x = position[0]
            transform_msg.transform.translation.y = position[1]
            transform_msg.transform.translation.z = position[2]
            
            transform_msg.transform.rotation.w = qw
            transform_msg.transform.rotation.x = qx
            transform_msg.transform.rotation.y = qy
            transform_msg.transform.rotation.z = qz
            
            # Broadcast transform
            self.tf_broadcaster.sendTransform(transform_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing pose: {e}')
    
    def publish_map(self):
        """Publish the current 2D occupancy grid map"""
        try:
            # Get 2D map from SLAM
            with self.slam_lock:
                grid_map = self.slam.get_2d_map(
                    resolution=self.map_resolution,
                    size=self.map_size
                )
            
            if grid_map is None:
                return
                
            # Create occupancy grid message
            map_msg = OccupancyGrid()
            map_msg.header = Header()
            map_msg.header.stamp = self.get_clock().now().to_msg()
            map_msg.header.frame_id = "map"
            
            # Set map metadata
            map_msg.info.resolution = self.map_resolution
            map_msg.info.width = grid_map.shape[1]
            map_msg.info.height = grid_map.shape[0]
            map_msg.info.origin.position.x = -self.map_size / 2
            map_msg.info.origin.position.y = -self.map_size / 2
            map_msg.info.origin.position.z = 0.0
            map_msg.info.origin.orientation.w = 1.0
            
            # Convert grid map to occupancy grid data
            # OpenCV uses 0-255 values, ROS OccupancyGrid uses -1 (unknown) or 0-100 (probability)
            data = []
            for i in range(grid_map.shape[0]):
                for j in range(grid_map.shape[1]):
                    value = grid_map[i, j]
                    if value == 0:
                        data.append(0)  # Free
                    else:
                        occupancy = int(value / 255.0 * 100)
                        data.append(occupancy)  # Occupied (0-100)
            
            map_msg.data = data
            
            # Publish map
            self.map_pub.publish(map_msg)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing map: {e}')
    
    def rotation_matrix_to_quaternion(self, R):
        """
        Convert a rotation matrix to a quaternion
        
        Args:
            R (numpy.ndarray): 3x3 rotation matrix
            
        Returns:
            tuple: Quaternion as (w, x, y, z)
        """
        # Check if rotation matrix is valid
        if R.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
            
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (R[2, 1] - R[1, 2]) / S
            qy = (R[0, 2] - R[2, 0]) / S
            qz = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            
        return qw, qx, qy, qz
    
    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.slam is not None:
            self.slam.stop()
        super().destroy_node()


def main(args=None):
    """Main function"""
    rclpy.init(args=args)
    node = PCNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 