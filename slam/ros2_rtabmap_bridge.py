#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS 2 Bridge for RTAB-Map integration
This module connects to ROS 2 and subscribes to RTAB-Map topics
"""

import rclpy
from rclpy.node import Node
import numpy as np
import threading
import time
import logging
import json
import base64
from io import BytesIO
import cv2
from PIL import Image

# ROS 2 message types
from rtabmap_msgs.msg import MapData
from sensor_msgs.msg import PointCloud2, Image as ROSImage
from geometry_msgs.msg import TransformStamped
from tf2_msgs.msg import TFMessage
from nav_msgs.msg import OccupancyGrid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ROS2RTABMapBridge(Node):
    """
    Bridge class for connecting to ROS 2 and subscribing to RTAB-Map topics
    """
    
    def __init__(self):
        """Initialize the ROS 2 node and subscribers"""
        super().__init__('rtabmap_bridge')
        
        # Initialize data storage
        self.map_data = None
        self.occupancy_grid = None
        self.camera_pose = np.eye(4)  # Identity matrix as initial pose
        self.point_cloud = None
        self.latest_image = None
        
        # Thread locks
        self.map_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.cloud_lock = threading.Lock()
        self.image_lock = threading.Lock()
        
        # Create subscribers
        self.map_data_sub = self.create_subscription(
            MapData,
            '/rtabmap/mapData',
            self.map_data_callback,
            10)
        
        self.occupancy_grid_sub = self.create_subscription(
            OccupancyGrid,
            '/rtabmap/grid_map',
            self.occupancy_grid_callback,
            10)
        
        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10)
        
        self.point_cloud_sub = self.create_subscription(
            PointCloud2,
            '/rtabmap/cloud_map',
            self.point_cloud_callback,
            10)
        
        self.image_sub = self.create_subscription(
            ROSImage,
            '/camera/image_raw',
            self.image_callback,
            10)
        
        logger.info("ROS 2 RTAB-Map bridge initialized")
    
    def map_data_callback(self, msg):
        """
        Callback for /rtabmap/mapData topic
        
        Args:
            msg (rtabmap_msgs.msg.MapData): Map data message
        """
        with self.map_lock:
            self.map_data = msg
        logger.debug("Received map data")
    
    def occupancy_grid_callback(self, msg):
        """
        Callback for /rtabmap/grid_map topic
        
        Args:
            msg (nav_msgs.msg.OccupancyGrid): Occupancy grid message
        """
        with self.map_lock:
            self.occupancy_grid = msg
        logger.debug("Received occupancy grid")
    
    def tf_callback(self, msg):
        """
        Callback for /tf topic to get camera pose
        
        Args:
            msg (tf2_msgs.msg.TFMessage): TF message
        """
        for transform in msg.transforms:
            # Look for the camera transform
            if transform.header.frame_id == "map" and transform.child_frame_id == "camera_link":
                with self.pose_lock:
                    # Convert ROS transform to 4x4 matrix
                    translation = transform.transform.translation
                    rotation = transform.transform.rotation
                    
                    # Create translation matrix
                    trans_matrix = np.array([
                        [1, 0, 0, translation.x],
                        [0, 1, 0, translation.y],
                        [0, 0, 1, translation.z],
                        [0, 0, 0, 1]
                    ])
                    
                    # Create rotation matrix from quaternion
                    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
                    rot_matrix = np.array([
                        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w), 0],
                        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w), 0],
                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y), 0],
                        [0, 0, 0, 1]
                    ])
                    
                    # Combine translation and rotation
                    self.camera_pose = np.matmul(trans_matrix, rot_matrix)
                
                logger.debug("Updated camera pose")
    
    def point_cloud_callback(self, msg):
        """
        Callback for /rtabmap/cloud_map topic
        
        Args:
            msg (sensor_msgs.msg.PointCloud2): Point cloud message
        """
        with self.cloud_lock:
            self.point_cloud = msg
        logger.debug("Received point cloud")
    
    def image_callback(self, msg):
        """
        Callback for /camera/image_raw topic
        
        Args:
            msg (sensor_msgs.msg.Image): Image message
        """
        with self.image_lock:
            self.latest_image = msg
        logger.debug("Received camera image")
    
    def get_camera_pose(self):
        """
        Get the current camera pose
        
        Returns:
            numpy.ndarray: 4x4 transformation matrix
        """
        with self.pose_lock:
            return self.camera_pose.copy()
    
    def get_2d_map(self):
        """
        Get the 2D occupancy grid map as an image
        
        Returns:
            numpy.ndarray: 2D occupancy grid map as an image
        """
        with self.map_lock:
            if self.occupancy_grid is None:
                # Return empty map if no data available
                return np.zeros((100, 100), dtype=np.uint8)
            
            # Convert occupancy grid to image
            width = self.occupancy_grid.info.width
            height = self.occupancy_grid.info.height
            grid_data = np.array(self.occupancy_grid.data).reshape(height, width)
            
            # Convert from -1 (unknown), 0 (free), 100 (occupied) to grayscale image
            # -1 -> 128 (gray), 0 -> 255 (white), 100 -> 0 (black)
            image = np.zeros((height, width), dtype=np.uint8)
            image[grid_data == -1] = 128  # Unknown -> gray
            image[grid_data == 0] = 255   # Free -> white
            image[grid_data == 100] = 0   # Occupied -> black
            
            return image
    
    def get_point_cloud_data(self):
        """
        Get the point cloud data in a format suitable for Three.js
        
        Returns:
            dict: Point cloud data with positions, colors, etc.
        """
        with self.cloud_lock:
            if self.point_cloud is None:
                return None
            
            # Extract point cloud data
            # This is a simplified version - in a real implementation,
            # you would need to parse the PointCloud2 message properly
            
            # For now, return a placeholder
            return {
                "positions": [],  # Will contain [x, y, z, x, y, z, ...]
                "colors": [],     # Will contain [r, g, b, r, g, b, ...]
                "count": 0
            }
    
    def get_camera_image(self):
        """
        Get the latest camera image as base64 encoded JPEG
        
        Returns:
            str: Base64 encoded JPEG image
        """
        with self.image_lock:
            if self.latest_image is None:
                return None
            
            # Convert ROS image to OpenCV format
            # This is a simplified version - in a real implementation,
            # you would use cv_bridge or similar
            
            # For now, return a placeholder
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', dummy_image)
            return base64.b64encode(buffer).decode('utf-8')

# Example usage
if __name__ == "__main__":
    rclpy.init()
    bridge = ROS2RTABMapBridge()
    
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    
    bridge.destroy_node()
    rclpy.shutdown() 