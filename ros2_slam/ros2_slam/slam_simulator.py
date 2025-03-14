#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SLAM Simulator for testing without ORB-SLAM3
"""

import numpy as np
import cv2
import threading
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SLAMSimulator:
    """
    Simulates SLAM functionality for testing without ORB-SLAM3
    """
    
    def __init__(self):
        """Initialize the simulator"""
        self.is_running = False
        self.current_pose = np.eye(4)  # Identity matrix as initial pose
        self.map_points = []
        self.map_lock = threading.Lock()
        logger.info("SLAM simulator initialized")
    
    def start(self):
        """Start the simulator"""
        self.is_running = True
        return True
    
    def stop(self):
        """Stop the simulator"""
        self.is_running = False
        return True
    
    def process_frame(self, frame):
        """Simulate processing a frame"""
        if not self.is_running:
            return False
        
        # Simulate some processing delay
        time.sleep(0.01)
        
        # Update pose with some random movement
        delta = np.random.normal(0, 0.01, 3)
        self.current_pose[0:3, 3] += delta
        
        # Generate some random map points
        if len(self.map_points) < 1000 and np.random.random() < 0.1:
            new_points = np.random.normal(0, 1, (10, 3))
            with self.map_lock:
                self.map_points.extend(new_points)
        
        return True
    
    def get_current_pose(self):
        """Get the current pose"""
        return self.current_pose.copy()
    
    def get_map_points(self):
        """Get the current map points"""
        with self.map_lock:
            return self.map_points.copy()
    
    def get_2d_map(self, resolution=0.05, size=10.0):
        """Generate a 2D occupancy grid map"""
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