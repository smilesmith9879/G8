#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch file for Raspberry Pi SLAM data collection node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for Raspberry Pi node"""
    
    # Declare launch arguments
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device ID'
    )
    
    camera_width_arg = DeclareLaunchArgument(
        'camera_width',
        default_value='640',
        description='Camera frame width'
    )
    
    camera_height_arg = DeclareLaunchArgument(
        'camera_height',
        default_value='480',
        description='Camera frame height'
    )
    
    camera_fps_arg = DeclareLaunchArgument(
        'camera_fps',
        default_value='30',
        description='Camera frames per second'
    )
    
    imu_enabled_arg = DeclareLaunchArgument(
        'imu_enabled',
        default_value='true',
        description='Enable IMU data collection'
    )
    
    compression_quality_arg = DeclareLaunchArgument(
        'compression_quality',
        default_value='80',
        description='JPEG compression quality (0-100)'
    )
    
    image_publish_rate_arg = DeclareLaunchArgument(
        'image_publish_rate',
        default_value='15.0',
        description='Image publishing rate in Hz'
    )
    
    imu_publish_rate_arg = DeclareLaunchArgument(
        'imu_publish_rate',
        default_value='100.0',
        description='IMU data publishing rate in Hz'
    )
    
    # Create node
    raspberry_node = Node(
        package='ros2_slam',
        executable='raspberry_node',
        name='raspberry_slam_node',
        output='screen',
        parameters=[{
            'camera_id': LaunchConfiguration('camera_id'),
            'camera_width': LaunchConfiguration('camera_width'),
            'camera_height': LaunchConfiguration('camera_height'),
            'camera_fps': LaunchConfiguration('camera_fps'),
            'imu_enabled': LaunchConfiguration('imu_enabled'),
            'compression_quality': LaunchConfiguration('compression_quality'),
            'image_publish_rate': LaunchConfiguration('image_publish_rate'),
            'imu_publish_rate': LaunchConfiguration('imu_publish_rate'),
        }]
    )
    
    # Return launch description
    return LaunchDescription([
        camera_id_arg,
        camera_width_arg,
        camera_height_arg,
        camera_fps_arg,
        imu_enabled_arg,
        compression_quality_arg,
        image_publish_rate_arg,
        imu_publish_rate_arg,
        raspberry_node
    ]) 