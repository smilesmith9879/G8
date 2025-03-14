#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Launch file for PC SLAM processing node
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    """Generate launch description for PC SLAM node"""
    
    # Declare launch arguments
    slam_config_file_arg = DeclareLaunchArgument(
        'slam_config_file',
        default_value='',
        description='Path to ORB-SLAM3 configuration file'
    )
    
    map_publish_rate_arg = DeclareLaunchArgument(
        'map_publish_rate',
        default_value='1.0',
        description='Map publishing rate in Hz'
    )
    
    map_resolution_arg = DeclareLaunchArgument(
        'map_resolution',
        default_value='0.05',
        description='Map resolution in meters per pixel'
    )
    
    map_size_arg = DeclareLaunchArgument(
        'map_size',
        default_value='10.0',
        description='Map size in meters'
    )
    
    # Create node
    pc_node = Node(
        package='ros2_slam',
        executable='pc_slam_node',
        name='pc_slam_node',
        output='screen',
        parameters=[{
            'slam_config_file': LaunchConfiguration('slam_config_file'),
            'map_publish_rate': LaunchConfiguration('map_publish_rate'),
            'map_resolution': LaunchConfiguration('map_resolution'),
            'map_size': LaunchConfiguration('map_size'),
        }]
    )
    
    # Return launch description
    return LaunchDescription([
        slam_config_file_arg,
        map_publish_rate_arg,
        map_resolution_arg,
        map_size_arg,
        pc_node
    ]) 