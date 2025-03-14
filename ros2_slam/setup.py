import os
from setuptools import setup, find_packages

package_name = 'ros2_slam'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'smbus2',  # For MPU6050 on Raspberry Pi
    ],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Distributed SLAM system using ROS 2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'raspberry_node = ros2_slam.raspberry_node:main',
            'pc_slam_node = ros2_slam.pc_slam_node:main',
        ],
    },
) 