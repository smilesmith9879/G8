#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for the LOBOROBOT library
This script demonstrates the basic movement capabilities of the robot
"""

import time
from LOBOROBOT import LOBOROBOT

def main():
    print("Initializing robot...")
    robot = LOBOROBOT()
    
    try:
        # Initialize servo gimbal to default position
        print("Setting gimbal to default position...")
        robot.set_servo_angle(9, 80)  # Horizontal (PWM9)
        robot.set_servo_angle(10, 40)  # Vertical (PWM10)
        time.sleep(1)
        
        # Test basic movements
        print("Testing basic movements...")
        
        # Forward
        print("Moving forward...")
        robot.t_up(20, 2)
        
        # Backward
        print("Moving backward...")
        robot.t_down(20, 2)
        
        # Left
        print("Moving left...")
        robot.moveLeft(20, 2)
        
        # Right
        print("Moving right...")
        robot.moveRight(20, 2)
        
        # Turn left
        print("Turning left...")
        robot.turnLeft(15, 2)
        
        # Turn right
        print("Turning right...")
        robot.turnRight(15, 2)
        
        # Stop
        print("Stopping...")
        robot.t_stop(1)
        
        # Test gimbal movement
        print("Testing gimbal movement...")
        
        # Move horizontal servo from left to right
        print("Moving horizontal servo from left to right...")
        for angle in range(35, 126, 5):
            robot.set_servo_angle(9, angle)
            print(f"Horizontal angle: {angle}°")
            time.sleep(0.2)
        
        # Reset to center
        robot.set_servo_angle(9, 80)
        time.sleep(1)
        
        # Move vertical servo from down to up
        print("Moving vertical servo from down to up...")
        for angle in range(0, 86, 5):
            robot.set_servo_angle(10, angle)
            print(f"Vertical angle: {angle}°")
            time.sleep(0.2)
        
        # Reset to center
        robot.set_servo_angle(10, 40)
        time.sleep(1)
        
        # Test diagonal movements
        print("Testing diagonal movements...")
        
        # Forward-left
        print("Moving forward-left...")
        robot.forward_Left(20, 2)
        
        # Forward-right
        print("Moving forward-right...")
        robot.forward_Right(20, 2)
        
        # Backward-left
        print("Moving backward-left...")
        robot.backward_Left(20, 2)
        
        # Backward-right
        print("Moving backward-right...")
        robot.backward_Right(20, 2)
        
        # Final stop
        print("Final stop...")
        robot.t_stop(1)
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Ensure robot stops even if an error occurs
        robot.t_stop(0)
        print("Robot stopped.")

if __name__ == "__main__":
    main() 