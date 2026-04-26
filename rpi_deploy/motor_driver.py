"""
NexusPilot: Robust Motor Driver
Standardized version using gpiozero.Robot for LOBOROBOT board.
"""
import os
import sys
import time
from enum import Enum
from typing import Optional

# CRITICAL: Must set environment variable BEFORE importing gpiozero
os.environ["LG_CHIP"] = "0"

try:
    from gpiozero import Robot, Motor
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    Robot = None
    Motor = None

# Path setup
if __name__ == "__main__" or __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rpi_deploy.hardware_config import hardware_config
else:
    from .hardware_config import hardware_config

class MotorController:
    """
    Differential drive controller using the Robot container.
    """
    def __init__(self):
        self.config = hardware_config.motor
        self._robot: Optional[Robot] = None
        
        if GPIO_AVAILABLE:
            try:
                self._robot = Robot(
                    left=Motor(forward=self.config.left_forward, backward=self.config.left_backward, enable=self.config.left_enable),
                    right=Motor(forward=self.config.right_forward, backward=self.config.right_backward, enable=self.config.right_enable)
                )
                print(f"[Motor] Robot initialized on Pi 5. Factory: {self._robot.pin_factory}")
            except Exception as e:
                print(f"[Motor] Init Error: {e}")

    def curve_move(self, linear_speed: float, angular_rate: float):
        if not self._robot: return
        # Simple differential mixing
        turn_factor = 0.5
        l_speed = max(-1.0, min(1.0, linear_speed + angular_rate * turn_factor))
        r_speed = max(-1.0, min(1.0, linear_speed - angular_rate * turn_factor))
        
        self._robot.left_motor.value = l_speed
        self._robot.right_motor.value = r_speed

    def stop(self):
        if self._robot: self._robot.stop()

    def emergency_stop(self):
        self.stop()

    def move_backward(self, speed: float):
        if self._robot: self._robot.backward(speed)

    def cleanup(self):
        self.stop()
        if self._robot: self._robot.close()

if __name__ == "__main__":
    print("Testing hardware drive (Robot mode)...")
    ctrl = MotorController()
    ctrl.curve_move(0.4, 0.0)
    time.sleep(1.0)
    ctrl.stop()
    ctrl.cleanup()
