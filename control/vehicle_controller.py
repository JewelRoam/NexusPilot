"""
Vehicle Controller - converts APF planner output to vehicle control commands.
Optimized for zero-lag hardware execution and deadband compensation.
"""
import time
import math
import numpy as np
from typing import Optional, Tuple

class VehicleController:
    """
    Translates APF PlannerOutput into hardware-specific commands.
    Features: 
      - Direct feed-forward control for RPi (zero-lag)
      - PID regulation for CARLA simulation
      - Motor deadband compensation
    """

    def __init__(self, config: dict, platform: str = "carla"):
        self.platform = platform
        self.config = config
        self.base_pwm = config.get("base_pwm", 0.3) # Min power to move

    def compute_control(self, planner_output, current_speed_kmh: float = 0.0) -> dict:
        """
        Main entry point for control computation.
        """
        if planner_output.emergency_brake:
            return self._emergency_stop()

        if self.platform == "carla":
            return self._compute_carla(planner_output, current_speed_kmh)
        elif self.platform == "raspberry_pi":
            return self._compute_rpi(planner_output)
        
        return {}

    def _compute_carla(self, out, current_v: float) -> dict:
        """Simple proportional control for CARLA."""
        target_v = out.target_speed
        steer = float(np.clip(out.target_steering, -1.0, 1.0))
        
        if target_v < 0: # Recovery mode
            return {"throttle": 0.4, "steer": -steer, "brake": 0.0, "reverse": True}
        
        error = target_v - current_v
        throttle = float(np.clip(error / 10.0, 0.1, 0.8))
        brake = float(np.clip(-error / 20.0, 0.0, 1.0))
        
        return {"throttle": throttle, "steer": steer, "brake": brake, "reverse": False}

    def _compute_rpi(self, out) -> dict:
        """
        Zero-lag feed-forward control for Raspberry Pi.
        Maps APF output directly to motor PWM with deadband compensation.
        """
        # target_speed is 0-30. Scale to 0-1.
        speed_raw = out.target_speed / 20.0 
        
        if speed_raw < 0.05:
            return self._emergency_stop()

        # Apply deadband compensation
        # Effective PWM = Base + Raw * (1 - Base)
        speed_comp = self.base_pwm + speed_raw * (1.0 - self.base_pwm)
        speed_comp = np.clip(speed_comp, 0.0, 1.0)

        # target_steering: -1 (left) to 1 (right)
        # For differential drive, steer_factor modifies left/right ratio
        steer = np.clip(out.target_steering, -1.0, 1.0)
        
        # Logic: Steering positive (right) -> slow down right wheel
        if steer >= 0:
            left_pwm = speed_comp
            right_pwm = speed_comp * (1.0 - abs(steer) * 0.7)
        else:
            left_pwm = speed_comp * (1.0 - abs(steer) * 0.7)
            right_pwm = speed_comp
            
        return {
            "left_pwm": float(left_pwm),
            "right_pwm": float(right_pwm),
            "direction": "forward" if out.target_speed >= 0 else "backward"
        }

    def _emergency_stop(self) -> dict:
        if self.platform == "carla":
            return {"throttle": 0.0, "steer": 0.0, "brake": 1.0, "reverse": False}
        return {"left_pwm": 0.0, "right_pwm": 0.0, "direction": "stop"}
