#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派小车避障模块
Obstacle Avoidance Module for Raspberry Pi Robot Car

三种避障模式:
  1. simple  – 前方超声波仅, 基本后退转向 (案例6风格)
  2. servo   – 舵机云台扫描左右, 智能选择转向 (案例7风格)
  3. apf     – APF人工势场避障, 融合项目planning/apf_planner思想

基于rpi_deploy现有模块:
  - motor_driver.MotorController   (4WD电机控制)
  - servo_controller.ServoController (PCA9685舵机)
  - ultrasonic_sensor.UltrasonicSensor (HC-SR04测距)
  - hardware_config.HardwareConfig    (统一配置)

Usage:
  python -m rpi_deploy.obstacle_avoidance               # 默认servo模式
  python -m rpi_deploy.obstacle_avoidance --mode simple  # 简单避障
  python -m rpi_deploy.obstacle_avoidance --mode servo   # 舵机增强
  python -m rpi_deploy.obstacle_avoidance --mode apf     # APF势场
"""

import time
import math
import argparse
from typing import Optional, Tuple

from .hardware_config import hardware_config
from .motor_driver import MotorController, Direction
from .servo_controller import ServoController
from .ultrasonic_sensor import UltrasonicSensor, DistanceReading

# Button & LED support (graceful fallback for PC simulation)
try:
    from gpiozero import Button, LED
    _GPIO_AVAILABLE = True
except ImportError:
    _GPIO_AVAILABLE = False
    print("[WARNING] gpiozero not available, button/LED running in simulation mode")


# ---------------------------------------------------------------------------
# Configuration (overrides from hardware_config)
# ---------------------------------------------------------------------------

# Distance thresholds (cm)
OBSTACLE_DISTANCE_CM = hardware_config.control.warning_distance_cm   # 50cm warning
SAFE_DISTANCE_CM = hardware_config.control.safe_distance_cm          # 30cm safe
EMERGENCY_DISTANCE_CM = 15.0  # emergency stop

# Speed settings (0.0-1.0, matching gpiozero.Robot convention)
FORWARD_SPEED = 0.15       # cruising speed (case 6: 0.15, case 7: 0.2)
AVOID_SPEED = 0.3          # turning / reversing speed
APF_FORWARD_SPEED = 0.2    # APF mode cruising speed
APF_TURN_SPEED = 0.3       # APF mode turning speed

# Timing (seconds)
SERVO_SETTLE_TIME = 0.4    # wait for servo to reach position
BACKUP_DURATION = 0.5      # how long to back up
TURN_DURATION = 0.5        # how long to turn
SCAN_PAUSE = 0.3           # pause between loop iterations


# ---------------------------------------------------------------------------
# Obstacle Avoidance Controller
# ---------------------------------------------------------------------------

class ObstacleAvoidanceController:
    """
    避障控制器 – 封装电机、舵机、超声波硬件

    Provides a unified interface for the three avoidance modes,
    delegating hardware access to the existing rpi_deploy modules.
    """

    def __init__(self):
        self.motor = MotorController()
        self.servo = ServoController()
        self.sensor = UltrasonicSensor()

        # Ensure servo starts at center (front-facing)
        self.servo.center_ultrasonic()
        time.sleep(0.3)

        # Button & LED (graceful fallback for PC simulation)
        self._btn = None
        self._green_led = None
        self._red_led = None
        self._keyflag = False
        self._running = False

        btn_cfg = hardware_config.button_led
        if _GPIO_AVAILABLE:
            self._btn = Button(btn_cfg.button_pin, pull_up=True)
            self._green_led = LED(btn_cfg.green_led_pin)
            self._red_led = LED(btn_cfg.red_led_pin)
            self._btn.when_pressed = self._on_button_press
            self._btn.when_released = self._on_button_release

        print("[ObstacleAvoidance] Hardware initialized")

    # -- Button callbacks --------------------------------------------------

    def _on_button_press(self):
        self._keyflag = True
        self._running = True
        if self._red_led:
            self._red_led.on()
        if self._green_led:
            self._green_led.off()
        print("[Button] PRESSED – robot STARTED")

    def _on_button_release(self):
        if self._red_led:
            self._red_led.off()
        if self._green_led:
            self._green_led.on()
        print("[Button] released")

    # -- Sensor helpers ----------------------------------------------------

    def distance_front_cm(self) -> float:
        """Measure distance straight ahead (cm)."""
        reading = self.sensor.measure_average(count=3)
        return reading.distance_cm if reading.valid else 999.0

    def scan_front_cm(self) -> float:
        """Point ultrasonic forward and measure (cm)."""
        self.servo.center_ultrasonic()
        time.sleep(SERVO_SETTLE_TIME)
        return self.distance_front_cm()

    def scan_left_cm(self) -> float:
        """Point ultrasonic left (~155°) and measure (cm)."""
        self.servo.set_ultrasonic_angle(155)
        time.sleep(SERVO_SETTLE_TIME)
        dist = self.distance_front_cm()
        return dist

    def scan_right_cm(self) -> float:
        """Point ultrasonic right (~85°) and measure (cm)."""
        self.servo.set_ultrasonic_angle(85)
        time.sleep(SERVO_SETTLE_TIME)
        dist = self.distance_front_cm()
        return dist

    # -- Motor helpers (delegate to MotorController) -----------------------

    def forward(self, speed: float = FORWARD_SPEED):
        self.motor.move_forward(speed)

    def backward(self, speed: float = AVOID_SPEED):
        self.motor.move_backward(speed)

    def left(self, speed: float = AVOID_SPEED):
        self.motor.rotate_left(speed)

    def right(self, speed: float = AVOID_SPEED):
        self.motor.rotate_right(speed)

    def stop(self):
        self.motor.stop()

    def emergency_stop(self):
        self.motor.emergency_stop()

    # -- State -------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        self._running = True
        print("[ObstacleAvoidance] STARTED")

    def stop_running(self):
        self._running = False
        self.stop()
        print("[ObstacleAvoidance] STOPPED")

    def wait_for_start(self):
        """Block until the button is pressed (or auto-start in sim mode)."""
        if not _GPIO_AVAILABLE:
            print("[Sim] Auto-starting in 2 s …")
            time.sleep(2)
            self._keyflag = True
            self._running = True
            return
        print("[Robot] Press the on-board button to START …")
        while not self._keyflag:
            time.sleep(0.05)

    def cleanup(self):
        self.stop()
        self.servo.center_ultrasonic()
        time.sleep(0.2)
        self.motor.cleanup()
        self.servo.cleanup()
        self.sensor.cleanup()
        if self._green_led:
            self._green_led.off()
        if self._red_led:
            self._red_led.off()
        print("[ObstacleAvoidance] Cleanup done")


# ===================================================================== #
#  MODE 1 – Simple ultrasonic avoidance (案例6风格)                     #
# ===================================================================== #

def mode_simple(ctrl: ObstacleAvoidanceController):
    """
    简单超声波避障 – 仅使用前方超声波

    Algorithm:
      1. Measure distance ahead.
      2. If obstacle < threshold → back up, then turn right.
      3. Otherwise → drive forward.

    对应案例: resources/makerobo_code/6.机器人超声波避障.ipynb
    """
    print("=" * 50)
    print("  Mode: SIMPLE ultrasonic avoidance")
    print(f"  Obstacle threshold: {OBSTACLE_DISTANCE_CM:.0f} cm")
    print("=" * 50)

    ctrl.wait_for_start()

    try:
        while ctrl.is_running:
            dis = ctrl.distance_front_cm()

            if dis < OBSTACLE_DISTANCE_CM:
                # Obstacle ahead → back up then turn right
                while dis < OBSTACLE_DISTANCE_CM:
                    ctrl.backward(AVOID_SPEED)
                    time.sleep(BACKUP_DURATION)
                    ctrl.right(AVOID_SPEED)
                    time.sleep(TURN_DURATION)
                    dis = ctrl.distance_front_cm()
            else:
                ctrl.forward(FORWARD_SPEED)

            print(f"[Simple] Distance: {dis:.1f} cm")
            time.sleep(SCAN_PAUSE)

    except KeyboardInterrupt:
        pass
    finally:
        ctrl.cleanup()


# ===================================================================== #
#  MODE 2 – Servo-enhanced ultrasonic avoidance (案例7风格)             #
# ===================================================================== #

def mode_servo(ctrl: ObstacleAvoidanceController):
    """
    舵机云台超声波避障

    Algorithm:
      1. Scan front.  If clear → drive forward.
      2. If obstacle ahead → stop, back up.
      3. Scan left and right with the servo.
      4. Both sides blocked → spin left (180° pivot).
      5. Otherwise → turn toward the side with more clearance.

    对应案例: resources/makerobo_code/7.机器人超声波(带舵机)避障.ipynb
    """
    print("=" * 50)
    print("  Mode: SERVO-enhanced ultrasonic avoidance")
    print(f"  Obstacle threshold: {OBSTACLE_DISTANCE_CM:.0f} cm")
    print("=" * 50)

    ctrl.wait_for_start()

    try:
        while ctrl.is_running:
            dis_front = ctrl.scan_front_cm()

            if dis_front < OBSTACLE_DISTANCE_CM:
                # Obstacle ahead → stop & back up
                ctrl.stop()
                time.sleep(0.2)
                ctrl.backward(AVOID_SPEED)
                time.sleep(BACKUP_DURATION)
                ctrl.stop()
                time.sleep(0.2)

                # Scan both sides
                dis_left = ctrl.scan_left_cm()
                dis_right = ctrl.scan_right_cm()

                if dis_left < OBSTACLE_DISTANCE_CM and dis_right < OBSTACLE_DISTANCE_CM:
                    # Both sides blocked → pivot left ~180°
                    ctrl.left(AVOID_SPEED)
                    time.sleep(1.0)
                    ctrl.stop()
                    time.sleep(0.1)
                elif dis_left > dis_right:
                    # More room on the left
                    ctrl.left(AVOID_SPEED)
                    time.sleep(TURN_DURATION)
                    ctrl.stop()
                    time.sleep(0.1)
                else:
                    # More room on the right
                    ctrl.right(AVOID_SPEED)
                    time.sleep(TURN_DURATION)
                    ctrl.stop()
                    time.sleep(0.1)

                # Resume front-facing
                ctrl.scan_front_cm()
            else:
                ctrl.forward(FORWARD_SPEED)

            print(f"[Servo] Front: {dis_front:.1f} cm")
            time.sleep(SCAN_PAUSE)

    except KeyboardInterrupt:
        pass
    finally:
        ctrl.cleanup()


# ===================================================================== #
#  MODE 3 – APF-inspired potential-field avoidance                       #
# ===================================================================== #

class SimpleAPFController:
    """
    简化版人工势场控制器 – 适配超声波三方向离散采样

    Adapts the APF concept from planning/apf_planner.py to the
    Raspberry Pi robot's discrete ultrasonic sensor capabilities.

    The ultrasonic + servo system provides three distance samples
    (front, left, right) rather than a continuous obstacle field.
    We map these into attractive / repulsive forces:

      - Attractive force: always pulls the robot forward (goal = ahead).
      - Repulsive force:  pushes the robot away from detected obstacles,
                          magnitude inversely proportional to distance^2.

    Force mapping:
      front_obstacle → repulsive vector pointing backward  (−x)
      left_obstacle  → repulsive vector pointing right     (+y)
      right_obstacle → repulsive vector pointing left      (−y)

    The net force is classified into: FORWARD, LEFT, RIGHT, BACKWARD, STOP
    """

    # Repulsive force gain (higher = stronger avoidance)
    K_REP: float = 60.0
    # Attractive force gain (forward pull)
    K_ATT: float = 1.0
    # Maximum influence distance for repulsive force (cm)
    D0: float = 80.0
    # Emergency brake distance (cm)
    EMERGENCY_DIST: float = EMERGENCY_DISTANCE_CM
    # Lateral scaling factor for side obstacles
    LATERAL_SCALE: float = 0.6

    def __init__(self):
        self._state = "normal"  # normal / avoiding / emergency

    def _repulsive_magnitude(self, distance_cm: float) -> float:
        """
        Compute repulsive force magnitude for a given distance (cm).

        Uses simplified APF formula (no 1/d² gradient term) because
        our distances are in centimeters (1-400 cm), not meters,
        making the classic 1/d² term produce negligibly small values.

        F_rep = K_REP * (1/d - 1/D0)   for d < D0
              = 0                       for d >= D0
        """
        if distance_cm >= self.D0:
            return 0.0
        d = max(distance_cm, 1.0)
        return self.K_REP * (1.0 / d - 1.0 / self.D0)

    def compute(self, dist_front: float, dist_left: float, dist_right: float) -> dict:
        """
        Compute APF control from three distance readings (cm).

        Returns:
            dict with keys:
              action   – "forward", "left", "right", "backward", "stop"
              speed    – float 0-100 (motor percentage)
              state    – "normal", "avoiding", "emergency"
              fx, fy   – net force components (debugging / logging)
        """
        # --- Attractive force (always pulls forward, +x) ---
        fx_att = self.K_ATT
        fy_att = 0.0

        # --- Repulsive forces ---
        fx_rep = 0.0
        fy_rep = 0.0

        # Front obstacle → repels backward (−x)
        fx_rep -= self._repulsive_magnitude(dist_front)

        # Left obstacle → repels to the right (+y)
        fy_rep += self._repulsive_magnitude(dist_left) * self.LATERAL_SCALE

        # Right obstacle → repels to the left (−y)
        fy_rep -= self._repulsive_magnitude(dist_right) * self.LATERAL_SCALE

        # --- Net force ---
        fx = fx_att + fx_rep
        fy = fy_att + fy_rep

        # --- Emergency check ---
        min_dist = min(dist_front, dist_left, dist_right)
        if min_dist < self.EMERGENCY_DIST:
            self._state = "emergency"
            return dict(action="stop", speed=0.0, state=self._state,
                        fx=fx, fy=fy)

        # --- Determine action from force vector ---
        if fx <= 0 and abs(fy) < 0.01:
            # Strong front repulsion, no lateral preference → back up
            self._state = "avoiding"
            return dict(action="backward", speed=APF_AVOID_SPEED,
                        state=self._state, fx=fx, fy=fy)

        force_mag = math.sqrt(fx * fx + fy * fy)
        force_angle = math.degrees(math.atan2(fy, fx))
        # 0° = straight, +90° = right, −90° = left

        if force_mag < 0.01:
            self._state = "normal"
            return dict(action="stop", speed=0.0, state=self._state,
                        fx=fx, fy=fy)

        # Speed proportional to forward component
        speed_factor = max(fx / (self.K_ATT + 1.0), 0.0)
        speed = APF_FORWARD_SPEED * speed_factor
        speed = max(speed, 0.05)  # minimum crawling speed

        # Classify angle into action
        if abs(force_angle) < 30:
            action = "forward"
            self._state = "normal"
        elif force_angle >= 30:
            action = "right"
            speed = APF_TURN_SPEED
            self._state = "avoiding"
        elif force_angle <= -30:
            action = "left"
            speed = APF_TURN_SPEED
            self._state = "avoiding"
        else:
            # Safety fallback
            action = "forward"
            speed = APF_FORWARD_SPEED
            self._state = "normal"

        return dict(action=action, speed=speed, state=self._state,
                    fx=fx, fy=fy)


# APF mode specific speed
APF_AVOID_SPEED = 0.3


def mode_apf(ctrl: ObstacleAvoidanceController):
    """
    APF人工势场避障 – 舵机扫描三方向

    Each cycle:
      1. Scan front / left / right distances via servo.
      2. Feed distances into the APF controller.
      3. Execute the returned action.

    融合项目 planning/apf_planner.py 的势场算法思想,
    适配超声波三方向离散采样场景.
    """
    print("=" * 50)
    print("  Mode: APF potential-field avoidance")
    print(f"  Emergency distance: {EMERGENCY_DISTANCE_CM:.0f} cm")
    print("=" * 50)

    apf = SimpleAPFController()
    ctrl.wait_for_start()

    # Action duration constants (seconds)
    ACTION_FORWARD_DT = 0.2
    ACTION_TURN_DT = 0.3
    ACTION_BACKUP_DT = 0.4

    try:
        while ctrl.is_running:
            # --- Scan three directions ---
            dist_front = ctrl.scan_front_cm()
            dist_left = ctrl.scan_left_cm()
            dist_right = ctrl.scan_right_cm()

            # --- Compute APF control ---
            cmd = apf.compute(dist_front, dist_left, dist_right)

            # --- Execute ---
            action = cmd["action"]
            speed = cmd["speed"]

            if action == "forward":
                ctrl.forward(speed)
                time.sleep(ACTION_FORWARD_DT)
            elif action == "left":
                ctrl.left(speed)
                time.sleep(ACTION_TURN_DT)
            elif action == "right":
                ctrl.right(speed)
                time.sleep(ACTION_TURN_DT)
            elif action == "backward":
                ctrl.backward(speed)
                time.sleep(ACTION_BACKUP_DT)
            elif action == "stop":
                ctrl.stop()
                time.sleep(0.2)

            # --- Log ---
            print(f"[APF] F={dist_front:.0f} L={dist_left:.0f} R={dist_right:.0f} cm | "
                  f"action={action} speed={speed:.2f} state={cmd['state']} | "
                  f"fx={cmd['fx']:.2f} fy={cmd['fy']:.2f}")

    except KeyboardInterrupt:
        pass
    finally:
        ctrl.cleanup()


# ===================================================================== #
#  Entry point                                                           #
# ===================================================================== #

MODE_MAP = {
    "simple": mode_simple,
    "servo": mode_servo,
    "apf": mode_apf,
}


def main(argv=None):
    """Command-line entry point for obstacle avoidance."""
    parser = argparse.ArgumentParser(
        description="Makerobo 4WD Robot – Obstacle Avoidance (rpi_deploy)"
    )
    parser.add_argument(
        "--mode", choices=MODE_MAP.keys(), default="servo",
        help="Avoidance mode: simple (front-only), servo (scan left/right), "
             "apf (potential-field). Default: servo",
    )
    args = parser.parse_args(argv)

    print()
    print("*" * 50)
    print("  Makerobo 4WD Obstacle Avoidance")
    print(f"  Mode: {args.mode}")
    print("*" * 50)
    print()

    ctrl = ObstacleAvoidanceController()

    try:
        MODE_MAP[args.mode](ctrl)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user")
    finally:
        ctrl.cleanup()


if __name__ == "__main__":
    main()