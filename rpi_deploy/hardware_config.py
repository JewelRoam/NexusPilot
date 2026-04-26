#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hardware Configuration Module for Raspberry Pi Robot Car

Based on Chuang Le Bo LOBOROBOT expansion board
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MotorPins:
    """Motor control pin configuration (gpiozero.Robot 2-motor architecture)"""
    # Left motor group (front+rear paired)
    left_forward: int = 22   # Left forward pin
    left_backward: int = 27  # Left backward pin
    left_enable: int = 18    # Left PWM enable pin
    
    # Right motor group (front+rear paired)
    right_forward: int = 25  # Right forward pin
    right_backward: int = 24 # Right backward pin
    right_enable: int = 23   # Right PWM enable pin


@dataclass
class UltrasonicConfig:
    """Ultrasonic sensor configuration"""
    trigger_pin: int = 20
    echo_pin: int = 21
    servo_channel: int = 0  # PCA9685 channel
    min_angle: int = 0      # Minimum angle
    max_angle: int = 180    # Maximum angle
    center_angle: int = 120  # Center position (physical offset: 90° was too far right)


@dataclass
class CameraConfig:
    """Camera configuration"""
    device_id: int = 0           # USB camera device ID
    width: int = 640             # Resolution width
    height: int = 480            # Resolution height
    fps: int = 30                # Framerate
    pan_servo_channel: int = 1   # Pan servo channel
    tilt_servo_channel: int = 2  # Tilt servo channel
    pan_min: int = 0             # Minimum pan angle
    pan_max: int = 180           # Maximum pan angle
    tilt_min: int = 0          # Minimum tilt angle
    tilt_max: int = 90           # Maximum tilt angle


@dataclass
class I2CConfig:
    """I2C bus configuration"""
    bus_number: int = 1          # I2C bus number (RPi 5 uses bus 1)
    pca9685_address: int = 0x40  # Default PCA9685 address
    pwm_frequency: int = 50      # Servo PWM frequency (50Hz = 20ms period)


@dataclass
class NetworkConfig:
    """Network communication configuration"""
    # Remote control server
    control_host: str = "0.0.0.0"
    control_port: int = 5000
    
    # Video stream server
    video_port: int = 8080
    
    # V2V communication (WiFi UDP broadcast)
    v2v_enabled: bool = True
    v2v_broadcast_ip: str = "192.168.137.255"  # Adjust according to actual subnet
    v2v_port: int = 9999
    vehicle_id: str = "RPI_CAR_001"


@dataclass
class ButtonLEDConfig:
    """Button and LED configuration"""
    button_pin: int = 19        # Start/Stop button
    green_led_pin: int = 5      # Running status LED
    red_led_pin: int = 6        # Obstacle avoidance LED


@dataclass
class ControlConfig:
    """Motion control configuration"""
    # Speed limits
    max_speed: float = 100.0     # Maximum speed percentage
    min_speed: float = 20.0      # Minimum effective speed
    
    # Steering parameters
    turn_factor: float = 0.5     # Turn factor
    
    # Safety distance
    safe_distance_cm: float = 30.0   # Safe stopping distance
    warning_distance_cm: float = 50.0  # Warning distance
    
    # Scanning parameters
    scan_angles: Tuple[int, ...] = (30, 75, 120, 165, 180)  # Scan angles (centered at 120°)
    scan_delay: float = 0.3      # Delay for each scan step


class HardwareConfig:
    """
    Hardware configuration management class
    Centralized management of all hardware configuration parameters
    """
    
    def __init__(self):
        self.motor = MotorPins()
        self.ultrasonic = UltrasonicConfig()
        self.camera = CameraConfig()
        self.i2c = I2CConfig()
        self.network = NetworkConfig()
        self.button_led = ButtonLEDConfig()
        self.control = ControlConfig()
        
        # Load override configuration from environment variables
        self._load_from_env()
    
    def _load_from_env(self):
        """Load configuration overrides from environment variables"""
        # Network configuration
        if os.getenv('RPI_CONTROL_PORT'):
            self.network.control_port = int(os.getenv('RPI_CONTROL_PORT'))
        if os.getenv('RPI_VIDEO_PORT'):
            self.network.video_port = int(os.getenv('RPI_VIDEO_PORT'))
        if os.getenv('RPI_VEHICLE_ID'):
            self.network.vehicle_id = os.getenv('RPI_VEHICLE_ID')
        if os.getenv('RPI_V2V_BROADCAST'):
            self.network.v2v_broadcast_ip = os.getenv('RPI_V2V_BROADCAST')
        
        # Camera configuration
        if os.getenv('RPI_CAMERA_WIDTH'):
            self.camera.width = int(os.getenv('RPI_CAMERA_WIDTH'))
        if os.getenv('RPI_CAMERA_HEIGHT'):
            self.camera.height = int(os.getenv('RPI_CAMERA_HEIGHT'))
        
        # Safety distance
        if os.getenv('RPI_SAFE_DISTANCE'):
            self.control.safe_distance_cm = float(os.getenv('RPI_SAFE_DISTANCE'))
    
    def to_dict(self) -> dict:
        """Export as dictionary format"""
        return {
            'motor': self.motor.__dict__,
            'ultrasonic': self.ultrasonic.__dict__,
            'camera': self.camera.__dict__,
            'i2c': self.i2c.__dict__,
            'network': self.network.__dict__,
            'button_led': self.button_led.__dict__,
            'control': self.control.__dict__
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'HardwareConfig':
        """Create configuration from dictionary"""
        config = cls()
        if 'motor' in config_dict:
            config.motor = MotorPins(**config_dict['motor'])
        if 'ultrasonic' in config_dict:
            config.ultrasonic = UltrasonicConfig(**config_dict['ultrasonic'])
        if 'camera' in config_dict:
            config.camera = CameraConfig(**config_dict['camera'])
        if 'i2c' in config_dict:
            config.i2c = I2CConfig(**config_dict['i2c'])
        if 'network' in config_dict:
            config.network = NetworkConfig(**config_dict['network'])
        if 'button_led' in config_dict:
            config.button_led = ButtonLEDConfig(**config_dict['button_led'])
        if 'control' in config_dict:
            config.control = ControlConfig(**config_dict['control'])
        return config


# Global configuration instance
hardware_config = HardwareConfig()
