#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派小车硬件配置模块
Hardware Configuration Module for Raspberry Pi Robot Car

基于创乐博LOBOROBOT扩展板的硬件配置
Based on Chuang Le Bo LOBOROBOT expansion board
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MotorPins:
    """电机控制引脚配置"""
    # 左前轮
    left_front_pwm: int = 12   # PWM速度控制
    left_front_dir1: int = 5   # 方向控制1
    left_front_dir2: int = 6   # 方向控制2
    
    # 右前轮
    right_front_pwm: int = 13
    right_front_dir1: int = 20
    right_front_dir2: int = 21
    
    # 左后轮
    left_rear_pwm: int = 18
    left_rear_dir1: int = 23
    left_rear_dir2: int = 24
    
    # 右后轮
    right_rear_pwm: int = 19
    right_rear_dir1: int = 16
    right_rear_dir2: int = 26


@dataclass
class UltrasonicConfig:
    """超声波传感器配置"""
    trigger_pin: int = 4
    echo_pin: int = 17
    servo_channel: int = 0  # PCA9685通道
    min_angle: int = 0      # 最小角度
    max_angle: int = 180    # 最大角度
    center_angle: int = 90  # 中心位置


@dataclass
class CameraConfig:
    """摄像头配置"""
    device_id: int = 0           # USB摄像头设备ID
    width: int = 640             # 分辨率宽度
    height: int = 480            # 分辨率高度
    fps: int = 30                # 帧率
    pan_servo_channel: int = 1   # 水平舵机通道
    tilt_servo_channel: int = 2  # 俯仰舵机通道
    pan_min: int = 0             # 水平最小角度
    pan_max: int = 180           # 水平最大角度
    tilt_min: int = -10          # 俯仰最小角度
    tilt_max: int = 90           # 俯仰最大角度


@dataclass
class I2CConfig:
    """I2C总线配置"""
    bus_number: int = 1          # I2C总线号 (RPi 5使用bus 1)
    pca9685_address: int = 0x40  # PCA9685默认地址
    pwm_frequency: int = 50      # 舵机PWM频率 (50Hz = 20ms周期)


@dataclass
class NetworkConfig:
    """网络通信配置"""
    # 远程控制服务器
    control_host: str = "0.0.0.0"
    control_port: int = 5000
    
    # 视频流服务器
    video_port: int = 8080
    
    # V2V通信 (WiFi UDP广播)
    v2v_enabled: bool = True
    v2v_broadcast_ip: str = "192.168.137.255"  # 根据实际网段调整
    v2v_port: int = 9999
    vehicle_id: str = "RPI_CAR_001"


@dataclass
class ButtonLEDConfig:
    """按键和LED配置"""
    button_pin: int = 19        # 启动/停止按键
    green_led_pin: int = 5      # 运行指示灯
    red_led_pin: int = 6        # 避障指示灯


@dataclass
class ControlConfig:
    """运动控制配置"""
    # 速度限制
    max_speed: float = 100.0     # 最大速度百分比
    min_speed: float = 20.0      # 最小有效速度
    
    # 转向参数
    turn_factor: float = 0.5     # 转向系数
    
    # 安全距离
    safe_distance_cm: float = 30.0   # 安全停止距离
    warning_distance_cm: float = 50.0  # 警告距离
    
    # 扫描参数
    scan_angles: Tuple[int, ...] = (0, 45, 90, 135, 180)  # 扫描角度序列
    scan_delay: float = 0.3      # 每次扫描延迟


class HardwareConfig:
    """
    硬件配置管理类
    统一管理所有硬件配置参数
    """
    
    def __init__(self):
        self.motor = MotorPins()
        self.ultrasonic = UltrasonicConfig()
        self.camera = CameraConfig()
        self.i2c = I2CConfig()
        self.network = NetworkConfig()
        self.button_led = ButtonLEDConfig()
        self.control = ControlConfig()
        
        # 从环境变量加载覆盖配置
        self._load_from_env()
    
    def _load_from_env(self):
        """从环境变量加载配置覆盖"""
        # 网络配置
        if os.getenv('RPI_CONTROL_PORT'):
            self.network.control_port = int(os.getenv('RPI_CONTROL_PORT'))
        if os.getenv('RPI_VIDEO_PORT'):
            self.network.video_port = int(os.getenv('RPI_VIDEO_PORT'))
        if os.getenv('RPI_VEHICLE_ID'):
            self.network.vehicle_id = os.getenv('RPI_VEHICLE_ID')
        if os.getenv('RPI_V2V_BROADCAST'):
            self.network.v2v_broadcast_ip = os.getenv('RPI_V2V_BROADCAST')
        
        # 摄像头配置
        if os.getenv('RPI_CAMERA_WIDTH'):
            self.camera.width = int(os.getenv('RPI_CAMERA_WIDTH'))
        if os.getenv('RPI_CAMERA_HEIGHT'):
            self.camera.height = int(os.getenv('RPI_CAMERA_HEIGHT'))
        
        # 安全距离
        if os.getenv('RPI_SAFE_DISTANCE'):
            self.control.safe_distance_cm = float(os.getenv('RPI_SAFE_DISTANCE'))
    
    def to_dict(self) -> dict:
        """导出为字典格式"""
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
        """从字典创建配置"""
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


# 全局配置实例
hardware_config = HardwareConfig()
