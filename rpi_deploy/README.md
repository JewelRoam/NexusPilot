# 树莓派小车部署模块 (Raspberry Pi Car Deployment Module)

本模块包含将自主避障系统部署到树莓派5 4WD小车平台的完整代码。

## 硬件配置

### 核心组件
- **主板**: Raspberry Pi 5
- **扩展板**: 创乐博LOBOROBOT智能机器人扩展板 V3.0
- **电机驱动**: L298N双H桥（集成在扩展板上）
- **舵机控制器**: PCA9685 I2C PWM控制器 (地址0x40)
- **超声波传感器**: HC-SR04 x 1（带舵机云台）
- **摄像头**: USB摄像头（支持640x480@30fps）
- **通信**: 以太网/WiFi

### 引脚映射
```
电机控制:
  左前轮: PWM=GPIO12, DIR1=GPIO5, DIR2=GPIO6
  右前轮: PWM=GPIO13, DIR1=GPIO20, DIR2=GPIO21
  左后轮: PWM=GPIO18, DIR1=GPIO23, DIR2=GPIO24
  右后轮: PWM=GPIO19, DIR1=GPIO16, DIR2=GPIO26

超声波传感器:
  TRIG=GPIO4, ECHO=GPIO17

按键 & LED:
  按键=GPIO19, 绿色LED=GPIO5, 红色LED=GPIO6

I2C (PCA9685):
  SDA=GPIO2, SCL=GPIO3, 地址=0x40

舵机通道:
  Ch0: 超声波云台 (0-180°)
  Ch1: 摄像头水平云台 (0-180°)
  Ch2: 摄像头俯仰云台 (-10° to 90°)
```

## 模块结构

```
rpi_deploy/
├── __init__.py              # 包初始化
├── hardware_config.py       # 硬件配置管理
├── motor_driver.py          # 4WD电机驱动
├── servo_controller.py      # PCA9685舵机控制
├── ultrasonic_sensor.py     # HC-SR04超声波传感器
├── camera_driver.py         # USB摄像头驱动
├── obstacle_avoidance.py    # 三模式避障（simple/servo/apf）
├── remote_control.py        # 网络远程控制
├── rpi_car_controller.py    # 主控制程序（原有）
└── tests/                   # 硬件测试脚本
    ├── __init__.py
    ├── test_motor.py        # 电机测试
    ├── test_servo.py        # 舵机测试
    ├── test_ultrasonic.py   # 超声波测试
    ├── test_camera.py       # 摄像头测试
    ├── test_obstacle_avoidance.py # 避障算法测试
    └── test_remote_control.py # 远程控制测试
```

## 快速开始

### 1. 安装依赖

在树莓派上执行：

```bash
# 系统依赖
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv i2c-tools

# 启用I2C接口
sudo raspi-config  # Interface Options -> I2C -> Enable

# Python依赖
pip3 install gpiozero smbus2 flask numpy
```

### 2. 克隆项目

```bash
cd ~
git clone <your-repo-url> QMC9_Project
cd QMC9_Project
```

### 3. 硬件测试

按顺序测试各硬件组件：

```bash
cd rpi_deploy/tests

# 1. 测试电机（确保小车有空间移动！）
python3 test_motor.py --duration 1.0

# 2. 测试舵机
python3 test_servo.py

# 3. 测试超声波传感器
python3 test_ultrasonic.py --scan

# 4. 测试摄像头
python3 test_camera.py --preview --save

# 5. 测试远程控制（树莓派端启动服务器）
python3 test_remote_control.py --server
```

### 4. PC端远程控制

在PC上运行客户端测试：

```bash
python rpi_deploy/tests/test_remote_control.py --client --host 192.168.137.33
```

浏览器查看视频流：
```
http://192.168.137.33:8080
```

## 使用主控制程序

### 基本避障模式

```bash
cd ~/QMC9_Project
python3 -m rpi_deploy.rpi_car_controller --mode obstacle_avoidance
```

### 协作模式（多车）

```bash
python3 -m rpi_deploy.rpi_car_controller --mode cooperative --v2v
```

### 远程遥控模式

```bash
python3 -m rpi_deploy.rpi_car_controller --mode remote
```

## 避障模块 (obstacle_avoidance)

基于 `resources/makerobo_code` 案例接口实现的三模式避障系统。

### 三种避障模式

| 模式 | 命令 | 对应案例 | 算法 |
|------|------|----------|------|
| **simple** | `--mode simple` | 案例6 | 前方超声波仅，障碍→后退+右转 |
| **servo** | `--mode servo` | 案例7 | 舵机扫描左/右，智能选择转向方向 |
| **apf** | `--mode apf` | apf_planner | APF人工势场，排斥力∝1/d，平滑避障 |

### 使用方法

```bash
cd ~/QMC9_Project

# 默认 servo 模式
python3 -m rpi_deploy.obstacle_avoidance

# 简单避障（案例6风格）
python3 -m rpi_deploy.obstacle_avoidance --mode simple

# 舵机增强避障（案例7风格）
python3 -m rpi_deploy.obstacle_avoidance --mode servo

# APF势场避障
python3 -m rpi_deploy.obstacle_avoidance --mode apf
```

按板载按键启动机器人，`Ctrl+C` 退出。

### APF算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| K_REP | 60.0 | 排斥力增益 |
| K_ATT | 1.0 | 吸引力增益 |
| D₀ | 80cm | 排斥力影响距离 |
| LATERAL_SCALE | 0.6 | 侧向力缩放因子 |
| EMERGENCY_DIST | 15cm | 紧急制动距离 |

> **注意**：使用简化APF公式 F = K_REP × (1/d − 1/D₀)，不含经典1/d²梯度项，因为距离单位为厘米(1–400cm)，1/d²项在此尺度下产生极小值导致排斥力不足。

### 仿真模式

当 `gpiozero` 不可用时（PC开发调试），程序自动进入仿真模式：电机动作输出到控制台，超声波返回固定安全距离，自动启动（无需按键）。

### 避障算法测试

```bash
cd ~/QMC9_Project
python3 -m unittest rpi_deploy.tests.test_obstacle_avoidance -v
```

## API参考

### MotorController (电机控制)

```python
from rpi_deploy.motor_driver import MotorController

motor = MotorController()
motor.move_forward(speed=50)    # 前进 50%速度
motor.turn_left(speed=40, turn_rate=0.5)  # 左转
motor.rotate_right(speed=30)    # 原地右转
motor.stop()                     # 停止
motor.emergency_stop()           # 紧急停止
```

### ServoController (舵机控制)

```python
from rpi_deploy.servo_controller import ServoController

servo = ServoController()
servo.set_ultrasonic_angle(90)   # 设置超声波云台角度
servo.set_camera_pan(45)         # 设置摄像头水平角度
servo.set_camera_tilt(30)        # 设置摄像头俯仰角度
servo.center_all()               # 所有舵机归中
```

### UltrasonicSensor (超声波传感器)

```python
from rpi_deploy.ultrasonic_sensor import UltrasonicSensor

sensor = UltrasonicSensor()
distance = sensor.measure_once().distance_cm  # 单次测量
filtered = sensor.measure_average().distance_cm  # 滤波测量
is_obstacle = sensor.is_obstacle_detected(threshold=30)  # 障碍物检测
```

### CameraDriver (摄像头)

```python
from rpi_deploy.camera_driver import CameraDriver

camera = CameraDriver()
camera.open()
camera.start_capture()
frame = camera.get_frame()  # 获取图像帧
camera.capture_image("photo.jpg")  # 保存图片
camera.close()
```

### RemoteControlServer (远程控制)

```python
from rpi_deploy.remote_control import RemoteControlServer

server = RemoteControlServer(host='0.0.0.0', port=5000)
server.register_handler('move', move_handler)
server.register_status_provider(get_status)
server.start()
```

## 配置文件

通过环境变量覆盖默认配置：

```bash
export RPI_CONTROL_PORT=5000        # 控制端口
export RPI_VIDEO_PORT=8080          # 视频流端口
export RPI_VEHICLE_ID="CAR_001"     # 车辆ID
export RPI_CAMERA_WIDTH=640         # 摄像头宽度
export RPI_CAMERA_HEIGHT=480        # 摄像头高度
export RPI_SAFE_DISTANCE=30.0       # 安全距离(cm)
```

## 故障排除

### I2C设备未找到
```bash
# 检查I2C是否启用
sudo i2cdetect -y 1

# 应显示0x40 (PCA9685地址)
```

### 电机不转
- 检查电源电压（需要7-12V）
- 确认gpiozero库已安装
- 检查电机接线

### 摄像头无法打开
```bash
# 列出可用摄像头
ls /dev/video*

# 测试摄像头
ffplay /dev/video0
```

### 权限问题
```bash
# 添加用户到gpio组
sudo usermod -a -G gpio pi

# 重新登录生效
```

## 技术路线说明

本模块遵循以下设计原则：

1. **厂商兼容**: 基于创乐博LOBOROBOT库的硬件控制逻辑
2. **模块化设计**: 每个硬件组件独立封装，便于测试和维护
3. **模拟模式**: 支持在非树莓派环境下进行代码开发和测试
4. **网络遥控**: TCP Socket实现PC远程控制，方便调试
5. **渐进式部署**: 从单硬件测试 → 组合测试 → 完整系统

## 与仿真系统的对应关系

| 仿真组件 | 实际硬件 |
|---------|---------|
| CARLA Vehicle | 4WD小车底盘 |
| RGB Camera | USB摄像头 |
| Depth Camera | 超声波+扫描 |
| Vehicle Control | 电机驱动器 |
| V2V Communication | WiFi UDP广播 |

## 许可证

本项目遵循MIT许可证。参考了创乐博LOBOROBOT开源示例代码。
