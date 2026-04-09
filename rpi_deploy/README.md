# Raspberry Pi Car Deployment Module

This module contains the complete codebase for deploying the NexusPilot autonomous obstacle avoidance system onto a Raspberry Pi 5 4WD robot platform.

## Hardware Configuration

### Core Components
- **Board**: Raspberry Pi 5
- **Expansion Board**: LOBOROBOT Smart Robot Expansion Board V3.0 (Chuangbo)
- **Motor Driver**: L298N Dual H-Bridge (integrated on expansion board)
- **Servo Controller**: PCA9685 I2C PWM Controller (address 0x40)
- **Ultrasonic Sensor**: HC-SR04 x 1 (with servo pan-tilt mount)
- **Camera**: USB Camera (640x480@30fps supported)
- **Communication**: Ethernet / WiFi

### Pin Mapping (consistent with makerobo_code reference)
```
Motor Control (gpiozero.Robot, left/right paired in parallel):
  Left motors:  forward=GPIO22, backward=GPIO27, enable=GPIO18
  Right motors: forward=GPIO25, backward=GPIO24, enable=GPIO23

Ultrasonic Sensor (gpiozero.DistanceSensor):
  TRIG=GPIO20, ECHO=GPIO21

Button & LED:
  Button=GPIO19, Green LED=GPIO5, Red LED=GPIO6

I2C (PCA9685):
  SDA=GPIO2, SCL=GPIO3, Address=0x40

Servo Channels:
  Ch0: Ultrasonic pan-tilt (0-180deg)
  Ch1: Camera horizontal pan (0-180deg)
  Ch2: Camera vertical tilt (-10deg to 90deg)
```

## Module Structure

```
rpi_deploy/
├── __init__.py              # Package initialization
├── hardware_config.py       # Hardware configuration management
├── motor_driver.py          # 4WD motor driver
├── servo_controller.py      # PCA9685 servo control
├── ultrasonic_sensor.py     # HC-SR04 ultrasonic sensor
├── camera_driver.py         # USB camera driver
├── obstacle_avoidance.py    # 3-mode avoidance (simple/servo/apf)
├── remote_control.py        # Network remote control
├── pc_remote_controller.py  # PC-side keyboard remote client
├── pc_v2v_coordinator.py    # PC-side V2V cooperation coordinator
├── rpi_car_controller.py    # Main controller program
└── tests/                   # Hardware test scripts
    ├── __init__.py
    ├── test_motor.py        # Motor test
    ├── test_servo.py        # Servo test
    ├── test_ultrasonic.py   # Ultrasonic sensor test
    ├── test_camera.py       # Camera test
    ├── test_obstacle_avoidance.py # Avoidance algorithm test
    └── test_remote_control.py # Remote control test
```

## Quick Start

### 1. Install Dependencies

On the Raspberry Pi:

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv i2c-tools

# Enable I2C interface
sudo raspi-config  # Interface Options -> I2C -> Enable

# Python dependencies
pip3 install gpiozero smbus2 flask numpy
```

### 2. Clone the Project

```bash
git clone https://github.com/JewelRoam/NexusPilot.git
cd NexusPilot
```

### 3. Hardware Testing

Test each hardware component in order (run from the project root `NexusPilot/`):

```bash
# 1. Test motors (ensure the car has room to move!)
python3 -m rpi_deploy.tests.test_motor --duration 1.0

# 2. Test servos
python3 -m rpi_deploy.tests.test_servo

# 3. Test ultrasonic sensor
python3 -m rpi_deploy.tests.test_ultrasonic --scan

# 4. Test camera
python3 -m rpi_deploy.tests.test_camera --preview --save

# 5. Test remote control (start server on RPi side)
python3 -m rpi_deploy.tests.test_remote_control --server
```

### 4. PC-Side Remote Control

Run the client test on PC:

```bash
python3 -m rpi_deploy.tests.test_remote_control --client --host 192.168.137.33
```

View video stream in browser:
```
http://192.168.137.33:8080
```

## Using the Main Controller

### Obstacle Avoidance Mode (Recommended)

```bash
cd ~/NexusPilot
python3 -m rpi_deploy.rpi_car_controller --mode obstacle_avoidance
```

### Remote Control Mode

Starts a TCP server that accepts remote control commands from a PC client (direction, speed, servo, ultrasonic scan):

```bash
python3 -m rpi_deploy.rpi_car_controller --mode remote
```

PC-side keyboard remote control:
```bash
python -m rpi_deploy.pc_remote_controller --host 192.168.137.33
```

Or programmatic usage:
```python
from rpi_deploy.pc_remote_controller import PCRemoteController
ctrl = PCRemoteController("192.168.137.33", port=5000)
ctrl.connect()
ctrl.send_move('forward')   # WASD direction control
ctrl.send_stop()            # Stop
ctrl.send_servo('ultrasonic', 90)  # Servo control
ctrl.query_status()         # Query vehicle status
ctrl.disconnect()
```

### V2V Cooperative Avoidance Mode

Ultrasonic avoidance + V2V communication, sharing obstacle detection info with the PC side for cooperative avoidance:

```bash
python3 -m rpi_deploy.rpi_car_controller --mode v2v --pc-host 192.168.1.50 --pc-port 5555
```

---

## PC-Side Control Scripts

### PC Keyboard Remote Control (pc_remote_controller.py)

Run an interactive keyboard remote control client on the PC to control the RPi car via TCP:

```bash
python -m rpi_deploy.pc_remote_controller --host 192.168.137.33
python -m rpi_deploy.pc_remote_controller --host 192.168.137.33 --port 5000
```

Keyboard control mapping:

| Key | Function | Key | Function |
|-----|----------|-----|----------|
| W / Up | Forward | S / Down | Backward |
| A / Left | Turn Left | D / Right | Turn Right |
| Space | Stop | Q / Esc | Quit |
| 1-5 | Speed gear (20%-100%) | +/- | Adjust speed |
| U / J | Ultrasonic servo Left/Right | I / O | Camera pan Left/Right |
| K / L | Camera tilt Up/Down | C | Center all servos |
| R | Query status | P | Ultrasonic scan |
| M | Toggle auto-avoidance | H | Help |

### PC V2V Cooperation Coordinator (pc_v2v_coordinator.py)

Run the V2V coordinator on the PC to receive RPi obstacle detections and share PC-side perception results:

```bash
# Standalone monitoring mode (no camera)
python -m rpi_deploy.pc_v2v_coordinator --rpi-host 192.168.137.33

# With camera + YOLO detection mode
python -m rpi_deploy.pc_v2v_coordinator --rpi-host 192.168.137.33 --camera

# Custom port
python -m rpi_deploy.pc_v2v_coordinator --rpi-host 192.168.1.10 --port 5555
```

Communication architecture:
```
RPi (v2v mode) <--UDP--> PC (pc_v2v_coordinator)
  |-- RPi broadcasts: ultrasonic detection results, driving intent, position
  |-- PC broadcasts: camera/YOLO detection results, cooperation requests
  +-- Bidirectional: V2VCommunicator socket mode
```

---

### Camera + YOLO Perception Mode

```bash
python3 -m rpi_deploy.rpi_car_controller --mode camera
# Enable V2V cooperation
python3 -m rpi_deploy.rpi_car_controller --mode camera --cooperative --pc-host 192.168.1.50
```

## Obstacle Avoidance Module (obstacle_avoidance)

A 3-mode obstacle avoidance system implemented based on the `resources/makerobo_code` reference interface.

### Three Avoidance Modes

| Mode | Command | Reference | Algorithm |
|------|---------|-----------|-----------|
| **simple** | `--mode simple` | Case 6 | Front ultrasonic only; obstacle -> reverse + turn right |
| **servo** | `--mode servo` | Case 7 | Servo scans left/right, intelligently chooses turn direction |
| **apf** | `--mode apf` | apf_planner | APF artificial potential field, repulsive force ~ 1/d, smooth avoidance |

### Usage

```bash
cd ~/NexusPilot

# Default servo mode
python3 -m rpi_deploy.obstacle_avoidance

# Simple avoidance (Case 6 style)
python3 -m rpi_deploy.obstacle_avoidance --mode simple

# Servo-enhanced avoidance (Case 7 style)
python3 -m rpi_deploy.obstacle_avoidance --mode servo

# APF potential field avoidance
python3 -m rpi_deploy.obstacle_avoidance --mode apf
```

Press the onboard button to start the robot, `Ctrl+C` to exit.

### APF Algorithm Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| K_REP | 60.0 | Repulsive force gain |
| K_ATT | 1.0 | Attractive force gain |
| D0 | 80cm | Repulsive force influence distance |
| LATERAL_SCALE | 0.6 | Lateral force scaling factor |
| EMERGENCY_DIST | 15cm | Emergency brake distance |

> **Note**: Uses a simplified APF formula F = K_REP x (1/d - 1/D0) without the classic 1/d^2 gradient term, because distances are in centimeters (1-400cm) where the 1/d^2 term produces negligibly small values, resulting in insufficient repulsive force.

### Simulation Mode

When `gpiozero` is unavailable (PC development/debugging), the program automatically enters simulation mode: motor actions are printed to the console, ultrasonic sensor returns a fixed safe distance, and auto-start is enabled (no button press required).

### Avoidance Algorithm Testing

```bash
cd ~/NexusPilot
python3 -m unittest rpi_deploy.tests.test_obstacle_avoidance -v
```

## API Reference

### MotorController (Motor Control, gpiozero.Robot)

```python
from rpi_deploy.motor_driver import MotorController

motor = MotorController()
motor.move_forward(speed=0.3)    # Forward, speed 0.0-1.0
motor.move_backward(speed=0.3)   # Backward
motor.turn_left(speed=0.3)       # Differential left turn
motor.turn_right(speed=0.3)      # Differential right turn
motor.rotate_left(speed=0.3)     # Pivot left turn
motor.rotate_right(speed=0.3)    # Pivot right turn
motor.curve_move(0.3, 0.1)       # Curve movement (linear, angular)
motor.stop()                      # Stop
motor.emergency_stop()            # Emergency stop
```

### ServoController (Servo Control)

```python
from rpi_deploy.servo_controller import ServoController

servo = ServoController()
servo.set_ultrasonic_angle(90)   # Set ultrasonic pan-tilt angle
servo.set_camera_pan(45)         # Set camera horizontal pan angle
servo.set_camera_tilt(30)        # Set camera vertical tilt angle
servo.center_all()               # Center all servos
```

### UltrasonicSensor (Ultrasonic Sensor, gpiozero.DistanceSensor)

```python
from rpi_deploy.ultrasonic_sensor import UltrasonicSensor

sensor = UltrasonicSensor()  # TRIG=GPIO20, ECHO=GPIO21
distance = sensor.measure_once().distance_cm  # Single measurement (cm)
filtered = sensor.measure_average().distance_cm  # Median-filtered measurement
is_obstacle = sensor.is_obstacle_detected(threshold=30)  # Obstacle detection
clear_angle = sensor.find_clear_direction(servo)  # Servo scan for clear direction
```

### CameraDriver (Camera)

```python
from rpi_deploy.camera_driver import CameraDriver

camera = CameraDriver()
camera.open()
camera.start_capture()
frame = camera.get_frame()  # Get image frame
camera.capture_image("photo.jpg")  # Save image
camera.close()
```

### RemoteControlServer (Remote Control)

```python
from rpi_deploy.remote_control import RemoteControlServer

server = RemoteControlServer(host='0.0.0.0', port=5000)
server.register_handler('move', move_handler)
server.register_status_provider(get_status)
server.start()
```

## Configuration

Override default configuration via environment variables:

```bash
export RPI_CONTROL_PORT=5000        # Control port
export RPI_VIDEO_PORT=8080          # Video stream port
export RPI_VEHICLE_ID="CAR_001"     # Vehicle ID
export RPI_CAMERA_WIDTH=640         # Camera width
export RPI_CAMERA_HEIGHT=480        # Camera height
export RPI_SAFE_DISTANCE=30.0       # Safe distance (cm)
```

## Troubleshooting

### I2C Device Not Found
```bash
# Check if I2C is enabled
sudo i2cdetect -y 1

# Should show 0x40 (PCA9685 address)
```

### Motors Not Spinning
- Check power supply voltage (requires 7-12V)
- Confirm gpiozero library is installed
- Check motor wiring

### Camera Cannot Open
```bash
# List available cameras
ls /dev/video*

# Test camera
ffplay /dev/video0
```

### Permission Issues
```bash
# Add user to gpio group
sudo usermod -a -G gpio pi

# Re-login to take effect
```

## Design Principles

This module follows these design principles:

1. **Vendor Compatibility**: Based on LOBOROBOT library hardware control logic
2. **Modular Design**: Each hardware component is independently encapsulated for easy testing and maintenance
3. **Simulation Mode**: Supports code development and testing on non-Raspberry Pi environments
4. **Network Remote Control**: TCP Socket-based PC remote control for convenient debugging
5. **Progressive Deployment**: From single hardware test -> integration test -> complete system

## Simulation-to-Real Mapping

| Simulation Component | Real Hardware |
|---------------------|---------------|
| CARLA Vehicle | 4WD car chassis |
| RGB Camera | USB camera |
| Depth Camera | Ultrasonic + scanning |
| Vehicle Control | Motor driver |
| V2V Communication | WiFi UDP broadcast |