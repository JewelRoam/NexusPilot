<div align="center">

# 🚗 NexusPilot

**Cooperative Autonomous Driving with V2V Communication**

*Nexus = Interconnected Hub · Pilot = Autonomous Driving*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.13-FF6F00?logo=carla&logoColor=white)](https://carla.org)
[![ROS2](https://img.shields.io/badge/ROS2-Jazzy-2259A3?logo=ros&logoColor=white)](https://docs.ros.org/en/jazzy/)
[![YOLO](https://img.shields.io/badge/YOLO-v8%2Fv11-00FFFF?logo=yolo&logoColor=black)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-CARLA%20%7C%20RPi5-blueviolet)]()

</div>

---

A modular **Perception → Planning → Control** pipeline for autonomous obstacle detection and avoidance, featuring **multi-vehicle V2V cooperative driving** as the core innovation. From CARLA simulation to Raspberry Pi 5 real-world deployment.

## ✨ Key Features

- 🎯 **YOLO-based Perception** — Real-time object detection with YOLOv8/v11 (PyTorch & ONNX)
- 🧭 **APF Path Planning** — Artificial Potential Field with Bézier smoothing and ethical VRU weighting
- 🎮 **PID Vehicle Control** — Unified controller for CARLA simulation and RPi differential drive
- 📡 **V2V Cooperation** — Distance-aware multi-vehicle coordination with occlusion compensation
- 🏎️ **CARLA Simulation** — Full sync-mode simulation with single & multi-vehicle demos
- 🍓 **RPi 5 Deployment** — 4WD robot with ultrasonic sensor, servo pan-tilt, and camera stream
- 🧪 **Automated Testing** — Scenario generation, metrics collection, and APF parameter auto-tuning
- 🤖 **ROS 2 Integration** — Dual-environment ROS 2 nodes with ZeroMQ bridge

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NexusPilot System Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │Perception│──▶│ Planning │──▶│ Control  │──▶│ Actuator │   │
│  │  (YOLO)  │   │  (APF)   │   │  (PID)   │   │CARLA/RPi │   │
│  └────┬─────┘   └────┬─────┘   └──────────┘   └──────────┘   │
│       │              │                                         │
│       │         ┌────┴─────┐                                   │
│       └────────▶│   V2V    │◀── Cooperative Obstacles          │
│  Shared Dets   │Cooperation│    Intent Sharing                 │
│                │ (Socket/  │    Intersection Negotiation       │
│                │ In-Proc)  │    Occlusion Compensation         │
│                └──────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
NexusPilot/
├── config/
│   └── config.yaml              # Unified configuration
├── perception/
│   ├── detector.py              # YOLO object detection (PT/ONNX)
│   └── depth_estimator.py       # Depth estimation (CARLA depth cam)
├── planning/
│   └── apf_planner.py           # Artificial Potential Field + Bézier smoothing
├── control/
│   └── vehicle_controller.py    # PID controller (CARLA/RPi)
├── cooperation/
│   ├── v2v_message.py           # V2V message protocol & communication
│   └── cooperative_planner.py   # Multi-vehicle coordination & scenarios
├── simulation/
│   ├── carla_env.py             # CARLA environment wrapper (sync mode)
│   ├── single_vehicle_demo.py   # Single-vehicle closed-loop demo
│   └── multi_vehicle_demo.py    # Multi-vehicle cooperative demo ★
├── testing/                     # Automated Testing Framework
│   ├── scenario_generator.py    # Programmatic scenario generation
│   ├── metrics_collector.py     # Performance metrics collection
│   ├── parameter_tuner.py       # APF parameter auto-tuning
│   ├── report_generator.py      # HTML/Markdown test reports
│   ├── automated_runner.py      # Main test orchestrator
│   └── example_usage.py         # Usage examples
├── ros2_nodes/                  # ROS 2 Integration (Dual-Environment)
│   ├── carla_bridge/            # CARLA ↔ ZeroMQ bridge
│   ├── perception_node.py       # ROS 2 perception node
│   ├── planning_node.py         # ROS 2 planning node
│   ├── control_node.py          # ROS 2 control node
│   ├── cooperation_node.py      # ROS 2 multi-vehicle cooperation
│   └── zmq_to_ros_node.py       # ZeroMQ → ROS 2 converter
├── rpi_deploy/                  # Raspberry Pi 5 real-world deployment
│   ├── hardware_config.py       # Pin mappings & configuration
│   ├── motor_driver.py          # 4WD motor control (gpiozero)
│   ├── servo_controller.py      # PCA9685 servo (I2C)
│   ├── ultrasonic_sensor.py     # HC-SR04 distance sensor
│   ├── camera_driver.py         # USB camera + Flask MJPEG stream
│   ├── obstacle_avoidance.py    # 3-mode avoidance (simple/servo/apf)
│   ├── remote_control.py        # TCP JSON remote control server
│   ├── rpi_car_controller.py    # Main controller (4 modes)
│   ├── pc_remote_controller.py  # PC-side keyboard remote client
│   ├── pc_v2v_coordinator.py    # PC-side V2V cooperation coordinator
│   └── tests/                   # Hardware test scripts
├── model/
│   ├── train.py                 # YOLOv11 training on KITTI
│   ├── export_model.py          # ONNX export
│   ├── yolov8n.pt               # Pre-trained weights
│   ├── yolo11n.pt               # YOLOv11 weights
│   └── yolo11n.onnx             # Exported ONNX model
├── utils/
│   └── logger.py                # Logging & performance metrics
├── docs/                        # Documentation
├── requirements.txt             # Python dependencies
└── LICENSE                      # MIT License
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+ with venv
- CARLA 0.9.13 simulator
- NVIDIA GPU with CUDA (for YOLO inference)

### 1️⃣ Single-Vehicle Demo (CARLA)

```bash
# Start CARLA server
CARLA_0.9.13\WindowsNoEditor\CarlaUE4.exe

# Run single-vehicle closed-loop demo
python -m simulation.single_vehicle_demo
```

Demonstrates: YOLO detection → depth estimation → APF planning → PID control → real-time HUD

### 2️⃣ Multi-Vehicle Cooperative Demo ★

```bash
# Start CARLA server first, then:
python -m simulation.multi_vehicle_demo
```

Demonstrates the **core innovation** — V2V cooperation:
- Multiple vehicles with independent perception pipelines
- V2V message exchange (position, detections, intent)
- Cooperative obstacle fusion (occlusion compensation)
- Intersection coordination and deadlock resolution
- Split-screen view with Bird's Eye View overlay

### 3️⃣ Automated Testing & Parameter Tuning

```bash
# Quick test suite (5 scenarios)
python -m testing.automated_runner --suite quick

# Auto-tune APF parameters
python -m testing.automated_runner --tune --iterations 20

# Apply optimized parameters
python testing/apply_best_config.py
```

| Feature | Benefit |
|---------|---------|
| Headless execution | No GUI needed — generates HTML reports |
| Quantitative metrics | Collision rate, speed, latency, etc. |
| Auto parameter search | Finds optimal APF coefficients |
| A/B comparison | Side-by-side config evaluation |

### 4️⃣ Raspberry Pi Deployment

See [rpi_deploy/README.md](rpi_deploy/README.md) for full documentation.

```bash
# Obstacle avoidance mode
python3 -m rpi_deploy.rpi_car_controller --mode obstacle_avoidance

# Remote control mode (PC keyboard)
python3 -m rpi_deploy.rpi_car_controller --mode remote

# V2V cooperative mode
python3 -m rpi_deploy.rpi_car_controller --mode v2v --pc-host 192.168.1.50

# Camera + YOLO perception mode
python3 -m rpi_deploy.rpi_car_controller --mode camera
```

## 📖 Module Details

### 🎯 Perception (`perception/`)
- **YOLODetector** — Wraps Ultralytics YOLO (PT/ONNX), classifies vehicles/pedestrians/cyclists/signs
- **DepthEstimator** — Converts 2D detections to 3D positions via CARLA depth camera

### 🧭 Planning (`planning/`)
- **APFPlanner** — Attractive force toward goal, repulsive force from obstacles, cooperative obstacle integration, emergency brake logic
- **BezierSmoother** — Quintic Bézier curve path smoothing

### 🎮 Control (`control/`)
- **VehicleController** — Unified PID controller with CARLA mode (throttle/steer/brake) and RPi mode (differential drive PWM)

### 📡 Cooperation (`cooperation/`) ★ Core Innovation
- **V2VCommunicator** — In-process (CARLA) and socket (PC ↔ RPi) modes with configurable latency/dropout
- **CooperativePlanner** — Distance-aware coordination:
  - `>50m` INACTIVE → `30-50m` MONITORING → `15-30m` ACTIVE → `8-15m` WARNING → `<8m` CRITICAL
  - TTI-based intersection priority (Time-to-Intersection)
  - Occlusion compensation, platooning, deadlock resolution

See [docs/DISTANCE_AWARE_COORDINATION.md](docs/DISTANCE_AWARE_COORDINATION.md) for details.

### 🧪 Testing Framework (`testing/`)

```
testing/
├── automated_runner.py      # Main orchestrator
├── scenario_generator.py    # Generate test scenarios
├── metrics_collector.py     # Collect performance metrics
├── parameter_tuner.py       # Auto-tune APF parameters
├── report_generator.py      # HTML/Markdown reports
└── example_usage.py         # Usage examples
```

**Evaluation Metrics:**

| Metric | Description | Target |
|--------|-------------|--------|
| Collision Count | Collisions per scenario | 0 |
| Near Misses | Close calls (< 3m) | < 2 |
| Average Speed | Mean driving speed | > 15 km/h |
| Min Obstacle Distance | Closest approach | > 2 m |
| Lane Deviation | Distance from center | < 0.5 m |
| Completion Rate | Scenarios passed | > 80% |
| Perception Latency | YOLO inference time | < 50 ms |
| Planning Latency | APF computation time | < 10 ms |
| Cooperative Gain | V2V improvement | > 20% |

## ⚙️ Configuration

All parameters are centralized in `config/config.yaml`:
- CARLA connection and simulation settings
- YOLO model path and detection thresholds
- APF planner coefficients (k_att, k_rep, d0)
- PID controller gains
- V2V communication protocol and parameters
- Raspberry Pi hardware pin mappings

## 📅 Development Timeline

| Phase | Target | Status |
|-------|--------|--------|
| Sensor Data Integration | CARLA + ROS 2 data stream | ✅ Done |
| Obstacle Detection Model | YOLOv11n trained on KITTI | ✅ Done |
| PC Single-Vehicle Loop | Perception→APF→Control in CARLA | 🔨 Ready |
| PC Multi-Vehicle Cooperation | V2V cooperative avoidance | 🔨 Ready |
| RPi Car Deployment | Real-world obstacle avoidance | ✅ Done |
| Multi-Vehicle Testing | V2V cooperative scenarios | 📋 Planned |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Simulation | CARLA 0.9.13 |
| Perception | YOLOv8 / YOLOv11 (Ultralytics) |
| Planning | Artificial Potential Field + Bézier |
| Control | PID Controller |
| Communication | V2V Socket / UDP |
| Middleware | ROS 2 Jazzy + ZeroMQ |
| Real-world | Raspberry Pi 5 + gpiozero + PCA9685 |
| Language | Python 3.10+ |

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.