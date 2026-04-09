# QMC9 - Intelligent Obstacle Detection and Avoidance System for Autonomous Cars

## Project Overview

A modular **Perception → Planning → Control** pipeline for autonomous obstacle detection and avoidance, with **multi-vehicle V2V cooperative** capabilities as the key innovation.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    System Architecture                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   │
│  │Perception│──▶│ Planning │──▶│ Control  │──▶│ Actuator │   │
│  │  (YOLO)  │   │  (APF)   │   │  (PID)   │   │(CARLA/RPi│   │
│  └────┬─────┘   └────┬─────┘   └──────────┘   └──────────┘   │
│       │              │                                         │
│       │         ┌────┴─────┐                                   │
│       └────────▶│   V2V    │◀── Cooperative Obstacles          │
│  Shared Dets   │Cooperation│    Intent Sharing                 │
│                 │ (Socket/  │    Intersection Negotiation       │
│                 │ In-Proc)  │    Occlusion Compensation         │
│                 └──────────┘                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
QMC9_Project/
├── config/
│   └── config.yaml              # Unified configuration
├── perception/
│   ├── detector.py              # YOLO object detection (PT/ONNX)
│   └── depth_estimator.py       # Depth estimation (CARLA depth cam)
├── planning/
│   └── apf_planner.py           # Artificial Potential Field + Bézier smoothing
├── control/
│   └── vehicle_controller.py    # PID controller (CARLA/RPi/STM32)
├── cooperation/
│   ├── v2v_message.py           # V2V message protocol & communication
│   └── cooperative_planner.py   # Multi-vehicle coordination & scenarios
├── simulation/
│   ├── carla_env.py             # CARLA environment wrapper (sync mode)
│   ├── single_vehicle_demo.py   # Single-vehicle closed-loop demo
│   └── multi_vehicle_demo.py    # Multi-vehicle cooperative demo ★
├── testing/                     # NEW: Automated Testing Framework
│   ├── scenario_generator.py    # Programmatic scenario generation
│   ├── metrics_collector.py     # Performance metrics collection
│   ├── parameter_tuner.py       # APF parameter auto-tuning
│   ├── report_generator.py      # HTML/Markdown test reports
│   ├── automated_runner.py      # Main test orchestrator
│   └── example_usage.py         # Usage examples
├── ros2_nodes/                  # NEW: ROS 2 Integration (Dual-Environment)
│   ├── carla_bridge/            # CARLA ↔ ZeroMQ bridge (Python 3.7)
│   ├── perception_node.py       # ROS 2 perception node
│   ├── planning_node.py         # ROS 2 planning node
│   ├── control_node.py          # ROS 2 control node
│   ├── cooperation_node.py      # ROS 2 multi-vehicle cooperation
│   ├── zmq_to_ros_node.py       # ZeroMQ → ROS 2 converter
│   ├── run_carla_bridge.bat     # Launch CARLA bridge
│   ├── run_ros_stack.bat        # Launch ROS 2 nodes
│   └── README.md                # ROS 2 setup guide
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
│   ├── yolov8n.pt               # Pre-trained weights (CARLA py37 compatible)
│   ├── yolo11n.pt               # YOLOv11 weights (for newer Python envs)
│   └── yolo11n.onnx             # Exported ONNX model
├── utils/
│   └── logger.py                # Logging & performance metrics
├── pc_simulation.py             # Original CARLA+YOLO detection demo
└── README.md                    # This file
```

## Quick Start

### Prerequisites

- Python 3.10+ with venv
- CARLA 0.9.13 simulator (already installed at `CARLA_0.9.13/`)
- NVIDIA GPU with CUDA (for YOLO inference)

### Step 1: PC Simulation - Single Vehicle (Closed-Loop)

```bash
# 1. Start CARLA server
CARLA_0.9.13\WindowsNoEditor\CarlaUE4.exe

# 2. Run single-vehicle demo (full perception→planning→control loop)
python -m simulation.single_vehicle_demo
```

This demonstrates:
- YOLO object detection on CARLA camera feed
- Depth estimation via CARLA depth camera
- APF obstacle avoidance planning
- PID vehicle control
- Real-time HUD with detection info, planning status, and control output

### Step 2: PC Simulation - Multi-Vehicle Cooperative (Innovation)

```bash
# Start CARLA server first, then:
python -m simulation.multi_vehicle_demo
```

This demonstrates the **key innovation** - V2V cooperation:
- Multiple vehicles with independent perception pipelines
- V2V message exchange (position, detections, intent)
- Cooperative obstacle fusion (occlusion compensation)
- Intersection coordination and deadlock resolution
- Split-screen view with Bird's Eye View overlay

### Step 3: Automated Testing & Parameter Tuning (NEW)

The testing framework allows you to evaluate and optimize your algorithm **without manual debugging**:

```bash
# Quick test suite - runs 5 scenarios automatically
python -m testing.automated_runner --suite quick

# Auto-tune APF parameters (finds optimal k_att, k_rep, d0)
python -m testing.automated_runner --tune --iterations 20

# Run examples showing all features
python testing/example_usage.py --example 1  # Quick test
python testing/example_usage.py --example 2  # Parameter tuning
python testing/example_usage.py --example 3  # Compare configs
python testing/example_usage.py --example 4  # Single scenario debug
```

**Benefits:**
- ✅ **No GUI needed** - runs headless and generates HTML reports
- ✅ **Quantitative metrics** - collision rate, speed, latency, etc.
- ✅ **Auto parameter search** - finds best APF coefficients automatically
- ✅ **Scenario library** - intersection, occlusion, pedestrian crossing, etc.
- ✅ **A/B comparison** - compare two configurations side-by-side

### Step 4: Raspberry Pi Deployment

See [rpi_deploy/README.md](rpi_deploy/README.md) for full documentation.

```bash
# Obstacle avoidance mode (recommended)
python3 -m rpi_deploy.rpi_car_controller --mode obstacle_avoidance

# Remote control mode (PC keyboard control)
python3 -m rpi_deploy.rpi_car_controller --mode remote

# V2V cooperative mode (PC as Edge Cloud)
python3 -m rpi_deploy.rpi_car_controller --mode v2v --pc-host 192.168.1.50

# Camera + YOLO perception mode
python3 -m rpi_deploy.rpi_car_controller --mode camera

# PC-side keyboard remote controller
python -m rpi_deploy.pc_remote_controller --host 192.168.137.33
```

## Module Details

### Perception (`perception/`)
- **YOLODetector**: Wraps Ultralytics YOLO for object detection
  - Supports PyTorch (.pt) and ONNX (.onnx) formats
  - Classifies: vehicles, pedestrians, cyclists, traffic signs
  - Configurable image size and confidence threshold
- **DepthEstimator**: Converts 2D detections to 3D positions
  - CARLA depth camera decoding
  - Pixel-to-world coordinate transformation

### Planning (`planning/`)
- **APFPlanner**: Artificial Potential Field obstacle avoidance
  - Attractive force toward goal waypoint
  - Repulsive force from obstacles (with ethical VRU weighting)
  - Cooperative obstacle integration from V2V
  - Emergency brake logic
- **BezierSmoother**: Quintic Bézier curve path smoothing

### Control (`control/`)
- **VehicleController**: Unified controller with PID
  - CARLA mode: throttle/steer/brake
  - RPi mode: differential drive PWM
  - Emergency brake handling

### Cooperation (`cooperation/`) ★ Innovation
- **V2VCommunicator**: Message passing between vehicles
  - In-process mode (CARLA simulation)
  - Socket mode (PC ↔ RPi real-world)
  - Configurable latency/dropout for robustness testing
- **CooperativePlanner**: Multi-vehicle coordination with **Distance-Aware Coordination**
  - **Distance-based activation**: Only coordinates when vehicles are close enough to benefit
    - `>50m`: INACTIVE - no coordination overhead
    - `30-50m`: MONITORING - passive tracking, platooning possible
    - `15-30m`: ACTIVE - full coordination (intersection, deadlock detection)
    - `8-15m`: WARNING - close proximity alert
    - `<8m`: CRITICAL - emergency braking
  - **TTI-based intersection priority**: Time-to-Intersection calculation for fair priority assignment
    - Vehicle arriving FIRST at conflict point gets priority
    - Close timing (<2s difference) uses deterministic ID tiebreaker
  - Occlusion compensation (shared detection fusion)
  - Platooning (convoy following)
  - Deadlock resolution (yield negotiation)

See [docs/DISTANCE_AWARE_COORDINATION.md](docs/DISTANCE_AWARE_COORDINATION.md) for details.

## Automated Testing Framework (`testing/`)

The testing framework provides **automated evaluation and parameter optimization** to reduce debugging costs.

### Framework Structure

```
testing/
├── automated_runner.py      # Main orchestrator - run test suites
├── scenario_generator.py    # Generate diverse test scenarios
├── metrics_collector.py     # Collect performance metrics
├── parameter_tuner.py       # Auto-tune APF parameters
├── report_generator.py      # Generate HTML/Markdown reports
├── apply_best_config.py     # Apply tuned parameters to config.yaml
└── example_usage.py         # Usage examples
```

### Core Components

#### 1. Scenario Generator (`scenario_generator.py`)
Programmatically generates test scenarios:
- **Straight road** (easy/medium/hard) - basic lane keeping with obstacles
- **Intersection** (easy/medium/hard) - crossing traffic, right-of-way
- **Occlusion** - hidden obstacle behind leading vehicle (V2V test)
- **Pedestrian crossing** (low/medium/high density) - VRU priority testing
- **Randomized** - fully random scenarios for robustness testing

```python
from testing.scenario_generator import ScenarioGenerator

gen = ScenarioGenerator(env)
scenario = gen.generate_intersection_scenario(difficulty='medium')
suite = gen.generate_test_suite("quick")  # 5 scenarios
```

#### 2. Metrics Collector (`metrics_collector.py`)
Collects comprehensive performance data:
- **Safety**: collision count, near misses, min obstacle distance
- **Performance**: average/max speed, distance traveled, time stopped
- **Lane keeping**: deviation from center, time out of lane
- **Latency**: perception/planning/control timing breakdown
- **V2V**: cooperative gain metrics

```python
from testing.metrics_collector import MetricsCollector

collector = MetricsCollector()
collector.start_recording()
# ... run simulation ...
summary = collector.stop_recording()  # Returns TestRunSummary
```

#### 3. Parameter Tuner (`parameter_tuner.py`)
Automatically finds optimal APF parameters using random search:

**Search Space:**
- `k_attractive`: [0.5, 3.0] - goal attraction strength
- `k_repulsive`: [50, 300] - obstacle repulsion strength
- `d0`: [8, 25] - repulsion influence distance
- `max_speed`: [20, 40] - maximum vehicle speed
- `emergency_distance`: [2, 5] - emergency brake threshold

**Scoring Function:**
```
score = safety_score * 0.4 + collision_avoidance * 0.3 + completion_rate * 0.2 + efficiency * 0.1
```

```python
from testing.parameter_tuner import quick_tune_apf

best_config = quick_tune_apf(env, num_scenarios=5, num_configs=20)
print(f"Best k_repulsive: {best_config.k_repulsive}")
```

#### 4. Report Generator (`report_generator.py`)
Generates visual reports:
- **HTML report** with tables, charts, and color-coded pass/fail
- **Markdown summary** for quick viewing
- **Comparison tables** for A/B testing

```python
from testing.report_generator import ReportGenerator

gen = ReportGenerator("My Test Results")
gen.add_run(summary1, "Config A")
gen.add_run(summary2, "Config B")
gen.generate_html("output/report.html")
```

### Complete Workflow Example

```bash
# 1. Start CARLA server
CARLA_0.9.13\WindowsNoEditor\CarlaUE4.exe

# 2. Run parameter tuning (finds best APF parameters)
python -m testing.automated_runner --tune --iterations 20
# Output: output/best_apf_config.json

# 3. Apply best configuration to config.yaml
python testing/apply_best_config.py
# This updates config.yaml with optimized parameters

# 4. Verify with single vehicle demo
python -m simulation.single_vehicle_demo

# 5. Run full test suite to validate
python -m testing.automated_runner --suite standard
# Generates: output/test_report.html
```

### Command Line Usage

```bash
# Quick test (5 scenarios, ~5 minutes)
python -m testing.automated_runner --suite quick

# Standard test (15 scenarios, ~15 minutes)
python -m testing.automated_runner --suite standard

# Full test (30 scenarios including randomized, ~30 minutes)
python -m testing.automated_runner --suite full

# Parameter tuning only
python -m testing.automated_runner --tune --iterations 30

# Apply tuned parameters to config.yaml
python -m testing.apply_best_config

# Compare two configurations
python testing/example_usage.py --example 3
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Collision Count | Number of collisions per scenario | 0 |
| Near Misses | Close calls (< 3m to obstacle) | < 2 |
| Average Speed | Mean driving speed | > 15 km/h |
| Min Obstacle Distance | Closest approach to any obstacle | > 2 m |
| Lane Deviation | Average distance from lane center | < 0.5 m |
| Completion Rate | % of scenarios passed | > 80% |
| Perception Latency | YOLO inference time | < 50 ms |
| Planning Latency | APF computation time | < 10 ms |
| Cooperative Gain | Improvement with V2V enabled | > 20% |

### Benefits Summary

| Problem | Solution |
|---------|----------|
| Manual debugging is slow | Automated scenario execution |
| Can't see algorithm behavior | Quantitative metrics + HTML reports |
| Parameter tuning by trial-and-error | Random/grid search with scoring |
| Hard to compare configurations | A/B testing with side-by-side tables |
| No record of improvements | JSON export + trend analysis |

## Configuration

All parameters are centralized in `config/config.yaml`:
- CARLA connection and simulation settings
- YOLO model path and detection thresholds
- APF planner coefficients (k_att, k_rep, d0)
- PID controller gains
- V2V communication protocol and parameters
- Raspberry Pi hardware pin mappings

## Development Timeline (aligned with Gantt chart)

| Phase | Target | Status |
|-------|--------|--------|
| Sensor Data Integration | CARLA + ROS 2 data stream | ✅ Done |
| Obstacle Detection Model | YOLOv11n trained on KITTI | ✅ Done |
| PC Single-Vehicle Loop | Perception→APF→Control in CARLA | 🔨 Ready |
| PC Multi-Vehicle Cooperation | V2V cooperative avoidance | 🔨 Ready |
| RPi Car Deployment | Real-world obstacle avoidance | ✅ Done |
| Multi-Vehicle Testing | V2V cooperative scenarios | 📋 Planned |
| Mid-term Evaluation | Performance metrics & report | 📋 Planned |
