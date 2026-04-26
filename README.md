<div align="center">

# 🚗 NexusPilot

**Cooperative Autonomous Driving with V2V Communication**

*Interconnected Hub (Nexus) · Autonomous Navigation (Pilot)*

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![CARLA](https://img.shields.io/badge/CARLA-0.9.13-FF6F00?logo=carla&logoColor=white)](https://carla.org)
[![YOLO](https://img.shields.io/badge/YOLO-v11n-00FFFF?logo=yolo&logoColor=black)](https://ultralytics.com)
[![Platform](https://img.shields.io/badge/Platform-CARLA%20%7C%20RPi5-blueviolet)]()

</div>

---

NexusPilot is an integrated autonomous driving system that combines **AI-based vision perception**, **Improved Artificial Potential Field (APF) planning**, and **decentralized V2V cooperative logic**. The system has been rigorously validated on both the CARLA 0.9.13 simulator and a physical Raspberry Pi 5 4WD platform.

## 🏆 Final Performance Metrics

| Metric | Simulation (CARLA) | Hardware (RPi 5) | Status |
|--------|-------------------|------------------|--------|
| **Success Rate (OASR)** | **90.4%** (250 trials) | **75.0%** (20 trials) | ✅ Validated |
| **Pipeline Throughput** | 19.3 FPS | **12.7 FPS** | ✅ Real-time |
| **V2V Cooperative Gain**| **45.0%** collision reduction | Verified (2-car) | ✅ Proven |
| **Local Minima Escape** | 85.7% Success | Integrated | ✅ Robust |

## ✨ Core Innovations

- 🎯 **Advanced Perception:** YOLO11n fine-tuned on KITTI, optimized via OpenVINO for ARM NEON acceleration.
- 🧭 **Improved APF Planner:** Features a **Rotational Field** mechanism for deadlock escape and **C1 Bézier smoothing** for jitter-free steering.
- 📡 **V2V Cooperation:** A distance-aware coordination model using **TTI (Time-to-Intersection)** and temporal extrapolation for latency compensation.
- 🍓 **Sim-to-Real Ready:** Modular architecture with high-fidelity hardware abstraction layers for seamless deployment.

## 🏗️ System Architecture

```mermaid
graph TD
    S[Sensors: Cam/Ultra] --> P[Perception: YOLO/Tracking]
    P --> Pl[Planning: Improved APF]
    Pl --> C[Control: PID/Pure Pursuit]
    C --> A[Actuators: Motor/Servo]
    
    Coop[Cooperation: V2V Protocol] <--> P
    Coop <--> Pl
```

## 📂 Project Structure

- `rpi_deploy/`: Deployment scripts for Raspberry Pi 5 (Motor, Servo, Ultrasonic, Camera).
- `planning/`: Implementation of the Improved APF and Bézier path generation.
- `perception/`: YOLO11n detector with OpenVINO backend and IOU-based object tracking.
- `cooperation/`: TTI-based V2V communication protocol and coordination logic.
- `simulation/`: CARLA wrappers and multi-vehicle benchmarking scenarios.
- `testing/`: Automated metrics collection and parameter tuning framework.

## 🚀 Deployment (Raspberry Pi 5)

Ensure you are in the `.pi_env` and run:

```bash
# Core Autonomous Mode
python rpi_deploy/rpi_car_controller.py

# Hardware Diagnostics
python rpi_deploy/ultrasonic_sensor.py
python rpi_deploy/motor_driver.py
```

## 📄 Documentation

The technical details, mathematical derivations, and comprehensive experimental analysis are documented in the final thesis:
- 📖 **[QMC9_Final_Report.pdf](https://github.com/JewelRoam/QMC9_project_docs)**

---
*This project is submitted as part of the undergraduate degree programme at Queen Mary University of London and Beijing University of Posts and Telecommunications.*
