"""
NexusPilot: Robust RPi Car Controller
High-stability version with FPS capping, log rate-limiting, and RPi 5 GPIO fixes.
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np

# Force Raspberry Pi 5 to use the correct GPIO chip (gpiochip0)
os.environ["LG_CHIP"] = "0"

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.detector import YOLODetector
from planning.apf_planner import APFPlanner, Obstacle
from rpi_deploy.motor_driver import MotorController
from rpi_deploy.ultrasonic_sensor import UltrasonicSensor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    # --- Constants & Calibration ---
    BASE_PWM = 0.32           # Min power for RPi 5 motors
    CAMERA_OFFSET_X = 0.12    # Offset from axle (meters)
    MAX_FPS = 15              # Throttle to keep CPU cool and SSH alive
    FRAME_INTERVAL = 1.0 / MAX_FPS
    
    # 1. Hardware Init
    print("[INFO] Initializing Hardware...")
    motor = MotorController()
    ultrasonic = UltrasonicSensor()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. Software Init
    print("[INFO] Loading Perception Model...")
    detector = YOLODetector({"use_onnx": True, "imgsz": 320, "confidence_threshold": 0.4})
    planner = APFPlanner({
        "k_attractive": 1.2,
        "k_repulsive": 160.0,
        "d0": 2.2,
        "emergency_distance": 0.35, # 35cm
        "max_speed": 10.0 # Slow & Safe for Demo
    })

    print("\n[READY] System Stabilized. Watching for obstacles...")
    dist_buffer = {}
    last_hard_stop_time = 0

    try:
        while True:
            start_time = time.perf_counter()

            # --- Safety Layer 1: Ultrasonic ---
            u_reading = ultrasonic.measure_once()
            if u_reading.valid and u_reading.distance_cm < 20.0:
                motor.emergency_stop()
                if time.time() - last_hard_stop_time > 1.0: # Rate limit logs
                    print(f"!!! ULTRASONIC HARD STOP: {u_reading.distance_cm:.1f}cm")
                    last_hard_stop_time = time.time()
                time.sleep(0.1) # Yield CPU
                continue

            # --- Perception ---
            ret, frame = cap.read()
            if not ret: continue
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect(rgb)
            obstacles = detector.get_obstacles(detections)
            
            apf_obs_list = []
            for det in obstacles:
                raw_dist = 480.0 / (det.height + 1e-6) * 0.32 
                tid = det.track_id
                if tid not in dist_buffer: dist_buffer[tid] = raw_dist
                dist_buffer[tid] = 0.7 * dist_buffer[tid] + 0.3 * raw_dist
                
                corrected_x = dist_buffer[tid] + CAMERA_OFFSET_X
                lateral = (det.center[0] - 320) / 320.0 * corrected_x * 0.55
                apf_obs_list.append(Obstacle(x=corrected_x, y=lateral, distance=corrected_x, category=det.category))

            # --- Planning ---
            out = planner.compute(0, 0, 0, 8.0, 2.5, 0, apf_obs_list)

            # --- Execution ---
            if out.emergency_brake or out.status == "emergency":
                motor.stop()
            elif out.status == "recovering":
                motor.move_backward(0.4)
            else:
                clamped_speed = speed_req * (1.0 - BASE_PWM) + BASE_PWM
                if out.target_speed < 0.1: 
                    motor.stop()
                else: 
                    if time.time() - last_move_log_time > 2.0:
                        print(f"[MOVE] PWM:{clamped_speed:.2f} Steer:{out.target_steering:.2f} Status:{out.status}")
                        last_move_log_time = time.time()
                    motor.curve_move(clamped_speed, out.target_steering)

            # --- FPS Throttling (Keep System Stable) ---
            if not args.headless:
                cv2.imshow("Security Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
            
            elapsed = time.perf_counter() - start_time
            if elapsed < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - elapsed)

    except KeyboardInterrupt:
        print("\n[INFO] Manual Shutdown Received.")
    finally:
        motor.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
