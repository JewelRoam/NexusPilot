"""
NexusPilot: Professional RPi Car Controller
Fixed import ordering to prevent RPi 5 system hangs.
"""
import os
# CRITICAL: LG_CHIP must be set BEFORE any hardware library imports
os.environ["LG_CHIP"] = "0"

import sys
import time
import argparse
import cv2
import numpy as np

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

    # 1. Hardware Init
    print("[INIT] Booting Hardware Subsystems...")
    motor = MotorController()
    ultrasonic = UltrasonicSensor()
    
    # Quick pulse for confirmation
    motor.curve_move(0.4, 0.0)
    time.sleep(0.2)
    motor.stop()

    # 2. Camera Init
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

    # 3. Software Init
    print("[INIT] Loading YOLOv11n (approx. 5s)...")
    detector = YOLODetector({"onnx_model_path": "model/yolo11n.onnx", "imgsz": 320})
    planner = APFPlanner({
        "k_attractive": 1.5,
        "k_repulsive": 180.0,
        "d0": 2.2,
        "emergency_distance": 0.30,
        "max_speed": 10.0
    })

    # Warmup
    detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))
    
    print("\n" + "="*40)
    print(" NEXUSPILOT: READY ")
    print("="*40)
    
    BASE_PWM = 0.35
    dist_buffer = {}
    last_log_time = 0

    try:
        while True:
            loop_start = time.perf_counter()

            u_res = ultrasonic.measure_once()
            if u_res.valid and u_res.distance_cm < 15.0:
                motor.emergency_stop()
                if time.time() - last_log_time > 2.0:
                    print(f"!! SAFETY LOCK: {u_res.distance_cm:.1f}cm")
                    last_log_time = time.time()
                continue

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
                
                cx = dist_buffer[tid] + 0.12
                cy = (det.center[0] - 160) / 160.0 * cx * 0.55
                apf_obs_list.append(Obstacle(x=cx, y=cy, distance=cx, category=det.category))

            out = planner.compute(0, 0, 0, 8.0, 2.5, 0, apf_obs_list)

            if out.status == "emergency" or out.emergency_brake:
                motor.stop()
            elif out.status == "recovering":
                motor.move_backward(0.4)
            else:
                pwm = (out.target_speed / 10.0) * (1.0 - BASE_PWM) + BASE_PWM
                if out.target_speed < 0.1: motor.stop()
                else: motor.curve_move(pwm, out.target_steering)

            if not args.headless:
                cv2.imshow("Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            elapsed = time.perf_counter() - loop_start
            if elapsed < 0.066: time.sleep(0.066 - elapsed)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        motor.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
