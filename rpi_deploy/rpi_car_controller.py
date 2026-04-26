"""
NexusPilot: RPi Car Controller v2
- YOLO vision (throttled to 3 FPS) + servo-enhanced ultrasonic scanning
- State-machine based obstacle avoidance (inspired by makerobo Case 7)
- CPU-optimized for Raspberry Pi 5
"""
import os
import sys
import time
import argparse
import cv2
import numpy as np

# Force RPi 5 GPIO chip index
os.environ["LG_CHIP"] = "0"

# Path setup
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.detector import YOLODetector
from planning.apf_planner import APFPlanner, Obstacle
from rpi_deploy.motor_driver import MotorController
from rpi_deploy.ultrasonic_sensor import UltrasonicSensor
from rpi_deploy.servo_controller import ServoController

# ---- Constants ----

# Ultrasonic thresholds (cm)
EMERGENCY_DIST = 15.0      # Hard stop
OBSTACLE_DIST = 40.0       # Trigger scan + avoidance
WARNING_DIST = 60.0        # Slow down, APF steering

# Speed settings (0.0-1.0 gpiozero scale)
CRUISE_SPEED = 0.2
AVOID_BACKUP_SPEED = 0.3
AVOID_TURN_SPEED = 0.3
RECOVERY_BACKUP_DURATION = 0.5
RECOVERY_TURN_DURATION = 1.0

# Servo settle time (seconds)
SERVO_SETTLE = 0.4

# Camera
CAM_WIDTH = 320
CAM_HEIGHT = 320
CAM_CENTER_X = CAM_WIDTH / 2.0

# YOLO inference throttling: run detection every N frames
YOLO_FRAME_INTERVAL = 4    # ~3 FPS at 12 Hz main loop
YOLO_CONFIDENCE_THRESHOLD = 0.45  # Higher threshold = fewer detections = less CPU

# Main loop target: 12 Hz (ultrasonic-only frames are ~80 µs, YOLO frames are ~200 ms)
MAIN_LOOP_INTERVAL = 0.083  # ~12 FPS for ultrasonic safety checks

# State machine
STATE_CRUISE = "CRUISE"
STATE_BLOCKED = "BLOCKED"
STATE_AVOID = "AVOID"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="Run without X11 window display (saves ~10% CPU)")
    args = parser.parse_args()

    # ---- Hardware Init ----
    print("[INIT] Starting Hardware...")
    motor = MotorController()
    ultrasonic = UltrasonicSensor()
    servo = ServoController()
    servo.center_ultrasonic()
    time.sleep(0.5)
    print("[INIT] Hardware OK.")

    # ---- Camera Init ----
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open camera device.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    # Reduce camera FPS to save CPU
    cap.set(cv2.CAP_PROP_FPS, 15)

    # ---- AI Init ----
    print("[INIT] Loading YOLO11n ONNX...")
    detector = YOLODetector({
        "use_onnx": True,
        "imgsz": 320,
        "confidence_threshold": YOLO_CONFIDENCE_THRESHOLD,
    })
    planner = APFPlanner({
        "k_attractive": 1.5,
        "k_repulsive": 180.0,
        "d0": 2.2,
        "emergency_distance": 0.30,
        "max_speed": 10.0,
    })

    # Warmup
    print("[INIT] Warming up inference...")
    detector.detect(np.zeros((320, 320, 3), dtype=np.uint8))

    print("\n" + "=" * 40)
    print(" NEXUSPILOT: READY FOR MISSION ")
    print("=" * 40)

    # ---- Runtime State ----
    state = STATE_CRUISE
    dist_buffer = {}
    last_log_time = 0
    avoid_action = None
    avoid_start_time = 0
    consecutive_blocked = 0

    # Cached detections (reused between YOLO frames)
    cached_detections = []
    cached_apf_obs = []
    yolo_frame_count = 0

    try:
        while True:
            loop_start = time.perf_counter()

            # ---- Safety: Emergency ultrasonic check ----
            u_res = ultrasonic.measure_once()
            if u_res.valid and u_res.distance_cm < EMERGENCY_DIST:
                motor.emergency_stop()
                state = STATE_BLOCKED
                consecutive_blocked += 1
                if time.time() - last_log_time > 1.0:
                    print(f"!! EMERGENCY STOP: {u_res.distance_cm:.1f}cm")
                    last_log_time = time.time()
                # Flush stale camera frame
                cap.read()
                # Sleep briefly to avoid tight spin loop
                time.sleep(0.05)
                continue

            # ---- Read camera (only on YOLO frames or GUI frames) ----
            frame = None
            run_yolo = (yolo_frame_count % YOLO_FRAME_INTERVAL == 0) and state == STATE_CRUISE

            if run_yolo or not args.headless:
                ret, frame = cap.read()
                if not ret:
                    frame = None

            if run_yolo and frame is not None:
                # ---- YOLO Perception (throttled to ~3 FPS) ----
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cached_detections = detector.detect(rgb)
                obstacles = detector.get_obstacles(cached_detections)

                # Rebuild APF obstacle list (cached until next YOLO frame)
                cached_apf_obs = []
                for det in obstacles:
                    raw_dist = 480.0 / (det.height + 1e-6) * 0.32
                    tid = det.track_id
                    if tid not in dist_buffer:
                        dist_buffer[tid] = raw_dist
                    dist_buffer[tid] = 0.7 * dist_buffer[tid] + 0.3 * raw_dist
                    corrected_x = dist_buffer[tid] + 0.12
                    lateral = (det.center[0] - CAM_CENTER_X) / CAM_CENTER_X * corrected_x * 0.55
                    cached_apf_obs.append(Obstacle(
                        x=corrected_x, y=lateral,
                        distance=corrected_x, category=det.category
                    ))

            yolo_frame_count += 1

            # ---- State Machine ----
            front_dist = u_res.distance_cm if u_res.valid else 999.0

            if state == STATE_CRUISE:
                if front_dist < OBSTACLE_DIST:
                    motor.stop()
                    state = STATE_BLOCKED
                    consecutive_blocked += 1
                    if time.time() - last_log_time > 1.0:
                        print(f"[BLOCKED] Front: {front_dist:.1f}cm")
                        last_log_time = time.time()

                elif front_dist < WARNING_DIST:
                    # Warning zone -> APF gentle steering (cached obstacles)
                    out = planner.compute(0, 0, 0, 5.0, 3.0, 0, cached_apf_obs)
                    steer = out.target_steering * 0.3
                    motor.curve_move(CRUISE_SPEED * 0.7, steer)
                    if time.time() - last_log_time > 2.0:
                        print(f"[WARN] Front: {front_dist:.1f}cm, APF steering")
                        last_log_time = time.time()

                else:
                    # Clear -> forward with APF fine-tuning
                    out = planner.compute(0, 0, 0, 8.0, 3.0, 0, cached_apf_obs)
                    if not out.emergency_brake and out.target_speed > 0.1:
                        motor.curve_move(CRUISE_SPEED, out.target_steering * 0.3)
                    else:
                        motor.stop()

                    if time.time() - last_log_time > 3.0:
                        print(f"[CRUISE] Front: {front_dist:.1f}cm | Det: {len(cached_apf_obs)}")
                        last_log_time = time.time()

            elif state == STATE_BLOCKED:
                # Step 1: Back up to create clearance
                motor.move_backward(AVOID_BACKUP_SPEED)
                time.sleep(RECOVERY_BACKUP_DURATION)
                motor.stop()
                time.sleep(0.2)

                # Step 2: Scan left and right with ultrasonic servo
                dis_left = _scan_left(ultrasonic, servo)
                dis_right = _scan_right(ultrasonic, servo)

                # Re-center servo
                servo.center_ultrasonic()
                time.sleep(SERVO_SETTLE)

                # Step 3: Decide direction
                if dis_left < OBSTACLE_DIST and dis_right < OBSTACLE_DIST:
                    if consecutive_blocked >= 3:
                        print("[RECOVER] Both blocked x3, spinning out")
                        motor.rotate_left(AVOID_TURN_SPEED)
                        time.sleep(1.5)
                        motor.stop()
                        consecutive_blocked = 0
                        state = STATE_CRUISE
                    else:
                        print(f"[RECOVER] Both blocked L={dis_left:.0f} R={dis_right:.0f}, spin")
                        motor.rotate_left(AVOID_TURN_SPEED)
                        time.sleep(1.0)
                        motor.stop()
                        state = STATE_CRUISE
                elif dis_left > dis_right:
                    avoid_action = "left"
                    print(f"[AVOID] Turn LEFT (L={dis_left:.0f} > R={dis_right:.0f})")
                    state = STATE_AVOID
                    avoid_start_time = time.time()
                else:
                    avoid_action = "right"
                    print(f"[AVOID] Turn RIGHT (R={dis_right:.0f} > L={dis_left:.0f})")
                    state = STATE_AVOID
                    avoid_start_time = time.time()

            elif state == STATE_AVOID:
                if avoid_action == "left":
                    motor.rotate_left(AVOID_TURN_SPEED)
                else:
                    motor.rotate_right(AVOID_TURN_SPEED)

                elapsed = time.time() - avoid_start_time
                if elapsed >= RECOVERY_TURN_DURATION:
                    motor.stop()
                    time.sleep(0.1)
                    front_after = ultrasonic.measure_once()
                    if front_after.valid and front_after.distance_cm < OBSTACLE_DIST:
                        state = STATE_BLOCKED
                    else:
                        consecutive_blocked = max(0, consecutive_blocked - 1)
                        state = STATE_CRUISE
                        print(f"[CRUISE] Resumed (front={front_after.distance_cm:.1f}cm)")

            # ---- Visual Overlay (only on frames we already read) ----
            if not args.headless and frame is not None:
                _draw_overlay(frame, cached_detections, state, u_res)
                cv2.imshow("NexusPilot Monitor", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # FPS throttle
            elapsed = time.perf_counter() - loop_start
            if elapsed < MAIN_LOOP_INTERVAL:
                time.sleep(MAIN_LOOP_INTERVAL - elapsed)

    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Manual halt received.")
    finally:
        motor.cleanup()
        servo.cleanup()
        cap.release()
        cv2.destroyAllWindows()


# ---- Servo Scan Helpers ----

def _scan_left(sensor: UltrasonicSensor, servo: ServoController) -> float:
    """Scan left (~175 degrees) and return distance in cm."""
    servo.set_ultrasonic_angle(175)
    time.sleep(SERVO_SETTLE)
    reading = sensor.measure_once()
    return reading.distance_cm if reading.valid else 999.0


def _scan_right(sensor: UltrasonicSensor, servo: ServoController) -> float:
    """Scan right (~5 degrees) and return distance in cm."""
    servo.set_ultrasonic_angle(5)
    time.sleep(SERVO_SETTLE)
    reading = sensor.measure_once()
    return reading.distance_cm if reading.valid else 999.0


# ---- Overlay Drawing ----

def _draw_overlay(frame, detections, state, u_res):
    """Draw detection boxes and status on frame."""
    for det in detections:
        cv2.rectangle(frame,
                      (det.bbox[0], det.bbox[1]),
                      (det.bbox[2], det.bbox[3]),
                      (0, 255, 0), 2)
        label = f"{det.category} {det.confidence:.1f}"
        cv2.putText(frame, label, (det.bbox[0], det.bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    color = {
        STATE_CRUISE: (0, 255, 0),
        STATE_BLOCKED: (0, 0, 255),
        STATE_AVOID: (0, 165, 255),
    }.get(state, (255, 255, 255))
    dist_str = f"{u_res.distance_cm:.0f}cm" if u_res.valid else "---"
    cv2.putText(frame, f"{state} | {dist_str}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


if __name__ == "__main__":
    main()
