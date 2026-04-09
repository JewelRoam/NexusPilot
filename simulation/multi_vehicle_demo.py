"""
Multi-Vehicle Cooperative Demo - V2V Cooperative Perception & Planning in CARLA.
Demonstrates the KEY INNOVATION: inter-vehicle collaboration.

Scenarios:
  1. Intersection coordination - two vehicles negotiate priority
  2. Occlusion compensation - shared detection of hidden obstacles
  3. Platooning - convoy following with shared perception
  4. Deadlock resolution - narrow road yield negotiation

Usage:
  1. Start CARLA server: CARLA_0.9.13/WindowsNoEditor/CarlaUE4.exe
  2. Run: python -m simulation.multi_vehicle_demo
"""
import os
import sys
import time
import yaml
import cv2
import math
import traceback
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.carla_env import CarlaEnv
from perception.detector import YOLODetector
from perception.depth_estimator import DepthEstimator
from planning.apf_planner import APFPlanner
from control.vehicle_controller import VehicleController
from cooperation.v2v_message import V2VCommunicator
from cooperation.cooperative_planner import CooperativePlanner
from utils.logger import setup_logger, FPSCounter, PerformanceMetrics


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class CooperativeVehicleAgent:
    """
    A single cooperative vehicle agent with full pipeline:
    Perception → Depth → V2V Comm → Cooperative Planning → APF → Control
    """

    def __init__(self, vehicle_id: str, config: dict, managed_vehicle):
        self.vehicle_id = vehicle_id
        self.config = config
        self.vehicle = managed_vehicle

        # Perception
        self.detector = YOLODetector(config['perception'])
        self.depth_estimator = DepthEstimator(
            config['depth'], config['sensors']['rgb_camera']
        )

        # Planning & Control
        self.apf_planner = APFPlanner(config['planning']['apf'])
        self.controller = VehicleController(config['control'], platform="carla")

        # Cooperation
        comm_config = config.get('cooperation', {}).get('communication', {})
        comm_config['protocol'] = 'in_process'  # Force in-process for simulation
        self.v2v_comm = V2VCommunicator(vehicle_id, comm_config)
        self.coop_planner = CooperativePlanner(vehicle_id, config.get('cooperation', {}))

        # State
        self.latest_detections = []
        self.latest_planner_output = None
        self.latest_control = {}
        self.latest_coop_decision = None
        self.frame_count = 0

    def step(self):
        """Execute one full pipeline step."""
        # 1. Get sensor data
        rgb_image = self.vehicle.get_rgb_image()
        depth_data = self.vehicle.get_depth_data()

        if rgb_image is None:
            return None

        # 2. Update depth
        if depth_data is not None:
            self.depth_estimator.update_depth_image(depth_data)

        # 3. PERCEPTION
        detections = self.detector.detect(rgb_image)
        obstacles_det = self.detector.get_obstacles(detections)
        self.depth_estimator.enrich_detections(obstacles_det, self.vehicle.get_transform())
        self.latest_detections = detections

        # 4. Get ego state
        ego_x, ego_y, _ = self.vehicle.get_location()
        ego_yaw = self.vehicle.get_yaw()
        ego_speed = self.vehicle.get_speed_kmh()

        # 5. V2V COMMUNICATION - Broadcast our state and detections
        shared_dets = self.v2v_comm.detections_to_shared(obstacles_det, ego_x, ego_y)
        intent = self._determine_intent(ego_speed)
        msg = self.v2v_comm.create_message(
            position=(ego_x, ego_y, ego_yaw),
            velocity=(ego_speed, ego_yaw),
            detections=shared_dets,
            intent=intent,
        )
        self.v2v_comm.broadcast(msg)

        # 6. V2V RECEIVE - Get messages from other vehicles
        v2v_messages = self.v2v_comm.receive_all()

        # 7. COOPERATIVE PLANNING - Process V2V info
        self.latest_coop_decision = self.coop_planner.process(
            ego_x, ego_y, ego_yaw, ego_speed,
            obstacles_det, v2v_messages
        )

        # 8. Advance waypoint
        self.vehicle.advance_waypoint(threshold=5.0)
        goal = self.vehicle.get_next_waypoint()

        if goal:
            goal_x, goal_y = goal

            # 9. APF PLANNING with cooperative obstacles
            local_obstacles = self.apf_planner.detections_to_obstacles(
                obstacles_det, ego_x, ego_y, ego_yaw
            )
            # Merge cooperative shared obstacles
            coop_obstacles = self.latest_coop_decision.shared_obstacles if self.latest_coop_decision else []

            self.latest_planner_output = self.apf_planner.compute(
                ego_x, ego_y, ego_yaw, ego_speed,
                goal_x, goal_y, local_obstacles,
                cooperative_obstacles=coop_obstacles,
            )

            # Apply cooperative speed adjustment
            if self.latest_coop_decision:
                adj = self.latest_coop_decision.speed_adjustment
                self.latest_planner_output.target_speed *= adj

            # 10. CONTROL
            self.latest_control = self.controller.compute_control(
                self.latest_planner_output, ego_speed
            )
            self.controller.apply_carla_control(self.vehicle.actor, self.latest_control)

        self.frame_count += 1
        return rgb_image

    def broadcast_v2v(self):
        """
        V2V Phase 1: Broadcast our current state and detections.
        Called BEFORE step_with_v2v() for all agents in sync.
        """
        # Get sensor data
        rgb_image = self.vehicle.get_rgb_image()
        if rgb_image is None:
            return
        
        depth_data = self.vehicle.get_depth_data()
        if depth_data is not None:
            self.depth_estimator.update_depth_image(depth_data)

        # PERCEPTION
        detections = self.detector.detect(rgb_image)
        obstacles_det = self.detector.get_obstacles(detections)
        self.depth_estimator.enrich_detections(obstacles_det, self.vehicle.get_transform())
        self.latest_detections = detections

        # Get ego state
        ego_x, ego_y, _ = self.vehicle.get_location()
        ego_yaw = self.vehicle.get_yaw()
        ego_speed = self.vehicle.get_speed_kmh()

        # Broadcast V2V message
        shared_dets = self.v2v_comm.detections_to_shared(obstacles_det, ego_x, ego_y)
        intent = self._determine_intent(ego_speed)
        msg = self.v2v_comm.create_message(
            position=(ego_x, ego_y, ego_yaw),
            velocity=(ego_speed, ego_yaw),
            detections=shared_dets,
            intent=intent,
        )
        self.v2v_comm.broadcast(msg)
        
        # Store state for next phase
        self._ego_x = ego_x
        self._ego_y = ego_y
        self._ego_yaw = ego_yaw
        self._ego_speed = ego_speed
        self._obstacles_det = obstacles_det

    def step_with_v2v(self):
        """
        V2V Phase 2: Receive V2V messages and run full pipeline.
        Called AFTER broadcast_all() for all agents in sync.
        """
        # Receive V2V messages from other vehicles
        v2v_messages = self.v2v_comm.receive_all()

        # COOPERATIVE PLANNING - Process V2V info
        self.latest_coop_decision = self.coop_planner.process(
            self._ego_x, self._ego_y, self._ego_yaw, self._ego_speed,
            self._obstacles_det, v2v_messages
        )

        # Advance waypoint
        self.vehicle.advance_waypoint(threshold=5.0)
        goal = self.vehicle.get_next_waypoint()

        if goal:
            goal_x, goal_y = goal

            # APF PLANNING with cooperative obstacles
            local_obstacles = self.apf_planner.detections_to_obstacles(
                self._obstacles_det, self._ego_x, self._ego_y, self._ego_yaw
            )
            # Merge cooperative shared obstacles
            coop_obstacles = self.latest_coop_decision.shared_obstacles if self.latest_coop_decision else []

            self.latest_planner_output = self.apf_planner.compute(
                self._ego_x, self._ego_y, self._ego_yaw, self._ego_speed,
                goal_x, goal_y, local_obstacles,
                cooperative_obstacles=coop_obstacles,
            )

            # Apply cooperative speed adjustment
            if self.latest_coop_decision:
                adj = self.latest_coop_decision.speed_adjustment
                self.latest_planner_output.target_speed *= adj

            # CONTROL
            self.latest_control = self.controller.compute_control(
                self.latest_planner_output, self._ego_speed
            )
            self.controller.apply_carla_control(self.vehicle.actor, self.latest_control)

        self.frame_count += 1

    def _determine_intent(self, speed: float) -> str:
        """Determine current driving intent."""
        if self.controller.is_emergency:
            return "emergency_brake"
        if speed < 1.0:
            return "stopped"
        if self.latest_planner_output and self.latest_planner_output.status == "avoiding":
            steer = self.latest_planner_output.target_steering
            if steer < -0.3:
                return "turning_left"
            elif steer > 0.3:
                return "turning_right"
        return "cruising"


def draw_bird_eye_view(agents: list, bev_size: int = 300, scale: float = 2.0) -> np.ndarray:
    """Draw a simple bird's eye view showing vehicle positions and cooperation."""
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)
    cv2.putText(bev, "Bird's Eye View", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if not agents:
        return bev

    # Find center (average of all vehicle positions)
    positions = []
    for agent in agents:
        x, y, _ = agent.vehicle.get_location()
        positions.append((x, y))

    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)

    colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for i, agent in enumerate(agents):
        x, y, _ = agent.vehicle.get_location()
        yaw = agent.vehicle.get_yaw()

        # Map to BEV coordinates
        bx = int((x - cx) * scale + bev_size // 2)
        by = int((y - cy) * scale + bev_size // 2)

        if 10 < bx < bev_size - 10 and 10 < by < bev_size - 10:
            color = colors[i % len(colors)]

            # Draw vehicle as triangle
            yaw_rad = math.radians(yaw)
            pts = np.array([
                [bx + int(8 * math.cos(yaw_rad)), by + int(8 * math.sin(yaw_rad))],
                [bx + int(5 * math.cos(yaw_rad + 2.5)), by + int(5 * math.sin(yaw_rad + 2.5))],
                [bx + int(5 * math.cos(yaw_rad - 2.5)), by + int(5 * math.sin(yaw_rad - 2.5))],
            ], dtype=np.int32)
            cv2.fillPoly(bev, [pts], color)

            # Label
            cv2.putText(bev, agent.vehicle_id[:4], (bx - 15, by - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

            # Draw shared obstacles from V2V
            if agent.latest_coop_decision:
                for obs in agent.latest_coop_decision.shared_obstacles:
                    ox = int((obs.x - cx) * scale + bev_size // 2)
                    oy = int((obs.y - cy) * scale + bev_size // 2)
                    if 5 < ox < bev_size - 5 and 5 < oy < bev_size - 5:
                        cv2.circle(bev, (ox, oy), 3, (0, 165, 255), -1)

    return bev


def main():
    config = load_config()
    logger = setup_logger("MultiVehicle", config.get("logging", {}).get("level", "INFO"))
    metrics = PerformanceMetrics()
    fps_counter = FPSCounter()

    # Reset V2V shared bus
    V2VCommunicator.reset_shared_bus()

    # Initialize CARLA
    logger.info("Connecting to CARLA...")
    env = CarlaEnv(config)
    env.connect()

    agents = []

    try:
        num_coop_vehicles = config.get('cooperation', {}).get('num_cooperative_vehicles', 2)
        sensor_cfg = config.get('sensors', {})

        logger.info(f"Spawning {num_coop_vehicles} cooperative vehicles...")
        
        # Spawn cooperative vehicles - CLOSE TOGETHER for V2V coordination testing
        # Step 1: Spawn all vehicles at random locations first
        # Step 2: Wait one tick for positions to settle
        # Step 3: Teleport second vehicle to be close to first vehicle
        managed_vehicles = []
        
        logger.info(f"Spawning {num_coop_vehicles} vehicles at random locations...")
        for i in range(num_coop_vehicles):
            vid = f"vehicle_{i}"
            managed = env.spawn_vehicle(vid)
            managed_vehicles.append(managed)
        
        # Wait one tick for spawn positions to settle
        env.tick()
        time.sleep(0.1)
        
        # Log actual positions after spawn
        for i, managed in enumerate(managed_vehicles):
            loc = managed.actor.get_location()
            logger.info(f"  vehicle_{i} actual position: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
        
        # Step 3: Teleport second vehicle to be close to first vehicle (within 30m)
        if len(managed_vehicles) >= 2:
            import carla
            first_transform = managed_vehicles[0].actor.get_transform()
            first_loc = first_transform.location
            first_yaw_rad = math.radians(first_transform.rotation.yaw)
            
            logger.info(f"First vehicle location: ({first_loc.x:.1f}, {first_loc.y:.1f}), yaw: {first_transform.rotation.yaw:.1f}")
            
            # Calculate target position: 20m behind and to the side
            offset_distance = 20.0  # meters
            target_x = first_loc.x - offset_distance * math.sin(first_yaw_rad)
            target_y = first_loc.y + offset_distance * math.cos(first_yaw_rad)
            target_z = first_loc.z + 0.5
            
            logger.info(f"Target position for vehicle_1: ({target_x:.1f}, {target_y:.1f})")
            
            # Create teleport transform
            teleport_loc = carla.Location(x=target_x, y=target_y, z=target_z)
            teleport_transform = carla.Transform(
                teleport_loc,
                carla.Rotation(yaw=first_transform.rotation.yaw)
            )
            
            # Teleport second vehicle
            managed_vehicles[1].actor.set_transform(teleport_transform)
            env.tick()  # Ensure teleport takes effect
            time.sleep(0.1)
            
            # Verify distance
            new_loc = managed_vehicles[1].actor.get_location()
            actual_dist = math.sqrt(
                (new_loc.x - first_loc.x)**2 + (new_loc.y - first_loc.y)**2
            )
            logger.info(f"Teleported vehicle_1 to ({new_loc.x:.1f}, {new_loc.y:.1f})")
            logger.info(f"V2V initial distance: {actual_dist:.1f}m (target: {offset_distance}m)")
            
            if actual_dist > 35:
                logger.warning(f"Distance still too large ({actual_dist:.1f}m), teleporting closer...")
                # Force closer position
                target_x = first_loc.x - 15.0 * math.sin(first_yaw_rad)
                target_y = first_loc.y + 15.0 * math.cos(first_yaw_rad)
                teleport_loc = carla.Location(x=target_x, y=target_y, z=first_loc.z + 0.5)
                teleport_transform = carla.Transform(
                    teleport_loc,
                    carla.Rotation(yaw=first_transform.rotation.yaw)
                )
                managed_vehicles[1].actor.set_transform(teleport_transform)
                env.tick()
                time.sleep(0.1)
                new_loc = managed_vehicles[1].actor.get_location()
                actual_dist = math.sqrt(
                    (new_loc.x - first_loc.x)**2 + (new_loc.y - first_loc.y)**2
                )
                logger.info(f"Final distance after re-teleport: {actual_dist:.1f}m")
        
        # Now create agents with the spawned vehicles
        for i, managed in enumerate(managed_vehicles):
            vid = f"vehicle_{i}"
            
            logger.info(f"Attaching RGB camera to {vid}...")
            managed.attach_rgb_camera(sensor_cfg.get('rgb_camera', {}))
            logger.info(f"Attaching depth camera to {vid}...")
            managed.attach_depth_camera(sensor_cfg.get('depth_camera', {}))
            logger.info(f"Generating route for {vid}...")
            env.generate_route(managed, sampling_resolution=4.0)

            logger.info(f"Creating agent for {vid}...")
            agent = CooperativeVehicleAgent(vid, config, managed)
            agents.append(agent)
            logger.info(f"Created cooperative agent: {vid}")

        # Wait for sensor data
        logger.info("Waiting for sensor initialization...")
        max_wait_frames = 60  # Max 3 seconds at 20 FPS
        for i in range(max_wait_frames):
            env.tick()
            # Check if all agents have received sensor data
            all_ready = True
            for agent in agents:
                rgb = agent.vehicle.get_rgb_image()
                if rgb is None:
                    all_ready = False
                    logger.debug(f"Agent {agent.vehicle_id} waiting for RGB data...")
            if all_ready:
                logger.info(f"All sensors ready after {i+1} ticks")
                break
            time.sleep(0.05)
        else:
            logger.warning("Sensor initialization timeout - proceeding anyway")

        logger.info("Setting up display windows...")

        # Create separate windows for each vehicle + bird's eye view
        window_names = []
        for i, agent in enumerate(agents):
            win_name = f'Vehicle {i+1}: {agent.vehicle_id}'
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win_name, 800, 600)
            window_names.append(win_name)
        
        # Bird's eye view window
        bev_win_name = "Bird's Eye View (V2V Cooperation)"
        cv2.namedWindow(bev_win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(bev_win_name, 600, 600)
        window_names.append(bev_win_name)

        logger.info("="*60)
        logger.info("=== Multi-Vehicle Cooperative Demo Started ===")
        logger.info(f"=== {num_coop_vehicles} cooperative vehicles active ===")
        logger.info("=== Press 'q' or close window to stop ===")
        logger.info("="*60)

        frame_count = 0

        logger.info("Entering main loop...")
        last_debug_time = time.time()
        
        # V2V Sync: Collect messages before processing
        # Phase 1: All agents broadcast their current state
        def broadcast_all():
            for agent in agents:
                try:
                    agent.broadcast_v2v()
                except Exception as e:
                    logger.error(f"Broadcast error for {agent.vehicle_id}: {e}")
        
        # Phase 2: All agents process V2V messages and run full pipeline
        def process_all():
            for agent in agents:
                try:
                    agent.step_with_v2v()
                except Exception as e:
                    logger.error(f"Step error for {agent.vehicle_id}: {e}")
        
        while True:
            try:
                metrics.start_timer("total_frame")

                # Tick simulation
                env.tick()

                # V2V Sync: Step 1 - All agents broadcast first
                broadcast_all()
                
                # V2V Sync: Step 2 - All agents receive and process
                process_all()

                # Debug output every 5 seconds - SHOW V2V DISTANCE
                current_time = time.time()
                if current_time - last_debug_time > 5.0:
                    # Calculate distance between vehicles
                    if len(agents) >= 2:
                        v0_x, v0_y, _ = agents[0].vehicle.get_location()
                        v1_x, v1_y, _ = agents[1].vehicle.get_location()
                        v2v_distance = ((v0_x - v1_x)**2 + (v0_y - v1_y)**2)**0.5
                        
                        # Get coordination status for both
                        coord_0, n_0 = agents[0].coop_planner._get_coordination_status(v0_x, v0_y)
                        coord_1, n_1 = agents[1].coop_planner._get_coordination_status(v1_x, v1_y)
                        
                        logger.info(f"🔗 V2V DISTANCE: {v2v_distance:.1f}m | "
                                  f"V0: {coord_0.upper()} | V1: {coord_1.upper()}")
                    
                    for agent in agents:
                        speed = agent.vehicle.get_speed_kmh()
                        loc = agent.vehicle.get_location()
                        logger.info(f"  {agent.vehicle_id}: pos=({loc[0]:.1f}, {loc[1]:.1f}), speed={speed:.1f}km/h")
                    last_debug_time = current_time

                # Display all agent views
                for i, agent in enumerate(agents):
                    metrics.start_timer(f"agent_{agent.vehicle_id}")

                    # Prepare and display in separate window
                    win_name = window_names[i]
                    if rgb is not None:
                        # Draw detections and info on image
                        display = rgb.copy()
                        h, w = display.shape[:2]
                        
                        # Draw detections
                        for det in agent.latest_detections:
                            x1, y1, x2, y2 = det.bbox
                            color = {"vehicle": (0,255,0), "pedestrian": (0,0,255),
                                     "cyclist": (255,165,0)}.get(det.category, (128,128,128))
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                            lbl = f"{det.class_name} {det.confidence:.2f}"
                            if det.distance > 0:
                                lbl += f" {det.distance:.1f}m"
                            cv2.putText(display, lbl, (x1, y1-5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        
                        # Add info overlay
                        speed = agent.vehicle.get_speed_kmh()
                        status = agent.latest_planner_output.status if agent.latest_planner_output else "init"
                        coop = agent.latest_coop_decision
                        
                        # Get coordination status from planner
                        coord_status, num_nearby = agent.coop_planner._get_coordination_status(
                            *agent.vehicle.get_location()[:2]
                        )
                        
                        # Status colors
                        status_colors = {
                            "inactive": (128, 128, 128),    # Gray
                            "monitoring": (0, 255, 255),     # Cyan
                            "active": (0, 255, 0),           # Green
                            "warning": (0, 165, 255),        # Orange
                            "critical": (0, 0, 255),         # Red
                        }
                        coord_color = status_colors.get(coord_status, (255, 255, 255))
                        
                        # Top info bar
                        cv2.rectangle(display, (0, 0), (w, 120), (0, 0, 0), -1)
                        cv2.putText(display, f"{agent.vehicle_id} | {speed:.1f} km/h | {status.upper()}", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Coordination status indicator
                        cv2.putText(display, f"Coord: {coord_status.upper()} ({num_nearby} vehicles)", 
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, coord_color, 2)
                        
                        if coop:
                            color = {
                                "proceed": (0, 255, 0), "yield": (0, 0, 255),
                                "slow_down": (0, 165, 255), "follow": (255, 255, 0),
                                "stop": (0, 0, 255),
                            }.get(coop.action, (255, 255, 255))
                            cv2.putText(display, f"V2V Action: {coop.action.upper()}", 
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.putText(display, f"Reason: {coop.reason[:50]}", 
                                        (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                            if coop.shared_obstacles:
                                cv2.putText(display, f"Shared obstacles: {len(coop.shared_obstacles)}", 
                                            (w-250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        # Show in window
                        cv2.imshow(win_name, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
                    else:
                        # No data - show black screen with message
                        blank = np.zeros((600, 800, 3), dtype=np.uint8)
                        cv2.putText(blank, f"{win_name}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        cv2.putText(blank, "No camera data", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow(win_name, blank)

                # Update bird's eye view
                bev = draw_bird_eye_view(agents, bev_size=600, scale=3.0)
                # Add legend
                cv2.putText(bev, "Green: Vehicle 1 | Blue: Vehicle 2 | Orange dots: Shared obstacles", 
                            (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.imshow(bev_win_name, bev)

                fps_counter.tick()
                metrics.stop_timer("total_frame")
                metrics.increment("frames")

                # Periodic logging
                frame_count += 1
                if frame_count % 60 == 0:
                    log_parts = [f"Frame {frame_count} | FPS: {fps_counter.fps:.1f}"]
                    for agent in agents:
                        speed = agent.vehicle.get_speed_kmh()
                        status = agent.latest_planner_output.status if agent.latest_planner_output else "?"
                        coop_action = agent.latest_coop_decision.action if agent.latest_coop_decision else "?"
                        log_parts.append(f"{agent.vehicle_id}: {speed:.0f}km/h {status}/{coop_action}")
                    logger.info(" | ".join(log_parts))

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("'q' pressed - exiting")
                    break
                    
                # Check if any window was closed - only after first frame
                if frame_count > 1:
                    any_window_closed = False
                    for win_name in window_names:
                        try:
                            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                                any_window_closed = True
                                logger.info(f"Window '{win_name}' closed - exiting")
                                break
                        except cv2.error:
                            pass
                    if any_window_closed:
                        break
                        
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error in main try block: {e}")
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        V2VCommunicator.reset_shared_bus()
        env.cleanup()

        metrics_path = os.path.join(config.get("logging", {}).get("output_dir", "output"),
                                    "multi_vehicle_metrics.json")
        metrics.save(metrics_path)
        logger.info(f"Metrics saved to {metrics_path}")
        logger.info("Multi-vehicle cooperative demo finished.")


if __name__ == '__main__':
    main()
