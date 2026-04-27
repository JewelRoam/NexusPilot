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
import random
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
        self._latest_rgb = rgb_image

    def step_with_v2v(self):
        """
        V2V Phase 2: Receive V2V messages and run full pipeline.
        Called AFTER broadcast_all() for all agents in sync.
        """
        # Receive V2V messages from other vehicles
        v2v_messages = self.v2v_comm.receive_all()
        self._last_v2v_messages = v2v_messages  # Store for BEV visualization

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


def _draw_arrow(img, p1, p2, color, thickness=2, tip_len=10, tip_angle=25):
    """Draw an arrow from p1 to p2 with an arrowhead."""
    cv2.line(img, p1, p2, color, thickness)
    angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    rad = math.radians(tip_angle)
    p3 = (p2[0] - int(tip_len * math.cos(angle - rad)),
          p2[1] - int(tip_len * math.sin(angle - rad)))
    p4 = (p2[0] - int(tip_len * math.cos(angle + rad)),
          p2[1] - int(tip_len * math.sin(angle + rad)))
    cv2.fillPoly(img, [np.array([p2, p3, p4], dtype=np.int32)], color)


def _draw_dashed_line(img, p1, p2, color, thickness=1, dash_len=6, gap_len=4):
    """Draw a dashed line from p1 to p2."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 1:
        return
    steps = int(dist / (dash_len + gap_len))
    if steps == 0:
        cv2.line(img, p1, p2, color, thickness)
        return
    ux, uy = dx / dist, dy / dist
    for i in range(steps):
        s = i * (dash_len + gap_len)
        e = min(s + dash_len, dist)
        sp = (int(p1[0] + ux * s), int(p1[1] + uy * s))
        ep = (int(p1[0] + ux * e), int(p1[1] + uy * e))
        cv2.line(img, sp, ep, color, thickness)


def draw_bird_eye_view(agents: list, env, bev_size: int = 500, scale: float = 1.5) -> np.ndarray:
    """Enhanced bird's eye view with V2V communication, routes, and road topology."""
    bev = np.zeros((bev_size, bev_size, 3), dtype=np.uint8)

    if not agents:
        cv2.putText(bev, "No agents", (bev_size // 2 - 40, bev_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        return bev

    # --- Center calculation ---
    positions = []
    for agent in agents:
        x, y, _ = agent.vehicle.get_location()
        positions.append((x, y))
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)

    def to_bev(wx, wy):
        """World coords to BEV pixel."""
        bx = int((wx - cx) * scale + bev_size // 2)
        by = int((wy - cy) * scale + bev_size // 2)
        return (bx, by)

    def in_view(bx, by, margin=5):
        return margin < bx < bev_size - margin and margin < by < bev_size - margin

    # --- Draw road topology from CARLA map ---
    if env and env.map:
        center_loc = agents[0].vehicle.actor.get_location()
        # Cache road edges — only recompute when center moves > 30m
        if not hasattr(draw_bird_eye_view, '_road_cache'):
            draw_bird_eye_view._road_cache = {'center': None, 'edges': []}
        cache = draw_bird_eye_view._road_cache
        cache_center = cache['center']
        need_rebuild = (cache_center is None or
                        math.sqrt((center_loc.x - cache_center[0])**2 +
                                  (center_loc.y - cache_center[1])**2) > 30.0)

        if need_rebuild:
            center_wp = env.map.get_waypoint(center_loc, project_to_road=True)
            road_edges = []
            queue = [center_wp]
            visited = {center_wp.road_id}
            bfs_count = 0
            while queue and bfs_count < 500:
                wp = queue.pop(0)
                bfs_count += 1
                wx, wy = wp.transform.location.x, wp.transform.location.y
                for nxt in wp.next(8.0):
                    nx, ny = nxt.transform.location.x, nxt.transform.location.y
                    road_edges.append(((wx, wy), (nx, ny)))
                    if nxt.road_id not in visited:
                        visited.add(nxt.road_id)
                        queue.append(nxt)
                for prev in wp.previous(8.0):
                    px, py = prev.transform.location.x, prev.transform.location.y
                    road_edges.append(((wx, wy), (px, py)))
                    if prev.road_id not in visited:
                        visited.add(prev.road_id)
                        queue.append(prev)
            cache['center'] = (center_loc.x, center_loc.y)
            cache['edges'] = road_edges
        else:
            road_edges = cache['edges']

        for (x1, y1), (x2, y2) in road_edges:
            b1 = to_bev(x1, y1)
            b2 = to_bev(x2, y2)
            if in_view(b1[0], b1[1], margin=-20) or in_view(b2[0], b2[1], margin=-20):
                cv2.line(bev, b1, b2, (45, 45, 45), 2)

    # --- Draw routes for each agent ---
    route_colors = [(0, 80, 0), (0, 0, 80), (80, 80, 0), (80, 0, 80)]
    for i, agent in enumerate(agents):
        route_wps = agent.vehicle._route_waypoints
        if not route_wps:
            continue
        prev_bev = None
        wp_index = agent.vehicle._current_wp_index
        for j, wp in enumerate(route_wps):
            bx, by = to_bev(wp.transform.location.x, wp.transform.location.y)
            if not in_view(bx, by, margin=0):
                prev_bev = (bx, by)
                continue
            # Upcoming route is brighter, past route is dimmer
            is_future = j >= wp_index
            color = route_colors[i % len(route_colors)]
            if is_future:
                brightness = (255, 255, 255)
            else:
                brightness = (30, 30, 30)
            if prev_bev and in_view(prev_bev[0], prev_bev[1], margin=0):
                cv2.line(bev, prev_bev, (bx, by), brightness if is_future else color, 1)
            prev_bev = (bx, by)

    vehicle_colors = [(0, 255, 0), (0, 100, 255), (255, 255, 0), (255, 0, 255)]
    vehicle_positions = []

    # --- Draw vehicles as larger arrows ---
    for i, agent in enumerate(agents):
        x, y, _ = agent.vehicle.get_location()
        yaw = agent.vehicle.get_yaw()
        bx, by = to_bev(x, y)
        vehicle_positions.append((bx, by, agent))

        if not in_view(bx, by):
            continue

        color = vehicle_colors[i % len(vehicle_colors)]
        yaw_rad = math.radians(yaw)

        # Draw arrow (larger, more visible)
        arrow_len = 18
        tip = (bx + int(arrow_len * math.cos(yaw_rad)),
               by + int(arrow_len * math.sin(yaw_rad)))
        tail_left = (bx + int(10 * math.cos(yaw_rad + 2.3)),
                     by + int(10 * math.sin(yaw_rad + 2.3)))
        tail_right = (bx + int(10 * math.cos(yaw_rad - 2.3)),
                      by + int(10 * math.sin(yaw_rad - 2.3)))
        cv2.fillPoly(bev, [np.array([tip, tail_left, tail_right], dtype=np.int32)], color)

        # Speed text
        speed = agent.vehicle.get_speed_kmh()
        cv2.putText(bev, f"{agent.vehicle_id}", (bx - 20, by - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(bev, f"{speed:.0f}km/h", (bx - 18, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    # --- Draw NPC vehicles and pedestrians ---
    if env and env._traffic_actors:
        for actor in env._traffic_actors:
            if actor is None:
                continue
            type_id = actor.type_id
            loc = actor.get_location()
            bx, by = to_bev(loc.x, loc.y)
            if not in_view(bx, by):
                continue
            if 'walker' in type_id or 'pedestrian' in type_id:
                cv2.circle(bev, (bx, by), 3, (200, 200, 200), -1)
            elif 'vehicle' in type_id:
                yaw = actor.get_transform().rotation.yaw
                yaw_rad = math.radians(yaw)
                # Small triangle for NPC vehicles
                pts = np.array([
                    [bx + int(6 * math.cos(yaw_rad)), by + int(6 * math.sin(yaw_rad))],
                    [bx + int(4 * math.cos(yaw_rad + 2.3)), by + int(4 * math.sin(yaw_rad + 2.3))],
                    [bx + int(4 * math.cos(yaw_rad - 2.3)), by + int(4 * math.sin(yaw_rad - 2.3))],
                ], dtype=np.int32)
                cv2.fillPoly(bev, [pts], (128, 128, 128))

    # --- V2V Communication Links ---
    for i, (bx, by, agent) in enumerate(vehicle_positions):
        if not agent.latest_coop_decision:
            continue

        # Get V2V messages this agent received
        v2v_msgs = getattr(agent, '_last_v2v_messages', {})
        if not v2v_msgs:
            continue

        for sender_id, msg in v2v_msgs.items():
            # Find sender vehicle position
            sender_pos = None
            for j, (sbx, sby, sagent) in enumerate(vehicle_positions):
                if sagent.vehicle_id == sender_id:
                    sender_pos = (sbx, sby)
                    break

            if sender_pos is None:
                # Use position from V2V message
                sx, sy, _ = msg.position
                sender_pos = to_bev(sx, sy)

            if not in_view(sender_pos[0], sender_pos[1]) and not in_view(bx, by):
                continue

            # Determine link color based on coordination status
            action = agent.latest_coop_decision.action
            if action == "yield":
                link_color = (0, 0, 255)  # Red
            elif action == "slow_down":
                link_color = (0, 165, 255)  # Orange
            elif action == "follow":
                link_color = (255, 255, 0)  # Yellow
            elif action == "stop":
                link_color = (0, 0, 200)  # Dark red
            else:
                link_color = (0, 255, 128)  # Green (proceed)

            # Draw dashed communication link
            _draw_dashed_line(bev, sender_pos, (bx, by), link_color, thickness=1)

            # Draw arrow from sender to receiver
            mid_x = (sender_pos[0] + bx) // 2
            mid_y = (sender_pos[1] + by) // 2
            _draw_arrow(bev, sender_pos, (mid_x, mid_y), link_color, thickness=1, tip_len=6)

            # Draw shared obstacle dots from V2V
            if agent.latest_coop_decision.shared_obstacles:
                for obs in agent.latest_coop_decision.shared_obstacles:
                    ox, oy = to_bev(obs.x, obs.y)
                    if in_view(ox, oy):
                        cv2.circle(bev, (ox, oy), 4, (0, 165, 255), -1)
                        cv2.circle(bev, (ox, oy), 6, (0, 165, 255), 1)

    # --- Info Panel (top-left) ---
    panel_y = 10
    cv2.rectangle(bev, (0, 0), (bev_size - 1, 85), (20, 20, 20), -1)
    cv2.putText(bev, "V2V Cooperative Perception", (10, panel_y + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # V2V distance
    if len(agents) >= 2:
        v0_loc = agents[0].vehicle.get_location()
        v1_loc = agents[1].vehicle.get_location()
        v2v_dist = math.sqrt((v0_loc[0]-v1_loc[0])**2 + (v0_loc[1]-v1_loc[1])**2)
        cv2.putText(bev, f"V2V Distance: {v2v_dist:.1f}m", (10, panel_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 200), 1)

    # Per-vehicle V2V status
    for i, agent in enumerate(agents):
        y_off = panel_y + 48 + i * 16
        loc = agent.vehicle.get_location()
        if loc is None:
            continue
        coord_status, n_nearby = agent.coop_planner._get_coordination_status(
            loc[0], loc[1]
        )
        action = agent.latest_coop_decision.action if agent.latest_coop_decision else "N/A"
        shared_n = len(agent.latest_coop_decision.shared_obstacles) if agent.latest_coop_decision else 0
        v_color = vehicle_colors[i % len(vehicle_colors)]
        cv2.putText(bev, f"[V{i}] {coord_status.upper()} | act:{action} | shared:{shared_n} | n:{n_nearby}",
                    (10, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.32, v_color, 1)

    # --- Legend (bottom-right) ---
    leg_x = bev_size - 170
    leg_y = bev_size - 70
    cv2.rectangle(bev, (leg_x - 5, leg_y - 15), (bev_size - 1, bev_size - 1), (20, 20, 20), -1)
    cv2.putText(bev, "--- V2V Link ---", (leg_x, leg_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
    _draw_dashed_line(bev, (leg_x, leg_y + 15), (leg_x + 60, leg_y + 15), (0, 255, 128), 1)
    cv2.putText(bev, "proceed", (leg_x + 65, leg_y + 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 128), 1)
    _draw_dashed_line(bev, (leg_x, leg_y + 28), (leg_x + 60, leg_y + 28), (0, 165, 255), 1)
    cv2.putText(bev, "slow_down", (leg_x + 65, leg_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 165, 255), 1)
    _draw_dashed_line(bev, (leg_x, leg_y + 41), (leg_x + 60, leg_y + 41), (0, 0, 255), 1)
    cv2.putText(bev, "yield", (leg_x + 65, leg_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    return bev


def spawn_npc_traffic(env, agents, num_vehicles: int = 8, num_pedestrians: int = 5,
                      spawn_radius: float = 60.0):
    """
    Spawn NPC vehicles and pedestrians near the ego vehicles' routes.
    Safe for CARLA synchronous mode — avoids get_random_location_from_navigation().
    """
    import carla
    bp_library = env.world.get_blueprint_library()
    spawn_points = env.map.get_spawn_points()
    vehicle_bps = bp_library.filter('vehicle')
    walker_bps = bp_library.filter('walker.pedestrian.*')

    # Collect a sample of route waypoints as spawn anchors (avoid O(n*m))
    hot_zones = []
    for agent in agents:
        route = agent.vehicle._route_waypoints
        if route:
            step = max(1, len(route) // 15)
            for wp in route[::step]:
                hot_zones.append((wp.transform.location.x, wp.transform.location.y))

    if not hot_zones:
        return 0, 0

    # --- NPC Vehicles ---
    spawned_vehicles = 0
    random.shuffle(spawn_points)
    for sp in spawn_points:
        if spawned_vehicles >= num_vehicles:
            break
        # Check if spawn point is near any route waypoint
        too_far = True
        for hx, hy in hot_zones:
            if (sp.location.x - hx) ** 2 + (sp.location.y - hy) ** 2 < spawn_radius ** 2:
                too_far = False
                break
        if too_far:
            continue

        bp = random.choice(vehicle_bps)
        if bp.has_attribute('color'):
            bp.set_attribute('color', random.choice(bp.get_attribute('color').recommended_values))
        actor = env.world.try_spawn_actor(bp, sp)
        if actor:
            # Don't enable autopilot — TrafficManager conflicts with synchronous mode
            # NPCs act as static obstacles for YOLO perception testing
            env._traffic_actors.append(actor)
            spawned_vehicles += 1

    # Pedestrians disabled — walker AI crashes CARLA in sync mode
    spawned_pedestrians = 0

    return spawned_vehicles, spawned_pedestrians


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
        
        # Step 3: Teleport second vehicle behind first vehicle, ON THE ROAD
        if len(managed_vehicles) >= 2:
            import carla
            first_transform = managed_vehicles[0].actor.get_transform()
            first_loc = first_transform.location
            first_yaw_rad = math.radians(first_transform.rotation.yaw)

            logger.info(f"First vehicle location: ({first_loc.x:.1f}, {first_loc.y:.1f}), yaw: {first_transform.rotation.yaw:.1f}")

            # Use CARLA waypoint API to find a valid on-road position behind vehicle_0
            wp0 = env.map.get_waypoint(first_loc, project_to_road=True)
            target_wp = wp0
            offset_distance = 20.0  # meters
            traveled = 0.0
            step = 2.0  # walk backwards in 2m increments

            while traveled < offset_distance and target_wp is not None:
                prev_wps = target_wp.previous(step)
                if not prev_wps:
                    break
                target_wp = prev_wps[0]
                traveled += step

            if target_wp is not None:
                teleport_transform = target_wp.transform
                teleport_transform.location.z += 0.5  # raise slightly to avoid ground clip
                logger.info(f"Teleporting vehicle_1 to waypoint at ({teleport_transform.location.x:.1f}, {teleport_transform.location.y:.1f}), traveled {traveled:.0f}m along road")
            else:
                # Fallback: place 10m behind using forward vector
                target_x = first_loc.x - offset_distance * math.cos(first_yaw_rad)
                target_y = first_loc.y - offset_distance * math.sin(first_yaw_rad)
                target_z = first_loc.z + 0.5
                teleport_transform = carla.Transform(
                    carla.Location(x=target_x, y=target_y, z=target_z),
                    carla.Rotation(yaw=first_transform.rotation.yaw)
                )
                logger.info(f"Waypoint walk failed, using fallback position: ({target_x:.1f}, {target_y:.1f})")

            # Teleport second vehicle
            managed_vehicles[1].actor.set_transform(teleport_transform)
            managed_vehicles[1].spawn_transform = None  # Clear so generate_route uses current position
            env.tick()
            time.sleep(0.5)

            # Verify distance
            new_loc = managed_vehicles[1].actor.get_location()
            actual_dist = math.sqrt(
                (new_loc.x - first_loc.x)**2 + (new_loc.y - first_loc.y)**2
            )
            logger.info(f"Teleported vehicle_1 to ({new_loc.x:.1f}, {new_loc.y:.1f})")
            logger.info(f"V2V initial distance: {actual_dist:.1f}m (target: {offset_distance:.0f}m)")

        # Now create agents with the spawned vehicles
        # IMPORTANT: vehicle_0 gets a fresh route; vehicle_1's route is regenerated
        # AFTER teleportation to match vehicle_0's direction (same end_location)
        vehicle_0_end_location = None

        for i, managed in enumerate(managed_vehicles):
            vid = f"vehicle_{i}"

            logger.info(f"Attaching RGB camera to {vid}...")
            managed.attach_rgb_camera(sensor_cfg.get('rgb_camera', {}))
            logger.info(f"Attaching depth camera to {vid}...")
            managed.attach_depth_camera(sensor_cfg.get('depth_camera', {}))

            if i == 0:
                logger.info(f"Generating route for {vid}...")
                env.generate_route(managed, sampling_resolution=4.0)
                # Remember vehicle_0's end location so vehicle_1 heads the same direction
                if managed._route_waypoints:
                    vehicle_0_end_location = managed._route_waypoints[-1].transform.location
                    logger.info(f"Vehicle 0 heading toward: ({vehicle_0_end_location.x:.1f}, {vehicle_0_end_location.y:.1f})")
            elif vehicle_0_end_location is not None:
                # Regenerate route from CURRENT (teleported) position toward vehicle_0's goal
                logger.info(f"Generating route for {vid} (from teleported position, same direction as vehicle_0)...")
                env.generate_route(managed, end_location=vehicle_0_end_location, sampling_resolution=4.0)
            else:
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

        # Spawn NPC traffic and pedestrians near ego vehicle routes
        npc_cfg = config.get('cooperation', {})
        num_npc_vehicles = npc_cfg.get('num_npc_vehicles', 8)
        num_npc_pedestrians = npc_cfg.get('num_npc_pedestrians', 5)
        if num_npc_vehicles > 0 or num_npc_pedestrians > 0:
            logger.info(f"Spawning NPC traffic ({num_npc_vehicles} vehicles, {num_npc_pedestrians} pedestrians)...")
            try:
                nv, np = spawn_npc_traffic(env, agents,
                                           num_vehicles=num_npc_vehicles,
                                           num_pedestrians=num_npc_pedestrians)
                logger.info(f"  Spawned {nv} NPC vehicles, {np} pedestrians")
            except Exception as e:
                logger.warning(f"NPC spawn failed: {e}")

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
                    rgb = agent._latest_rgb if hasattr(agent, '_latest_rgb') else None
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
                bev = draw_bird_eye_view(agents, env, bev_size=600, scale=3.0)
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
