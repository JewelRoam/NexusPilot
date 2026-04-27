"""
Artificial Potential Field (APF) Planner for obstacle avoidance.
Hybrid version: Rotational Fields + Recovery State Machine.
Ensures both smooth escape and hard-recovery capabilities.
"""
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Obstacle:
    """Obstacle representation for APF computation."""
    x: float                # world x position (meters)
    y: float                # world y position (meters)
    distance: float         # distance from ego (meters)
    category: str           # "vehicle", "pedestrian", "cyclist"
    confidence: float = 1.0
    velocity: Tuple[float, float] = (0.0, 0.0)  # vx, vy
    source: str = "local"   # "local" or "v2v" (cooperative)
    ethical_weight: float = 1.0


@dataclass
class PlannerOutput:
    """Output of the APF planner for one planning step."""
    target_steering: float
    target_speed: float
    emergency_brake: bool
    status: str = "normal"   # "normal", "avoiding", "emergency", "recovering"
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    recovery_mode: str = "none"
    debug_info: dict = field(default_factory=dict)
    nearest_obstacle_dist: float = 999.0


class APFPlanner:
    def __init__(self, config: dict):
        # 1. Base APF Parameters
        self.k_att = config.get("k_attractive", 1.0)
        self.k_rep = config.get("k_repulsive", 80.0)
        self.k_rot = config.get("k_rotational", 20.0)
        self.d0 = config.get("d0", 15.0)
        self.emergency_distance = config.get("emergency_distance", 3.0)
        self.max_speed = config.get("max_speed", 30.0)
        self.min_speed = config.get("min_speed", 5.0)
        self.goal_tolerance = config.get("goal_tolerance", 5.0)
        self.steer_smoothing = config.get("steer_smoothing", 0.3)  # EMA factor

        # Steering rate limiter
        self._last_steer = 0.0
        self._max_steer_delta = config.get("max_steer_delta", 0.15)  # per frame

        # 2. Ethical Weights for VRUs
        self._ethical_weights = {"pedestrian": 2.5, "cyclist": 2.0, "vehicle": 1.0}

        # 3. Recovery Logic (Restored and Integrated)
        self._recovery_state = {
            "is_recovering": False,
            "recovery_mode": "none", # "reversing", "turning"
            "start_time": 0.0,
            "stuck_pos": None,
            "attempts": 0
        }
        self.stuck_timeout = 5.0 # Seconds
        self.recovery_reverse_speed = -6.0

    def check_stuck(self, ego_pos: np.ndarray, ego_speed: float) -> bool:
        """Check if vehicle is in a local minimum (stuck)."""
        now = time.time()
        if not hasattr(self, '_last_pos_check'):
            self._last_pos_check = (ego_pos, now)
            return False

        last_pos, last_time = self._last_pos_check
        if now - last_time > self.stuck_timeout:
            dist = np.linalg.norm(ego_pos - last_pos)
            self._last_pos_check = (ego_pos, now)
            return dist < 0.5 and ego_speed < 2.0
        return False

    def detections_to_obstacles(self, detections, ego_x: float, ego_y: float,
                                 ego_yaw: float) -> List[Obstacle]:
        """Convert enriched DetectedObject list to APF Obstacle list."""
        obstacles = []
        ego_yaw_rad = math.radians(ego_yaw)
        for det in detections:
            if det.world_position is not None and det.distance > 0:
                ox, oy = det.world_position[0], det.world_position[1]
            elif det.distance > 0:
                # Approximate world position from ego pose + distance
                cx, cy = det.center
                # Rough direction from image center
                img_center_x = 320.0  # assumes 640 width
                angle_offset = (cx - img_center_x) / 320.0 * math.radians(55.0)  # ~110° FOV
                abs_angle = ego_yaw_rad + angle_offset
                ox = ego_x + det.distance * math.cos(abs_angle)
                oy = ego_y + det.distance * math.sin(abs_angle)
            else:
                continue  # skip detections without depth

            obs = Obstacle(
                x=ox,
                y=oy,
                distance=det.distance,
                category=det.category,
                confidence=det.confidence,
                velocity=det.velocity,
            )
            obs.ethical_weight = self._ethical_weights.get(det.category, 1.0)
            obstacles.append(obs)
        return obstacles

    def _calculate_forces(self, ego_pos: np.ndarray, goal_pos: np.ndarray, 
                           obstacles: List[Obstacle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 1. Attractive Force
        vec_to_goal = goal_pos - ego_pos
        dist_to_goal = np.linalg.norm(vec_to_goal)
        f_att = self.k_att * vec_to_goal / (dist_to_goal + 1e-6)
        if dist_to_goal > 10.0: f_att = f_att * 10.0 / dist_to_goal

        # 2. Repulsive & Rotational Forces
        f_rep_total = np.zeros(2)
        f_rot_total = np.zeros(2)
        
        for obs in obstacles:
            obs_pos = np.array([obs.x, obs.y])
            vec_from_obs = ego_pos - obs_pos
            dist = np.linalg.norm(vec_from_obs)
            
            if dist < self.d0:
                ew = self._ethical_weights.get(obs.category, 1.0)
                rep_mag = self.k_rep * ew * (1.0/dist - 1.0/self.d0) * (1.0/(dist**2))
                f_rep = rep_mag * vec_from_obs / dist
                f_rep_total += f_rep
                
                # Rotational component to escape local minima
                obs_to_goal = goal_pos - obs_pos
                cross_z = np.cross(vec_from_obs, obs_to_goal)
                steering_sign = 1.0 if cross_z > 0 else -1.0
                f_rot = steering_sign * self.k_rot * np.array([-f_rep[1], f_rep[0]]) / dist
                f_rot_total += f_rot
                
        return f_att, f_rep_total, f_rot_total

    def compute(self, ego_x: float, ego_y: float, ego_yaw: float,
                ego_speed: float, goal_x: float, goal_y: float,
                obstacles: List[Obstacle],
                cooperative_obstacles: Optional[List[Obstacle]] = None) -> PlannerOutput:
        
        ego_pos = np.array([ego_x, ego_y])
        goal_pos = np.array([goal_x, goal_y])
        all_obs = list(obstacles)
        if cooperative_obstacles: all_obs.extend(cooperative_obstacles)
        
        # === 1. Recovery Mode Logic (Fail-safe) ===
        if self._recovery_state["is_recovering"]:
            elapsed = time.time() - self._recovery_state["start_time"]
            if elapsed > 4.0: # End recovery after 4 seconds
                self._recovery_state["is_recovering"] = False
            else:
                return PlannerOutput(0.0, self.recovery_reverse_speed, False, "recovering", recovery_mode=self._recovery_state["recovery_mode"], nearest_obstacle_dist=999.0)

        if self.check_stuck(ego_pos, ego_speed):
            self._recovery_state.update({"is_recovering": True, "start_time": time.time(), "recovery_mode": "reversing"})
            return PlannerOutput(0.0, self.recovery_reverse_speed, False, "recovering", recovery_mode="reversing", nearest_obstacle_dist=999.0)

        # === 2. Normal Planning (Improved APF) ===
        f_att, f_rep, f_rot = self._calculate_forces(ego_pos, goal_pos, all_obs)
        f_total = f_att + f_rep + f_rot
        
        force_dir = math.atan2(f_total[1], f_total[0])
        ego_yaw_rad = math.radians(ego_yaw)
        angle_diff = math.atan2(math.sin(force_dir - ego_yaw_rad), math.cos(force_dir - ego_yaw_rad))
        
        raw_steer = np.clip(angle_diff / (math.pi/3), -1.0, 1.0)

        # Smooth steering with exponential moving average + rate limit
        smoothed = self.steer_smoothing * raw_steer + (1.0 - self.steer_smoothing) * self._last_steer
        delta = np.clip(smoothed - self._last_steer, -self._max_steer_delta, self._max_steer_delta)
        target_steering = self._last_steer + delta
        self._last_steer = target_steering
        
        # Distance logic
        min_dist = min([obs.distance for obs in all_obs]) if all_obs else self.d0
        emergency = min_dist < self.emergency_distance and len(all_obs) > 0
        
        if emergency:
            target_speed = 0.0
            status = "emergency"
        elif min_dist < self.d0:
            speed_factor = np.clip((min_dist - self.emergency_distance)/(self.d0 - self.emergency_distance), 0.2, 1.0)
            target_speed = self.max_speed * speed_factor
            status = "avoiding"
        else:
            target_speed = self.max_speed
            status = "normal"
            
        traj = self._generate_trajectory(ego_pos, ego_yaw_rad, f_total)

        return PlannerOutput(
            target_steering=float(target_steering),
            target_speed=float(target_speed),
            emergency_brake=emergency,
            status=status,
            trajectory=traj,
            debug_info={"f_att": f_att.tolist(), "f_rep": f_rep.tolist(), "f_rot": f_rot.tolist()},
            nearest_obstacle_dist=min_dist,
        )

    def _generate_trajectory(self, start_pos: np.ndarray, start_yaw: float,
                              force_vec: np.ndarray) -> List[Tuple[float, float]]:
        p0 = start_pos
        p1 = p0 + 2.0 * np.array([math.cos(start_yaw), math.sin(start_yaw)])
        p3 = p0 + np.clip(force_vec, -10, 10)
        p2 = p3 - 2.0 * (force_vec / (np.linalg.norm(force_vec) + 1e-6))
        
        points = []
        for t in np.linspace(0, 1, 10):
            pt = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t)*t**2 * p2 + t**3 * p3
            points.append((float(pt[0]), float(pt[1])))
        return points
