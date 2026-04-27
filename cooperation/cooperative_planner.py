"""
Cooperative Planner - Multi-vehicle coordination and conflict resolution.
Refined with Time-to-Intersection (TTI) formal logic and detection extrapolation.
"""
import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from cooperation.v2v_message import V2VMessage, SharedDetection
from planning.apf_planner import Obstacle


@dataclass
class CooperativeDecision:
    """Result of cooperative planning."""
    action: str             # "proceed", "yield", "slow_down", "stop", "follow"
    reason: str             # Human-readable explanation
    priority: int           # Vehicle priority (lower = higher priority)
    shared_obstacles: List[Obstacle]  # Obstacles from V2V to add to APF
    speed_adjustment: float  # Multiplier for target speed (0.0 - 1.0)
    cooperative_intent: str  # Intent to broadcast to others


class CooperativePlanner:
    def __init__(self, vehicle_id: str, config: dict):
        self.vehicle_id = vehicle_id
        self.config = config

        self.min_distance = 8.0
        self.intersection_radius = 15.0
        self.coordination_range = config.get('coordination_range', 50.0)
        self.active_range = config.get('active_coordination_range', 30.0)
        self.warning_range = config.get('warning_range', 15.0)

        self._deadlock_timer = {}
        self._nearby_vehicles = {}  # Track nearby vehicles for status queries

    def _calculate_tti(self, x: float, y: float, yaw: float, speed: float, 
                        cx: float, cy: float) -> float:
        """Calculate TTI with basic turn penalty."""
        dist = math.sqrt((cx - x)**2 + (cy - y)**2)
        v_ms = speed / 3.6
        if v_ms < 0.1: return 999.0
        
        # Calculate angle to conflict point to estimate turning overhead
        angle_to_cp = math.degrees(math.atan2(cy - y, cx - x))
        yaw_err = abs(yaw - angle_to_cp)
        yaw_err = min(yaw_err, 360 - yaw_err)
        
        # Heuristic: 0.2s extra per 15 deg turn
        turn_delay = (yaw_err / 15.0) * 0.2
        return dist / v_ms + turn_delay

    def _fuse_and_extrapolate(self, ego_x: float, ego_y: float, 
                               nearby_messages: Dict[str, V2VMessage]) -> List[Obstacle]:
        """
        Fuse detections and extrapolate positions based on latency to reduce 'ghosting'.
        """
        obstacles = []
        now = time.time()
        
        for vid, msg in nearby_messages.items():
            latency = now - msg.timestamp
            for det in msg.detections:
                # 1. Temporal Extrapolation (p_new = p_old + v * dt)
                # Note: msg.velocity is [vx, vy]
                dt = max(0, latency)
                extrapolated_x = det.world_x + msg.velocity[0] * dt
                extrapolated_y = det.world_y + msg.velocity[1] * dt
                
                dist = math.sqrt((extrapolated_x - ego_x)**2 + (extrapolated_y - ego_y)**2)
                
                if 1.0 < dist < 40.0:
                    obstacles.append(Obstacle(
                        x=extrapolated_x,
                        y=extrapolated_y,
                        distance=dist,
                        category=det.category,
                        confidence=det.confidence * 0.7, # Higher discount for older data
                        source="v2v"
                    ))
        return obstacles

    def process(self, ego_x: float, ego_y: float, ego_yaw: float,
                ego_speed: float, obstacles: List = None,
                v2v_messages: Dict[str, V2VMessage] = None) -> CooperativeDecision:
        if v2v_messages is None:
            v2v_messages = {}

        # 1. Filter nearby vehicles and update tracking state
        nearby = {}
        min_dist = 999.0
        for vid, msg in v2v_messages.items():
            ox, oy, _ = msg.position
            d = math.sqrt((ox - ego_x)**2 + (oy - ego_y)**2)
            if d <= self.coordination_range:
                nearby[vid] = msg
                min_dist = min(min_dist, d)

        # Store for _get_coordination_status queries
        self._nearby_vehicles = {}
        for vid, msg in nearby.items():
            ox, oy, _ = msg.position
            d = math.sqrt((ox - ego_x)**2 + (oy - ego_y)**2)
            self._nearby_vehicles[vid] = {
                "distance": d,
                "position": (ox, oy),
                "speed": msg.velocity[0],
            }

        # 2. Status detection
        if not nearby or min_dist > self.coordination_range:
            return CooperativeDecision("proceed", "No nearby vehicles", 0, [], 1.0, "cruising")

        # 3. Detection Fusion with extrapolation
        shared_obs = self._fuse_and_extrapolate(ego_x, ego_y, nearby)

        # 4. Intersection Negotiation (TTI Logic)
        if min_dist < self.intersection_radius:
            for vid, msg in nearby.items():
                # Simple CP estimation: midpoint
                cp_x, cp_y = (ego_x + msg.position[0])/2, (ego_y + msg.position[1])/2
                ego_tti = self._calculate_tti(ego_x, ego_y, ego_yaw, ego_speed, cp_x, cp_y)
                other_tti = self._calculate_tti(msg.position[0], msg.position[1], msg.position[2], msg.velocity[0], cp_x, cp_y)

                if abs(ego_tti - other_tti) < 2.0: # Conflict threshold
                    if ego_tti > other_tti or (abs(ego_tti - other_tti) < 0.2 and self.vehicle_id < vid):
                        return CooperativeDecision("yield", f"Yielding to {vid} (TTI diff: {ego_tti-other_tti:.1f}s)", 2, shared_obs, 0.2, "yielding")
                    else:
                        return CooperativeDecision("slow_down", "Arriving earlier, proceeding with caution", 1, shared_obs, 0.6, "cruising")

        return CooperativeDecision("proceed", "Coordination active", 0, shared_obs, 1.0, "cruising")

    def _get_coordination_status(self, ego_x: float, ego_y: float) -> Tuple[str, int]:
        """Determine coordination status based on nearby vehicles."""
        num_nearby = len(self._nearby_vehicles)
        if num_nearby == 0:
            return ("inactive", 0)

        min_dist = min(v["distance"] for v in self._nearby_vehicles.values())

        if min_dist < self.warning_range:
            status = "critical"
        elif min_dist < self.intersection_radius:
            status = "active"
        elif min_dist < self.active_range:
            status = "monitoring"
        else:
            status = "inactive"

        return (status, num_nearby)

    def _find_conflict_point(self, ego_x: float, ego_y: float, ego_yaw: float,
                              other_x: float, other_y: float, other_yaw: float) -> Optional[Tuple[float, float]]:
        """Estimate conflict point as intersection of two heading rays."""
        ego_dir = (math.cos(math.radians(ego_yaw)), math.sin(math.radians(ego_yaw)))
        other_dir = (math.cos(math.radians(other_yaw)), math.sin(math.radians(other_yaw)))

        # Solve: ego_pos + t * ego_dir = other_pos + s * other_dir
        dx, dy = other_x - ego_x, other_y - ego_y
        det = ego_dir[0] * other_dir[1] - ego_dir[1] * other_dir[0]

        if abs(det) < 1e-6:
            return None  # Parallel headings

        t = (dx * other_dir[1] - dy * other_dir[0]) / det
        s = (dx * ego_dir[1] - dy * ego_dir[0]) / det

        if t < 0 or s < 0:
            return None  # Conflict behind one of the vehicles

        cx = ego_x + t * ego_dir[0]
        cy = ego_y + t * ego_dir[1]
        return (cx, cy)

    def _calculate_time_to_intersection(self, x: float, y: float, yaw: float,
                                         speed: float, cx: float, cy: float) -> float:
        """Calculate time to reach a conflict point. Alias for _calculate_tti."""
        return self._calculate_tti(x, y, yaw, speed, cx, cy)
