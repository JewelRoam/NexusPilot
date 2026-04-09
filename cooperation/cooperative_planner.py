"""
Cooperative Planner - Multi-vehicle coordination and conflict resolution.
Implements cooperative behaviours for the V2V system:
  1. Intersection coordination (priority negotiation)
  2. Occlusion compensation (shared detection fusion)
  3. Platooning (convoy following)
  4. Deadlock resolution (narrow road yield)

This is the key innovation module addressing the feedback to enhance
inter-vehicle collaboration.
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
    """
    Handles multi-vehicle cooperative decision making.
    Processes V2V messages and outputs cooperative decisions that
    integrate with the APF planner.
    
    Enhanced with distance-aware coordination - only activates when
    vehicles are close enough to benefit from coordination.
    """

    def __init__(self, vehicle_id: str, config: dict):
        """
        Args:
            vehicle_id: this vehicle's ID
            config: cooperation section of config.yaml
        """
        self.vehicle_id = vehicle_id
        self.config = config

        # Safety parameters
        self.min_inter_vehicle_distance = 8.0  # meters
        self.intersection_radius = 15.0        # meters
        self.platooning_distance = 10.0        # meters
        self.deadlock_timeout = 3.0            # seconds

        # Distance-aware coordination thresholds
        self.coordination_range = config.get('coordination_range', 50.0)  # Max range for coordination
        self.active_coordination_range = config.get('active_coordination_range', 30.0)  # Active coordination
        self.warning_range = config.get('warning_range', 15.0)  # Warning/critical range

        # State tracking
        self._deadlock_timer = {}  # vehicle_id -> first_detected_time
        self._nearby_vehicles = {}  # vehicle_id -> {'distance': float, 'last_seen': timestamp}

    def _update_nearby_vehicles(self, ego_x: float, ego_y: float,
                                 v2v_messages: Dict[str, V2VMessage]) -> Dict[str, dict]:
        """
        Update tracking of nearby vehicles based on distance.
        Returns filtered messages only for vehicles within coordination range.
        
        This is the key enhancement for distance-aware coordination.
        """
        now = time.time()
        nearby = {}
        
        # Debug log received messages
        if not v2v_messages:
            print(f"[Coop DEBUG] {self.vehicle_id}: No V2V messages received this frame")
        else:
            print(f"[Coop DEBUG] {self.vehicle_id}: Received {len(v2v_messages)} V2V messages")
        
        for vid, msg in v2v_messages.items():
            other_x, other_y, _ = msg.position
            dist = math.sqrt((other_x - ego_x) ** 2 + (other_y - ego_y) ** 2)
            
            # Update tracking
            self._nearby_vehicles[vid] = {
                'distance': dist,
                'last_seen': now,
                'position': (other_x, other_y),
                'velocity': msg.velocity,
                'intent': msg.intent,
            }
            
            print(f"[Coop DEBUG] {self.vehicle_id}: Tracking {vid} at dist={dist:.1f}m")
            
            # Only include vehicles within coordination range
            if dist <= self.coordination_range:
                nearby[vid] = msg
                print(f"[Coop DEBUG] {self.vehicle_id}: {vid} IN RANGE (coord_range={self.coordination_range}m)")
        
        # Clean up stale entries (>5 seconds old)
        stale = [vid for vid, data in self._nearby_vehicles.items() 
                 if now - data['last_seen'] > 5.0]
        for vid in stale:
            del self._nearby_vehicles[vid]
            self._deadlock_timer.pop(vid, None)
        
        return nearby

    def _get_coordination_status(self, ego_x: float, ego_y: float) -> Tuple[str, int]:
        """
        Determine coordination status based on nearest vehicle distance.
        
        Returns:
            (status_str, num_nearby_vehicles)
            status_str: "inactive", "monitoring", "active", "warning", "critical"
        """
        if not self._nearby_vehicles:
            return "inactive", 0
        
        min_dist = min(data['distance'] for data in self._nearby_vehicles.values())
        num_nearby = sum(1 for d in self._nearby_vehicles.values() 
                        if d['distance'] <= self.coordination_range)
        
        if min_dist > self.coordination_range:
            return "inactive", num_nearby
        elif min_dist > self.active_coordination_range:
            return "monitoring", num_nearby
        elif min_dist > self.warning_range:
            return "active", num_nearby
        elif min_dist > self.min_inter_vehicle_distance:
            return "warning", num_nearby
        else:
            return "critical", num_nearby

    def process(self, ego_x: float, ego_y: float, ego_yaw: float,
                ego_speed: float, ego_detections: list,
                v2v_messages: Dict[str, V2VMessage]) -> CooperativeDecision:
        """
        Main cooperative planning step with distance-aware coordination.
        Only activates full coordination when vehicles are close enough.

        Args:
            ego_x, ego_y, ego_yaw: ego vehicle state
            ego_speed: ego speed in km/h
            ego_detections: local DetectedObject list
            v2v_messages: received V2V messages from other vehicles

        Returns:
            CooperativeDecision with action and shared obstacles
        """
        # STEP 1: Always update nearby vehicles tracking (even with empty messages)
        nearby_messages = self._update_nearby_vehicles(ego_x, ego_y, v2v_messages)
        coord_status, num_nearby = self._get_coordination_status(ego_x, ego_y)

        # If no vehicles in coordination range, return inactive
        if not nearby_messages or coord_status == "inactive":
            return CooperativeDecision(
                action="proceed",
                reason=f"No vehicles in coordination range ({num_nearby} vehicles beyond {self.coordination_range:.0f}m)",
                priority=0,
                shared_obstacles=[],
                speed_adjustment=1.0,
                cooperative_intent="cruising",
            )

        # STEP 2: Gather shared obstacles from nearby vehicles only
        shared_obstacles = self._fuse_shared_detections(
            ego_x, ego_y, nearby_messages
        )

        # STEP 3: Check for various cooperative scenarios
        # Priority: emergency > critical proximity > intersection > deadlock > platooning > normal

        # 1. Check for emergency from other vehicles (always check)
        emergency = self._check_emergency(nearby_messages)
        if emergency:
            return CooperativeDecision(
                action="slow_down",
                reason=f"Emergency! Vehicle {emergency} broadcasting emergency",
                priority=1,
                shared_obstacles=shared_obstacles,
                speed_adjustment=0.3,
                cooperative_intent="yielding",
            )

        # 2. Check critical proximity (very close vehicles)
        if coord_status == "critical":
            closest_vid = min(self._nearby_vehicles.keys(),
                            key=lambda v: self._nearby_vehicles[v]['distance'])
            closest_dist = self._nearby_vehicles[closest_vid]['distance']
            return CooperativeDecision(
                action="slow_down",
                reason=f"Critical proximity to {closest_vid} ({closest_dist:.1f}m)",
                priority=1,
                shared_obstacles=shared_obstacles,
                speed_adjustment=0.4,
                cooperative_intent="yielding",
            )

        # 3. Check intersection conflict (only in active/warning range)
        if coord_status in ["active", "warning"]:
            intersection = self._check_intersection_conflict(
                ego_x, ego_y, ego_yaw, ego_speed, nearby_messages
            )
            if intersection:
                return intersection.to_decision(shared_obstacles)

        # 4. Check deadlock scenario (only when close)
        if coord_status in ["active", "warning", "critical"]:
            deadlock = self._check_deadlock(
                ego_x, ego_y, ego_speed, nearby_messages
            )
            if deadlock:
                return CooperativeDecision(
                    action=deadlock["action"],
                    reason=deadlock["reason"],
                    priority=deadlock["priority"],
                    shared_obstacles=shared_obstacles,
                    speed_adjustment=deadlock["speed_adj"],
                    cooperative_intent=deadlock["intent"],
                )

        # 5. Check platooning opportunity (can work at longer ranges)
        if coord_status in ["monitoring", "active"]:
            platoon = self._check_platooning(
                ego_x, ego_y, ego_yaw, ego_speed, nearby_messages
            )
            if platoon:
                return CooperativeDecision(
                    action="follow",
                    reason=platoon["reason"],
                    priority=1,
                    shared_obstacles=shared_obstacles,
                    speed_adjustment=platoon["speed_adj"],
                    cooperative_intent="following",
                )

        # Default: monitoring mode - aware but not actively coordinating
        status_desc = {
            "monitoring": f"Monitoring {num_nearby} vehicles",
            "active": f"Active coordination with {num_nearby} vehicles",
            "warning": f"Warning: close proximity to {num_nearby} vehicles",
        }.get(coord_status, "Cooperative awareness active")
        
        return CooperativeDecision(
            action="proceed",
            reason=status_desc,
            priority=0,
            shared_obstacles=shared_obstacles,
            speed_adjustment=1.0,
            cooperative_intent="cruising",
        )

    def _fuse_shared_detections(self, ego_x: float, ego_y: float,
                                 v2v_messages: Dict[str, V2VMessage]) -> List[Obstacle]:
        """
        Convert shared detections from V2V messages into Obstacle objects
        for the APF planner. This is the OCCLUSION COMPENSATION feature.

        Key innovation: obstacles that ego cannot see but other vehicles can
        are fused into the local planning pipeline.
        """
        obstacles = []
        for vid, msg in v2v_messages.items():
            for det in msg.detections:
                # Calculate distance from ego
                dx = det.world_x - ego_x
                dy = det.world_y - ego_y
                dist = math.sqrt(dx * dx + dy * dy)

                # Only include if within reasonable range and
                # not too far (likely irrelevant)
                if 2.0 < dist < 50.0:
                    obstacles.append(Obstacle(
                        x=det.world_x,
                        y=det.world_y,
                        distance=dist,
                        category=det.category,
                        confidence=det.confidence * 0.8,  # Discount shared info
                        source="v2v",
                    ))

        return obstacles

    def _check_emergency(self, v2v_messages: Dict[str, V2VMessage]) -> Optional[str]:
        """Check if any vehicle is broadcasting an emergency."""
        for vid, msg in v2v_messages.items():
            if msg.intent == "emergency_brake":
                return vid
        return None

    def _calculate_time_to_intersection(self, x: float, y: float, yaw: float, 
                                        speed: float, intersection_x: float, 
                                        intersection_y: float) -> float:
        """
        Calculate estimated time to reach intersection point.
        
        Args:
            x, y: current position
            yaw: heading in degrees
            speed: speed in km/h
            intersection_x, intersection_y: intersection point
            
        Returns:
            Estimated time in seconds to reach intersection
        """
        if speed < 0.5:  # Almost stopped
            return float('inf')
        
        # Calculate distance to intersection
        dx = intersection_x - x
        dy = intersection_y - y
        distance = math.sqrt(dx * dx + dy * dy)
        
        # Convert speed to m/s
        speed_ms = speed / 3.6
        
        # Basic time = distance / speed
        time_to_intersection = distance / speed_ms
        
        # Add penalty for large heading deviation (need to turn more)
        angle_to_intersection = math.degrees(math.atan2(dy, dx))
        heading_error = abs(yaw - angle_to_intersection)
        heading_error = min(heading_error, 360 - heading_error)
        
        # Add 0.5s penalty per 45 degrees of heading error
        turn_penalty = (heading_error / 45.0) * 0.5
        
        return time_to_intersection + turn_penalty

    def _find_conflict_point(self, ego_x: float, ego_y: float, ego_yaw: float,
                             other_x: float, other_y: float, other_yaw: float) -> Optional[Tuple[float, float]]:
        """
        Estimate the conflict point where two vehicle paths may intersect.
        
        Returns:
            (x, y) of estimated conflict point, or None if paths don't cross
        """
        # Convert headings to radians
        ego_yaw_rad = math.radians(ego_yaw)
        other_yaw_rad = math.radians(other_yaw)
        
        # Direction vectors
        ego_dx = math.cos(ego_yaw_rad)
        ego_dy = math.sin(ego_yaw_rad)
        other_dx = math.cos(other_yaw_rad)
        other_dy = math.sin(other_yaw_rad)
        
        # Check if paths are roughly parallel
        cross_product = ego_dx * other_dy - ego_dy * other_dx
        if abs(cross_product) < 0.1:  # Nearly parallel
            return None
        
        # Find intersection of two lines
        # Line 1: ego_x + t1 * ego_dx, ego_y + t1 * ego_dy
        # Line 2: other_x + t2 * other_dx, other_y + t2 * other_dy
        
        dx = other_x - ego_x
        dy = other_y - ego_y
        
        t1 = (dx * other_dy - dy * other_dx) / cross_product
        
        # Conflict point is ahead on ego's path
        conflict_x = ego_x + t1 * ego_dx
        conflict_y = ego_y + t1 * ego_dy
        
        return (conflict_x, conflict_y)

    def _check_intersection_conflict(self, ego_x: float, ego_y: float,
                                      ego_yaw: float, ego_speed: float,
                                      v2v_messages: Dict[str, V2VMessage]) -> Optional[CooperativeDecision]:
        """
        Detect potential intersection collision and negotiate priority.
        
        ENHANCED: Uses Time-to-Intersection (TTI) based priority:
        - Vehicle that will arrive at conflict point FIRST has priority
        - If arrival times are similar (< 2s difference), use tiebreaker rules
        - Considers speed, distance, and heading alignment
        """
        for vid, msg in v2v_messages.items():
            other_x, other_y, other_yaw = msg.position
            other_speed = msg.velocity[0]

            # Distance between vehicles
            dx = other_x - ego_x
            dy = other_y - ego_y
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > self.intersection_radius * 2:
                continue

            # Check if paths might cross (heading difference > 30 degrees)
            heading_diff = abs(ego_yaw - other_yaw)
            heading_diff = min(heading_diff, 360 - heading_diff)

            if heading_diff > 30 and dist < self.intersection_radius:
                # Find conflict point
                conflict_point = self._find_conflict_point(
                    ego_x, ego_y, ego_yaw, other_x, other_y, other_yaw
                )
                
                if conflict_point is None:
                    # Paths don't cross, treat as normal proximity
                    continue
                
                conflict_x, conflict_y = conflict_point
                
                # Calculate time to intersection for both vehicles
                ego_tti = self._calculate_time_to_intersection(
                    ego_x, ego_y, ego_yaw, ego_speed, conflict_x, conflict_y
                )
                other_tti = self._calculate_time_to_intersection(
                    other_x, other_y, other_yaw, other_speed, conflict_x, conflict_y
                )
                
                # Time difference
                time_diff = ego_tti - other_tti
                
                # Decision logic based on arrival times
                if abs(time_diff) < 2.0:  # Arriving within 2 seconds of each other
                    # Close timing - need careful coordination
                    if time_diff < 0:
                        # We arrive slightly earlier
                        if time_diff < -1.0:
                            # We arrive significantly earlier - proceed
                            return _IntersectionResult(
                                action="slow_down",
                                reason=f"Intersection with {vid}: arriving {abs(time_diff):.1f}s earlier, proceeding",
                                priority=0,
                                speed_adjustment=0.6,
                                cooperative_intent="cruising",
                            )
                        else:
                            # Too close to call - use ID tiebreaker for determinism
                            if self.vehicle_id < vid:
                                return _IntersectionResult(
                                    action="yield",
                                    reason=f"Intersection with {vid}: close timing ({abs(time_diff):.1f}s), yielding by ID",
                                    priority=2,
                                    speed_adjustment=0.0,
                                    cooperative_intent="yielding",
                                )
                            else:
                                return _IntersectionResult(
                                    action="slow_down",
                                    reason=f"Intersection with {vid}: close timing ({abs(time_diff):.1f}s), proceeding by ID",
                                    priority=0,
                                    speed_adjustment=0.5,
                                    cooperative_intent="cruising",
                                )
                    else:
                        # Other arrives earlier
                        if time_diff > 1.0:
                            # They arrive significantly earlier - we yield
                            return _IntersectionResult(
                                action="yield",
                                reason=f"Intersection with {vid}: arriving {time_diff:.1f}s later, yielding",
                                priority=2,
                                speed_adjustment=0.0,
                                cooperative_intent="yielding",
                            )
                        else:
                            # Too close to call - use ID tiebreaker
                            if self.vehicle_id < vid:
                                return _IntersectionResult(
                                    action="yield",
                                    reason=f"Intersection with {vid}: close timing ({time_diff:.1f}s), yielding by ID",
                                    priority=2,
                                    speed_adjustment=0.0,
                                    cooperative_intent="yielding",
                                )
                            else:
                                return _IntersectionResult(
                                    action="slow_down",
                                    reason=f"Intersection with {vid}: close timing ({time_diff:.1f}s), proceeding by ID",
                                    priority=0,
                                    speed_adjustment=0.5,
                                    cooperative_intent="cruising",
                                )
                elif time_diff < 0:
                    # We arrive clearly earlier
                    return _IntersectionResult(
                        action="proceed",
                        reason=f"Intersection with {vid}: arriving {abs(time_diff):.1f}s earlier, clear priority",
                        priority=0,
                        speed_adjustment=0.8,
                        cooperative_intent="cruising",
                    )
                else:
                    # They arrive clearly earlier
                    return _IntersectionResult(
                        action="yield",
                        reason=f"Intersection with {vid}: arriving {time_diff:.1f}s later, yielding",
                        priority=2,
                        speed_adjustment=0.0,
                        cooperative_intent="yielding",
                    )

        return None

    def _check_deadlock(self, ego_x: float, ego_y: float, ego_speed: float,
                        v2v_messages: Dict[str, V2VMessage]) -> Optional[dict]:
        """
        Detect deadlock: two vehicles facing each other, both stopped.
        Resolution: vehicle with lower ID backs up / yields.
        """
        for vid, msg in v2v_messages.items():
            other_x, other_y, other_yaw = msg.position
            other_speed = msg.velocity[0]

            dist = math.sqrt((other_x - ego_x) ** 2 + (other_y - ego_y) ** 2)

            # Both slow and close
            if dist < self.min_inter_vehicle_distance and ego_speed < 2 and other_speed < 2:
                if vid not in self._deadlock_timer:
                    self._deadlock_timer[vid] = time.time()

                elapsed = time.time() - self._deadlock_timer[vid]
                if elapsed > self.deadlock_timeout:
                    # Deadlock confirmed
                    if self.vehicle_id < vid:
                        return {
                            "action": "yield",
                            "reason": f"Deadlock with {vid}, yielding",
                            "priority": 2,
                            "speed_adj": 0.0,
                            "intent": "yielding",
                        }
                    else:
                        return {
                            "action": "proceed",
                            "reason": f"Deadlock with {vid}, proceeding",
                            "priority": 0,
                            "speed_adj": 0.3,
                            "intent": "cruising",
                        }
            else:
                self._deadlock_timer.pop(vid, None)

        return None

    def _check_platooning(self, ego_x: float, ego_y: float, ego_yaw: float,
                          ego_speed: float,
                          v2v_messages: Dict[str, V2VMessage]) -> Optional[dict]:
        """
        Check if platooning (convoy) behaviour is appropriate.
        Conditions: same heading, sequential positions, moderate speed.
        """
        for vid, msg in v2v_messages.items():
            other_x, other_y, other_yaw = msg.position
            other_speed = msg.velocity[0]

            dist = math.sqrt((other_x - ego_x) ** 2 + (other_y - ego_y) ** 2)

            # Check heading alignment (within 15 degrees)
            heading_diff = abs(ego_yaw - other_yaw)
            heading_diff = min(heading_diff, 360 - heading_diff)

            if heading_diff < 15 and self.platooning_distance < dist < self.platooning_distance * 3:
                # Check if other vehicle is in front
                dx = other_x - ego_x
                dy = other_y - ego_y
                angle_to_other = math.degrees(math.atan2(dy, dx))
                angle_diff = abs(ego_yaw - angle_to_other)
                angle_diff = min(angle_diff, 360 - angle_diff)

                if angle_diff < 30:  # Other vehicle is ahead
                    target_speed_adj = min(other_speed / max(ego_speed, 1.0), 1.0)
                    return {
                        "reason": f"Platooning with {vid} at {dist:.1f}m",
                        "speed_adj": target_speed_adj,
                    }

        return None


@dataclass
class _IntersectionResult:
    """Helper for intersection check results."""
    action: str
    reason: str
    priority: int
    speed_adjustment: float
    cooperative_intent: str
    shared_obstacles: List[Obstacle] = None

    def to_decision(self, shared_obstacles: List[Obstacle]) -> CooperativeDecision:
        return CooperativeDecision(
            action=self.action,
            reason=self.reason,
            priority=self.priority,
            shared_obstacles=shared_obstacles,
            speed_adjustment=self.speed_adjustment,
            cooperative_intent=self.cooperative_intent,
        )
