"""
Test script for distance-aware coordination.
Tests the CooperativePlanner with simulated V2V messages at different distances.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cooperation.cooperative_planner import CooperativePlanner
from cooperation.v2v_message import V2VMessage, SharedDetection


def create_test_message(vehicle_id: str, x: float, y: float, yaw: float = 0.0, 
                        speed: float = 10.0, intent: str = "cruising") -> V2VMessage:
    """Create a test V2V message."""
    return V2VMessage(
        timestamp=__import__('time').time(),
        vehicle_id=vehicle_id,
        position=(x, y, yaw),
        velocity=(speed, yaw),
        detections=[],
        intent=intent,
        confidence=1.0,
        cooperation_request=None,
    )


def test_distance_coordination():
    """Test coordination at different distances."""
    print("=" * 60)
    print("Testing Distance-Aware Coordination")
    print("=" * 60)
    
    # Create planner for ego vehicle
    config = {
        'coordination_range': 50.0,
        'active_coordination_range': 30.0,
        'warning_range': 15.0,
    }
    planner = CooperativePlanner("ego_vehicle", config)
    
    # Ego vehicle at origin
    ego_x, ego_y, ego_yaw = 0.0, 0.0, 0.0
    ego_speed = 15.0
    
    test_cases = [
        ("No other vehicles", {}),
        ("Vehicle at 100m (beyond range)", {"other_1": create_test_message("other_1", 100.0, 0.0)}),
        ("Vehicle at 50m (at boundary)", {"other_1": create_test_message("other_1", 50.0, 0.0)}),
        ("Vehicle at 40m (monitoring range)", {"other_1": create_test_message("other_1", 40.0, 0.0)}),
        ("Vehicle at 25m (active range)", {"other_1": create_test_message("other_1", 25.0, 0.0)}),
        ("Vehicle at 12m (warning range)", {"other_1": create_test_message("other_1", 12.0, 0.0)}),
        ("Vehicle at 5m (critical range)", {"other_1": create_test_message("other_1", 5.0, 0.0)}),
        ("Two vehicles at different distances", {
            "other_1": create_test_message("other_1", 45.0, 0.0),  # monitoring
            "other_2": create_test_message("other_2", 20.0, 10.0),  # active
        }),
    ]
    
    for description, v2v_messages in test_cases:
        print(f"\n--- Test: {description} ---")
        decision = planner.process(ego_x, ego_y, ego_yaw, ego_speed, [], v2v_messages)
        
        # Get coordination status
        coord_status, num_nearby = planner._get_coordination_status(ego_x, ego_y)
        
        print(f"  Coordination Status: {coord_status}")
        print(f"  Nearby Vehicles: {num_nearby}")
        print(f"  Action: {decision.action}")
        print(f"  Reason: {decision.reason}")
        print(f"  Speed Adjustment: {decision.speed_adjustment:.2f}")
        print(f"  Shared Obstacles: {len(decision.shared_obstacles)}")
        
        # Show tracked vehicles
        if planner._nearby_vehicles:
            print(f"  Tracked Vehicles:")
            for vid, data in planner._nearby_vehicles.items():
                print(f"    - {vid}: {data['distance']:.1f}m")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)


def test_intersection_coordination():
    """Test intersection coordination with distance awareness."""
    print("\n" + "=" * 60)
    print("Testing Intersection Coordination with Distance Awareness")
    print("=" * 60)
    
    config = {
        'coordination_range': 50.0,
        'active_coordination_range': 30.0,
        'warning_range': 15.0,
    }
    planner = CooperativePlanner("ego_vehicle", config)
    
    # Ego vehicle heading east
    ego_x, ego_y, ego_yaw = 0.0, 0.0, 90.0
    ego_speed = 15.0
    
    # Other vehicle heading north (perpendicular, potential intersection conflict)
    other_yaw = 0.0
    
    test_distances = [60.0, 40.0, 20.0, 10.0]
    
    for dist in test_distances:
        # Place other vehicle to the right (east) of ego
        other_x = dist
        other_y = 0.0
        
        v2v_messages = {
            "other_1": create_test_message("other_1", other_x, other_y, other_yaw, 15.0)
        }
        
        decision = planner.process(ego_x, ego_y, ego_yaw, ego_speed, [], v2v_messages)
        coord_status, _ = planner._get_coordination_status(ego_x, ego_y)
        
        print(f"\nDistance: {dist:.0f}m | Status: {coord_status}")
        print(f"  Action: {decision.action} | Intent: {decision.cooperative_intent}")
        print(f"  Reason: {decision.reason}")


def test_tti_based_priority():
    """Test Time-to-Intersection based priority calculation."""
    print("\n" + "=" * 60)
    print("Testing Time-to-Intersection (TTI) Based Priority")
    print("=" * 60)
    
    config = {
        'coordination_range': 50.0,
        'active_coordination_range': 30.0,
        'warning_range': 15.0,
    }
    planner = CooperativePlanner("ego_vehicle", config)
    
    # Scenario: Two vehicles approaching an intersection from different directions
    # Ego: at (0, -20), heading north (0°), speed 30 km/h
    # Other: at (-20, 0), heading east (90°), varying speeds
    
    ego_x, ego_y, ego_yaw = 0.0, -20.0, 0.0
    ego_speed = 30.0  # km/h
    
    test_cases = [
        ("Other faster (arrives first)", -20.0, 0.0, 90.0, 50.0),
        ("Other same speed (close timing)", -20.0, 0.0, 90.0, 30.0),
        ("Other slower (we arrive first)", -20.0, 0.0, 90.0, 15.0),
        ("Other stopped", -20.0, 0.0, 90.0, 0.0),
    ]
    
    for description, other_x, other_y, other_yaw, other_speed in test_cases:
        print(f"\n--- Test: {description} ---")
        print(f"  Ego: ({ego_x}, {ego_y}) heading {ego_yaw}° at {ego_speed} km/h")
        print(f"  Other: ({other_x}, {other_y}) heading {other_yaw}° at {other_speed} km/h")
        
        # Calculate conflict point and TTI
        conflict = planner._find_conflict_point(ego_x, ego_y, ego_yaw, other_x, other_y, other_yaw)
        if conflict:
            ego_tti = planner._calculate_time_to_intersection(ego_x, ego_y, ego_yaw, ego_speed, *conflict)
            other_tti = planner._calculate_time_to_intersection(other_x, other_y, other_yaw, other_speed, *conflict)
            print(f"  Conflict point: ({conflict[0]:.1f}, {conflict[1]:.1f})")
            print(f"  Ego TTI: {ego_tti:.1f}s, Other TTI: {other_tti:.1f}s")
            print(f"  Time difference: {ego_tti - other_tti:.1f}s")
        
        v2v_messages = {
            "other_1": create_test_message("other_1", other_x, other_y, other_yaw, other_speed)
        }
        
        decision = planner.process(ego_x, ego_y, ego_yaw, ego_speed, [], v2v_messages)
        print(f"  Decision: {decision.action.upper()}")
        print(f"  Reason: {decision.reason}")


if __name__ == "__main__":
    test_distance_coordination()
    test_intersection_coordination()
    test_tti_based_priority()
