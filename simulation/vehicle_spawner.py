"""
V2V测试常驻服务 - 初始化车辆和环境并保持运行

功能：
1. 初始化CARLA环境和车辆
2. 创建V2V通信总线
3. 提供车辆控制接口供远程脚本调用
4. 监控车辆状态和V2V通信

使用方法：
  python -m simulation.vehicle_spawner

然后在其他终端运行 vehicle_controller.py 来控制车辆
"""
import os
import sys
import time
import json
import yaml
import zmq
import threading
import pickle
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.carla_env import CarlaEnv
from cooperation.v2v_message import V2VCommunicator
from utils.logger import setup_logger


class VehicleSpawner:
    """
    常驻车辆管理器
    提供ZMQ接口供外部脚本控制车辆
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.logger = setup_logger("VehicleSpawner", "INFO")
        self.config = self.load_config(config_path)
        
        # CARLA environment
        self.env = None
        self.vehicles = {}  # vehicle_id -> ManagedVehicle
        self.agents = {}    # vehicle_id -> VehicleAgent (if active)
        
        # V2V communication
        V2VCommunicator.reset_shared_bus()
        
        # Control server
        self.control_server = None
        self.running = False
        
        # Vehicle state cache for remote queries
        self.vehicle_states = {}
        self.lock = threading.Lock()

    def load_config(self, config_path: str) -> dict:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def initialize(self, num_vehicles: int = 2):
        """Initialize CARLA and spawn vehicles."""
        self.logger.info("=" * 60)
        self.logger.info("=== Initializing V2V Test Environment ===")
        self.logger.info("=" * 60)
        
        # Initialize CARLA
        self.logger.info("Connecting to CARLA...")
        self.env = CarlaEnv(self.config)
        self.env.connect()
        
        # Spawn vehicles in close proximity for testing
        sensor_cfg = self.config.get('sensors', {})
        
        self.logger.info(f"Spawning {num_vehicles} vehicles for V2V testing...")
        
        # Spawn first vehicle
        vehicle_0 = self.env.spawn_vehicle("vehicle_0")
        vehicle_0.attach_rgb_camera(sensor_cfg.get('rgb_camera', {}))
        vehicle_0.attach_depth_camera(sensor_cfg.get('depth_camera', {}))
        self.env.generate_route(vehicle_0, sampling_resolution=2.0)
        self.vehicles["vehicle_0"] = vehicle_0
        self.logger.info(f"Spawned vehicle_0 at {vehicle_0.actor.get_location()}")
        
        # Spawn second vehicle close to the first one
        # Get a spawn point near vehicle_0
        spawn_points = self.env.map.get_spawn_points()
        v0_loc = vehicle_0.actor.get_location()
        
        # Find spawn points within 10-30m (close enough for immediate V2V coordination)
        nearby_spawns = []
        for i, sp in enumerate(spawn_points):
            dist = abs(sp.location.x - v0_loc.x) + abs(sp.location.y - v0_loc.y)
            if 10 < dist < 30:  # Between 10-30m for active V2V testing
                nearby_spawns.append((i, sp, dist))
        
        # Sort by distance and try to spawn
        nearby_spawns.sort(key=lambda x: x[2])
        
        vehicle_1 = None
        for idx, sp, dist in nearby_spawns[:5]:  # Try top 5 closest
            try:
                vehicle_1 = self.env.spawn_vehicle("vehicle_1", spawn_index=idx)
                break
            except RuntimeError:
                continue
        
        if vehicle_1 is None:
            # Fallback: spawn at a random but close point
            vehicle_1 = self.env.spawn_vehicle("vehicle_1")
        
        vehicle_1.attach_rgb_camera(sensor_cfg.get('rgb_camera', {}))
        vehicle_1.attach_depth_camera(sensor_cfg.get('depth_camera', {}))
        self.env.generate_route(vehicle_1, sampling_resolution=2.0)
        self.vehicles["vehicle_1"] = vehicle_1
        self.logger.info(f"Spawned vehicle_1 at {vehicle_1.actor.get_location()}")
        
        # Calculate actual distance between vehicles
        v0_x, v0_y, _ = vehicle_0.get_location()
        v1_x, v1_y, _ = vehicle_1.get_location()
        distance = ((v0_x - v1_x)**2 + (v0_y - v1_y)**2)**0.5
        self.logger.info(f"Distance between vehicles: {distance:.1f}m")
        
        # Initialize vehicle states
        self.update_vehicle_states()
        
        # Wait for sensor initialization
        self.logger.info("Waiting for sensor initialization...")
        for _ in range(20):
            self.env.tick()
            time.sleep(0.05)
        
        self.logger.info("Sensor initialization complete")
        
        # Save spawn info for remote controller
        self.save_spawn_info()
        
        return True

    def update_vehicle_states(self):
        """Update cached vehicle states."""
        with self.lock:
            for vid, vehicle in self.vehicles.items():
                x, y, z = vehicle.get_location()
                yaw = vehicle.get_yaw()
                speed = vehicle.get_speed_kmh()
                wp = vehicle.get_next_waypoint()
                
                self.vehicle_states[vid] = {
                    'position': (x, y, z),
                    'yaw': yaw,
                    'speed': speed,
                    'next_waypoint': wp,
                    'timestamp': time.time()
                }

    def save_spawn_info(self):
        """Save spawn information for remote controller."""
        spawn_info = {
            'host': 'localhost',
            'control_port': 5559,
            'state_port': 5560,
            'vehicles': list(self.vehicles.keys()),
            'initial_distance': self.get_distance_between_vehicles()
        }
        
        with open('output/spawn_info.json', 'w') as f:
            json.dump(spawn_info, f, indent=2)
        
        self.logger.info(f"Spawn info saved to output/spawn_info.json")
        self.logger.info(f"  Control port: {spawn_info['control_port']}")
        self.logger.info(f"  State port: {spawn_info['state_port']}")

    def get_distance_between_vehicles(self) -> float:
        """Get distance between vehicle_0 and vehicle_1."""
        if len(self.vehicles) < 2:
            return -1
        
        v0 = self.vehicles.get("vehicle_0")
        v1 = self.vehicles.get("vehicle_1")
        
        if v0 and v1:
            x0, y0, _ = v0.get_location()
            x1, y1, _ = v1.get_location()
            return ((x0 - x1)**2 + (y0 - y1)**2)**0.5
        
        return -1

    def apply_control(self, vehicle_id: str, throttle: float, steer: float, brake: float = 0.0):
        """Apply control to a specific vehicle."""
        if vehicle_id not in self.vehicles:
            return False
        
        vehicle = self.vehicles[vehicle_id]
        control = {
            'throttle': throttle,
            'steer': steer,
            'brake': brake
        }
        vehicle.actor.apply_control(vehicle.actor.get_world().ActorType.t4)
        
        # Simpler direct control
        from carla import VehicleControl
        c = VehicleControl(throttle=throttle, steer=steer, brake=brake)
        vehicle.actor.apply_control(c)
        
        return True

    def move_to_location(self, vehicle_id: str, target_x: float, target_y: float, 
                        speed: float = 10.0):
        """Move vehicle toward target location."""
        if vehicle_id not in self.vehicles:
            return False
        
        vehicle = self.vehicles[vehicle_id]
        x, y, _ = vehicle.get_location()
        
        # Calculate direction
        dx = target_x - x
        dy = target_y - y
        distance = (dx**2 + dy**2)**0.5
        
        if distance < 1.0:
            return True  # Already at target
        
        # Calculate angle to target
        import math
        target_angle = math.degrees(math.atan2(dy, dx))
        current_yaw = vehicle.get_yaw()
        
        # Calculate steering
        angle_diff = target_angle - current_yaw
        while angle_diff > 180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360
        
        steer = max(-1.0, min(1.0, angle_diff / 45.0))
        
        # Apply control
        from carla import VehicleControl
        c = VehicleControl(throttle=min(1.0, speed / 20.0), steer=steer, brake=0.0)
        vehicle.actor.apply_control(c)
        
        return distance

    def start_control_server(self, port: int = 5559):
        """Start ZMQ control server."""
        context = zmq.Context()
        server = context.socket(zmq.REP)
        server.bind(f"tcp://*:{port}")
        server.setsockopt(zmq.RCVTIMEO, 1000)  # 1 second timeout for recv
        
        self.logger.info(f"Control server listening on port {port}")
        
        # Server loop - runs until stop command
        while True:
            try:
                # Wait for command
                message = server.recv_json()
                
                cmd = message.get('command')
                response = {'status': 'ok'}
                
                if cmd == 'get_states':
                    self.update_vehicle_states()
                    with self.lock:
                        response['states'] = self.vehicle_states
                    response['distance'] = self.get_distance_between_vehicles()
                    
                elif cmd == 'control':
                    vid = message.get('vehicle_id')
                    throttle = message.get('throttle', 0.0)
                    steer = message.get('steer', 0.0)
                    brake = message.get('brake', 0.0)
                    self.apply_control(vid, throttle, steer, brake)
                    
                elif cmd == 'move_to':
                    vid = message.get('vehicle_id')
                    tx = message.get('target_x')
                    ty = message.get('target_y')
                    speed = message.get('speed', 10.0)
                    remaining = self.move_to_location(vid, tx, ty, speed)
                    response['remaining_distance'] = remaining
                    
                elif cmd == 'tick':
                    self.env.tick()
                    self.update_vehicle_states()
                    response['ticked'] = True
                    
                elif cmd == 'stop':
                    response['status'] = 'stopping'
                    server.send_json(response)
                    server.close()
                    context.term()
                    self.logger.info("Control server stopped")
                    return
                    
                else:
                    response['status'] = 'unknown_command'
                
                server.send_json(response)
                
            except zmq.Again:
                continue  # Timeout, continue loop
            except Exception as e:
                self.logger.error(f"Control server error: {e}")
                break
        
        server.close()
        context.term()
        self.logger.info("Control server stopped")

    def run(self):
        """Main run loop."""
        try:
            # Initialize
            self.initialize()
            
            # Start control server in background thread
            server_thread = threading.Thread(target=self.start_control_server, daemon=True)
            server_thread.start()
            
            self.logger.info("=" * 60)
            self.logger.info("=== V2V Test Environment Ready ===")
            self.logger.info(f"=== {len(self.vehicles)} vehicles spawned ===")
            self.logger.info("=== Waiting for control commands... ===")
            self.logger.info("=== Press Ctrl+C to stop ===")
            self.logger.info("=" * 60)
            
            # Main tick loop
            while self.running:
                self.env.tick()
                self.update_vehicle_states()
                
                # Log distance periodically
                dist = self.get_distance_between_vehicles()
                if dist > 0:
                    self.logger.info(f"Distance between vehicles: {dist:.1f}m")
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Cleanup resources."""
        self.logger.info("Cleaning up...")
        self.running = False
        
        if self.control_server:
            self.control_server.close()
        
        if self.env:
            V2VCommunicator.reset_shared_bus()
            self.env.cleanup()
        
        self.logger.info("Cleanup complete")


def main():
    spawner = VehicleSpawner()
    spawner.run()


if __name__ == '__main__':
    main()
