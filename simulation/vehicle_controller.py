"""
远程车辆控制器 - 与 vehicle_spawner.py 配合使用

功能：
1. 连接到常驻的spawner服务
2. 控制两个车辆向彼此移动
3. 监控V2V协调状态
4. 记录测试数据

使用方法：
  python -m simulation.vehicle_controller

前提条件：
  1. 先启动 vehicle_spawner.py
  2. 确保两个车辆已spawn并初始化
"""
import os
import sys
import time
import json
import zmq
import math
from typing import Dict, List, Tuple, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import setup_logger


class RemoteVehicleController:
    """
    远程车辆控制器
    通过ZMQ连接到spawner并控制车辆
    """

    def __init__(self, host: str = "localhost", port: int = 5559):
        self.logger = setup_logger("RemoteController", "INFO")
        self.host = host
        self.port = port
        
        # ZMQ socket
        self.context = zmq.Context()
        self.socket = self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")
        self.socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5 second timeout
        
        # State
        self.running = False
        self.vehicle_states = {}
        
        # Test data
        self.test_data = []
        self.start_time = None

    def send_command(self, cmd: dict) -> dict:
        """Send command to spawner and get response."""
        try:
            self.socket.send_json(cmd)
            response = self.socket.recv_json()
            return response
        except zmq.Again:
            self.logger.error("Timeout waiting for response from spawner")
            return {'status': 'timeout'}
        except Exception as e:
            self.logger.error(f"Error communicating with spawner: {e}")
            return {'status': 'error', 'message': str(e)}

    def get_states(self) -> Dict:
        """Get current vehicle states."""
        response = self.send_command({'command': 'get_states'})
        if 'states' in response:
            self.vehicle_states = response['states']
        return response

    def get_distance(self) -> float:
        """Get distance between vehicles."""
        response = self.send_command({'command': 'get_states'})
        return response.get('distance', -1)

    def tick(self):
        """Tick simulation."""
        response = self.send_command({'command': 'tick'})
        return response.get('ticked', False)

    def move_toward_other(self, vehicle_id: str, speed: float = 15.0):
        """
        Move one vehicle toward the other.
        This is the key test for V2V coordination.
        """
        if vehicle_id not in self.vehicle_states:
            self.get_states()
        
        if vehicle_id not in self.vehicle_states:
            self.logger.error(f"Vehicle {vehicle_id} not found")
            return False
        
        # Get our position
        our_state = self.vehicle_states[vehicle_id]
        our_x, our_y, _ = our_state['position']
        our_yaw = our_state['yaw']
        
        # Find other vehicle
        other_id = "vehicle_1" if vehicle_id == "vehicle_0" else "vehicle_0"
        if other_id not in self.vehicle_states:
            self.logger.warning(f"Other vehicle {other_id} not found")
            return False
        
        other_state = self.vehicle_states[other_id]
        other_x, other_y, _ = other_state['position']
        other_yaw = other_state['yaw']
        
        # Calculate direction to other vehicle
        dx = other_x - our_x
        dy = other_y - our_y
        distance = math.sqrt(dx**2 + dy**2)
        angle_to_target = math.degrees(math.atan2(dy, dx))
        
        # Calculate steering
        angle_diff = angle_to_target - our_yaw
        while angle_diff > 180: angle_diff -= 360
        while angle_diff < -180: angle_diff += 360
        
        steer = max(-1.0, min(1.0, angle_diff / 45.0))
        
        # Apply throttle (full speed toward other vehicle)
        throttle = min(1.0, speed / 20.0)
        
        # Send control command
        response = self.send_command({
            'command': 'control',
            'vehicle_id': vehicle_id,
            'throttle': throttle,
            'steer': steer,
            'brake': 0.0
        })
        
        return {
            'vehicle_id': vehicle_id,
            'our_pos': (our_x, our_y),
            'other_pos': (other_x, other_y),
            'distance': distance,
            'angle_diff': angle_diff,
            'steer': steer,
            'throttle': throttle
        }

    def record_data(self, v0_state: dict, v1_state: dict, distance: float):
        """Record test data point."""
        data_point = {
            'timestamp': time.time() - self.start_time,
            'v0': {
                'position': v0_state.get('position'),
                'speed': v0_state.get('speed'),
                'yaw': v0_state.get('yaw')
            },
            'v1': {
                'position': v1_state.get('position'),
                'speed': v1_state.get('speed'),
                'yaw': v1_state.get('yaw')
            },
            'distance': distance
        }
        self.test_data.append(data_point)

    def run_intersection_test(self, duration: int = 60):
        """
        Run intersection test: both vehicles move toward each other.
        This tests V2V coordination and priority negotiation.
        """
        self.logger.info("=" * 60)
        self.logger.info("=== V2V Intersection Test Started ===")
        self.logger.info("=== Both vehicles will move toward each other ===")
        self.logger.info("=" * 60)
        
        self.start_time = time.time()
        self.running = True
        
        initial_distance = self.get_distance()
        self.logger.info(f"Initial distance: {initial_distance:.1f}m")
        
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while self.running and (time.time() - start_time) < duration:
                # Get states
                response = self.get_states()
                distance = response.get('distance', -1)
                states = response.get('states', {})
                
                # Log every 5 seconds
                current_time = time.time()
                if current_time - last_log_time > 5.0:
                    elapsed = current_time - start_time
                    
                    if 'vehicle_0' in states and 'vehicle_1' in states:
                        v0 = states['vehicle_0']
                        v1 = states['vehicle_1']
                        
                        self.logger.info(f"[{elapsed:.1f}s] Distance: {distance:.1f}m | "
                                       f"V0: {v0.get('speed', 0):.1f}km/h | "
                                       f"V1: {v1.get('speed', 0):.1f}km/h")
                        
                        # Record data
                        self.record_data(v0, v1, distance)
                    
                    last_log_time = current_time
                
                # Move both vehicles toward each other
                self.move_toward_other("vehicle_0", speed=15.0)
                self.move_toward_other("vehicle_1", speed=15.0)
                
                # Tick simulation
                self.tick()
                
                # Check for emergency stop
                if distance < 5.0:
                    self.logger.warning(f"⚠️  Emergency distance reached: {distance:.1f}m")
                    self.logger.warning("Stopping both vehicles...")
                    
                    # Emergency stop
                    self.send_command({
                        'command': 'control',
                        'vehicle_id': 'vehicle_0',
                        'throttle': 0.0,
                        'steer': 0.0,
                        'brake': 1.0
                    })
                    self.send_command({
                        'command': 'control',
                        'vehicle_id': 'vehicle_1',
                        'throttle': 0.0,
                        'steer': 0.0,
                        'brake': 1.0
                    })
                    
                    break
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Test interrupted by user")
        finally:
            self.running = False
            self.save_test_data()
            
            final_distance = self.get_distance()
            self.logger.info("=" * 60)
            self.logger.info("=== Test Complete ===")
            self.logger.info(f"Initial distance: {initial_distance:.1f}m")
            self.logger.info(f"Final distance: {final_distance:.1f}m")
            self.logger.info(f"Data points: {len(self.test_data)}")
            self.logger.info("=" * 60)

    def run_platooning_test(self, duration: int = 60):
        """
        Run platooning test: one vehicle follows the other.
        This tests V2V detection sharing and cooperative planning.
        """
        self.logger.info("=" * 60)
        self.logger.info("=== V2V Platooning Test Started ===")
        self.logger.info("=== Vehicle_1 will follow Vehicle_0 ===")
        self.logger.info("=" * 60)
        
        self.start_time = time.time()
        self.running = True
        
        initial_distance = self.get_distance()
        self.logger.info(f"Initial distance: {initial_distance:.1f}m")
        
        start_time = time.time()
        last_log_time = start_time
        
        try:
            while self.running and (time.time() - start_time) < duration:
                # Get states
                response = self.get_states()
                distance = response.get('distance', -1)
                states = response.get('states', {})
                
                # Log every 5 seconds
                current_time = time.time()
                if current_time - last_log_time > 5.0:
                    elapsed = current_time - start_time
                    
                    if 'vehicle_0' in states and 'vehicle_1' in states:
                        v0 = states['vehicle_0']
                        v1 = states['vehicle_1']
                        
                        self.logger.info(f"[{elapsed:.1f}s] Distance: {distance:.1f}m | "
                                       f"V0: {v0.get('speed', 0):.1f}km/h | "
                                       f"V1: {v1.get('speed', 0):.1f}km/h")
                        
                        # Record data
                        self.record_data(v0, v1, distance)
                    
                    last_log_time = current_time
                
                # Vehicle_0 moves forward, Vehicle_1 follows
                self.move_toward_other("vehicle_0", speed=15.0)
                
                # Vehicle_1 tries to maintain distance
                if distance < 15.0:
                    # Too close, slow down
                    self.send_command({
                        'command': 'control',
                        'vehicle_id': 'vehicle_1',
                        'throttle': 0.0,
                        'steer': 0.0,
                        'brake': 0.5
                    })
                else:
                    self.move_toward_other("vehicle_1", speed=15.0)
                
                # Tick simulation
                self.tick()
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            self.logger.info("Test interrupted by user")
        finally:
            self.running = False
            self.save_test_data()
            
            final_distance = self.get_distance()
            self.logger.info("=" * 60)
            self.logger.info("=== Test Complete ===")
            self.logger.info(f"Initial distance: {initial_distance:.1f}m")
            self.logger.info(f"Final distance: {final_distance:.1f}m")
            self.logger.info(f"Data points: {len(self.test_data)}")
            self.logger.info("=" * 60)

    def save_test_data(self):
        """Save test data to file."""
        if not self.test_data:
            return
        
        output_file = "output/v2v_test_data.json"
        
        test_summary = {
            'test_type': 'intersection',
            'duration': time.time() - self.start_time if self.start_time else 0,
            'data_points': len(self.test_data),
            'initial_distance': self.test_data[0]['distance'] if self.test_data else 0,
            'final_distance': self.test_data[-1]['distance'] if self.test_data else 0,
            'min_distance': min(d['distance'] for d in self.test_data) if self.test_data else 0,
            'max_distance': max(d['distance'] for d in self.test_data) if self.test_data else 0,
            'data': self.test_data
        }
        
        os.makedirs("output", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(test_summary, f, indent=2)
        
        self.logger.info(f"Test data saved to {output_file}")

    def close(self):
        """Close connection."""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Remote vehicle controller for V2V testing')
    parser.add_argument('--host', type=str, default='localhost',
                       help='Spawner host (default: localhost)')
    parser.add_argument('--port', type=int, default=5559,
                       help='Spawner port (default: 5559)')
    parser.add_argument('--test', type=str, choices=['intersection', 'platooning'],
                       default='intersection',
                       help='Test type to run')
    parser.add_argument('--duration', type=int, default=60,
                       help='Test duration in seconds (default: 60)')
    
    args = parser.parse_args()
    
    controller = RemoteVehicleController(host=args.host, port=args.port)
    
    try:
        # Wait for spawner to be ready
        logger = setup_logger("RemoteController", "INFO")
        logger.info(f"Connecting to spawner at {args.host}:{args.port}...")
        
        # Try to connect
        response = controller.send_command({'command': 'get_states'})
        if response.get('status') in ['timeout', 'error']:
            logger.error("Failed to connect to spawner. Make sure vehicle_spawner.py is running.")
            return
        
        logger.info("Connected successfully!")
        
        # Run test
        if args.test == 'intersection':
            controller.run_intersection_test(duration=args.duration)
        elif args.test == 'platooning':
            controller.run_platooning_test(duration=args.duration)
            
    finally:
        controller.close()


if __name__ == '__main__':
    main()
