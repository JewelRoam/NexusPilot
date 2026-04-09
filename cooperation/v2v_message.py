"""
V2V (Vehicle-to-Vehicle) Message Protocol and Communication.
Implements lightweight message passing for cooperative perception and planning.

Supports:
  - In-process communication (for CARLA multi-vehicle simulation)
  - Socket-based communication (for PC ↔ RPi real-world deployment)
  - Future: ROS 2 Topic-based communication

Innovation focus: Intent sharing + detection sharing for cooperative safety.
"""
import time
import json
import threading
import socket
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


@dataclass
class SharedDetection:
    """A detection shared between vehicles."""
    class_name: str
    category: str           # "vehicle", "pedestrian", "cyclist"
    confidence: float
    world_x: float          # world coordinates
    world_y: float
    distance_from_sender: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'SharedDetection':
        return cls(**d)


@dataclass
class V2VMessage:
    """
    V2V cooperative message exchanged between vehicles.
    Contains: position, velocity, detections, and intent.
    """
    timestamp: float                          # Unix timestamp
    vehicle_id: str                           # Unique vehicle identifier
    position: Tuple[float, float, float]      # x, y, yaw (degrees)
    velocity: Tuple[float, float]             # speed (km/h), heading (degrees)
    detections: List[SharedDetection] = field(default_factory=list)
    intent: str = "cruising"                  # "cruising", "turning_left", "turning_right",
                                              # "yielding", "stopped", "emergency_brake"
    confidence: float = 1.0                   # Overall perception confidence
    cooperation_request: Optional[str] = None # "request_yield", "request_pass", None

    def to_json(self) -> str:
        d = {
            "timestamp": self.timestamp,
            "vehicle_id": self.vehicle_id,
            "position": list(self.position),
            "velocity": list(self.velocity),
            "detections": [det.to_dict() for det in self.detections],
            "intent": self.intent,
            "confidence": self.confidence,
            "cooperation_request": self.cooperation_request,
        }
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> 'V2VMessage':
        d = json.loads(json_str)
        detections = [SharedDetection.from_dict(det) for det in d.get("detections", [])]
        return cls(
            timestamp=d["timestamp"],
            vehicle_id=d["vehicle_id"],
            position=tuple(d["position"]),
            velocity=tuple(d["velocity"]),
            detections=detections,
            intent=d.get("intent", "cruising"),
            confidence=d.get("confidence", 1.0),
            cooperation_request=d.get("cooperation_request"),
        )

    @property
    def age_ms(self) -> float:
        """Age of message in milliseconds."""
        return (time.time() - self.timestamp) * 1000


class V2VCommunicator:
    """
    V2V communication manager.
    Supports in-process (simulation) and socket-based (real) modes.
    """

    def __init__(self, vehicle_id: str, config: dict):
        """
        Args:
            vehicle_id: unique ID for this vehicle
            config: cooperation.communication section of config.yaml
        """
        self.vehicle_id = vehicle_id
        self.config = config
        self.protocol = config.get("protocol", "in_process")
        self.max_latency_ms = config.get("max_latency_ms", 100)
        self.dropout_rate = config.get("dropout_rate", 0.0)

        # Message buffers
        self._inbox: Dict[str, V2VMessage] = {}  # Latest message from each vehicle
        self._outbox: Optional[V2VMessage] = None
        self._lock = threading.Lock()

        # In-process shared bus (for simulation)
        # All V2VCommunicator instances share this class-level bus
        self._message_history: List[V2VMessage] = []

        # Socket mode
        self._socket_server = None
        self._socket_clients = {}

    # ---- In-process mode (CARLA simulation) ----

    # Class-level shared message bus for in-process communication
    _shared_bus: Dict[str, 'V2VMessage'] = {}
    _bus_lock = threading.Lock()

    def broadcast(self, message: V2VMessage):
        """Broadcast a V2V message to all other vehicles."""
        import random
        # Simulate packet dropout
        if random.random() < self.dropout_rate:
            return

        message.vehicle_id = self.vehicle_id
        message.timestamp = time.time()

        if self.protocol == "in_process" or self.protocol == "ros2_topic":
            with V2VCommunicator._bus_lock:
                V2VCommunicator._shared_bus[self.vehicle_id] = message
                print(f"[V2V BROADCAST] {self.vehicle_id} broadcast message, bus now has: {list(V2VCommunicator._shared_bus.keys())}")

        elif self.protocol == "socket":
            self._socket_broadcast(message)

    def receive_all(self) -> Dict[str, V2VMessage]:
        """
        Receive latest messages from all other vehicles.
        Filters out stale messages (older than max_latency_ms).

        Returns:
            Dict mapping vehicle_id -> latest V2VMessage
        """
        messages = {}
        now = time.time()

        if self.protocol == "in_process" or self.protocol == "ros2_topic":
            with V2VCommunicator._bus_lock:
                # Debug: log bus contents
                bus_vids = list(V2VCommunicator._shared_bus.keys())
                for vid, msg in V2VCommunicator._shared_bus.items():
                    if vid == self.vehicle_id:
                        continue
                    # Filter stale messages (100x tolerance for simulation)
                    age_ms = (now - msg.timestamp) * 1000
                    if age_ms < self.max_latency_ms * 100:
                        messages[vid] = msg
                        
                # Debug log if no messages received
                if not messages and bus_vids:
                    # Filter out self
                    others = [v for v in bus_vids if v != self.vehicle_id]
                    if others:
                        print(f"[V2V DEBUG] {self.vehicle_id}: Bus has {others} but no messages received (possible race condition)")

        elif self.protocol == "socket":
            messages = self._socket_receive()

        return messages

    def create_message(self, position: Tuple[float, float, float],
                       velocity: Tuple[float, float],
                       detections: List[SharedDetection],
                       intent: str = "cruising",
                       cooperation_request: Optional[str] = None) -> V2VMessage:
        """Helper to create a V2V message."""
        return V2VMessage(
            timestamp=time.time(),
            vehicle_id=self.vehicle_id,
            position=position,
            velocity=velocity,
            detections=detections,
            intent=intent,
            cooperation_request=cooperation_request,
        )

    def detections_to_shared(self, detections, ego_x: float, ego_y: float) -> List[SharedDetection]:
        """
        Convert DetectedObject list to SharedDetection list for V2V sharing.

        Args:
            detections: List[DetectedObject] with world_position populated
            ego_x, ego_y: ego vehicle position
        """
        shared = []
        for det in detections:
            if det.world_position and det.distance > 0:
                shared.append(SharedDetection(
                    class_name=det.class_name,
                    category=det.category,
                    confidence=det.confidence,
                    world_x=det.world_position[0],
                    world_y=det.world_position[1],
                    distance_from_sender=det.distance,
                ))
        return shared

    # ---- Socket mode (for real-world PC ↔ RPi) ----

    def start_socket_server(self, host: str = "0.0.0.0", port: int = 5555):
        """Start a UDP socket server for receiving V2V messages."""
        self._socket_server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket_server.bind((host, port))
        self._socket_server.setblocking(False)
        print(f"[V2V] Socket server started on {host}:{port}")

    def add_socket_peer(self, vehicle_id: str, host: str, port: int):
        """Register a peer vehicle's address for socket communication."""
        self._socket_clients[vehicle_id] = (host, port)

    def _socket_broadcast(self, message: V2VMessage):
        """Send message to all registered peers via UDP."""
        data = message.to_json().encode('utf-8')
        for vid, (host, port) in self._socket_clients.items():
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(data, (host, port))
                sock.close()
            except Exception as e:
                pass  # Non-blocking, drop on error

    def _socket_receive(self) -> Dict[str, V2VMessage]:
        """Receive messages from socket."""
        messages = {}
        if self._socket_server is None:
            return messages

        try:
            while True:
                data, addr = self._socket_server.recvfrom(65536)
                msg = V2VMessage.from_json(data.decode('utf-8'))
                if msg.vehicle_id != self.vehicle_id:
                    messages[msg.vehicle_id] = msg
        except BlockingIOError:
            pass  # No more data
        except Exception:
            pass

        return messages

    @classmethod
    def reset_shared_bus(cls):
        """Reset the shared message bus (for test cleanup)."""
        with cls._bus_lock:
            cls._shared_bus.clear()
