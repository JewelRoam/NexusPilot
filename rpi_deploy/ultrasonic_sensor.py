"""
Robust Ultrasonic Sensor module with hard timeouts and DistanceReading class.
Fixes the ImportError in dependent modules.
"""
import time
from dataclasses import dataclass

try:
    import RPi.GPIO as GPIO
except ImportError:
    GPIO = None

# Handle direct execution vs package import
if __name__ == "__main__" or __package__ is None:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from rpi_deploy.hardware_config import hardware_config
else:
    from .hardware_config import hardware_config

@dataclass
class DistanceReading:
    """Standardized distance measurement result."""
    distance_cm: float
    valid: bool
    timestamp: float = field(default_factory=time.time)

# To ensure compatibility with code that doesn't use dataclasses yet
# manually defining the class for older python versions if needed, 
# but RPi 5 uses Python 3.11+, so dataclass is fine.
from dataclasses import field

class UltrasonicSensor:
    def __init__(self, trig=None, echo=None):
        self.config = hardware_config.ultrasonic
        self.trig = trig if trig is not None else self.config.trigger_pin
        self.echo = echo if echo is not None else self.config.echo_pin
        
        if GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.trig, GPIO.OUT)
            GPIO.setup(self.echo, GPIO.IN)
            print(f"[Ultrasonic] Initialized on Trig:{self.trig}, Echo:{self.echo}")

    def measure_once(self) -> DistanceReading:
        """Measure distance once with a 30ms hard timeout."""
        if not GPIO:
            return DistanceReading(999.0, False)
        
        GPIO.output(self.trig, False)
        time.sleep(0.01)
        GPIO.output(self.trig, True)
        time.sleep(0.00001)
        GPIO.output(self.trig, False)

        # Timeout logic to prevent loop lockup
        start_wait = time.time()
        pulse_start = time.time()
        
        # Wait for ECHO to go HIGH
        while GPIO.input(self.echo) == 0:
            pulse_start = time.time()
            if pulse_start - start_wait > 0.03: # 30ms timeout
                return DistanceReading(999.0, False)

        # Wait for ECHO to go LOW
        while GPIO.input(self.echo) == 1:
            pulse_end = time.time()
            if pulse_end - pulse_start > 0.03: # 30ms timeout
                break
        else:
            pulse_end = time.time()

        duration = pulse_end - pulse_start
        distance = (duration * 34300) / 2
        
        # Basic sanity check
        is_valid = 2.0 < distance < 400.0
        return DistanceReading(distance if is_valid else 999.0, is_valid)

    def measure_average(self, count=5) -> DistanceReading:
        """Median filtered measurement."""
        readings = []
        for _ in range(count):
            r = self.measure_once()
            if r.valid: readings.append(r.distance_cm)
            time.sleep(0.01)
        
        if not readings: return DistanceReading(999.0, False)
        return DistanceReading(float(np.median(readings)), True)

    def cleanup(self):
        print("[Ultrasonic] Cleanup done")

if __name__ == "__main__":
    import numpy as np # Needed for median
    print("Testing Ultrasonic Sensor (Direct Run)...")
    sensor = UltrasonicSensor()
    try:
        while True:
            res = sensor.measure_once()
            print(f"Distance: {res.distance_cm:.1f} cm {'[VALID]' if res.valid else '[INVALID]'}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        sensor.cleanup()
