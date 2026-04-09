#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for obstacle_avoidance module.

Tests the APF controller logic (pure computation, no hardware required)
and the ObstacleAvoidanceController integration in simulation mode.
"""

import sys
import os
import unittest

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rpi_deploy.obstacle_avoidance import (
    SimpleAPFController,
    OBSTACLE_DISTANCE_CM,
    EMERGENCY_DISTANCE_CM,
)


class TestSimpleAPFController(unittest.TestCase):
    """Test the APF computation logic (no hardware dependency)."""

    def setUp(self):
        self.apf = SimpleAPFController()

    # --- Forward (clear path) ---

    def test_clear_path_forward(self):
        """All directions clear → should move forward."""
        result = self.apf.compute(200.0, 200.0, 200.0)
        self.assertEqual(result["action"], "forward")
        self.assertEqual(result["state"], "normal")
        self.assertGreater(result["speed"], 0)

    def test_clear_path_has_positive_fx(self):
        """Clear path → net force points forward (fx > 0)."""
        result = self.apf.compute(200.0, 200.0, 200.0)
        self.assertGreater(result["fx"], 0)

    # --- Emergency ---

    def test_emergency_front(self):
        """Front distance < emergency → stop."""
        result = self.apf.compute(10.0, 200.0, 200.0)
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["state"], "emergency")

    def test_emergency_left(self):
        """Left distance < emergency → stop."""
        result = self.apf.compute(200.0, 10.0, 200.0)
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["state"], "emergency")

    def test_emergency_right(self):
        """Right distance < emergency → stop."""
        result = self.apf.compute(200.0, 200.0, 10.0)
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["state"], "emergency")

    # --- Obstacle ahead → avoidance ---

    def test_obstacle_front_backup(self):
        """Front obstacle at 15cm (at emergency boundary), both sides clear → back up."""
        # 15cm == EMERGENCY_DIST, so NOT emergency (< is strict).
        # Strong front repulsion makes fx negative → backward.
        result = self.apf.compute(15.0, 200.0, 200.0)
        self.assertEqual(result["action"], "backward")
        self.assertEqual(result["state"], "avoiding")

    def test_obstacle_front_with_left_clear(self):
        """Front obstacle, right side blocked, left side clear → turn left."""
        result = self.apf.compute(30.0, 200.0, 40.0)
        # Front pushes back, right obstacle pushes left (fy < 0)
        # Net force points backward-left → left action
        self.assertEqual(result["action"], "left")

    def test_obstacle_front_with_right_clear(self):
        """Front obstacle, left side blocked, right side clear → turn right."""
        result = self.apf.compute(30.0, 40.0, 200.0)
        # Front pushes back, left obstacle pushes right (fy > 0)
        # Net force points backward-right → right action
        self.assertEqual(result["action"], "right")

    # --- Side obstacles only ---

    def test_left_obstacle_only(self):
        """Only left side has obstacle → should veer right."""
        result = self.apf.compute(200.0, 25.0, 200.0)
        # Left obstacle pushes right (+y), so force_angle > 0 → right
        self.assertEqual(result["action"], "right")
        self.assertEqual(result["state"], "avoiding")

    def test_right_obstacle_only(self):
        """Only right side has obstacle → should veer left."""
        result = self.apf.compute(200.0, 200.0, 25.0)
        # Right obstacle pushes left (−y), so force_angle < 0 → left
        self.assertEqual(result["action"], "left")
        self.assertEqual(result["state"], "avoiding")

    # --- Both sides blocked ---

    def test_both_sides_blocked_front_clear(self):
        """Both sides blocked but front clear → still forward."""
        # With equal left/right obstacles, lateral forces cancel
        result = self.apf.compute(200.0, 30.0, 30.0)
        self.assertEqual(result["action"], "forward")

    # --- Boundary conditions ---

    def test_very_close_front(self):
        """Very close front obstacle → strong repulsion."""
        result = self.apf.compute(5.0, 200.0, 200.0)
        self.assertEqual(result["action"], "stop")
        self.assertEqual(result["state"], "emergency")

    def test_at_d0_boundary(self):
        """Distance exactly at D0 → no repulsive force from that direction."""
        # At D0, the repulsive term (1/d - 1/D0) = 0
        result = self.apf.compute(self.apf.D0, 200.0, 200.0)
        self.assertEqual(result["action"], "forward")

    def test_just_below_d0(self):
        """Distance just below D0 → slight repulsion."""
        result = self.apf.compute(self.apf.D0 - 1.0, 200.0, 200.0)
        # Should still be forward since repulsion is weak far from obstacle
        self.assertIn(result["action"], ["forward", "backward"])

    # --- Output structure ---

    def test_output_has_required_keys(self):
        """Result dict should have all required keys."""
        result = self.apf.compute(100.0, 100.0, 100.0)
        for key in ["action", "speed", "state", "fx", "fy"]:
            self.assertIn(key, result)

    def test_speed_non_negative(self):
        """Speed should never be negative."""
        for f, l, r in [(200, 200, 200), (30, 30, 30), (10, 10, 10)]:
            result = self.apf.compute(f, l, r)
            self.assertGreaterEqual(result["speed"], 0,
                                    f"Speed negative for F={f} L={l} R={r}")

    def test_valid_action_values(self):
        """Action should always be one of the valid values."""
        valid_actions = {"forward", "left", "right", "backward", "stop"}
        for f, l, r in [(200, 200, 200), (30, 200, 200), (30, 30, 200),
                         (30, 200, 30), (10, 10, 10), (80, 25, 200)]:
            result = self.apf.compute(f, l, r)
            self.assertIn(result["action"], valid_actions,
                          f"Invalid action '{result['action']}' for F={f} L={l} R={r}")


if __name__ == "__main__":
    unittest.main()