"""
Automated Test Runner - Main entry point for testing framework.
Provides a unified interface to:
  - Run batch tests across multiple scenarios
  - Compare different configurations
  - Auto-tune parameters
  - Generate comprehensive reports

Usage:
  # Quick test suite
  python -m testing.automated_runner --suite quick

  # Parameter tuning
  python -m testing.automated_runner --tune --iterations 20

  # Compare two configs
  python -m testing.automated_runner --compare config1.yaml config2.yaml
"""
import os
import sys
import yaml
import time
import argparse
from typing import List, Dict, Optional
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.carla_env import CarlaEnv
from testing.scenario_generator import ScenarioGenerator, ScenarioConfig
from testing.metrics_collector import MetricsCollector, FrameMetrics, TestRunSummary
from testing.parameter_tuner import APFParameterTuner, ParameterConfig
from testing.report_generator import ReportGenerator
from perception.detector import YOLODetector
from perception.depth_estimator import DepthEstimator
from planning.apf_planner import APFPlanner
from control.vehicle_controller import VehicleController


class AutomatedTestRunner:
    """
    Main orchestrator for automated testing.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.env: Optional[CarlaEnv] = None
        self.metrics = MetricsCollector()
        self.results: List[TestRunSummary] = []

    def connect(self):
        """Connect to CARLA server."""
        print("[Runner] Connecting to CARLA...")
        self.env = CarlaEnv(self.config)
        self.env.connect()
        return self

    def disconnect(self):
        """Disconnect and cleanup."""
        if self.env:
            self.env.cleanup()
            self.env = None

    def run_scenario(self, scenario: ScenarioConfig,
                     planner_config: Optional[Dict] = None,
                     duration_seconds: float = 60.0) -> TestRunSummary:
        """
        Run a single scenario and collect metrics.

        Args:
            scenario: Scenario configuration
            planner_config: Optional override for APF parameters
            duration_seconds: How long to run the scenario

        Returns:
            TestRunSummary with performance metrics
        """
        print(f"\n[Runner] Running scenario: {scenario.name}")
        print(f"  Description: {scenario.description}")

        # Initialize modules
        detector = YOLODetector(self.config['perception'])
        depth_estimator = DepthEstimator(self.config['depth'], self.config['sensors']['rgb_camera'])

        planner_cfg = planner_config or self.config['planning']['apf']
        planner = APFPlanner(planner_cfg)
        controller = VehicleController(self.config['control'], platform="carla")

        # Spawn ego vehicle at scenario spawn point
        ego_spawn = scenario.spawn_points[0]
        ego = self.env.spawn_vehicle(
            "ego",
            blueprint_filter=self.config.get('ego_vehicle', {}).get('blueprint_filter'),
            spawn_index=None  # We'll set position manually
        )

        # Set initial transform
        import carla
        transform = carla.Transform(
            carla.Location(x=ego_spawn[0], y=ego_spawn[1], z=0.5),
            carla.Rotation(yaw=ego_spawn[2])
        )
        ego.actor.set_transform(transform)

        # Attach sensors
        ego.attach_rgb_camera(self.config['sensors'].get('rgb_camera', {}))
        ego.attach_depth_camera(self.config['sensors'].get('depth_camera', {}))

        # Generate route
        self.env.generate_route(ego, sampling_resolution=2.0)

        # Spawn NPCs based on scenario
        if scenario.npc_vehicles > 0 or scenario.pedestrians > 0:
            self.env.spawn_traffic(
                num_vehicles=scenario.npc_vehicles,
                num_walkers=scenario.pedestrians
            )

        # Wait for sensors
        for _ in range(10):
            self.env.tick()
            time.sleep(0.05)

        # Start recording metrics
        self.metrics.start_recording()
        start_time = time.time()

        try:
            while time.time() - start_time < duration_seconds:
                # Tick simulation
                self.env.tick()

                # Get sensor data
                rgb_image = ego.get_rgb_image()
                depth_data = ego.get_depth_data()

                if rgb_image is None:
                    continue

                if depth_data is not None:
                    depth_estimator.update_depth_image(depth_data)

                # Perception
                detections = detector.detect(rgb_image)
                obstacles_det = detector.get_obstacles(detections)
                depth_estimator.enrich_detections(obstacles_det, ego.get_transform())

                # Ego state
                ego_x, ego_y, _ = ego.get_location()
                ego_yaw = ego.get_yaw()
                ego_speed = ego.get_speed_kmh()

                # Advance waypoint
                ego.advance_waypoint(threshold=6.0)
                goal = ego.get_next_waypoint()

                frame_metrics = FrameMetrics(
                    timestamp=time.time() - start_time,
                    frame_id=0,  # Will be set by collector
                    ego_x=ego_x,
                    ego_y=ego_y,
                    ego_yaw=ego_yaw,
                    ego_speed=ego_speed,
                    num_detections=len(detections),
                    num_obstacles=len(obstacles_det),
                )

                if goal:
                    goal_x, goal_y = goal

                    # Planning
                    apf_obstacles = planner.detections_to_obstacles(
                        obstacles_det, ego_x, ego_y, ego_yaw
                    )
                    planner_output = planner.compute(
                        ego_x, ego_y, ego_yaw, ego_speed,
                        goal_x, goal_y, apf_obstacles
                    )

                    # Control
                    control_dict = controller.compute_control(planner_output, ego_speed)
                    controller.apply_carla_control(ego.actor, control_dict)

                    # Update frame metrics
                    frame_metrics.target_speed = planner_output.target_speed
                    frame_metrics.planner_status = planner_output.status
                    frame_metrics.nearest_obstacle_dist = planner_output.nearest_obstacle_dist
                    frame_metrics.throttle = control_dict.get('throttle', 0)
                    frame_metrics.brake = control_dict.get('brake', 0)
                    frame_metrics.steer = control_dict.get('steer', 0)

                # Record frame
                self.metrics.record_frame(frame_metrics)

        except KeyboardInterrupt:
            print("[Runner] Interrupted by user")
        finally:
            # Cleanup scenario-specific actors
            ego.destroy()

        summary = self.metrics.stop_recording()
        summary.scenario_name = scenario.name
        return summary

    def run_test_suite(self, suite_name: str = "quick",
                       planner_config: Optional[Dict] = None) -> List[TestRunSummary]:
        """
        Run a complete test suite.

        Args:
            suite_name: "quick", "standard", or "full"
            planner_config: Optional APF parameter override

        Returns:
            List of TestRunSummary for each scenario
        """
        print(f"\n{'='*60}")
        print(f"Running Test Suite: {suite_name.upper()}")
        print(f"{'='*60}")

        # Generate scenarios
        gen = ScenarioGenerator(self.env)
        scenarios = gen.generate_test_suite(suite_name)

        results = []
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n[{i}/{len(scenarios)}] ", end="")
            result = self.run_scenario(
                scenario,
                planner_config=planner_config,
                duration_seconds=scenario.duration_seconds
            )
            results.append(result)

            status = "[PASS]" if result.passed else "[FAIL]"
            print(f"Result: {status} | Collisions: {result.collision_count} | "
                  f"Avg Speed: {result.average_speed:.1f} km/h")

        self.results.extend(results)
        return results

    def tune_parameters(self, iterations: int = 20) -> ParameterConfig:
        """
        Auto-tune APF parameters using random search.

        Args:
            iterations: Number of random configurations to try

        Returns:
            Best ParameterConfig found
        """
        from testing.parameter_tuner import quick_tune_apf

        print(f"\n{'='*60}")
        print(f"Starting Parameter Tuning ({iterations} iterations)")
        print(f"{'='*60}")

        best_config = quick_tune_apf(self.env, num_scenarios=5, num_configs=iterations)
        return best_config

    def generate_report(self, output_dir: str = "output"):
        """Generate HTML report of all results."""
        os.makedirs(output_dir, exist_ok=True)

        gen = ReportGenerator(title="Autonomous Driving Test Results")

        for i, result in enumerate(self.results):
            gen.add_run(result, f"Run {i+1}: {result.scenario_name}")

        html_path = os.path.join(output_dir, "test_report.html")
        gen.generate_html(html_path)

        md_path = os.path.join(output_dir, "test_report.md")
        gen.generate_markdown(md_path)

        print(f"\n[Runner] Reports generated:")
        print(f"  HTML: {html_path}")
        print(f"  Markdown: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Automated Testing Framework")
    parser.add_argument("--suite", choices=["quick", "standard", "full"],
                       default="quick", help="Test suite to run")
    parser.add_argument("--tune", action="store_true",
                       help="Run parameter tuning")
    parser.add_argument("--iterations", type=int, default=20,
                       help="Number of tuning iterations")
    parser.add_argument("--report", action="store_true", default=True,
                       help="Generate HTML report")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for reports")
    parser.add_argument("--config", default="config/config.yaml",
                       help="Path to config file")

    args = parser.parse_args()

    # Create runner
    runner = AutomatedTestRunner(args.config)

    try:
        runner.connect()

        if args.tune:
            # Run parameter tuning
            best_config = runner.tune_parameters(iterations=args.iterations)
            print("\n" + "="*60)
            print("Best Configuration Found:")
            print("="*60)
            print(f"  k_attractive: {best_config.k_attractive}")
            print(f"  k_repulsive: {best_config.k_repulsive}")
            print(f"  d0: {best_config.d0}")
            print(f"  max_speed: {best_config.max_speed}")

            # Run full test suite with best config
            print("\nRunning full test suite with optimized parameters...")
            runner.run_test_suite(args.suite, planner_config=best_config.to_planner_config())
        else:
            # Run standard test suite
            runner.run_test_suite(args.suite)

        # Generate report
        if args.report:
            runner.generate_report(args.output_dir)

    except Exception as e:
        print(f"[Runner] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        runner.disconnect()


if __name__ == '__main__':
    main()
