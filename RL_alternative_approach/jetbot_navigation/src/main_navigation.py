import cv2
import numpy as np
import time
from jetbot import Camera
import threading
import queue
from typing import Optional, Tuple, List

from .environment import RealWorldEnvironment
from .pathfinding import AStarPathfinder
from .target_detection import TargetDetector
from .obstacle_avoidance import ObstacleDetector
from .lane_following import LaneFollower
from .jetbot_controller import JetBotController


class NavigationSystem:
    """
    Main navigation system integrating all components
    """

    def __init__(self, target_description: str = "red object"):
        """
        Initialize complete navigation system

        Args:
            target_description: Description of target to find
        """
        # Initialize components
        self.camera = Camera.instance(width=224, height=224)
        self.environment = RealWorldEnvironment()
        self.pathfinder = AStarPathfinder(self.environment)
        self.target_detector = TargetDetector()
        self.obstacle_detector = ObstacleDetector()
        self.lane_follower = LaneFollower()
        self.robot_controller = JetBotController()

        # Set target
        self.target_detector.set_target_description(target_description)

        # Navigation state
        self.current_target_position = None
        self.current_path = []
        self.target_found = False
        self.navigation_active = False

        # Image processing
        self.latest_image = None
        self.image_lock = threading.Lock()

        # Camera callback
        self.camera.observe(self._camera_callback, names="value")

        print(f"Navigation system initialized. Looking for: {target_description}")

    def _camera_callback(self, change):
        """Camera image callback"""
        with self.image_lock:
            self.latest_image = change["new"]

    def start_navigation(self):
        """Start autonomous navigation"""
        self.navigation_active = True
        self.robot_controller.resume()

        print("Starting autonomous navigation...")

        try:
            while self.navigation_active:
                self._navigation_step()
                time.sleep(0.1)  # 10Hz update rate

        except KeyboardInterrupt:
            print("\nNavigation interrupted by user")
        except Exception as e:
            print(f"Navigation error: {e}")
        finally:
            self.stop_navigation()

    def stop_navigation(self):
        """Stop navigation and cleanup"""
        self.navigation_active = False
        self.robot_controller.stop()
        print("Navigation stopped")

    def _navigation_step(self):
        """Single navigation step"""
        # Get current image
        with self.image_lock:
            if self.latest_image is None:
                return
            current_image = self.latest_image.copy()

        # Update environment with obstacles
        self._update_environment(current_image)

        # Check for emergency obstacles
        if not self.obstacle_detector.is_path_clear(current_image):
            print("Emergency stop: Obstacle detected!")
            self.robot_controller.stop()
            time.sleep(1.0)
            self.robot_controller.resume()
            return

        # Detect target
        target_bbox = self.target_detector.detect_target(current_image)

        if target_bbox:
            # Target found - update target position
            self.target_found = True
            world_pos = self.target_detector.get_target_center_world_position(
                target_bbox
            )

            # Convert to absolute world coordinates
            robot_pos = self.robot_controller.get_position()
            robot_heading = self.robot_controller.get_heading()

            # Transform relative position to absolute
            cos_h = np.cos(robot_heading)
            sin_h = np.sin(robot_heading)
            abs_x = robot_pos[0] + world_pos[0] * cos_h - world_pos[1] * sin_h
            abs_y = robot_pos[1] + world_pos[0] * sin_h + world_pos[1] * cos_h

            self.current_target_position = (abs_x, abs_y)

            print(f"Target detected at: ({abs_x:.2f}, {abs_y:.2f})")

            # Plan new path to target
            self._plan_path_to_target()

        # Execute current path
        if self.current_path:
            self._follow_current_path(current_image)
        else:
            # No path - search for target
            self._search_for_target()

    def _update_environment(self, image: np.ndarray):
        """Update environment map with detected obstacles"""
        # For RGB-only obstacle detection
        obstacles = self.obstacle_detector.detect_obstacles_from_rgb(image)

        robot_pos = self.robot_controller.get_position()
        robot_heading = self.robot_controller.get_heading()

        # Add detected obstacles to environment
        for bbox in obstacles:
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2

            # Estimate distance (simplified)
            distance = 1.0  # Default distance estimate

            # Convert to world coordinates
            world_pos = self.obstacle_detector._pixel_to_world_position(
                center_x, center_y, distance
            )

            # Transform to absolute coordinates
            cos_h = np.cos(robot_heading)
            sin_h = np.sin(robot_heading)
            abs_x = robot_pos[0] + world_pos[0] * cos_h - world_pos[1] * sin_h
            abs_y = robot_pos[1] + world_pos[0] * sin_h + world_pos[1] * cos_h

            self.environment.add_obstacle_at_position(abs_x, abs_y)

    def _plan_path_to_target(self):
        """Plan path from current position to target"""
        if not self.current_target_position:
            return

        robot_pos = self.robot_controller.get_position()

        # Convert positions to grid coordinates
        start_grid = self.environment.world_to_grid(robot_pos[0], robot_pos[1])
        target_grid = self.environment.world_to_grid(
            self.current_target_position[0], self.current_target_position[1]
        )

        # Find path
        grid_path = self.pathfinder.find_path(start_grid, target_grid)

        if grid_path:
            # Convert to world coordinates and smooth
            self.current_path = self.pathfinder.smooth_path(grid_path)
            print(f"Path planned with {len(self.current_path)} waypoints")
        else:
            print("No path found to target")
            self.current_path = []

    def _follow_current_path(self, image: np.ndarray):
        """Follow the current planned path"""
        if not self.current_path:
            return

        robot_pos = self.robot_controller.get_position()
        robot_heading = self.robot_controller.get_heading()

        # Check if we've reached the target
        target_distance = np.sqrt(
            (self.current_target_position[0] - robot_pos[0]) ** 2
            + (self.current_target_position[1] - robot_pos[1]) ** 2
        )

        if target_distance < 0.3:  # 30cm tolerance
            print("Target reached!")
            self.robot_controller.stop()
            self.current_path = []
            return

        # Use lane following for path tracking
        steering, speed = self.lane_follower.follow_planned_path(
            robot_pos, self.current_path, robot_heading
        )

        # Convert to robot commands
        linear_speed = speed
        angular_speed = steering * 2.0  # Scale steering to angular velocity

        self.robot_controller.set_speeds(linear_speed, angular_speed)

        print(f"Following path: speed={linear_speed:.2f}, steering={angular_speed:.2f}")

    def _search_for_target(self):
        """Search behavior when no target is found"""
        print("Searching for target...")

        # Simple search pattern - rotate slowly
        self.robot_controller.set_speeds(0.0, 0.5)  # Turn in place

    def set_target_description(self, description: str):
        """Update target description"""
        self.target_detector.set_target_description(description)
        print(f"Target updated to: {description}")

    def get_status(self) -> dict:
        """Get current navigation status"""
        robot_pos = self.robot_controller.get_position()
        robot_heading = self.robot_controller.get_heading()

        return {
            "position": robot_pos,
            "heading": robot_heading * 180 / np.pi,  # Convert to degrees
            "target_found": self.target_found,
            "target_position": self.current_target_position,
            "path_length": len(self.current_path),
            "navigation_active": self.navigation_active,
        }


def main():
    """Main function for standalone execution"""
    print("JetBot Navigation System")
    print("========================")

    # Get target description from user
    target_description = input(
        "Enter target description (e.g., 'red object', 'person', 'blue ball'): "
    )

    if not target_description.strip():
        target_description = "red object"

    # Create and start navigation system
    nav_system = NavigationSystem(target_description)

    try:
        nav_system.start_navigation()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        nav_system.stop_navigation()


if __name__ == "__main__":
    main()
