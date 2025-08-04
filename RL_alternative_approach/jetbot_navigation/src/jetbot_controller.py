import time
import numpy as np
from jetbot import Robot
from typing import Tuple, Optional
import threading
import queue


class JetBotController:
    """
    Low-level controller for JetBot movement
    Handles motor control, odometry, and safety
    """

    def __init__(self, wheel_diameter=0.065, wheel_base=0.11, max_speed=0.5):
        """
        Initialize JetBot controller

        Args:
            wheel_diameter: Wheel diameter in meters
            wheel_base: Distance between wheels in meters
            max_speed: Maximum speed in m/s
        """
        self.robot = Robot()

        # Physical parameters
        self.wheel_diameter = wheel_diameter
        self.wheel_base = wheel_base
        self.max_speed = max_speed

        # Odometry
        self.position = (0.0, 0.0)  # (x, y) in meters
        self.heading = 0.0  # heading in radians
        self.last_time = time.time()

        # Safety
        self.emergency_stop = False
        self.max_turn_rate = 2.0  # rad/s

        # Command queue for smooth operation
        self.command_queue = queue.Queue(maxsize=10)
        self.controller_thread = threading.Thread(target=self._controller_loop)
        self.controller_thread.daemon = True
        self.running = True
        self.controller_thread.start()

    def set_speeds(self, linear_speed: float, angular_speed: float):
        """
        Set robot speeds

        Args:
            linear_speed: Forward speed in m/s
            angular_speed: Turning speed in rad/s
        """
        if self.emergency_stop:
            linear_speed = 0.0
            angular_speed = 0.0

        # Limit speeds
        linear_speed = np.clip(linear_speed, -self.max_speed, self.max_speed)
        angular_speed = np.clip(angular_speed, -self.max_turn_rate, self.max_turn_rate)

        try:
            self.command_queue.put((linear_speed, angular_speed), timeout=0.01)
        except queue.Full:
            pass  # Skip if queue is full

    def move_to_position(
        self, target_x: float, target_y: float, timeout: float = 10.0
    ) -> bool:
        """
        Move to specific position using simple point-turn navigation

        Args:
            target_x: Target x coordinate in meters
            target_y: Target y coordinate in meters
            timeout: Maximum time to spend trying to reach target

        Returns:
            True if target reached, False if timeout
        """
        start_time = time.time()
        tolerance = 0.1  # meters

        while time.time() - start_time < timeout:
            if self.emergency_stop:
                self.stop()
                return False

            # Calculate distance and angle to target
            dx = target_x - self.position[0]
            dy = target_y - self.position[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance < tolerance:
                self.stop()
                return True

            # Calculate desired heading
            desired_heading = np.arctan2(dy, dx)
            heading_error = desired_heading - self.heading

            # Normalize heading error
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi

            # Control logic
            if abs(heading_error) > 0.2:  # Turn in place if large heading error
                angular_speed = 0.5 * np.sign(heading_error)
                linear_speed = 0.0
            else:  # Move forward while adjusting heading
                linear_speed = min(0.3, distance)  # Slow down when close
                angular_speed = 0.3 * heading_error

            self.set_speeds(linear_speed, angular_speed)
            time.sleep(0.05)

        self.stop()
        return False

    def follow_path(self, path_points: list, speed: float = 0.3) -> bool:
        """
        Follow a path of waypoints

        Args:
            path_points: List of (x, y) waypoints
            speed: Desired speed in m/s

        Returns:
            True if path completed successfully
        """
        for point in path_points:
            if not self.move_to_position(point[0], point[1]):
                return False

            if self.emergency_stop:
                return False

        return True

    def stop(self):
        """Emergency stop"""
        self.robot.stop()
        self.emergency_stop = True

    def resume(self):
        """Resume operation after stop"""
        self.emergency_stop = False

    def get_position(self) -> Tuple[float, float]:
        """Get current position estimate"""
        return self.position

    def get_heading(self) -> float:
        """Get current heading estimate"""
        return self.heading

    def _controller_loop(self):
        """Main controller loop running in separate thread"""
        while self.running:
            try:
                # Get latest command
                linear_speed, angular_speed = self.command_queue.get(timeout=0.1)

                if not self.emergency_stop:
                    # Convert to wheel speeds
                    left_speed, right_speed = self._diff_drive_kinematics(
                        linear_speed, angular_speed
                    )

                    # Set motor values (-1 to 1)
                    left_motor = np.clip(left_speed / self.max_speed, -1.0, 1.0)
                    right_motor = np.clip(right_speed / self.max_speed, -1.0, 1.0)

                    self.robot.set_motors(left_motor, right_motor)

                    # Update odometry
                    self._update_odometry(linear_speed, angular_speed)

            except queue.Empty:
                # No new commands, keep last command or stop
                if self.emergency_stop:
                    self.robot.stop()

    def _diff_drive_kinematics(
        self, linear_speed: float, angular_speed: float
    ) -> Tuple[float, float]:
        """
        Convert linear and angular speeds to wheel speeds

        Returns:
            (left_wheel_speed, right_wheel_speed) in m/s
        """
        left_speed = linear_speed - (angular_speed * self.wheel_base / 2)
        right_speed = linear_speed + (angular_speed * self.wheel_base / 2)

        return (left_speed, right_speed)

    def _update_odometry(self, linear_speed: float, angular_speed: float):
        """Update position estimate based on commanded speeds"""
        current_time = time.time()
        dt = current_time - self.last_time

        if dt > 0:
            # Update heading
            self.heading += angular_speed * dt

            # Normalize heading
            while self.heading > np.pi:
                self.heading -= 2 * np.pi
            while self.heading < -np.pi:
                self.heading += 2 * np.pi

            # Update position
            dx = linear_speed * np.cos(self.heading) * dt
            dy = linear_speed * np.sin(self.heading) * dt

            self.position = (self.position[0] + dx, self.position[1] + dy)

        self.last_time = current_time

    def shutdown(self):
        """Shutdown controller"""
        self.running = False
        self.robot.stop()
        if self.controller_thread.is_alive():
            self.controller_thread.join()
