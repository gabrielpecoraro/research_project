import cv2
import numpy as np
from typing import Tuple, Optional


class LaneFollower:
    """
    Lane following for path tracking
    Combines visual lane detection with planned path following
    """

    def __init__(self, camera_width=224, camera_height=224):
        self.camera_width = camera_width
        self.camera_height = camera_height

        # PID controller parameters
        self.kp = 0.3
        self.ki = 0.0
        self.kd = 0.1

        # PID state
        self.prev_error = 0
        self.integral = 0

        # Path following parameters
        self.lookahead_distance = 0.5  # meters
        self.path_width_tolerance = 0.3  # meters

    def detect_lane_center(self, rgb_image: np.ndarray) -> Optional[float]:
        """
        Detect lane center from RGB image

        Args:
            rgb_image: Input RGB image

        Returns:
            Lane center x-coordinate (normalized -1 to 1), or None if not detected
        """
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Focus on lower part of image (road area)
        height, width = gray.shape
        roi_top = int(height * 0.6)
        roi = gray[roi_top:, :]

        # Edge detection
        edges = cv2.Canny(roi, 50, 150)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=10
        )

        if lines is None:
            return None

        # Filter lines by angle (keep roughly vertical lines)
        lane_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 != 0:
                angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
                if abs(angle) > 30:  # Keep lines with steep angles
                    lane_lines.append(line[0])

        if not lane_lines:
            return None

        # Separate left and right lane lines
        left_lines = []
        right_lines = []

        for line in lane_lines:
            x1, y1, x2, y2 = line
            if x1 < width // 2 and x2 < width // 2:
                left_lines.append(line)
            elif x1 > width // 2 and x2 > width // 2:
                right_lines.append(line)

        # Calculate lane center
        left_x = None
        right_x = None

        if left_lines:
            left_x = np.mean([np.mean([line[0], line[2]]) for line in left_lines])

        if right_lines:
            right_x = np.mean([np.mean([line[0], line[2]]) for line in right_lines])

        if left_x is not None and right_x is not None:
            center_x = (left_x + right_x) / 2
        elif left_x is not None:
            center_x = left_x + width * 0.25  # Assume lane width
        elif right_x is not None:
            center_x = right_x - width * 0.25  # Assume lane width
        else:
            return None

        # Normalize to [-1, 1]
        normalized_center = (center_x - width / 2) / (width / 2)
        return np.clip(normalized_center, -1, 1)

    def follow_planned_path(
        self,
        current_position: Tuple[float, float],
        planned_path: list,
        current_heading: float,
    ) -> Tuple[float, float]:
        """
        Calculate steering commands to follow planned path

        Args:
            current_position: (x, y) current robot position in world coordinates
            planned_path: List of (x, y) waypoints in world coordinates
            current_heading: Current robot heading in radians

        Returns:
            (steering_angle, speed) commands
        """
        if not planned_path or len(planned_path) < 2:
            return (0.0, 0.0)

        # Find lookahead point
        lookahead_point = self._find_lookahead_point(current_position, planned_path)

        if lookahead_point is None:
            return (0.0, 0.0)

        # Calculate desired heading
        dx = lookahead_point[0] - current_position[0]
        dy = lookahead_point[1] - current_position[1]
        desired_heading = np.arctan2(dy, dx)

        # Calculate heading error
        heading_error = desired_heading - current_heading

        # Normalize error to [-pi, pi]
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        # PID control for steering
        steering_angle = self._pid_control(heading_error)

        # Speed control based on curvature
        speed = self._calculate_speed(heading_error)

        return (steering_angle, speed)

    def _find_lookahead_point(
        self, current_pos: Tuple[float, float], path: list
    ) -> Optional[Tuple[float, float]]:
        """Find the lookahead point on the path"""
        min_distance = float("inf")
        closest_index = 0

        # Find closest point on path
        for i, point in enumerate(path):
            distance = np.sqrt(
                (point[0] - current_pos[0]) ** 2 + (point[1] - current_pos[1]) ** 2
            )
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        # Find lookahead point
        for i in range(closest_index, len(path)):
            point = path[i]
            distance = np.sqrt(
                (point[0] - current_pos[0]) ** 2 + (point[1] - current_pos[1]) ** 2
            )

            if distance >= self.lookahead_distance:
                return point

        # If no point found, return last point
        return path[-1] if path else None

    def _pid_control(self, error: float) -> float:
        """PID controller for steering"""
        self.integral += error
        derivative = error - self.prev_error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error

        # Limit steering angle
        return np.clip(output, -1.0, 1.0)

    def _calculate_speed(self, heading_error: float) -> float:
        """Calculate speed based on path curvature"""
        base_speed = 0.3  # Base speed in m/s

        # Reduce speed for sharp turns
        error_factor = 1.0 - min(abs(heading_error) / np.pi, 0.8)

        return base_speed * error_factor
