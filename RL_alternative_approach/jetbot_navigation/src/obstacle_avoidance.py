import cv2
import numpy as np
from typing import List, Tuple
from .environment import RealWorldEnvironment


class ObstacleDetector:
    """
    Obstacle detection using computer vision
    Integrates with environment mapping
    """

    def __init__(self, camera_width=224, camera_height=224, min_obstacle_distance=0.5):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.min_obstacle_distance = min_obstacle_distance  # meters

        # Camera parameters (adjust based on your camera)
        self.camera_fov_horizontal = 62.2  # degrees
        self.camera_fov_vertical = 48.8  # degrees

    def detect_obstacles_from_depth(
        self, depth_image: np.ndarray, environment: RealWorldEnvironment
    ) -> List[Tuple[float, float]]:
        """
        Detect obstacles from depth image and update environment

        Args:
            depth_image: Depth image in mm
            environment: Environment to update

        Returns:
            List of obstacle positions in world coordinates
        """
        obstacles = []

        # Convert depth image to meters
        depth_meters = depth_image.astype(np.float32) / 1000.0

        # Find obstacles (close objects)
        obstacle_mask = (depth_meters > 0.1) & (
            depth_meters < self.min_obstacle_distance
        )

        # Find contours of obstacles
        obstacle_mask_uint8 = obstacle_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            obstacle_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:
            if cv2.contourArea(contour) > 50:  # Filter small noise
                # Get centroid of obstacle
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Convert to world coordinates
                    world_pos = self._pixel_to_world_position(
                        cx, cy, depth_meters[cy, cx]
                    )
                    obstacles.append(world_pos)

                    # Add obstacle to environment
                    environment.add_obstacle_at_position(world_pos[0], world_pos[1])

        return obstacles

    def detect_obstacles_from_rgb(
        self, rgb_image: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect obstacles from RGB image using edge detection

        Args:
            rgb_image: RGB image

        Returns:
            List of bounding boxes (x, y, w, h) of detected obstacles
        """
        # Convert to grayscale
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        obstacles = []
        for contour in contours:
            if cv2.contourArea(contour) > 200:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)

                # Filter by position (obstacles should be in lower part of image)
                if y + h > self.camera_height * 0.3:
                    obstacles.append((x, y, w, h))

        return obstacles

    def _pixel_to_world_position(
        self, pixel_x: int, pixel_y: int, distance: float
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world position

        Args:
            pixel_x: X coordinate in image
            pixel_y: Y coordinate in image
            distance: Distance in meters

        Returns:
            (x, y) world position in meters
        """
        # Normalize pixel coordinates to [-1, 1]
        norm_x = (pixel_x - self.camera_width / 2) / (self.camera_width / 2)
        norm_y = (pixel_y - self.camera_height / 2) / (self.camera_height / 2)

        # Convert to angles
        angle_x = norm_x * (self.camera_fov_horizontal / 2) * np.pi / 180
        angle_y = norm_y * (self.camera_fov_vertical / 2) * np.pi / 180

        # Calculate world position
        world_x = distance * np.tan(angle_x)
        world_y = distance * np.tan(angle_y)

        return (world_x, -world_y)

    def is_path_clear(self, rgb_image: np.ndarray, center_width_ratio=0.3) -> bool:
        """
        Check if the center path is clear for emergency obstacle avoidance

        Args:
            rgb_image: RGB image from camera
            center_width_ratio: Width ratio of center region to check

        Returns:
            True if path is clear, False if obstacle detected
        """
        height, width = rgb_image.shape[:2]

        # Define center region
        center_x_start = int(width * (0.5 - center_width_ratio / 2))
        center_x_end = int(width * (0.5 + center_width_ratio / 2))
        center_y_start = int(height * 0.4)  # Focus on lower part
        center_y_end = height

        # Extract center region
        center_region = rgb_image[
            center_y_start:center_y_end, center_x_start:center_x_end
        ]

        # Detect obstacles in center region
        obstacles = self.detect_obstacles_from_rgb(center_region)

        return len(obstacles) == 0
