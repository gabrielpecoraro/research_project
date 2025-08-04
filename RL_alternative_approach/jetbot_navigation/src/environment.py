import numpy as np
import cv2
from typing import List, Tuple, Optional


class RealWorldEnvironment:
    """
    Real-world environment representation for JetBot navigation
    Integrates camera vision with grid-based pathfinding
    """

    def __init__(self, width_meters=10.0, height_meters=10.0, grid_resolution=0.1):
        """
        Initialize environment

        Args:
            width_meters: Real world width in meters
            height_meters: Real world height in meters
            grid_resolution: Grid cell size in meters (0.1m = 10cm cells)
        """
        self.width_meters = width_meters
        self.height_meters = height_meters
        self.grid_resolution = grid_resolution

        # Grid dimensions
        self.grid_width = int(width_meters / grid_resolution)
        self.grid_height = int(height_meters / grid_resolution)

        # Occupancy grid (0 = free, 1 = occupied, 0.5 = unknown)
        self.occupancy_grid = np.full((self.grid_height, self.grid_width), 0.5)

        # Robot position in grid coordinates
        self.robot_grid_pos = (self.grid_width // 2, self.grid_height // 2)

        # Safety margin around obstacles (in grid cells)
        self.safety_margin = 3

    def world_to_grid(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to grid coordinates"""
        grid_x = int(world_x / self.grid_resolution)
        grid_y = int(world_y / self.grid_resolution)
        return (grid_x, grid_y)

    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates (meters)"""
        world_x = grid_x * self.grid_resolution
        world_y = grid_y * self.grid_resolution
        return (world_x, world_y)

    def update_occupancy_from_depth(self, depth_image: np.ndarray, camera_params: dict):
        """
        Update occupancy grid from depth camera data

        Args:
            depth_image: Depth image from camera
            camera_params: Camera intrinsic parameters
        """
        # This would integrate with your depth camera
        # For now, we'll use a simplified approach
        pass

    def add_obstacle_at_position(
        self, world_x: float, world_y: float, size: float = 0.3
    ):
        """Add circular obstacle at world position"""
        center_grid = self.world_to_grid(world_x, world_y)
        radius_grid = int(size / self.grid_resolution)

        for y in range(
            max(0, center_grid[1] - radius_grid),
            min(self.grid_height, center_grid[1] + radius_grid + 1),
        ):
            for x in range(
                max(0, center_grid[0] - radius_grid),
                min(self.grid_width, center_grid[0] + radius_grid + 1),
            ):
                dist = np.sqrt((x - center_grid[0]) ** 2 + (y - center_grid[1]) ** 2)
                if dist <= radius_grid + self.safety_margin:
                    self.occupancy_grid[y, x] = 1.0

    def is_valid_position(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid position is valid (free space)"""
        if (
            grid_x < 0
            or grid_x >= self.grid_width
            or grid_y < 0
            or grid_y >= self.grid_height
        ):
            return False

        return self.occupancy_grid[grid_y, grid_x] < 0.5

    def get_occupancy_map(self) -> np.ndarray:
        """Get current occupancy grid for visualization"""
        return self.occupancy_grid.copy()
