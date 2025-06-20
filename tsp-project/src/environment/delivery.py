import numpy as np
import random
import math
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
import matplotlib  # <— add
import logging

logger = logging.getLogger(__name__)
# force a simple sans‐serif so FT2Font won’t choke
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]

import torch
from utils.rewards import RewardModel


class Delivery:
    def __init__(
        self,
        xy,
        boundary_index,
        n_stops,
        max_box,
        fixed,
        grid_size=3,  # Changed to 3x3
        block_size=4,
        type="nfixed",
        end_point=None,
        start_point=None,
        **kwargs,
    ):
        self.xy = xy
        self.boundary_index = boundary_index
        self.n_stops = n_stops
        self.max_box = max_box
        self.fixed = fixed
        self.grid_size = grid_size
        self.block_size = block_size
        self.end_point = end_point
        self.start_point = start_point
        logger.info(f"End point set to: {self.end_point}")

        # Define the four cardinal directions: up, down, left, right
        self.directions = [
            (0, -1),  # Up (decrease y)
            (0, 1),  # Down (increase y)
            (-1, 0),  # Left (decrease x)
            (1, 0),  # Right (increase x)
        ]

        # Create grid structure
        self._generate_grid()
        self._generate_stops()
        self._generate_q_values()

        # Initialize reward model with information about goal
        self.reward_model = RewardModel(self.grid, self.xy, self.max_box)

        # Set goal state in reward model if end point is specified
        if self.end_point:
            # Find closest stop to the goal point
            goal_distances = np.sqrt(
                (self.xy[:, 0] - self.end_point[0]) ** 2
                + (self.xy[:, 1] - self.end_point[1]) ** 2
            )
            self.goal_stop = np.argmin(goal_distances)
            self.reward_model.goal_state = self.goal_stop
            logger.info(f"Goal stop set to stop #{self.goal_stop}")
        else:
            self.goal_stop = None

        self.render()
        self.reset()

    def _generate_grid(self):
        """Generate grid structure with streets and buildings including outer streets"""
        # Calculate total size including streets (add 2 for outer streets)
        total_size = (self.grid_size * self.block_size) + (self.grid_size - 1) + 2
        self.grid = np.zeros((total_size, total_size))  # Initialize with streets

        # Create buildings (1 represents building/obstacle)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Calculate building position with streets in between (offset by 1 for outer street)
                row_start = 1 + i * (self.block_size + 1)
                row_end = row_start + self.block_size
                col_start = 1 + j * (self.block_size + 1)
                col_end = col_start + self.block_size

                # Place building block
                self.grid[row_start:row_end, col_start:col_end] = 1

        # Streets are represented by 0s (already set)
        self.accessible_coords = np.where(self.grid == 0)

    def _generate_stops(self):
        self.x = self.xy[:, 0]
        self.y = self.xy[:, 1]

    def _generate_q_values(self, box_size=0.2):
        xy = np.column_stack([self.x, self.y])
        self.q_stops = cdist(xy, xy)

    def render(self, route=None, ax=None, return_img=False):
        """Render the environment with connected route"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.set_title("City Grid with Delivery Stops")
        ax.imshow(self.grid, cmap="binary")

        if hasattr(self, "x") and hasattr(self, "y"):
            # Plot all stops
            ax.scatter(
                self.x, self.y, c="lightgray", s=50, zorder=2, label="Unvisited Stops"
            )

            # Plot route if provided
            if route is not None and len(route) > 1:
                # Convert route indices to coordinates
                route_x = [self.x[i] for i in route]
                route_y = [self.y[i] for i in route]

                # Plot route with arrows to show direction
                for i in range(len(route) - 1):
                    ax.annotate(
                        "",
                        xy=(route_x[i + 1], route_y[i + 1]),
                        xytext=(route_x[i], route_y[i]),
                        arrowprops=dict(arrowstyle="->", color="blue", lw=2),
                        zorder=3,
                    )

                # Plot visited stops
                ax.scatter(
                    route_x[1:-1],
                    route_y[1:-1],
                    c="red",
                    s=50,
                    zorder=4,
                    label="Visited Stops",
                )

                # Highlight start and end
                ax.scatter(
                    route_x[0], route_y[0], c="green", s=100, zorder=5, label="Start"
                )
                ax.scatter(
                    route_x[-1], route_y[-1], c="blue", s=100, zorder=5, label="End"
                )

                # Add stop numbers
                for i, (x, y) in enumerate(zip(route_x, route_y)):
                    ax.annotate(
                        f"{i}", (x, y), xytext=(5, 5), textcoords="offset points"
                    )

        ax.grid(True)
        ax.set_xticks([])
        ax.set_yticks([])
        # wrap legend in try/except
        try:
            ax.legend()
        except Exception:
            pass

        if return_img:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

    def reset(self):
        """Reset environment and start agent at specified start point or bottom-left"""
        self.stops = []

        # Get grid dimensions
        grid_height, grid_width = self.grid.shape

        if self.start_point:
            # Use provided start point
            start_x, start_y = self.start_point
        else:
            # Default to bottom-left
            start_x = 0
            start_y = grid_height - 1

        # Make sure we're on a street (value 0)
        # If start point is not on a street, find closest street cell
        if 0 <= start_y < grid_height and 0 <= start_x < grid_width:
            if self.grid[int(start_y), int(start_x)] != 0:
                # Find nearest valid street cell
                valid_coords = np.argwhere(self.grid == 0)
                dists = np.sqrt(
                    (valid_coords[:, 0] - start_y) ** 2
                    + (valid_coords[:, 1] - start_x) ** 2
                )
                closest_idx = np.argmin(dists)
                start_y, start_x = valid_coords[closest_idx]
        else:
            # If start point is off grid, default to bottom-left
            start_x = 0
            start_y = grid_height - 1
            while start_x < grid_width and self.grid[start_y, start_x] != 0:
                start_x += 1

        # Set initial position and find closest stop
        self.current_pos = np.array([start_x, start_y])

        # Find closest stop to this position
        start_distances = np.sqrt(
            (self.xy[:, 0] - start_x) ** 2 + (self.xy[:, 1] - start_y) ** 2
        )
        start_stop = np.argmin(start_distances)

        # Set initial stop
        self.stops = [start_stop]

        return start_stop

    def step(self, action):
        """Take a step in the environment using directional movement"""
        state = self._get_state()
        invalid_moves = 0

        # Get current position
        current_x, current_y = self.current_pos

        # Apply movement in the selected direction (up, down, left, right)
        dx, dy = self.directions[action]
        new_x = current_x + dx
        new_y = current_y + dy

        # Ensure we stay within grid boundaries
        if (
            new_x < 0
            or new_x >= self.grid.shape[1]
            or new_y < 0
            or new_y >= self.grid.shape[0]
        ):
            # Out of bounds - return current state with penalty
            invalid_moves += 1
            reward = self.reward_model.building_penalty
            return state, float(reward)

        # Check if we hit a building (value 1 in grid)
        if self.grid[int(new_y), int(new_x)] == 1:
            # We hit a building - return current state with penalty
            invalid_moves += 1
            reward = self.reward_model.building_penalty
            return state, float(reward)

        # Update position to new valid position
        self.current_pos = np.array([new_x, new_y])

        # Find the nearest stop to the new position
        distances = np.sqrt((self.xy[:, 0] - new_x) ** 2 + (self.xy[:, 1] - new_y) ** 2)
        new_state = np.argmin(distances)

        # Calculate reward
        reward = self.reward_model.calculate_reward(
            state, new_state, self.stops, self.xy
        )

        # Only record the stop if it's a new one
        if new_state not in self.stops:
            self.stops.append(new_state)

        return new_state, float(reward)

    def _get_state(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def _get_reward(self, state, new_state):
        """Get reward for transition from state to new_state"""
        reward = self.reward_model.calculate_reward(
            state, new_state, self.stops, self.xy
        )
        return reward  # The RewardModel will handle returning a CPU tensor


if __name__ == "__main__":
    # Generate random coordinates along streets
    grid_size = 3  # Changed to 3x3
    block_size = 4
    n_stops = 5
    max_box = grid_size * block_size

    # Generate coordinates only on streets
    def generate_street_coordinates(n_stops, grid_size, block_size):
        coords = []
        while len(coords) < n_stops:
            x = np.random.randint(0, grid_size * block_size)
            y = np.random.randint(0, grid_size * block_size)
            # Check if coordinate is on a street
            if x % block_size == 0 or y % block_size == 0:
                coords.append([x, y])
        return np.array(coords)

    xy = generate_street_coordinates(n_stops, grid_size, block_size)

    # Create environment
    env = Delivery(
        xy=xy,
        boundary_index=list(range(n_stops)),
        n_stops=n_stops,
        max_box=max_box,
        fixed=False,
        grid_size=grid_size,
        block_size=block_size,
    )

    # Render environment
    env.render()
