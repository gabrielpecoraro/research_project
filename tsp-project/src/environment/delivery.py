import numpy as np
import random
import math
from scipy.spatial.distance import cdist
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
import torch


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
        **kwargs,
    ):
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.xy = xy
        self.max_box = max_box
        self.stops = []
        self.fixed = fixed
        self.boundary_index = boundary_index
        self.grid_size = grid_size  # number of blocks per side
        self.block_size = block_size  # size of each block

        # Create grid structure
        self._generate_grid()
        self._generate_stops()
        self._generate_q_values()
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
        ax.legend()

        if return_img:
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image

    def reset(self):
        self.stops = []
        if self.fixed:
            k = self.boundary_index
            first_stop = k[0]
        else:
            first_stop = random.randint(0, self.n_stops - 1)
        self.stops.append(first_stop)
        return first_stop

    def step(self, destination):
        state = self._get_state()
        new_state = destination
        reward = self._get_reward(state, new_state)
        self.stops.append(destination)
        return new_state, reward

    def _get_state(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def _get_reward(self, state, new_state):
        """Calculate reward for the transition from state to new_state"""
        # Get coordinates of current and next state
        current_pos = np.array([self.x[state], self.y[state]])
        next_pos = np.array([self.x[new_state], self.y[new_state]])

        # Base reward components
        completion_reward = 0
        distance_penalty = 0
        obstacle_penalty = 0
        revisit_penalty = 0

        # Check if path intersects with buildings
        num_samples = 20
        t = np.linspace(0, 1, num_samples)
        path_points = np.array(
            [
                current_pos[0] + t * (next_pos[0] - current_pos[0]),
                current_pos[1] + t * (next_pos[1] - current_pos[1]),
            ]
        ).T

        # Convert to grid coordinates
        path_coords = path_points.astype(int)

        # Check for building collisions
        for point in path_coords:
            if (
                point[0] < 0
                or point[1] < 0
                or point[0] >= self.grid.shape[0]
                or point[1] >= self.grid.shape[1]
            ):
                return torch.tensor(-1000.0)  # Immediate failure for out of bounds

            if self.grid[point[0], point[1]] == 1:
                obstacle_penalty = -1000.0  # Penalty for hitting building

        # Calculate distance penalty (normalized)
        distance = np.linalg.norm(next_pos - current_pos)
        max_possible_distance = np.sqrt(2) * self.max_box
        distance_penalty = -100 * (distance / max_possible_distance)

        # Penalty for revisiting states
        if new_state in self.stops:
            revisit_penalty = -500.0

        # Reward for completing the route
        if len(self.stops) == self.n_stops - 1:  # About to visit last stop
            completion_reward = 1000.0

        # Additional reward for efficient paths
        if len(self.stops) > 1:
            prev_pos = np.array([self.x[self.stops[-2]], self.y[self.stops[-2]]])
            angle = np.abs(
                np.arctan2(next_pos[1] - current_pos[1], next_pos[0] - current_pos[0])
                - np.arctan2(current_pos[1] - prev_pos[1], current_pos[0] - prev_pos[0])
            )
            angle_penalty = -50 * (angle / np.pi)  # Penalize sharp turns
        else:
            angle_penalty = 0

        # Combine all rewards
        total_reward = (
            completion_reward
            + distance_penalty
            + obstacle_penalty
            + revisit_penalty
            + angle_penalty
        )

        return torch.tensor(total_reward, dtype=torch.float32)


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
