import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib.animation as animation
import random

# JAX imports
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# Add these imports for Bezier curves
from scipy.interpolate import splprep, splev

#### REFERENCES FOR ARTICLE ####

# https://arxiv.org/pdf/2205.07772
# https://arxiv.org/abs/1705.08926


# JAX-accelerated functions
@jit
def jax_distance(point1, point2):
    """JAX-accelerated distance calculation"""
    return jnp.sqrt(jnp.sum((point1 - point2) ** 2))


@jit
def jax_batch_distances(points, target_point):
    """JAX-accelerated batch distance calculation"""
    return jnp.sqrt(jnp.sum((points - target_point) ** 2, axis=1))


@jit
def jax_check_position_validity(position, obstacles_array, env_width, env_height):
    """JAX-accelerated position validity check"""
    x, y = position[0], position[1]

    # Check bounds
    if x < 0 or x > env_width or y < 0 or y > env_height:
        return False

    # Check obstacles
    for i in range(obstacles_array.shape[0]):
        ox, oy, ow, oh = obstacles_array[i]
        if (ox <= x <= ox + ow) and (oy <= y <= oy + oh):
            return False

    return True


@jit
def jax_batch_position_validity(positions, obstacles_array, env_width, env_height):
    """JAX-accelerated batch position validity check"""
    x, y = positions[:, 0], positions[:, 1]

    # Check bounds
    valid_bounds = (x >= 0) & (x <= env_width) & (y >= 0) & (y <= env_height)

    # Check obstacles
    valid_obstacles = jnp.ones_like(x, dtype=bool)
    for i in range(obstacles_array.shape[0]):
        ox, oy, ow, oh = obstacles_array[i]
        inside_obstacle = (x >= ox) & (x <= ox + ow) & (y >= oy) & (y <= oy + oh)
        valid_obstacles = valid_obstacles & ~inside_obstacle

    return valid_bounds & valid_obstacles


@jit
def jax_compute_escape_routes(
    target_pos, agent_positions, obstacles_array, env_width, env_height
):
    """JAX-accelerated escape route computation"""
    # Generate 32 directions
    angles = jnp.linspace(0, 2 * jnp.pi, 32, endpoint=False)
    directions = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)

    escape_distance = 3.0
    escape_routes = 0
    blocked_directions = 0

    for i in range(32):
        dx, dy = directions[i]
        direction_blocked = False

        # Check multiple points along the escape path
        for step in range(1, 15):
            test_dist = escape_distance * step / 14
            test_pos = target_pos + jnp.array([dx, dy]) * test_dist

            # Check if position is valid
            x, y = test_pos[0], test_pos[1]
            if x < 0 or x > env_width or y < 0 or y > env_height:
                direction_blocked = True
                break

            # Check obstacles
            for j in range(obstacles_array.shape[0]):
                ox, oy, ow, oh = obstacles_array[j]
                if (ox <= x <= ox + ow) and (oy <= y <= oy + oh):
                    direction_blocked = True
                    break

            if direction_blocked:
                break

        if direction_blocked:
            blocked_directions += 1
        else:
            # Check if this is a valid escape (leads away from agents)
            final_pos = target_pos + jnp.array([dx, dy]) * escape_distance

            # Calculate distances to all agents from escape point
            agent_dists_from_escape = jax_batch_distances(agent_positions, final_pos)
            current_agent_dists = jax_batch_distances(agent_positions, target_pos)

            # Valid escape if it leads significantly away from closest agent
            min_escape_dist = jnp.min(agent_dists_from_escape)
            min_current_dist = jnp.min(current_agent_dists)

            if min_escape_dist > min_current_dist * 2.0:
                escape_routes += 1

    return escape_routes, blocked_directions


@jit
def jax_path_length(path_points):
    """JAX-accelerated path length calculation"""
    if path_points.shape[0] < 2:
        return 0.0

    diffs = path_points[1:] - path_points[:-1]
    distances = jnp.sqrt(jnp.sum(diffs**2, axis=1))
    return jnp.sum(distances)


@jit
def jax_normalize_vector(vector):
    """JAX-accelerated vector normalization"""
    magnitude = jnp.sqrt(jnp.sum(vector**2))
    return jnp.where(magnitude > 1e-10, vector / magnitude, jnp.zeros_like(vector))


class JAXEnvironment:
    def __init__(self, width, height, block_size=1.0):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.obstacles = []  # List of (x,y,width,height) tuples for blocks
        self.obstacles_array = jnp.array([]).reshape(
            0, 4
        )  # JAX array for fast computation

    def add_block(self, x, y, width, height):
        """Add a block obstacle at the specified position"""
        self.obstacles.append((x, y, width, height))
        # Update JAX array
        self.obstacles_array = jnp.array(self.obstacles)

    def is_valid_position(self, x, y):
        """Check if a position is valid (within bounds and not in an obstacle)"""
        if len(self.obstacles) == 0:
            return 0 <= x <= self.width and 0 <= y <= self.height

        position = jnp.array([x, y])
        return bool(
            jax_check_position_validity(
                position, self.obstacles_array, self.width, self.height
            )
        )

    def is_valid_positions_batch(self, positions):
        """JAX-accelerated batch position validity check"""
        if len(self.obstacles) == 0:
            return jnp.all(
                (positions >= 0) & (positions <= jnp.array([self.width, self.height])),
                axis=1,
            )

        return jax_batch_position_validity(
            positions, self.obstacles_array, self.width, self.height
        )

    def plot(self, path=None):
        """Visualize the environment with obstacles and path"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot obstacles
        for ox, oy, ow, oh in self.obstacles:
            rect = plt.Rectangle((ox, oy), ow, oh, color="gray")
            ax.add_patch(rect)

        # Plot path if provided
        if path:
            path_x = [p[0] for p in path]
            path_y = [p[1] for p in path]
            plt.plot(path_x, path_y, "r-", linewidth=2)
            plt.plot(path_x[0], path_y[0], "go", markersize=10)  # Start
            plt.plot(path_x[-1], path_y[-1], "bo", markersize=10)  # End

        plt.xlim(0, self.width)
        plt.ylim(0, self.height)
        plt.grid(True)
        plt.title("Neighborhood Pathfinding")
        plt.show()

    def animate_path(self, path, save_animation=False):
        """Animate a symbol moving along the path"""
        if not path:
            print("No path to animate")
            return

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot obstacles
        for ox, oy, ow, oh in self.obstacles:
            rect = plt.Rectangle((ox, oy), ow, oh, color="gray")
            ax.add_patch(rect)

        # Plot the full path
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, "r-", alpha=0.6, linewidth=2)

        # Plot start and goal
        ax.plot(path_x[0], path_y[0], "go", markersize=10)  # Start
        ax.plot(path_x[-1], path_y[-1], "bo", markersize=10)  # End

        # Create moving point (will be updated during animation)
        (point,) = ax.plot([], [], "ko", markersize=8)

        # Set plot limits and styling
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.grid(True)
        ax.set_title("Pathfinding Animation")

        def init():
            point.set_data([], [])
            return (point,)

        def update(frame):
            if frame < len(path):
                x, y = path[frame]
                point.set_data([x], [y])  # Wrap x and y in lists
            return (point,)

        # Create animation
        frames = len(path)
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=frames,
            init_func=init,
            interval=100,
            blit=True,
            repeat=False,
        )

        if save_animation:
            # Save animation as mp4 (requires ffmpeg)
            ani.save("path_animation.mp4", writer="ffmpeg", fps=10)

        plt.tight_layout()
        plt.show()


class JAXAStar:
    def __init__(self, environment):
        self.env = environment

    def heuristic(self, a, b):
        """JAX-accelerated Euclidean distance heuristic"""
        point_a = jnp.array(a)
        point_b = jnp.array(b)
        return float(jax_distance(point_a, point_b))

    def get_neighbors(self, current, step_size=0.5):
        """Get valid neighboring points in continuous space with JAX acceleration"""
        x, y = current
        neighbors = []

        # Generate neighbors in 8 directions
        offsets = jnp.array(
            [
                [-step_size, -step_size],
                [-step_size, 0],
                [-step_size, step_size],
                [0, -step_size],
                [0, step_size],
                [step_size, -step_size],
                [step_size, 0],
                [step_size, step_size],
            ]
        )

        neighbor_positions = jnp.array([x, y]) + offsets

        # Use JAX batch validation
        if len(self.env.obstacles) > 0:
            valid_mask = self.env.is_valid_positions_batch(neighbor_positions)
            valid_neighbors = neighbor_positions[valid_mask]
        else:
            # Simple bounds check for no obstacles case
            bounds_check = jnp.all(
                (neighbor_positions >= 0)
                & (neighbor_positions <= jnp.array([self.env.width, self.env.height])),
                axis=1,
            )
            valid_neighbors = neighbor_positions[bounds_check]

        return [tuple(pos) for pos in valid_neighbors]

    def find_path(self, start, goal):
        """Find path using A* algorithm with JAX-accelerated heuristic"""
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if self.heuristic(current, goal) < 0.5:  # Close enough to goal
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # Return reversed path

            for neighbor in self.get_neighbors(current):
                # Distance between current and neighbor
                tentative_g_score = g_score[current] + self.heuristic(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(
                        neighbor, goal
                    )

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found


class FleeingTarget:
    def __init__(self, env, initial_position, flee_distance=2.0, speed=0.3):
        self.env = env
        self.position = initial_position  # Initially at point B
        self.flee_distance = flee_distance  # How far to flee in each step
        self.speed = speed  # How fast the target moves

    def update_position(self, agent_position):
        """Move away from agent while avoiding obstacles"""
        # Calculate vector from agent to target
        agent_pos = jnp.array(agent_position)
        target_pos = jnp.array(self.position)

        direction_vector = target_pos - agent_pos
        normalized_direction = jax_normalize_vector(direction_vector)

        # Scale by speed
        movement = normalized_direction * self.speed

        # Try to move in direction away from agent
        new_position = target_pos + movement

        # If that position is valid, move there
        if self.env.is_valid_position(float(new_position[0]), float(new_position[1])):
            self.position = (float(new_position[0]), float(new_position[1]))
            return

        # If direct path is blocked, try alternative directions
        possible_moves = []
        for angle_offset in [0, 30, -30, 60, -60, 90, -90]:
            angle = (
                jnp.arctan2(normalized_direction[1], normalized_direction[0])
                + angle_offset * jnp.pi / 180
            )
            new_direction = jnp.array([jnp.cos(angle), jnp.sin(angle)]) * self.speed
            new_pos = target_pos + new_direction

            if self.env.is_valid_position(float(new_pos[0]), float(new_pos[1])):
                distance = float(jax_distance(new_pos, agent_pos))
                possible_moves.append(
                    (distance, (float(new_pos[0]), float(new_pos[1])))
                )

        # Choose the move that keeps furthest from agent
        if possible_moves:
            possible_moves.sort(reverse=True)  # Sort by distance (descending)
            self.position = possible_moves[0][1]


class SmartFleeingTarget:
    def __init__(self, env, initial_position, speed=0.3):
        self.env = env
        self.position = initial_position
        self.speed = speed
        self.last_strategy = "opposite"

    def is_near_edge(self, margin=1.5):
        """Check if target is near the edge of the environment"""
        x, y = self.position
        return (
            x < margin
            or x > self.env.width - margin
            or y < margin
            or y > self.env.height - margin
        )

    def get_edge_proximity(self):
        """Return which edges the target is near to"""
        x, y = self.position
        edges = []
        margin = 1.5

        if x < margin:
            edges.append("left")
        if x > self.env.width - margin:
            edges.append("right")
        if y < margin:
            edges.append("bottom")
        if y > self.env.height - margin:
            edges.append("top")

        return edges

    def opposite_direction_strategy(self, agent_position):
        """JAX-accelerated opposite direction strategy"""
        agent_pos = jnp.array(agent_position)
        target_pos = jnp.array(self.position)

        direction_vector = target_pos - agent_pos
        normalized_direction = jax_normalize_vector(direction_vector)
        movement = normalized_direction * self.speed

        return float(movement[0]), float(movement[1])

    def perpendicular_direction_strategy(self, agent_position):
        """JAX-accelerated perpendicular direction strategy"""
        agent_pos = jnp.array(agent_position)
        target_pos = jnp.array(self.position)

        direction_vector = target_pos - agent_pos

        # Get perpendicular vectors (two options)
        perp1 = jnp.array(
            [-direction_vector[1], direction_vector[0]]
        )  # 90 degrees clockwise
        perp2 = jnp.array(
            [direction_vector[1], -direction_vector[0]]
        )  # 90 degrees counter-clockwise

        # Normalize and scale both options
        perp1_normalized = jax_normalize_vector(perp1) * self.speed
        perp2_normalized = jax_normalize_vector(perp2) * self.speed

        # Check which perpendicular direction keeps the target on the grid
        new_pos1 = target_pos + perp1_normalized
        new_pos2 = target_pos + perp2_normalized

        valid1 = self.env.is_valid_position(float(new_pos1[0]), float(new_pos1[1]))
        valid2 = self.env.is_valid_position(float(new_pos2[0]), float(new_pos2[1]))

        if valid1 and valid2:
            # Choose based on edge proximity
            edges = self.get_edge_proximity()
            if "left" in edges or "right" in edges:
                if abs(perp1_normalized[0]) < abs(perp1_normalized[1]):
                    return float(perp1_normalized[0]), float(perp1_normalized[1])
                else:
                    return float(perp2_normalized[0]), float(perp2_normalized[1])
            else:
                if abs(perp1_normalized[1]) < abs(perp1_normalized[0]):
                    return float(perp1_normalized[0]), float(perp1_normalized[1])
                else:
                    return float(perp2_normalized[0]), float(perp2_normalized[1])
        elif valid1:
            return float(perp1_normalized[0]), float(perp1_normalized[1])
        elif valid2:
            return float(perp2_normalized[0]), float(perp2_normalized[1])
        else:
            return self.opposite_direction_strategy(agent_position)

    def random_direction_strategy(self):
        """Move in a random valid direction"""
        angles = jnp.linspace(0, 2 * jnp.pi, 8, endpoint=False)
        angles_shuffled = np.random.permutation(angles)

        for angle in angles_shuffled:
            direction = jnp.array([jnp.cos(angle), jnp.sin(angle)]) * self.speed
            new_pos = jnp.array(self.position) + direction

            if self.env.is_valid_position(float(new_pos[0]), float(new_pos[1])):
                return float(direction[0]), float(direction[1])

        return 0, 0

    def update_position(self, agent_position):
        """Update position using smart fleeing strategy with JAX acceleration"""
        near_edge = self.is_near_edge()

        if near_edge:
            dx, dy = self.perpendicular_direction_strategy(agent_position)
            self.last_strategy = "perpendicular"
        else:
            if random.random() < 0.2:
                dx, dy = self.random_direction_strategy()
                self.last_strategy = "random"
            else:
                dx, dy = self.opposite_direction_strategy(agent_position)
                self.last_strategy = "opposite"

        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        if self.env.is_valid_position(new_x, new_y):
            self.position = (new_x, new_y)
            return

        # If primary strategy failed, try others in sequence
        strategies = ["opposite", "perpendicular", "random"]
        strategies.remove(self.last_strategy)

        for strategy in strategies:
            if strategy == "opposite":
                dx, dy = self.opposite_direction_strategy(agent_position)
            elif strategy == "perpendicular":
                dx, dy = self.perpendicular_direction_strategy(agent_position)
            else:
                dx, dy = self.random_direction_strategy()

            new_x = self.position[0] + dx
            new_y = self.position[1] + dy

            if self.env.is_valid_position(new_x, new_y):
                self.position = (new_x, new_y)
                self.last_strategy = strategy
                return


class JAXVehicleTarget:
    def __init__(self, env, initial_position, speed=0.5, inertia=0.8):
        """JAX-accelerated vehicle-like target"""
        self.env = env
        self.position = initial_position
        self.speed = speed
        self.inertia = inertia
        self.direction = jnp.array([0.0, 0.0])  # JAX array for direction
        self.last_strategy = "straight"

    def update_position(self, agent_position):
        """JAX-accelerated movement with vehicle-like patterns"""
        agent_pos = jnp.array(agent_position)
        target_pos = jnp.array(self.position)

        # Calculate vector from agent to target
        direction_vector = target_pos - agent_pos
        normalized_direction = jax_normalize_vector(direction_vector)

        # Apply inertia - blend new direction with previous direction
        if jnp.allclose(self.direction, 0.0):  # First movement
            new_direction = normalized_direction
        else:
            blended_direction = (
                self.inertia * self.direction
                + (1 - self.inertia) * normalized_direction
            )
            new_direction = jax_normalize_vector(blended_direction)

        # Scale by speed
        movement = new_direction * self.speed
        new_position = target_pos + movement

        # Try to move in that direction
        if self.env.is_valid_position(float(new_position[0]), float(new_position[1])):
            self.position = (float(new_position[0]), float(new_position[1]))
            self.direction = new_direction
            self.last_strategy = "straight"
            return

        # If blocked, try adjusting direction slightly
        for angle_offset in [
            15,
            -15,
            30,
            -30,
            45,
            -45,
            60,
            -60,
            80,
            -80,
            100,
            -100,
            110,
            -110,
        ]:
            angle_rad = angle_offset * jnp.pi / 180
            rotation_matrix = jnp.array(
                [
                    [jnp.cos(angle_rad), -jnp.sin(angle_rad)],
                    [jnp.sin(angle_rad), jnp.cos(angle_rad)],
                ]
            )

            rotated_direction = rotation_matrix @ new_direction
            test_movement = rotated_direction * self.speed
            test_position = target_pos + test_movement

            if self.env.is_valid_position(
                float(test_position[0]), float(test_position[1])
            ):
                self.position = (float(test_position[0]), float(test_position[1]))
                self.direction = rotated_direction
                self.last_strategy = f"turning_{angle_offset}"
                return


def smooth_path_with_bezier(path, smoothing_factor=0.05, num_points=None):
    """
    Optimized smooth path generation with JAX acceleration where applicable
    """
    if len(path) < 3:
        return path

    if num_points is None:
        num_points = min(50, len(path) * 2)

    # Convert path to JAX array for faster processing
    path_array = jnp.array(path)

    # Use JAX to calculate path length for better performance
    path_length = float(jax_path_length(path_array))

    # Adjust number of points based on path length
    num_points = min(num_points, max(10, int(path_length * 5)))

    # Convert back to numpy for scipy spline
    path_np = np.array(path)
    x = path_np[:, 0]
    y = path_np[:, 1]

    try:
        tck, u = splprep([x, y], s=smoothing_factor, per=False)
        u_new = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_new, tck)
        smooth_path = list(zip(x_smooth, y_smooth))
        return smooth_path
    except:
        return path


def interpolate_agent_position(path, progress, use_smooth=True, env=None):
    """JAX-accelerated agent position interpolation"""
    if not path or len(path) == 0:
        return (0, 0)

    if len(path) == 1:
        return path[0]

    if use_smooth and len(path) >= 3:
        smooth_path = smooth_path_with_bezier(
            path, smoothing_factor=0.05, num_points=min(30, len(path) * 2)
        )

        if env is not None:
            # JAX-accelerated validation
            path_array = jnp.array(smooth_path)
            check_interval = max(1, len(smooth_path) // 10)
            check_indices = jnp.arange(0, len(smooth_path), check_interval)

            safe_smooth_path = []
            for i in check_indices:
                point = smooth_path[int(i)]
                if not env.is_valid_position(point[0], point[1]):
                    break
                safe_smooth_path = smooth_path[: int(i) + 1]

            if len(safe_smooth_path) < 3:
                return interpolate_agent_position(path, progress, False, env)
            smooth_path = safe_smooth_path

        # JAX-accelerated interpolation
        smooth_index = progress * (len(smooth_path) - 1)
        lower_idx = int(smooth_index)
        upper_idx = min(lower_idx + 1, len(smooth_path) - 1)

        if lower_idx == upper_idx:
            return smooth_path[lower_idx]

        t = smooth_index - lower_idx
        point1 = jnp.array(smooth_path[lower_idx])
        point2 = jnp.array(smooth_path[upper_idx])

        interpolated = point1 * (1 - t) + point2 * t
        return (float(interpolated[0]), float(interpolated[1]))
    else:
        # JAX-accelerated discrete interpolation
        index = progress * (len(path) - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(path) - 1)

        if lower_idx == upper_idx:
            return path[lower_idx]

        t = index - lower_idx
        point1 = jnp.array(path[lower_idx])
        point2 = jnp.array(path[upper_idx])

        interpolated = point1 * (1 - t) + point2 * t
        return (float(interpolated[0]), float(interpolated[1]))


def animate_multi_agent_pursuit(
    env,
    pathfinder,
    start,
    target_position=None,
    max_frames=500,
    agent1_delay=30,
    agent2_delay=100,
    target_speed=1.2,
    agent1_speed=0.8,
    agent2_speed=2.0,
    use_smooth_trajectories=True,
):
    """JAX-accelerated multi-agent pursuit simulation"""
    if target_position is None:
        target_position = (env.width - 0.5, env.height - 0.5)

    if not env.is_valid_position(target_position[0], target_position[1]):
        raise ValueError("Target position must be on a street, not inside a building")

    fig, ax = plt.subplots(figsize=(10, 10))

    for ox, oy, ow, oh in env.obstacles:
        rect = plt.Rectangle((ox, oy), ow, oh, color="gray")
        ax.add_patch(rect)

    # Use JAX-accelerated vehicle target
    target = JAXVehicleTarget(env, target_position, speed=target_speed, inertia=0.8)
    target_history = []

    agent1_pos = start
    agent1_path = pathfinder.find_path(agent1_pos, target.position)

    agent1_path_progress = 0.0
    agent2_path_progress = 0.0
    agent1_smooth_path = []
    agent2_smooth_path = []

    if use_smooth_trajectories and agent1_path and len(agent1_path) >= 3:
        agent1_smooth_path = smooth_path_with_bezier(agent1_path)

    agent2_pos = None
    agent2_path = None

    # Create plot elements
    (agent1_point,) = ax.plot([], [], "ro", markersize=8)
    (agent2_point,) = ax.plot([], [], "mo", markersize=8)
    (target_point,) = ax.plot([], [], "bs", markersize=8)
    (path1_line,) = ax.plot([], [], "r-", alpha=0.4)
    (path2_line,) = ax.plot([], [], "m-", alpha=0.4)
    (smooth_path1_line,) = ax.plot([], [], "r-", alpha=0.8, linewidth=2)
    (smooth_path2_line,) = ax.plot([], [], "m-", alpha=0.8, linewidth=2)
    (predicted_path,) = ax.plot([], [], "g--", alpha=0.6)
    (target_trail,) = ax.plot([], [], "b:", alpha=0.4)

    # Text elements
    strategy_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="black")
    info_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, color="black")
    agent1_status = ax.text(0.02, 0.90, "", transform=ax.transAxes, color="red")
    agent2_status = ax.text(0.02, 0.86, "", transform=ax.transAxes, color="magenta")
    time_text = ax.text(0.02, 0.82, "", transform=ax.transAxes, color="blue")
    speed_text = ax.text(0.02, 0.78, "", transform=ax.transAxes, color="green")
    capture_text = ax.text(
        0.5,
        0.5,
        "",
        transform=ax.transAxes,
        color="red",
        fontsize=24,
        ha="center",
        va="center",
        alpha=0,
        fontweight="bold",
    )

    target_captured = False

    def is_target_captured_jax():
        """JAX-accelerated capture detection"""
        if not interceptor_deployed or agent2_pos is None:
            info_text.set_text("Agent 2 not deployed yet - no capture possible")
            return False

        # Convert positions to JAX arrays
        target_pos_jax = jnp.array(target.position)
        agent1_pos_jax = jnp.array(agent1_pos)
        agent2_pos_jax = jnp.array(agent2_pos)
        agent_positions = jnp.stack([agent1_pos_jax, agent2_pos_jax])

        # JAX-accelerated distance calculations
        distances = jax_batch_distances(agent_positions, target_pos_jax)
        dist1, dist2 = float(distances[0]), float(distances[1])

        # Proximity check
        proximity_radius = 1.85
        if dist1 <= proximity_radius and dist2 <= proximity_radius:
            # JAX-accelerated escape route analysis
            escape_routes, blocked_directions = jax_compute_escape_routes(
                target_pos_jax,
                agent_positions,
                env.obstacles_array,
                env.width,
                env.height,
            )

            if escape_routes == 0:
                info_text.set_text(
                    "CAPTURED! Both agents very close and no escape routes"
                )
                return True
            else:
                info_text.set_text(
                    f"Both agents close (d1:{dist1:.1f}, d2:{dist2:.1f}) but {escape_routes} escape routes available"
                )
                return False
        else:
            info_text.set_text(
                f"Proximity capture failed - Agent distances: {dist1:.1f}, {dist2:.1f} (need both ≤{proximity_radius})"
            )

        # Single agent trap check
        min_agent_dist = min(dist1, dist2)
        closest_agent_name = "Agent 1" if dist1 < dist2 else "Agent 2"

        if min_agent_dist > 0.8:
            info_text.set_text(
                f"No agents close enough to trap - closest: {closest_agent_name} at {min_agent_dist:.1f} (need ≤0.8)"
            )
            return False

        # JAX-accelerated detailed escape analysis
        escape_routes, blocked_directions = jax_compute_escape_routes(
            target_pos_jax, agent_positions, env.obstacles_array, env.width, env.height
        )

        min_blocked_threshold = int(32 * 0.9)

        if escape_routes == 0 and blocked_directions >= min_blocked_threshold:
            info_text.set_text(
                f"TRULY TRAPPED! {closest_agent_name} cornered target. "
                f"Distance: {min_agent_dist:.1f}, Escapes: {escape_routes}, "
                f"Blocked: {blocked_directions}/32"
            )
            return True

        info_text.set_text(
            f"NOT TRAPPED - Agent 1: {dist1:.1f}, Agent 2: {dist2:.1f}, "
            f"Escape routes: {escape_routes}, Blocked: {blocked_directions}/32"
        )
        return False

    # Simplified pattern learning and prediction functions (keeping original logic)
    def learn_movement_pattern(history, min_length=5):
        nonlocal pattern_confidence
        if len(history) < min_length:
            pattern_confidence = 0
            return None

        recent = history[-min(10, len(history)) :]
        directions = []

        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i - 1][0]
            dy = recent[i][1] - recent[i - 1][1]

            if abs(dx) < 0.01 and abs(dy) < 0.01:
                continue

            angle = np.arctan2(dy, dx) * 180 / np.pi
            directions.append(angle)

        if len(directions) < 3:
            pattern_confidence = 30
            return None

        direction_changes = []
        for i in range(1, len(directions)):
            change = abs((directions[i] - directions[i - 1] + 180) % 360 - 180)
            direction_changes.append(change)

        if direction_changes:
            avg_change = np.mean(direction_changes)
            pattern_confidence = max(30, min(90, 100 - avg_change * 2))
        else:
            pattern_confidence = 30

        if len(directions) >= 2:
            return np.mean(directions[-2:])
        return None

    def predict_vehicle_path(history):
        if len(history) < 3:
            return [target.position]

        recent = history[-min(5, len(history)) :]
        if len(recent) < 2:
            return [target.position]

        dx = recent[-1][0] - recent[-2][0]
        dy = recent[-1][1] - recent[-2][1]

        predicted = [target.position]
        current = target.position

        for i in range(min(15, prediction_length)):
            next_x = current[0] + dx
            next_y = current[1] + dy

            if env.is_valid_position(next_x, next_y):
                current = (next_x, next_y)
                predicted.append(current)
            else:
                for angle_offset in [30, -30, 60, -60, 90, -90]:
                    angle = np.arctan2(dy, dx) + angle_offset * np.pi / 180
                    test_dx = np.cos(angle) * target.speed
                    test_dy = np.sin(angle) * target.speed

                    test_x = current[0] + test_dx
                    test_y = current[1] + test_dy

                    if env.is_valid_position(test_x, test_y):
                        current = (test_x, test_y)
                        predicted.append(current)
                        dx, dy = test_dx, test_dy
                        break
                else:
                    break

        if predicted:
            pred_x = [p[0] for p in predicted]
            pred_y = [p[1] for p in predicted]
            predicted_path.set_data(pred_x, pred_y)
        return predicted

    def find_intercept_point(agent_pos, predicted_path):
        if not predicted_path or len(predicted_path) < 3:
            return target.position

        if pattern_confidence > 60:
            index = min(len(predicted_path) - 1, int(len(predicted_path) * 0.7))
        else:
            index = min(len(predicted_path) - 1, int(len(predicted_path) * 0.5))
        return predicted_path[index]

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.grid(True)
    ax.set_title("JAX-Accelerated Multi-Agent Pursuit")

    frame_count = 0
    agent1_active = False
    pursuit_active = False
    interceptor_deployed = False
    agent1_path_index = 0
    agent2_path_index = 0
    update_frequency = 3
    prediction_length = 15
    pattern_confidence = 30

    def init():
        agent1_point.set_data([], [])
        agent2_point.set_data([], [])
        target_point.set_data([], [])
        path1_line.set_data([], [])
        path2_line.set_data([], [])
        smooth_path1_line.set_data([], [])
        smooth_path2_line.set_data([], [])
        predicted_path.set_data([], [])
        target_trail.set_data([], [])
        strategy_text.set_text("")
        info_text.set_text("")
        agent1_status.set_text("")
        agent2_status.set_text("")
        time_text.set_text("")
        speed_text.set_text("")
        capture_text.set_text("")
        capture_text.set_alpha(0)
        return (
            agent1_point,
            agent2_point,
            target_point,
            path1_line,
            path2_line,
            smooth_path1_line,
            smooth_path2_line,
            predicted_path,
            target_trail,
            strategy_text,
            info_text,
            agent1_status,
            agent2_status,
            time_text,
            speed_text,
            capture_text,
        )

    def update(frame):
        nonlocal agent1_pos, agent1_path, agent1_path_index, agent1_path_progress
        nonlocal agent2_pos, agent2_path, agent2_path_index, agent2_path_progress
        nonlocal agent1_smooth_path, agent2_smooth_path
        nonlocal frame_count, agent1_active, pursuit_active, interceptor_deployed
        nonlocal target_history, target_captured

        frame_count += 1

        if target_captured:
            if capture_text.get_alpha() < 1.0:
                capture_text.set_alpha(min(1.0, capture_text.get_alpha() + 0.05))
            return (
                agent1_point,
                agent2_point,
                target_point,
                path1_line,
                path2_line,
                smooth_path1_line,
                smooth_path2_line,
                predicted_path,
                target_trail,
                strategy_text,
                info_text,
                agent1_status,
                agent2_status,
                time_text,
                speed_text,
                capture_text,
            )

        # Timing displays
        if not agent1_active:
            time_text.set_text(
                f"Agent 1 starts in: {(agent1_delay - frame_count) / 10:.1f}s"
            )
        elif not interceptor_deployed:
            time_text.set_text(
                f"Agent 2 deploys in: {(agent1_delay + agent2_delay - frame_count) / 10:.1f}s"
            )

        speed_text.set_text(
            f"Speeds - Target: {target_speed}, Agent 1: {agent1_speed}, Agent 2: {agent2_speed}"
        )

        # Target history
        target_history.append(target.position)
        if len(target_history) > 50:
            target_history = target_history[-50:]

        # Update trail
        if len(target_history) > 1:
            trail_x = [p[0] for p in target_history[-20:]]
            trail_y = [p[1] for p in target_history[-20:]]
            target_trail.set_data(trail_x, trail_y)

        target_point.set_data([target.position[0]], [target.position[1]])

        # JAX-accelerated distance calculation
        if agent1_active:
            distance_to_target = float(
                jax_distance(jnp.array(agent1_pos), jnp.array(target.position))
            )
        else:
            distance_to_target = float("inf")

        # Agent activation
        if frame_count >= agent1_delay and not agent1_active:
            agent1_active = True
            agent1_status.set_text("Agent 1: Started pursuit")

        if distance_to_target < 1.0 and agent1_active and not pursuit_active:
            pursuit_active = True
            agent1_status.set_text("Agent 1: Target fleeing!")

        if (
            pursuit_active
            and frame_count >= (agent1_delay + agent2_delay)
            and not interceptor_deployed
        ):
            interceptor_deployed = True
            agent2_pos = start
            agent2_status.set_text("Agent 2: Deploying...")

            predicted_positions = predict_vehicle_path(target_history)
            intercept_point = find_intercept_point(agent2_pos, predicted_positions)

            agent2_path = pathfinder.find_path(agent2_pos, intercept_point)
            agent2_path_index = 0
            agent2_path_progress = 0.0

        # Target movement
        if pursuit_active:
            if interceptor_deployed and agent2_pos is not None:
                # JAX-accelerated distance calculations
                agent_positions = jnp.array([agent1_pos, agent2_pos])
                distances = jax_batch_distances(
                    agent_positions, jnp.array(target.position)
                )
                dist1, dist2 = float(distances[0]), float(distances[1])

                closest_agent_pos = agent1_pos if dist1 < dist2 else agent2_pos
                fleeing_from = "Agent 1" if dist1 < dist2 else "Agent 2"
            else:
                closest_agent_pos = agent1_pos
                fleeing_from = "Agent 1"

            target.update_position(closest_agent_pos)

            strategy_text.set_text(f"Strategy: {target.last_strategy}")
            info_text.set_text(
                f"Fleeing from: {fleeing_from}, Pattern confidence: {pattern_confidence}%"
            )

            # Path updates
            if frame_count % update_frequency == 0:
                predicted_positions = predict_vehicle_path(target_history)

                if agent1_active:
                    new_path = pathfinder.find_path(agent1_pos, target.position)
                    if new_path and len(new_path) > 2:
                        agent1_path = new_path
                        agent1_path_index = 0
                        agent1_path_progress = 0.0
                        if use_smooth_trajectories:
                            agent1_smooth_path = smooth_path_with_bezier(agent1_path)

                if interceptor_deployed and agent2_pos is not None:
                    if frame_count % (update_frequency * 2) == 0:
                        intercept_point = find_intercept_point(
                            agent2_pos, predicted_positions
                        )
                        new_path = pathfinder.find_path(agent2_pos, intercept_point)
                        if new_path and len(new_path) > 2:
                            agent2_path = new_path
                            agent2_path_index = 0
                            agent2_path_progress = 0.0
                            if use_smooth_trajectories:
                                agent2_smooth_path = smooth_path_with_bezier(
                                    agent2_path
                                )

        # JAX-accelerated agent movement
        if agent1_active and agent1_path:
            if use_smooth_trajectories and agent1_smooth_path:
                if len(agent1_smooth_path) > 1:
                    path_array = jnp.array(agent1_smooth_path)
                    total_path_length = float(jax_path_length(path_array))
                    progress_step = agent1_speed / max(0.1, total_path_length)
                else:
                    progress_step = 0.1

                agent1_path_progress = min(1.0, agent1_path_progress + progress_step)
                agent1_pos = interpolate_agent_position(
                    agent1_path, agent1_path_progress, True, env
                )

                if agent1_path_progress >= 1.0:
                    new_path = pathfinder.find_path(agent1_pos, target.position)
                    if new_path:
                        agent1_path = new_path
                        agent1_path_progress = 0.0
                        agent1_smooth_path = smooth_path_with_bezier(agent1_path)
                    agent1_status.set_text("Agent 1: Recalculating")
                else:
                    agent1_status.set_text("Agent 1: Pursuing (constant speed)")
            else:
                # Discrete movement with JAX acceleration
                if agent1_path and len(agent1_path) > 1:
                    path_array = jnp.array(agent1_path)
                    total_length = float(jax_path_length(path_array))

                    if total_length > 0:
                        indices_per_speed = len(agent1_path) / total_length
                        step_size = max(1, int(agent1_speed * indices_per_speed))
                    else:
                        step_size = 1

                    if agent1_path_index < len(agent1_path):
                        agent1_pos = agent1_path[
                            min(agent1_path_index, len(agent1_path) - 1)
                        ]
                        agent1_path_index += step_size
                        agent1_status.set_text("Agent 1: Pursuing (constant speed)")
                    else:
                        new_path = pathfinder.find_path(agent1_pos, target.position)
                        if new_path:
                            agent1_path = new_path
                            agent1_path_index = 0
                            if use_smooth_trajectories:
                                agent1_smooth_path = smooth_path_with_bezier(
                                    agent1_path
                                )
                        agent1_status.set_text("Agent 1: Recalculating")

        agent1_point.set_data([agent1_pos[0]], [agent1_pos[1]])

        # Update path visualizations
        if agent1_path:
            path1_x = [p[0] for p in agent1_path]
            path1_y = [p[1] for p in agent1_path]
            path1_line.set_data(path1_x, path1_y)

            if use_smooth_trajectories and agent1_smooth_path:
                smooth1_x = [p[0] for p in agent1_smooth_path]
                smooth1_y = [p[1] for p in agent1_smooth_path]
                smooth_path1_line.set_data(smooth1_x, smooth1_y)
            else:
                smooth_path1_line.set_data([], [])

        # Agent 2 movement (similar JAX acceleration)
        if interceptor_deployed and agent2_pos is not None:
            if agent2_path:
                if use_smooth_trajectories and agent2_smooth_path:
                    if len(agent2_smooth_path) > 1:
                        path_array = jnp.array(agent2_smooth_path)
                        total_path_length = float(jax_path_length(path_array))
                        progress_step = agent2_speed / max(0.1, total_path_length)
                    else:
                        progress_step = 0.1

                    agent2_path_progress = min(
                        1.0, agent2_path_progress + progress_step
                    )
                    agent2_pos = interpolate_agent_position(
                        agent2_path, agent2_path_progress, True, env
                    )

                    if agent2_path_progress >= 1.0:
                        predicted_positions = predict_vehicle_path(target_history)
                        intercept_point = find_intercept_point(
                            agent2_pos, predicted_positions
                        )
                        new_path = pathfinder.find_path(agent2_pos, intercept_point)
                        if new_path:
                            agent2_path = new_path
                            agent2_path_progress = 0.0
                            agent2_smooth_path = smooth_path_with_bezier(agent2_path)
                        agent2_status.set_text("Agent 2: Updating interception")
                    else:
                        agent2_status.set_text(
                            f"Agent 2: Intercepting (constant speed, confidence: {pattern_confidence}%)"
                        )
                else:
                    if agent2_path and len(agent2_path) > 1:
                        path_array = jnp.array(agent2_path)
                        total_length = float(jax_path_length(path_array))

                        if total_length > 0:
                            indices_per_speed = len(agent2_path) / total_length
                            step_size = max(1, int(agent2_speed * indices_per_speed))
                        else:
                            step_size = 1

                        if agent2_path_index < len(agent2_path):
                            agent2_pos = agent2_path[
                                min(agent2_path_index, len(agent2_path) - 1)
                            ]
                            agent2_path_index += step_size
                            agent2_status.set_text(
                                f"Agent 2: Intercepting (constant speed, confidence: {pattern_confidence}%)"
                            )
                        else:
                            predicted_positions = predict_vehicle_path(target_history)
                            intercept_point = find_intercept_point(
                                agent2_pos, predicted_positions
                            )
                            new_path = pathfinder.find_path(agent2_pos, intercept_point)
                            if new_path:
                                agent2_path = new_path
                                agent2_path_index = 0
                                if use_smooth_trajectories:
                                    agent2_smooth_path = smooth_path_with_bezier(
                                        agent2_path
                                    )
                            agent2_status.set_text("Agent 2: Updating interception")

            agent2_point.set_data([agent2_pos[0]], [agent2_pos[1]])

            if agent2_path:
                path2_x = [p[0] for p in agent2_path]
                path2_y = [p[1] for p in agent2_path]
                path2_line.set_data(path2_x, path2_y)

                if use_smooth_trajectories and agent2_smooth_path:
                    smooth2_x = [p[0] for p in agent2_smooth_path]
                    smooth2_y = [p[1] for p in agent2_smooth_path]
                    smooth_path2_line.set_data(smooth2_x, smooth2_y)
                else:
                    smooth_path2_line.set_data([], [])
        else:
            agent2_point.set_data([], [])
            path2_line.set_data([], [])
            smooth_path2_line.set_data([], [])
            if not interceptor_deployed:
                agent2_status.set_text("Agent 2: Waiting")

        # JAX-accelerated capture detection
        if pursuit_active and interceptor_deployed and agent2_pos is not None:
            if is_target_captured_jax():
                target_captured = True
                capture_text.set_text("TARGET CAPTURED!")
                capture_text.set_alpha(0.1)
                agent1_status.set_text("Agent 1: Target secured!")
                agent2_status.set_text("Agent 2: Target secured!")
                info_text.set_text("Mission accomplished")

        return (
            agent1_point,
            agent2_point,
            target_point,
            path1_line,
            path2_line,
            smooth_path1_line,
            smooth_path2_line,
            predicted_path,
            target_trail,
            strategy_text,
            info_text,
            agent1_status,
            agent2_status,
            time_text,
            speed_text,
            capture_text,
        )

    ani = animation.FuncAnimation(
        fig, update, frames=max_frames, init_func=init, interval=50, blit=True
    )

    plt.tight_layout()
    plt.show()
    return ani


def create_sample_neighborhood(width=20, height=20):
    """Create JAX-accelerated environment"""
    env = JAXEnvironment(width, height)

    # Add some blocks as obstacles
    env.add_block(0.5, 2, 7, 5)
    env.add_block(1, 8, 6.5, 5)
    env.add_block(1, 14, 6.5, 5)
    env.add_block(8, 2, 5, 5)
    env.add_block(8, 8, 5, 5)
    env.add_block(8, 14, 5, 5)
    env.add_block(14, 2, 5, 5)
    env.add_block(14, 8, 5, 5)
    env.add_block(14, 14, 5, 5)
    env.add_block(6.5, 15, 6, 6)
    return env


def main():
    """JAX-accelerated main function"""
    env = create_sample_neighborhood()
    start = (0.5, 0.5)
    target_position = (16.7, 13.6)
    pathfinder = JAXAStar(env)

    try:
        animate_multi_agent_pursuit(
            env,
            pathfinder,
            start,
            target_position,
            agent1_delay=30,
            agent2_delay=100,
            target_speed=0.25,
            agent1_speed=0.2,
            agent2_speed=0.3,
            use_smooth_trajectories=True,
        )
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
