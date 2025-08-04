import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib.animation as animation
import random

# Add these imports for Bezier curves
from scipy.interpolate import splprep, splev

#### REFERENCES FOR ARTICLE ####

# https://arxiv.org/pdf/2205.07772
# https://arxiv.org/abs/1705.08926


class Environment:
    def __init__(self, width, height, block_size=1.0):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.obstacles = []  # List of (x,y,width,height) tuples for blocks

    def add_block(self, x, y, width, height):
        """Add a block obstacle at the specified position"""
        self.obstacles.append((x, y, width, height))

    def is_valid_position(self, x, y):
        """Check if a position is valid (within bounds and not in an obstacle)"""
        if x < 0 or x > self.width or y < 0 or y > self.height:
            return False

        # Check if position is inside any obstacle
        for ox, oy, ow, oh in self.obstacles:
            if (ox <= x <= ox + ow) and (oy <= y <= oy + oh):
                return False

        return True

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


class AStar:
    def __init__(self, environment):
        self.env = environment

    def heuristic(self, a, b):
        """Euclidean distance heuristic"""
        return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)

    def get_neighbors(self, current, step_size=0.5):
        """Get valid neighboring points in continuous space"""
        x, y = current
        neighbors = []

        # Generate neighbors in 8 directions
        for dx in [-step_size, 0, step_size]:
            for dy in [-step_size, 0, step_size]:
                if dx == 0 and dy == 0:
                    continue

                nx, ny = x + dx, y + dy
                if self.env.is_valid_position(nx, ny):
                    neighbors.append((nx, ny))

        return neighbors

    def find_path(self, start, goal):
        """Find path using A* algorithm"""
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
        dx = self.position[0] - agent_position[0]
        dy = self.position[1] - agent_position[1]

        # Normalize and scale by speed
        distance = max(0.1, np.sqrt(dx**2 + dy**2))  # Avoid division by zero
        dx = dx / distance * self.speed
        dy = dy / distance * self.speed

        # Try to move in direction away from agent
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # If that position is valid, move there
        if self.env.is_valid_position(new_x, new_y):
            self.position = (new_x, new_y)
            return

        # If direct path is blocked, try alternative directions
        possible_moves = []
        for angle_offset in [0, 30, -30, 60, -60, 90, -90]:
            angle = np.arctan2(dy, dx) + angle_offset * np.pi / 180
            new_dx = np.cos(angle) * self.speed
            new_dy = np.sin(angle) * self.speed

            new_x = self.position[0] + new_dx
            new_y = self.position[1] + new_dy

            if self.env.is_valid_position(new_x, new_y):
                distance = np.sqrt(
                    (new_x - agent_position[0]) ** 2 + (new_y - agent_position[1]) ** 2
                )
                possible_moves.append((distance, (new_x, new_y)))

        # Choose the move that keeps furthest from agent
        if possible_moves:
            possible_moves.sort(reverse=True)  # Sort by distance (descending)
            self.position = possible_moves[0][1]


class SmartFleeingTarget:
    def __init__(self, env, initial_position, speed=0.3):
        self.env = env
        self.position = initial_position
        self.speed = speed
        self.last_strategy = "opposite"  # Track which strategy was last used

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
        """Move in the opposite direction from the agent"""
        # Vector from agent to target
        dx = self.position[0] - agent_position[0]
        dy = self.position[1] - agent_position[1]

        # Normalize and scale by speed
        distance = max(0.1, np.sqrt(dx**2 + dy**2))
        dx = dx / distance * self.speed
        dy = dy / distance * self.speed

        return dx, dy

    def perpendicular_direction_strategy(self, agent_position):
        """Move perpendicular to the agent's approach direction"""
        # Vector from agent to target
        dx = self.position[0] - agent_position[0]
        dy = self.position[1] - agent_position[1]

        # Get perpendicular vectors (two options)
        perp1_dx, perp1_dy = -dy, dx  # 90 degrees clockwise
        perp2_dx, perp2_dy = dy, -dx  # 90 degrees counter-clockwise

        # Normalize and scale both options
        length1 = max(0.1, np.sqrt(perp1_dx**2 + perp1_dy**2))
        perp1_dx = perp1_dx / length1 * self.speed
        perp1_dy = perp1_dy / length1 * self.speed

        length2 = max(0.1, np.sqrt(perp2_dx**2 + perp2_dy**2))
        perp2_dx = perp2_dx / length2 * self.speed
        perp2_dy = perp2_dy / length2 * self.speed

        # Check which perpendicular direction keeps the target on the grid
        new_x1 = self.position[0] + perp1_dx
        new_y1 = self.position[1] + perp1_dy
        new_x2 = self.position[0] + perp2_dx
        new_y2 = self.position[1] + perp2_dy

        valid1 = self.env.is_valid_position(new_x1, new_y1)
        valid2 = self.env.is_valid_position(new_x2, new_y2)

        if valid1 and valid2:
            # Choose the perpendicular direction that leads further from edges
            edges = self.get_edge_proximity()
            # Simple heuristic: if near left/right edge, prefer vertical movement
            # if near top/bottom edge, prefer horizontal movement
            if "left" in edges or "right" in edges:
                if abs(perp1_dx) < abs(perp1_dy):
                    return perp1_dx, perp1_dy
                else:
                    return perp2_dx, perp2_dy
            else:
                if abs(perp1_dy) < abs(perp1_dx):
                    return perp1_dx, perp1_dy
                else:
                    return perp2_dx, perp2_dy
        elif valid1:
            return perp1_dx, perp1_dy
        elif valid2:
            return perp2_dx, perp2_dy
        else:
            # Neither perpendicular direction works, fall back to opposite
            return self.opposite_direction_strategy(agent_position)

    def random_direction_strategy(self):
        """Move in a random valid direction"""
        # Try 8 evenly spaced directions
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        # Shuffle angles for randomness
        np.random.shuffle(angles)

        for angle in angles:
            dx = np.cos(angle) * self.speed
            dy = np.sin(angle) * self.speed

            new_x = self.position[0] + dx
            new_y = self.position[1] + dy

            if self.env.is_valid_position(new_x, new_y):
                return dx, dy

        # If no direction works, don't move
        return 0, 0

    def update_position(self, agent_position):
        """Update position using smart fleeing strategy"""
        # Determine which strategy to use
        near_edge = self.is_near_edge()

        # Get movement vector based on selected strategy
        if near_edge:
            dx, dy = self.perpendicular_direction_strategy(agent_position)
            self.last_strategy = "perpendicular"
        else:
            # Occasionally use random movement (20% chance)
            if random.random() < 0.2:
                dx, dy = self.random_direction_strategy()
                self.last_strategy = "random"
            else:
                dx, dy = self.opposite_direction_strategy(agent_position)
                self.last_strategy = "opposite"

        # Calculate new position
        new_x = self.position[0] + dx
        new_y = self.position[1] + dy

        # If position is valid, move there
        if self.env.is_valid_position(new_x, new_y):
            self.position = (new_x, new_y)
            return

        # If primary strategy failed, try others in sequence
        strategies = ["opposite", "perpendicular", "random"]
        # Remove the strategy we just tried
        strategies.remove(self.last_strategy)

        for strategy in strategies:
            if strategy == "opposite":
                dx, dy = self.opposite_direction_strategy(agent_position)
            elif strategy == "perpendicular":
                dx, dy = self.perpendicular_direction_strategy(agent_position)
            else:  # random
                dx, dy = self.random_direction_strategy()

            new_x = self.position[0] + dx
            new_y = self.position[1] + dy

            if self.env.is_valid_position(new_x, new_y):
                self.position = (new_x, new_y)
                self.last_strategy = strategy
                return


# Add this new class for a vehicle-like target
class VehicleTarget:
    def __init__(self, env, initial_position, speed=0.5, inertia=0.8):
        """
        Vehicle-like target that moves more directly

        Parameters:
        - env: Environment with obstacles
        - initial_position: Starting position
        - speed: Movement speed (higher = faster)
        - inertia: How much previous direction influences new direction (0-1)
        """
        self.env = env
        self.position = initial_position
        self.speed = speed
        self.inertia = inertia
        self.direction = (0, 0)  # Current direction vector
        self.last_strategy = "straight"

    def update_position(self, agent_position):
        """Move away from agent with vehicle-like movement patterns"""
        # Calculate vector from agent to target
        dx = self.position[0] - agent_position[0]
        dy = self.position[1] - agent_position[1]

        # Normalize the vector
        distance = max(0.1, np.sqrt(dx**2 + dy**2))
        dx = dx / distance
        dy = dy / distance

        # Apply inertia - blend new direction with previous direction
        if self.direction == (0, 0):  # First movement
            new_dir_x = dx
            new_dir_y = dy
        else:
            prev_dir_x, prev_dir_y = self.direction
            new_dir_x = self.inertia * prev_dir_x + (1 - self.inertia) * dx
            new_dir_y = self.inertia * prev_dir_y + (1 - self.inertia) * dy

            # Re-normalize
            mag = max(0.1, np.sqrt(new_dir_x**2 + new_dir_y**2))
            new_dir_x /= mag
            new_dir_y /= mag

        # Scale by speed
        move_x = new_dir_x * self.speed
        move_y = new_dir_y * self.speed

        # Try to move in that direction
        new_x = self.position[0] + move_x
        new_y = self.position[1] + move_y

        # If position is valid, update position and direction
        if self.env.is_valid_position(new_x, new_y):
            self.position = (new_x, new_y)
            self.direction = (new_dir_x, new_dir_y)
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
            angle_rad = angle_offset * np.pi / 180
            rot_x = new_dir_x * np.cos(angle_rad) - new_dir_y * np.sin(angle_rad)
            rot_y = new_dir_x * np.sin(angle_rad) + new_dir_y * np.cos(angle_rad)

            test_x = self.position[0] + rot_x * self.speed
            test_y = self.position[1] + rot_y * self.speed

            if self.env.is_valid_position(test_x, test_y):
                self.position = (test_x, test_y)
                self.direction = (rot_x, rot_y)
                self.last_strategy = f"turning_{angle_offset}"
                return


def smooth_path_with_bezier(path, smoothing_factor=0.05, num_points=None):
    """
    Optimized smooth path generation with fewer points for better performance

    Parameters:
    - path: List of (x, y) tuples representing the original path
    - smoothing_factor: How much to smooth (reduced default for performance)
    - num_points: Number of points in the smoothed path (auto-calculated if None)

    Returns:
    - List of (x, y) tuples representing the smoothed path
    """
    if len(path) < 3:
        return path

    # Auto-calculate fewer points for better performance
    if num_points is None:
        num_points = min(50, len(path) * 2)  # Much fewer points

    # Convert path to arrays
    path_array = np.array(path)
    x = path_array[:, 0]
    y = path_array[:, 1]

    try:
        # Create parametric representation with reduced smoothing
        tck, u = splprep([x, y], s=smoothing_factor, per=False)

        # Generate smooth curve with fewer points
        u_new = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_new, tck)

        # Convert back to list of tuples
        smooth_path = list(zip(x_smooth, y_smooth))

        return smooth_path
    except:
        # If spline fails, fall back to original path
        return path


def interpolate_agent_position(path, progress, use_smooth=True, env=None):
    """
    Optimized agent position interpolation with performance improvements
    """
    if not path or len(path) == 0:
        return (0, 0)

    if len(path) == 1:
        return path[0]

    if use_smooth and len(path) >= 3:
        # Generate smooth path with reduced points
        smooth_path = smooth_path_with_bezier(
            path, smoothing_factor=0.05, num_points=min(30, len(path) * 2)
        )

        # Quick safety check - only validate every few points for performance
        if env is not None:
            safe_smooth_path = []
            check_interval = max(1, len(smooth_path) // 10)  # Check every 10th point

            for i, point in enumerate(smooth_path):
                if i % check_interval == 0:  # Only check some points
                    if not env.is_valid_position(point[0], point[1]):
                        break
                safe_smooth_path.append(point)

            if len(safe_smooth_path) < 3:
                return interpolate_agent_position(path, progress, False, env)

            smooth_path = safe_smooth_path

        # Find position along smooth path
        smooth_index = progress * (len(smooth_path) - 1)
        lower_idx = int(smooth_index)
        upper_idx = min(lower_idx + 1, len(smooth_path) - 1)

        if lower_idx == upper_idx:
            return smooth_path[lower_idx]

        # Linear interpolation between smooth points
        t = smooth_index - lower_idx
        x = smooth_path[lower_idx][0] * (1 - t) + smooth_path[upper_idx][0] * t
        y = smooth_path[lower_idx][1] * (1 - t) + smooth_path[upper_idx][1] * t

        return (x, y)
    else:
        # Original discrete interpolation
        index = progress * (len(path) - 1)
        lower_idx = int(index)
        upper_idx = min(lower_idx + 1, len(path) - 1)

        if lower_idx == upper_idx:
            return path[lower_idx]

        t = index - lower_idx
        x = path[lower_idx][0] * (1 - t) + path[upper_idx][0] * t
        y = path[lower_idx][1] * (1 - t) + path[upper_idx][1] * t

        return (x, y)


def animate_multi_agent_pursuit(
    env,
    pathfinder,
    start,
    target_position=None,
    max_frames=500,
    agent1_delay=30,
    agent2_delay=100,
    target_speed=1.2,  # Much faster
    agent1_speed=0.8,  # Fast but slower than target
    agent2_speed=2.0,  # Much faster than both
    use_smooth_trajectories=True,
):
    """
    Multi-agent pursuit with CONSTANT speeds throughout execution
    """
    if target_position is None:
        target_position = (env.width - 0.5, env.height - 0.5)

    if not env.is_valid_position(target_position[0], target_position[1]):
        raise ValueError("Target position must be on a street, not inside a building")

    fig, ax = plt.subplots(figsize=(10, 10))

    for ox, oy, ow, oh in env.obstacles:
        rect = plt.Rectangle((ox, oy), ow, oh, color="gray")
        ax.add_patch(rect)

    target = VehicleTarget(env, target_position, speed=target_speed, inertia=0.8)
    target_history = []

    agent1_pos = start
    agent1_path = pathfinder.find_path(agent1_pos, target.position)

    # CONSTANT SPEED VARIABLES - No variable step sizes
    agent1_path_progress = 0.0
    agent2_path_progress = 0.0
    agent1_smooth_path = []
    agent2_smooth_path = []

    # Generate initial smooth path with fewer points
    if use_smooth_trajectories and agent1_path and len(agent1_path) >= 3:
        agent1_smooth_path = smooth_path_with_bezier(agent1_path)

    # CONSTANT MOVEMENT FREQUENCY - Every frame, agents move at their specified speed
    # No more variable frequencies - agents move every frame with their actual speed
    agent1_move_every_frame = True
    agent2_move_every_frame = True

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
    capture_distance = 2.5

    def is_target_captured():
        """
        Check if target is captured under two specific conditions:
        1. Target is trapped between an agent and a building (deadlocked street)
        2. Both agents are within very close proximity (radius 0.8) of the target

        EXTREMELY STRICT CONDITIONS - target must have absolutely NO escape routes
        """
        # CRITICAL: Only check capture if BOTH agents are actually deployed and active
        if not interceptor_deployed or agent2_pos is None:
            info_text.set_text("Agent 2 not deployed yet - no capture possible")
            return False

        # Calculate distances from both agents to target using ACTUAL positions
        dist1 = np.sqrt(
            (agent1_pos[0] - target.position[0]) ** 2
            + (agent1_pos[1] - target.position[1]) ** 2
        )
        dist2 = np.sqrt(
            (agent2_pos[0] - target.position[0]) ** 2
            + (agent2_pos[1] - target.position[1]) ** 2
        )

        # CONDITION 2: Both agents are within VERY close vicinity (radius 0.8) - EXTREMELY STRICT
        proximity_radius = 1.85  # Reduced from 1.0 to be even more strict
        if dist1 <= proximity_radius and dist2 <= proximity_radius:
            # Even if both agents are very close, target must have absolutely NO escape routes
            escape_routes = 0

            # Test ALL 32 directions thoroughly with very strict criteria
            for i in range(32):
                angle = i * (2 * np.pi / 32)
                dx = np.cos(angle)
                dy = np.sin(angle)

                # Target must not be able to escape even 2.0 units away (increased from 1.5)
                escape_distance = 1.5

                # Check if this direction is completely blocked
                path_blocked = False
                for step in range(
                    1, 8
                ):  # Check 7 points along the path (more thorough)
                    test_dist = escape_distance * step / 7
                    test_x = target.position[0] + dx * test_dist
                    test_y = target.position[1] + dy * test_dist

                    if not env.is_valid_position(test_x, test_y):
                        path_blocked = True
                        break

                if not path_blocked:
                    # This direction is clear - check if it leads away from BOTH agents
                    final_x = target.position[0] + dx * escape_distance
                    final_y = target.position[1] + dy * escape_distance

                    dist_to_agent1 = np.sqrt(
                        (final_x - agent1_pos[0]) ** 2 + (final_y - agent1_pos[1]) ** 2
                    )
                    dist_to_agent2 = np.sqrt(
                        (final_x - agent2_pos[0]) ** 2 + (final_y - agent2_pos[1]) ** 2
                    )

                    # Escape is valid only if it leads MUCH further from BOTH agents (increased threshold)
                    if dist_to_agent1 > dist1 * 1.8 and dist_to_agent2 > dist2 * 1.8:
                        escape_routes += 1

            # Only capture if absolutely NO escape routes exist
            if escape_routes == 0:
                info_text.set_text(
                    f"CAPTURED! Both agents very close and no escape routes"
                )
                return True
            else:
                info_text.set_text(
                    f"Both agents close (d1:{dist1:.1f}, d2:{dist2:.1f}) but {escape_routes} escape routes available"
                )
                return False
        else:
            # At least one agent is not close enough for proximity capture
            info_text.set_text(
                f"Proximity capture failed - Agent distances: {dist1:.1f}, {dist2:.1f} (need both ≤{proximity_radius})"
            )

        # CONDITION 1: Target trapped between agent and building - EXTREMELY STRICT
        # Check if any single agent has the target truly cornered
        min_agent_dist = min(dist1, dist2)
        closest_agent_name = "Agent 1" if dist1 < dist2 else "Agent 2"

        # At least one agent must be VERY close (within 0.8 units) to create a trap
        if min_agent_dist > 0.8:  # Reduced from 1.2 to be much more strict
            info_text.set_text(
                f"No agents close enough to trap - closest: {closest_agent_name} at {min_agent_dist:.1f} (need ≤0.8)"
            )
            return False

        # Count COMPLETELY blocked directions and valid escape routes
        blocked_directions = 0
        escape_routes = 0

        # Test 32 directions for maximum precision
        for i in range(32):
            angle = i * (2 * np.pi / 32)
            dx = np.cos(angle)
            dy = np.sin(angle)

            # Check if this direction is blocked by buildings
            escape_distance = 3.0  # Increased from 2.5 - target must be able to get even further to "escape"
            direction_blocked = False

            # Check the entire path for obstacles - more thorough checking
            for step in range(
                1, 15
            ):  # Check 14 points along the path (increased from 10)
                test_dist = escape_distance * step / 14
                test_x = target.position[0] + dx * test_dist
                test_y = target.position[1] + dy * test_dist

                if not env.is_valid_position(test_x, test_y):
                    direction_blocked = True
                    blocked_directions += 1
                    break

            if not direction_blocked:
                # This direction is clear - check if it's a valid escape
                final_x = target.position[0] + dx * escape_distance
                final_y = target.position[1] + dy * escape_distance

                # Calculate distances to BOTH agents from escape point
                escape_to_agent1 = np.sqrt(
                    (final_x - agent1_pos[0]) ** 2 + (final_y - agent1_pos[1]) ** 2
                )
                escape_to_agent2 = np.sqrt(
                    (final_x - agent2_pos[0]) ** 2 + (final_y - agent2_pos[1]) ** 2
                )

                # Valid escape only if it leads significantly away from CLOSEST agent
                closest_agent_dist = min(dist1, dist2)
                min_escape_dist = min(escape_to_agent1, escape_to_agent2)

                # Must lead at least 100% further away from closest agent (doubled from 50%)
                if min_escape_dist > closest_agent_dist * 2.0:
                    escape_routes += 1

        # EXTREMELY STRICT CAPTURE CONDITIONS:
        # 1. At least one agent within 0.8 units AND
        # 2. Target has exactly 0 escape routes AND
        # 3. At least 90% of directions are blocked by buildings (increased from 85%)

        min_blocked_threshold = int(
            32 * 0.9
        )  # 90% of directions must be blocked (29+ out of 32)

        if escape_routes == 0 and blocked_directions >= min_blocked_threshold:
            info_text.set_text(
                f"TRULY TRAPPED! {closest_agent_name} cornered target. "
                f"Distance: {min_agent_dist:.1f}, Escapes: {escape_routes}, "
                f"Blocked: {blocked_directions}/32"
            )
            return True

        # Target is NOT captured - provide detailed feedback showing ACTUAL agent positions
        info_text.set_text(
            f"NOT TRAPPED - Agent 1: {dist1:.1f}, Agent 2: {dist2:.1f}, "
            f"Escape routes: {escape_routes}, Blocked: {blocked_directions}/32"
        )

        return False

    # SIMPLIFIED PATTERN LEARNING FOR PERFORMANCE
    def learn_movement_pattern(history, min_length=5):
        """Simplified pattern learning for better performance"""
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
        """Simplified prediction for better performance"""
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
        """Simplified interception point calculation"""
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
    ax.set_title("High-Speed Multi-Agent Pursuit")

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

        # Target history (keep recent only for performance)
        target_history.append(target.position)
        if len(target_history) > 50:
            target_history = target_history[-50:]

        # Update trail
        if len(target_history) > 1:
            trail_x = [p[0] for p in target_history[-20:]]
            trail_y = [p[1] for p in target_history[-20:]]
            target_trail.set_data(trail_x, trail_y)

        target_point.set_data([target.position[0]], [target.position[1]])

        distance_to_target = np.sqrt(
            (agent1_pos[0] - target.position[0]) ** 2
            + (agent1_pos[1] - target.position[1]) ** 2
        )

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
                dist1 = np.sqrt(
                    (agent1_pos[0] - target.position[0]) ** 2
                    + (agent1_pos[1] - target.position[1]) ** 2
                )
                dist2 = np.sqrt(
                    (agent2_pos[0] - target.position[0]) ** 2
                    + (agent2_pos[1] - target.position[1]) ** 2
                )

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

            # FASTER PATH UPDATES: Update paths more frequently
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

        # CONSTANT SPEED AGENT 1 MOVEMENT - Move at exact speed every frame
        if agent1_active and agent1_path:
            if use_smooth_trajectories and agent1_smooth_path:
                # CONSTANT PROGRESS STEP based on actual speed
                # Calculate the exact progress step needed for the specified speed
                if len(agent1_smooth_path) > 1:
                    total_path_length = 0
                    for i in range(1, len(agent1_smooth_path)):
                        dx = agent1_smooth_path[i][0] - agent1_smooth_path[i - 1][0]
                        dy = agent1_smooth_path[i][1] - agent1_smooth_path[i - 1][1]
                        total_path_length += np.sqrt(dx * dx + dy * dy)

                    # Progress step = speed / total_length
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
                # CONSTANT DISCRETE MOVEMENT - calculate step based on speed
                if agent1_path and len(agent1_path) > 1:
                    # Calculate total path length
                    total_length = 0
                    for i in range(1, len(agent1_path)):
                        dx = agent1_path[i][0] - agent1_path[i - 1][0]
                        dy = agent1_path[i][1] - agent1_path[i - 1][1]
                        total_length += np.sqrt(dx * dx + dy * dy)

                    # Calculate how many indices to advance based on speed
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

        # CONSTANT SPEED AGENT 2 MOVEMENT - Move at exact speed every frame
        if interceptor_deployed and agent2_pos is not None:
            if agent2_path:
                if use_smooth_trajectories and agent2_smooth_path:
                    # CONSTANT PROGRESS STEP based on actual speed
                    if len(agent2_smooth_path) > 1:
                        total_path_length = 0
                        for i in range(1, len(agent2_smooth_path)):
                            dx = agent2_smooth_path[i][0] - agent2_smooth_path[i - 1][0]
                            dy = agent2_smooth_path[i][1] - agent2_smooth_path[i - 1][1]
                            total_path_length += np.sqrt(dx * dx + dy * dy)

                        # Progress step = speed / total_length
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
                    # CONSTANT DISCRETE MOVEMENT
                    if agent2_path and len(agent2_path) > 1:
                        # Calculate total path length
                        total_length = 0
                        for i in range(1, len(agent2_path)):
                            dx = agent2_path[i][0] - agent2_path[i - 1][0]
                            dy = agent2_path[i][1] - agent2_path[i - 1][1]
                            total_length += np.sqrt(dx * dx + dy * dy)

                        # Calculate how many indices to advance based on speed
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

        # Capture detection
        if pursuit_active and interceptor_deployed and agent2_pos is not None:
            if is_target_captured():
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


def create_sample_neighborhood(width=50, height=50):
    env = Environment(width, height)

    # Scale up the existing obstacles proportionally
    # Original blocks scaled from 20x20 to 50x50 (2.5x scale factor)
    scale = 2.5

    # Add scaled blocks as obstacles
    env.add_block(0.5 * scale, 2 * scale, 7 * scale, 5 * scale)
    env.add_block(1 * scale, 8 * scale, 6.5 * scale, 5 * scale)
    env.add_block(1 * scale, 14 * scale, 6.5 * scale, 5 * scale)
    env.add_block(8 * scale, 2 * scale, 5 * scale, 5 * scale)
    env.add_block(8 * scale, 8 * scale, 5 * scale, 5 * scale)
    env.add_block(8 * scale, 14 * scale, 5 * scale, 5 * scale)
    env.add_block(14 * scale, 2 * scale, 5 * scale, 5 * scale)
    env.add_block(14 * scale, 8 * scale, 5 * scale, 5 * scale)
    env.add_block(14 * scale, 14 * scale, 5 * scale, 5 * scale)
    env.add_block(6.5 * scale, 15 * scale, 6 * scale, 6 * scale)

    # Add additional buildings to fill the larger space
    env.add_block(25, 5, 8, 6)
    env.add_block(35, 5, 8, 6)
    env.add_block(25, 15, 8, 6)
    env.add_block(35, 15, 8, 6)
    env.add_block(25, 25, 8, 6)
    env.add_block(35, 25, 8, 6)
    env.add_block(25, 35, 8, 6)
    env.add_block(35, 35, 8, 6)
    env.add_block(5, 30, 10, 8)
    env.add_block(15, 40, 12, 7)
    env.add_block(30, 40, 10, 8)
    env.add_block(42, 30, 6, 12)

    return env


# UPDATED MAIN FUNCTION WITH CONSISTENT SPEEDS
def main():
    env = create_sample_neighborhood()
    start = (0.5, 0.5)
    target_position = (33, 33)
    pathfinder = AStar(env)

    try:
        animate_multi_agent_pursuit(
            env,
            pathfinder,
            start,
            target_position,
            agent1_delay=30,
            agent2_delay=100,
            target_speed=0.25,  # Constant speed throughout
            agent1_speed=0.2,  # Constant speed throughout
            agent2_speed=0.3,  # Constant speed throughout
            use_smooth_trajectories=True,
        )
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
