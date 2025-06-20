import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib.animation as animation
import random


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
        """
        # If all else fails, try reversing direction
        reverse_x = -new_dir_x * self.speed * 0.5  # Move slower when reversing
        reverse_y = -new_dir_y * self.speed * 0.5

        test_x = self.position[0] + reverse_x
        test_y = self.position[1] + reverse_y

        if self.env.is_valid_position(test_x, test_y):
            self.position = (test_x, test_y)
            self.direction = (-new_dir_x, -new_dir_y)
            self.last_strategy = "reversing"""


def animate_multi_agent_pursuit(
    env,
    pathfinder,
    start,
    target_position=None,
    max_frames=500,
    agent1_delay=30,
    agent2_delay=100,
    target_speed=0.5,
    agent1_speed=0.4,
    agent2_speed=0.7,
):
    """
    Animate two agents pursuing a fleeing target

    Parameters:
    - env: The environment with obstacles
    - pathfinder: The A* pathfinding algorithm
    - start: Starting position for both agents
    - target_position: Initial position for target
    - max_frames: Maximum animation frames
    - agent1_delay: Frames to wait before agent 1 starts pursuit (3s = 30 frames)
    - agent2_delay: Frames to wait before deploying agent 2 (10s = 100 frames)
    - target_speed: Movement speed for target
    - agent1_speed: Movement speed for agent 1
    - agent2_speed: Movement speed for agent 2
    """
    # If no target position is provided, use top right corner
    if target_position is None:
        target_position = (env.width - 0.5, env.height - 0.5)

    # Validate that target position is valid
    if not env.is_valid_position(target_position[0], target_position[1]):
        raise ValueError("Target position must be on a street, not inside a building")

    fig, ax = plt.subplots(figsize=(10, 10))

    # Set up environment visualization
    for ox, oy, ow, oh in env.obstacles:
        rect = plt.Rectangle((ox, oy), ow, oh, color="gray")
        ax.add_patch(rect)

    # Initialize target and agents
    target = VehicleTarget(env, target_position, speed=target_speed, inertia=0.8)

    # Store target's position history for predictions
    target_history = []

    # First agent starts at position but doesn't move yet
    agent1_pos = start
    agent1_path = pathfinder.find_path(agent1_pos, target.position)

    # For speed adjustment, we'll modify how often we move along the path
    agent1_move_frequency = max(
        1, int(10 / agent1_speed)
    )  # Lower number = faster movement
    agent2_move_frequency = max(1, int(10 / agent2_speed))

    # Second agent (interceptor) waits until deployed
    agent2_pos = None
    agent2_path = None

    # Create plot elements
    (agent1_point,) = ax.plot([], [], "ro", markersize=8)  # Red circle for first agent
    (agent2_point,) = ax.plot(
        [], [], "mo", markersize=8
    )  # Magenta circle for second agent
    (target_point,) = ax.plot([], [], "bs", markersize=8)  # Blue square for target
    (path1_line,) = ax.plot([], [], "r-", alpha=0.4)  # Path line for first agent
    (path2_line,) = ax.plot([], [], "m-", alpha=0.4)  # Path line for second agent
    (predicted_path,) = ax.plot([], [], "g--", alpha=0.6)  # Predicted target path
    (target_trail,) = ax.plot([], [], "b:", alpha=0.4)  # Target's movement trail

    # Text elements
    strategy_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, color="black")
    info_text = ax.text(0.02, 0.94, "", transform=ax.transAxes, color="black")
    agent1_status = ax.text(0.02, 0.90, "", transform=ax.transAxes, color="red")
    agent2_status = ax.text(0.02, 0.86, "", transform=ax.transAxes, color="magenta")
    time_text = ax.text(0.02, 0.82, "", transform=ax.transAxes, color="blue")
    speed_text = ax.text(0.02, 0.78, "", transform=ax.transAxes, color="green")

    # Add a new text element for capture status
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

    # Flag to track if target has been captured
    target_captured = False

    # Parameters for capture detection
    capture_distance = 2.5  # How close agents need to be to capture target
    min_angle_diff = 90  # Minimum angle difference between agents to consider target cornered (degrees)

    def is_target_captured():
        """Check if target is effectively captured by both agents"""
        if not interceptor_deployed or agent2_pos is None:
            return False

        # Check if both agents are close enough to the target
        dist1 = np.sqrt(
            (agent1_pos[0] - target.position[0]) ** 2
            + (agent1_pos[1] - target.position[1]) ** 2
        )
        dist2 = np.sqrt(
            (agent2_pos[0] - target.position[0]) ** 2
            + (agent2_pos[1] - target.position[1]) ** 2
        )

        if dist1 > capture_distance or dist2 > capture_distance:
            return False  # At least one agent is too far away

        # Check if agents are on opposite sides (or at least at a significant angle)
        angle1 = (
            np.arctan2(
                agent1_pos[1] - target.position[1], agent1_pos[0] - target.position[0]
            )
            * 180
            / np.pi
        )
        angle2 = (
            np.arctan2(
                agent2_pos[1] - target.position[1], agent2_pos[0] - target.position[0]
            )
            * 180
            / np.pi
        )

        angle_diff = abs((angle1 - angle2 + 180) % 360 - 180)

        # Check if target is near an obstacle that limits escape options
        obstacle_nearby = False
        for ox, oy, ow, oh in env.obstacles:
            # Calculate minimum distance to obstacle edges
            dx = max(ox - target.position[0], 0, target.position[0] - (ox + ow))
            dy = max(oy - target.position[1], 0, target.position[1] - (oy + oh))
            dist_to_obstacle = np.sqrt(dx * dx + dy * dy)

            if dist_to_obstacle < 2.0:  # Target is near an obstacle
                obstacle_nearby = True
                break

        # Check for available escape routes
        escape_routes = 0
        for angle in range(0, 360, 45):  # Check 8 directions
            rad = angle * np.pi / 180
            test_x = target.position[0] + np.cos(rad) * 1.5
            test_y = target.position[1] + np.sin(rad) * 1.5

            if env.is_valid_position(test_x, test_y):
                # Calculate distance from this point to both agents
                d1 = np.sqrt(
                    (test_x - agent1_pos[0]) ** 2 + (test_y - agent1_pos[1]) ** 2
                )
                d2 = np.sqrt(
                    (test_x - agent2_pos[0]) ** 2 + (test_y - agent2_pos[1]) ** 2
                )

                if d1 > 1.0 and d2 > 1.0:  # If point is not too close to either agent
                    escape_routes += 1

        # Target is captured if:
        # 1. Both agents are close AND
        # 2. Either: Agents are at a significant angle from each other OR
        #    Target is near an obstacle AND has few escape routes
        return (angle_diff > min_angle_diff) or (obstacle_nearby and escape_routes <= 3)

    # Set plot limits and styling
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.grid(True)
    ax.set_title("Multi-Agent Pursuit with Vehicle Target")

    # For tracking simulation state
    frame_count = 0
    agent1_active = False
    pursuit_active = False  # True when target starts fleeing
    interceptor_deployed = False
    agent1_path_index = 0
    agent2_path_index = 0
    update_frequency = 5  # How often to recalculate paths

    # Pattern learning parameters
    prediction_length = 30  # How many steps to predict ahead
    history_length = 15  # How many historical positions to use
    pattern_confidence = 0  # Confidence in pattern prediction (0-100)

    def learn_movement_pattern(history, min_length=10):
        """Learn and analyze target's movement patterns"""
        nonlocal pattern_confidence

        if len(history) < min_length:
            pattern_confidence = 0
            return None

        # Calculate consistency in movement direction
        directions = []
        for i in range(1, len(history)):
            dx = history[i][0] - history[i - 1][0]
            dy = history[i][1] - history[i - 1][1]

            if abs(dx) < 0.01 and abs(dy) < 0.01:  # Skip if barely moved
                continue

            angle = np.arctan2(dy, dx) * 180 / np.pi
            directions.append(angle)

        if len(directions) < 5:
            pattern_confidence = 0
            return None

        # Calculate standard deviation of direction changes
        direction_changes = []
        for i in range(1, len(directions)):
            change = (directions[i] - directions[i - 1] + 180) % 360 - 180
            direction_changes.append(abs(change))

        if not direction_changes:
            pattern_confidence = 0
            return None

        std_dev = np.std(direction_changes)

        # High std_dev = erratic movement, low std_dev = consistent movement
        consistency = max(0, 100 - std_dev * 2)

        # Look for repeated patterns in direction changes
        pattern_detected = False
        pattern_length = 0

        # Simple pattern detection - look for repeating sequences of direction
        if len(directions) >= 8:
            for seq_len in range(2, 5):  # Look for patterns of length 2-4
                for i in range(len(directions) - 2 * seq_len):
                    seq1 = directions[i : i + seq_len]
                    seq2 = directions[i + seq_len : i + 2 * seq_len]

                    # Check similarity between sequences
                    similarity = (
                        sum(abs(seq1[j] - seq2[j]) < 20 for j in range(seq_len))
                        / seq_len
                    )

                    if similarity > 0.7:  # 70% similar
                        pattern_detected = True
                        pattern_length = seq_len
                        break

                if pattern_detected:
                    break

        # Calculate overall confidence in our prediction
        if pattern_detected:
            pattern_confidence = int(
                min(100, consistency + 20)
            )  # Bonus for detected pattern
        else:
            pattern_confidence = int(
                min(100, consistency * 0.7)
            )  # Penalty for no pattern

        # Return dominant direction and expected next moves
        if len(directions) >= 3:
            recent_dirs = directions[-3:]
            avg_direction = sum(recent_dirs) / len(recent_dirs)
            return avg_direction

        return None

    def predict_vehicle_path(history):
        """Predict path for a vehicle-like target with inertia"""
        if len(history) < 5:
            return [target.position]

        # For vehicle target, analyze recent trajectory and extrapolate
        # with consideration for inertia and obstacles

        # First, learn patterns
        movement_angle = learn_movement_pattern(history)

        # Calculate recent velocity vector
        recent = history[-min(len(history), 5) :]
        dx_sum, dy_sum = 0, 0

        for i in range(1, len(recent)):
            dx_sum += recent[i][0] - recent[i - 1][0]
            dy_sum += recent[i][1] - recent[i - 1][1]

        if len(recent) <= 1:
            return [target.position]

        avg_dx = dx_sum / (len(recent) - 1)
        avg_dy = dy_sum / (len(recent) - 1)

        # If we have high confidence in pattern, adjust prediction
        if pattern_confidence > 60 and movement_angle is not None:
            # Blend actual velocity with predicted pattern
            blend_ratio = pattern_confidence / 100

            # Convert angle to direction vector
            pattern_dx = np.cos(movement_angle * np.pi / 180)
            pattern_dy = np.sin(movement_angle * np.pi / 180)

            # Normalize actual velocity
            actual_mag = max(0.001, np.sqrt(avg_dx * avg_dx + avg_dy * avg_dy))
            actual_dx = avg_dx / actual_mag
            actual_dy = avg_dy / actual_mag

            # Blend vectors
            dx = actual_dx * (1 - blend_ratio) + pattern_dx * blend_ratio
            dy = actual_dy * (1 - blend_ratio) + pattern_dy * blend_ratio

            # Normalize and scale by target speed
            mag = max(0.001, np.sqrt(dx * dx + dy * dy))
            dx = dx / mag * target.speed
            dy = dy / mag * target.speed
        else:
            # Just use recent velocity
            dx = avg_dx
            dy = avg_dy

        # Generate predicted path
        predicted = [target.position]
        current = target.position

        for i in range(prediction_length):
            next_x = current[0] + dx
            next_y = current[1] + dy

            # Check if valid
            if env.is_valid_position(next_x, next_y):
                current = (next_x, next_y)
                predicted.append(current)
            else:
                # Hit obstacle, try to find alternate path by sampling angles
                found_valid = False

                # Try at most 8 different angles
                for angle_offset in [0, 30, -30, 45, -45, 60, -60, 90, -90]:
                    angle = np.arctan2(dy, dx) + angle_offset * np.pi / 180
                    test_dx = np.cos(angle) * target.speed
                    test_dy = np.sin(angle) * target.speed

                    test_x = current[0] + test_dx
                    test_y = current[1] + test_dy

                    if env.is_valid_position(test_x, test_y):
                        current = (test_x, test_dy)
                        predicted.append(current)

                        # Update direction for next iteration
                        dx = test_dx * 0.9 + dx * 0.1  # Blend with previous direction
                        dy = test_dy * 0.9 + dy * 0.1

                        found_valid = True
                        break

                if not found_valid:
                    break  # No valid move found

        # Update visualization of predicted path
        if predicted:
            pred_x = [p[0] for p in predicted]
            pred_y = [p[1] for p in predicted]
            predicted_path.set_data(pred_x, pred_y)

        return predicted

    def find_intercept_point(agent_pos, predicted_path):
        """Find the best interception point for agent 2"""
        if not predicted_path or len(predicted_path) < 3:
            return target.position

        # Choose interception point based on pattern confidence
        if pattern_confidence > 70:
            # Higher confidence = intercept further ahead
            index = int(min(len(predicted_path) - 1, len(predicted_path) * 0.8))
        elif pattern_confidence > 40:
            # Medium confidence = intercept at middle point
            index = int(min(len(predicted_path) - 1, len(predicted_path) * 0.6))
        else:
            # Low confidence = intercept closer to current position
            index = int(min(len(predicted_path) - 1, len(predicted_path) * 0.4))

        return predicted_path[index]

    def init():
        agent1_point.set_data([], [])
        agent2_point.set_data([], [])
        target_point.set_data([], [])
        path1_line.set_data([], [])
        path2_line.set_data([], [])
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
        nonlocal agent1_pos, agent1_path, agent1_path_index
        nonlocal agent2_pos, agent2_path, agent2_path_index
        nonlocal frame_count, agent1_active, pursuit_active, interceptor_deployed
        nonlocal target_history, target_captured

        # Increment frame counter
        frame_count += 1

        # If target is already captured, just show capture message and return
        if target_captured:
            if capture_text.get_alpha() < 1.0:
                capture_text.set_alpha(min(1.0, capture_text.get_alpha() + 0.05))
            return (
                agent1_point,
                agent2_point,
                target_point,
                path1_line,
                path2_line,
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

        # Update timing displays
        if not agent1_active:
            time_text.set_text(
                f"Agent 1 starts in: {(agent1_delay - frame_count) / 10:.1f}s"
            )
        elif not interceptor_deployed:
            time_text.set_text(
                f"Agent 2 deploys in: {(agent1_delay + agent2_delay - frame_count) / 10:.1f}s"
            )

        # Show speed information
        speed_text.set_text(
            f"Speeds - Target: {target_speed}, Agent 1: {agent1_speed}, Agent 2: {agent2_speed}"
        )

        # Add current target position to history
        target_history.append(target.position)
        if len(target_history) > 100:
            target_history = target_history[-100:]

        # Update target trail visualization
        if len(target_history) > 1:
            trail_x = [p[0] for p in target_history]
            trail_y = [p[1] for p in target_history]
            target_trail.set_data(trail_x, trail_y)

        # Update target visualization
        target_point.set_data([target.position[0]], [target.position[1]])

        # Check distance between agent1 and target
        distance_to_target = np.sqrt(
            (agent1_pos[0] - target.position[0]) ** 2
            + (agent1_pos[1] - target.position[1]) ** 2
        )

        # ===== AGENT 1 ACTIVATION AFTER DELAY =====
        if frame_count >= agent1_delay and not agent1_active:
            agent1_active = True
            agent1_status.set_text("Agent 1: Started pursuit")

        # ===== ACTIVATE PURSUIT =====
        if distance_to_target < 1.0 and agent1_active and not pursuit_active:
            pursuit_active = True
            agent1_status.set_text("Agent 1: Target fleeing!")

        # ===== DEPLOY AGENT 2 AFTER DELAY =====
        if (
            pursuit_active
            and frame_count >= (agent1_delay + agent2_delay)
            and not interceptor_deployed
        ):
            interceptor_deployed = True
            agent2_pos = start
            agent2_status.set_text("Agent 2: Deploying...")

            # Predict target's path
            predicted_positions = predict_vehicle_path(target_history)

            # Find interception point
            intercept_point = find_intercept_point(agent2_pos, predicted_positions)

            # Calculate path to interception point
            agent2_path = pathfinder.find_path(agent2_pos, intercept_point)
            agent2_path_index = 0

        # ===== TARGET MOVEMENT =====
        if pursuit_active:
            # Determine which agent the target should flee from
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

            # Update target position
            target.update_position(closest_agent_pos)

            # Update status text
            strategy_text.set_text(f"Strategy: {target.last_strategy}")
            info_text.set_text(
                f"Fleeing from: {fleeing_from}, Pattern confidence: {pattern_confidence}%"
            )

            # Periodically predict path and recalculate agent paths
            if frame_count % update_frequency == 0:
                # Generate prediction
                predicted_positions = predict_vehicle_path(target_history)

                # Update agent paths if needed
                if agent1_active:
                    agent1_path = pathfinder.find_path(agent1_pos, target.position)
                    agent1_path_index = 0

                if interceptor_deployed and agent2_pos is not None:
                    if frame_count % (update_frequency * 3) == 0:
                        intercept_point = find_intercept_point(
                            agent2_pos, predicted_positions
                        )
                        agent2_path = pathfinder.find_path(agent2_pos, intercept_point)
                        agent2_path_index = 0

        # ===== AGENT 1 MOVEMENT =====
        if agent1_active and agent1_path and frame_count % agent1_move_frequency == 0:
            if agent1_path_index < len(agent1_path):
                agent1_pos = agent1_path[agent1_path_index]
                agent1_path_index += 1
                agent1_status.set_text("Agent 1: Pursuing")
            else:
                # If we've reached the end of path, recalculate
                agent1_path = pathfinder.find_path(agent1_pos, target.position)
                agent1_path_index = 0
                agent1_status.set_text("Agent 1: Recalculating")

        # Update agent1 visualization
        agent1_point.set_data([agent1_pos[0]], [agent1_pos[1]])

        # Update agent1 path visualization
        if agent1_path:
            path1_x = [p[0] for p in agent1_path]
            path1_y = [p[1] for p in agent1_path]
            path1_line.set_data(path1_x, path1_y)

        # ===== AGENT 2 MOVEMENT =====
        if interceptor_deployed and agent2_pos is not None:
            if agent2_path and frame_count % agent2_move_frequency == 0:
                if agent2_path_index < len(agent2_path):
                    agent2_pos = agent2_path[agent2_path_index]
                    agent2_path_index += 1
                    agent2_status.set_text(
                        f"Agent 2: Intercepting (confidence: {pattern_confidence}%)"
                    )
                else:
                    # Recalculate path when we reach the end
                    predicted_positions = predict_vehicle_path(target_history)
                    intercept_point = find_intercept_point(
                        agent2_pos, predicted_positions
                    )
                    agent2_path = pathfinder.find_path(agent2_pos, intercept_point)
                    agent2_path_index = 0
                    agent2_status.set_text("Agent 2: Updating interception")

            # Update agent2 visualization
            agent2_point.set_data([agent2_pos[0]], [agent2_pos[1]])

            # Update agent2 path visualization
            if agent2_path:
                path2_x = [p[0] for p in agent2_path]
                path2_y = [p[1] for p in agent2_path]
                path2_line.set_data(path2_x, path2_y)
        else:
            agent2_point.set_data([], [])
            path2_line.set_data([], [])
            if not interceptor_deployed:
                agent2_status.set_text("Agent 2: Waiting")

        # Check for target capture after moving agents and target
        if pursuit_active and interceptor_deployed and agent2_pos is not None:
            if is_target_captured():
                target_captured = True
                capture_text.set_text("TARGET CAPTURED!")
                capture_text.set_alpha(0.1)  # Start fade in
                agent1_status.set_text("Agent 1: Target secured!")
                agent2_status.set_text("Agent 2: Target secured!")
                info_text.set_text("Mission accomplished")

        return (
            agent1_point,
            agent2_point,
            target_point,
            path1_line,
            path2_line,
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
        fig, update, frames=max_frames, init_func=init, interval=100, blit=True
    )

    plt.tight_layout()
    plt.show()

    return ani


def create_sample_neighborhood(width=20, height=20):
    env = Environment(width, height)

    # Add some blocks as obstacles
    env.add_block(2, 2, 6, 6)
    env.add_block(2, 8, 6, 6)
    env.add_block(2, 14, 6, 6)
    env.add_block(8, 2, 6, 6)
    env.add_block(8, 8, 6, 6)
    env.add_block(8, 14, 4, 4)
    env.add_block(14, 2, 5, 5)
    env.add_block(14, 8, 4, 4)
    env.add_block(14, 14, 4, 4)
    env.add_block(6, 14, 6, 6)
    env.add_block(6, 6, 4, 4)
    return env


# Update main function with new parameters
def main():
    # Create environment on your own
    env = create_sample_neighborhood()

    # Define start and initial target position
    start = (0.5, 0.5)  # Bottom left
    target_position = (13, 18)  # Custom position

    # Create pathfinder
    pathfinder = AStar(env)

    # Start the simulation with new parameters:
    # - Agent 1 waits 3 seconds (30 frames)
    # - Agent 2 deploys after 10 seconds (100 frames) from when target starts fleeing
    # - Target is faster than Agent 1 but slower than Agent 2
    try:
        animate_multi_agent_pursuit(
            env,
            pathfinder,
            start,
            target_position,
            agent1_delay=30,  # 3 seconds
            agent2_delay=100,  # 10 seconds
            target_speed=0.5,  # Medium speed
            agent1_speed=8,  # Slower than target
            agent2_speed=14,  # Faster than target
        )
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
