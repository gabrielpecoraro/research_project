import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathfinding import VehicleTarget, create_sample_neighborhood
from rl_integration import RLPursuitSystem


def animate_rl_multi_agent_pursuit(
    env,
    start,
    target_position=None,
    model_path="qmix_model_final.pth",
    max_frames=500,
    agent_speeds=[0.4, 0.4],
    target_speed=0.3,
    use_trained_model=True,
):
    """
    Animate multi-agent pursuit using trained QMIX model
    """

    if target_position is None:
        target_position = (env.width - 0.5, env.height - 0.5)

    if not env.is_valid_position(target_position[0], target_position[1]):
        raise ValueError("Target position must be on a street, not inside a building")

    # Initialize RL system
    if use_trained_model:
        try:
            rl_system = RLPursuitSystem(model_path=model_path)
            rl_system.set_environment(env)
            print("Using trained QMIX model for coordination")
        except:
            print("Could not load trained model, using random actions")
            rl_system = RLPursuitSystem()
            rl_system.set_environment(env)
    else:
        print("Using untrained model")
        rl_system = RLPursuitSystem()
        rl_system.set_environment(env)

    # Initialize environment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Main environment plot
    for ox, oy, ow, oh in env.obstacles:
        rect = plt.Rectangle((ox, oy), ow, oh, color="gray")
        ax1.add_patch(rect)

    # Initialize agents and target
    agent_positions = [start, start]
    target = VehicleTarget(env, target_position, speed=target_speed, inertia=0.8)
    target_history = [target.position]

    # Create plot elements
    (agent1_point,) = ax1.plot([], [], "ro", markersize=10, label="Agent 1")
    (agent2_point,) = ax1.plot([], [], "bo", markersize=10, label="Agent 2")
    (target_point,) = ax1.plot([], [], "gs", markersize=10, label="Target")
    (target_trail,) = ax1.plot([], [], "g:", alpha=0.5, linewidth=2)

    # Set up main plot
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.grid(True)
    ax1.set_title("QMIX Multi-Agent Pursuit")
    ax1.legend()

    # Performance tracking plot
    rewards_history = []
    distances_history = []
    ax2.set_title("Real-time Performance")
    (reward_line,) = ax2.plot([], [], "b-", label="Cumulative Reward")
    (distance_line,) = ax2.plot([], [], "r-", label="Min Distance to Target")
    ax2.legend()
    ax2.grid(True)

    # Text displays
    info_text = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    frame_count = 0
    cumulative_reward = 0
    capture_distance = 2.0
    target_captured = False

    def is_target_captured():
        """Check if target is captured"""
        distances = [
            np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            )
            for pos in agent_positions
        ]
        return all(d < capture_distance for d in distances)

    def calculate_reward():
        """Calculate current reward"""
        distances = [
            np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            )
            for pos in agent_positions
        ]

        if is_target_captured():
            return 50.0

        min_distance = min(distances)
        max_distance = np.sqrt(env.width**2 + env.height**2)
        distance_reward = -min_distance / max_distance * 10

        # Coordination reward
        agent_distance = np.sqrt(
            (agent_positions[0][0] - agent_positions[1][0]) ** 2
            + (agent_positions[0][1] - agent_positions[1][1]) ** 2
        )
        ideal_distance = max_distance * 0.1
        coordination_reward = -abs(agent_distance - ideal_distance) / max_distance

        return distance_reward + coordination_reward - 0.01  # Time penalty

    def init():
        agent1_point.set_data([], [])
        agent2_point.set_data([], [])
        target_point.set_data([], [])
        target_trail.set_data([], [])
        reward_line.set_data([], [])
        distance_line.set_data([], [])
        info_text.set_text("")
        return (
            agent1_point,
            agent2_point,
            target_point,
            target_trail,
            reward_line,
            distance_line,
            info_text,
        )

    def update(frame):
        nonlocal \
            agent_positions, \
            target_history, \
            frame_count, \
            cumulative_reward, \
            target_captured

        frame_count += 1

        if target_captured:
            info_text.set_text(
                f"TARGET CAPTURED!\nFrame: {frame_count}\nTotal Reward: {cumulative_reward:.2f}"
            )
            return (
                agent1_point,
                agent2_point,
                target_point,
                target_trail,
                reward_line,
                distance_line,
                info_text,
            )

        # Get RL coordinated actions
        try:
            actions = rl_system.get_coordinated_actions(
                agent_positions, target.position, target_history
            )

            # Convert actions to movements
            agent_positions = rl_system.convert_actions_to_movements(
                actions, agent_positions, speed=agent_speeds[0]
            )

            action_info = f"Actions: {actions}"
        except Exception as e:
            # Fallback to random movement
            actions = [np.random.randint(0, 8) for _ in range(2)]
            directions = [
                (0, 1),
                (1, 1),
                (1, 0),
                (1, -1),
                (0, -1),
                (-1, -1),
                (-1, 0),
                (-1, 1),
            ]

            new_positions = []
            for action, current_pos in zip(actions, agent_positions):
                dx, dy = directions[action]
                new_x = current_pos[0] + dx * agent_speeds[0]
                new_y = current_pos[1] + dy * agent_speeds[0]

                if env.is_valid_position(new_x, new_y):
                    new_positions.append((new_x, new_y))
                else:
                    new_positions.append(current_pos)

            agent_positions = new_positions
            action_info = f"Random Actions: {actions}"

        # Update target
        closest_agent = min(
            agent_positions,
            key=lambda pos: np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            ),
        )
        target.update_position(closest_agent)
        target_history.append(target.position)

        # Keep history manageable
        if len(target_history) > 50:
            target_history = target_history[-50:]

        # Calculate reward and distances
        reward = calculate_reward()
        cumulative_reward += reward

        distances = [
            np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            )
            for pos in agent_positions
        ]
        min_distance = min(distances)

        # Update tracking
        rewards_history.append(cumulative_reward)
        distances_history.append(min_distance)

        # Check capture
        target_captured = is_target_captured()

        # Update plot elements
        agent1_point.set_data([agent_positions[0][0]], [agent_positions[0][1]])
        agent2_point.set_data([agent_positions[1][0]], [agent_positions[1][1]])
        target_point.set_data([target.position[0]], [target.position[1]])

        # Update target trail
        if len(target_history) > 1:
            trail_x = [p[0] for p in target_history[-20:]]
            trail_y = [p[1] for p in target_history[-20:]]
            target_trail.set_data(trail_x, trail_y)

        # Update performance plots
        if len(rewards_history) > 1:
            frames = list(range(len(rewards_history)))
            reward_line.set_data(frames, rewards_history)
            distance_line.set_data(frames, distances_history)

            ax2.relim()
            ax2.autoscale_view()

        # Update info text
        info_text.set_text(
            f"Frame: {frame_count}\n"
            f"Reward: {cumulative_reward:.2f}\n"
            f"Min Distance: {min_distance:.2f}\n"
            f"{action_info}\n"
            f"Target Strategy: {target.last_strategy}\n"
            f"Epsilon: {rl_system.agent.epsilon:.3f}"
        )

        return (
            agent1_point,
            agent2_point,
            target_point,
            target_trail,
            reward_line,
            distance_line,
            info_text,
        )

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_frames,
        init_func=init,
        interval=100,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()

    return ani


def main():
    """Main function to run the RL-powered animation"""
    # Create environment
    env = create_sample_neighborhood()
    start = (0.5, 0.5)
    target_position = (35, 35)

    # Try with trained model first, fall back to untrained if not available
    try:
        print("Attempting to use trained model...")
        ani = animate_rl_multi_agent_pursuit(
            env,
            start,
            target_position,
            model_path="qmix_model_final.pth",
            use_trained_model=True,
            max_frames=1000,
        )
    except:
        print("Trained model not found, using untrained model...")
        ani = animate_rl_multi_agent_pursuit(
            env, start, target_position, use_trained_model=False, max_frames=1000
        )

    return ani


if __name__ == "__main__":
    import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathfinding import VehicleTarget, create_sample_neighborhood
from rl_integration import RLPursuitSystem


def animate_rl_multi_agent_pursuit(
    env,
    start,
    target_position=None,
    model_path="qmix_model_final.pth",
    max_frames=500,
    agent_speeds=[0.4, 0.4],
    target_speed=0.3,
    use_trained_model=True,
):
    """
    Animate multi-agent pursuit using trained QMIX model
    """

    if target_position is None:
        target_position = (env.width - 0.5, env.height - 0.5)

    if not env.is_valid_position(target_position[0], target_position[1]):
        raise ValueError("Target position must be on a street, not inside a building")

    # Initialize RL system
    if use_trained_model:
        try:
            rl_system = RLPursuitSystem(model_path=model_path)
            rl_system.set_environment(env)
            print("Using trained QMIX model for coordination")
        except:
            print("Could not load trained model, using random actions")
            rl_system = RLPursuitSystem()
            rl_system.set_environment(env)
    else:
        print("Using untrained model")
        rl_system = RLPursuitSystem()
        rl_system.set_environment(env)

    # Initialize environment
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Main environment plot
    for ox, oy, ow, oh in env.obstacles:
        rect = plt.Rectangle((ox, oy), ow, oh, color="gray")
        ax1.add_patch(rect)

    # Initialize agents and target
    agent_positions = [start, start]
    target = VehicleTarget(env, target_position, speed=target_speed, inertia=0.8)
    target_history = [target.position]

    # Create plot elements
    (agent1_point,) = ax1.plot([], [], "ro", markersize=10, label="Agent 1")
    (agent2_point,) = ax1.plot([], [], "bo", markersize=10, label="Agent 2")
    (target_point,) = ax1.plot([], [], "gs", markersize=10, label="Target")
    (target_trail,) = ax1.plot([], [], "g:", alpha=0.5, linewidth=2)

    # Set up main plot
    ax1.set_xlim(0, env.width)
    ax1.set_ylim(0, env.height)
    ax1.grid(True)
    ax1.set_title("QMIX Multi-Agent Pursuit")
    ax1.legend()

    # Performance tracking plot
    rewards_history = []
    distances_history = []
    ax2.set_title("Real-time Performance")
    (reward_line,) = ax2.plot([], [], "b-", label="Cumulative Reward")
    (distance_line,) = ax2.plot([], [], "r-", label="Min Distance to Target")
    ax2.legend()
    ax2.grid(True)

    # Text displays
    info_text = ax1.text(
        0.02,
        0.98,
        "",
        transform=ax1.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    frame_count = 0
    cumulative_reward = 0
    capture_distance = 2.0
    target_captured = False

    def is_target_captured():
        """Check if target is captured"""
        distances = [
            np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            )
            for pos in agent_positions
        ]
        return all(d < capture_distance for d in distances)

    def calculate_reward():
        """Calculate current reward"""
        distances = [
            np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            )
            for pos in agent_positions
        ]

        if is_target_captured():
            return 50.0

        min_distance = min(distances)
        max_distance = np.sqrt(env.width**2 + env.height**2)
        distance_reward = -min_distance / max_distance * 10

        # Coordination reward
        agent_distance = np.sqrt(
            (agent_positions[0][0] - agent_positions[1][0]) ** 2
            + (agent_positions[0][1] - agent_positions[1][1]) ** 2
        )
        ideal_distance = max_distance * 0.1
        coordination_reward = -abs(agent_distance - ideal_distance) / max_distance

        return distance_reward + coordination_reward - 0.01  # Time penalty

    def init():
        agent1_point.set_data([], [])
        agent2_point.set_data([], [])
        target_point.set_data([], [])
        target_trail.set_data([], [])
        reward_line.set_data([], [])
        distance_line.set_data([], [])
        info_text.set_text("")
        return (
            agent1_point,
            agent2_point,
            target_point,
            target_trail,
            reward_line,
            distance_line,
            info_text,
        )

    def update(frame):
        nonlocal \
            agent_positions, \
            target_history, \
            frame_count, \
            cumulative_reward, \
            target_captured

        frame_count += 1

        if target_captured:
            info_text.set_text(
                f"TARGET CAPTURED!\nFrame: {frame_count}\nTotal Reward: {cumulative_reward:.2f}"
            )
            return (
                agent1_point,
                agent2_point,
                target_point,
                target_trail,
                reward_line,
                distance_line,
                info_text,
            )

        # Get RL coordinated actions
        try:
            actions = rl_system.get_coordinated_actions(
                agent_positions, target.position, target_history
            )

            # Convert actions to movements
            agent_positions = rl_system.convert_actions_to_movements(
                actions, agent_positions, speed=agent_speeds[0]
            )

            action_info = f"Actions: {actions}"
        except Exception as e:
            # Fallback to random movement
            actions = [np.random.randint(0, 8) for _ in range(2)]
            directions = [
                (0, 1),
                (1, 1),
                (1, 0),
                (1, -1),
                (0, -1),
                (-1, -1),
                (-1, 0),
                (-1, 1),
            ]

            new_positions = []
            for action, current_pos in zip(actions, agent_positions):
                dx, dy = directions[action]
                new_x = current_pos[0] + dx * agent_speeds[0]
                new_y = current_pos[1] + dy * agent_speeds[0]

                if env.is_valid_position(new_x, new_y):
                    new_positions.append((new_x, new_y))
                else:
                    new_positions.append(current_pos)

            agent_positions = new_positions
            action_info = f"Random Actions: {actions}"

        # Update target
        closest_agent = min(
            agent_positions,
            key=lambda pos: np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            ),
        )
        target.update_position(closest_agent)
        target_history.append(target.position)

        # Keep history manageable
        if len(target_history) > 50:
            target_history = target_history[-50:]

        # Calculate reward and distances
        reward = calculate_reward()
        cumulative_reward += reward

        distances = [
            np.sqrt(
                (pos[0] - target.position[0]) ** 2 + (pos[1] - target.position[1]) ** 2
            )
            for pos in agent_positions
        ]
        min_distance = min(distances)

        # Update tracking
        rewards_history.append(cumulative_reward)
        distances_history.append(min_distance)

        # Check capture
        target_captured = is_target_captured()

        # Update plot elements
        agent1_point.set_data([agent_positions[0][0]], [agent_positions[0][1]])
        agent2_point.set_data([agent_positions[1][0]], [agent_positions[1][1]])
        target_point.set_data([target.position[0]], [target.position[1]])

        # Update target trail
        if len(target_history) > 1:
            trail_x = [p[0] for p in target_history[-20:]]
            trail_y = [p[1] for p in target_history[-20:]]
            target_trail.set_data(trail_x, trail_y)

        # Update performance plots
        if len(rewards_history) > 1:
            frames = list(range(len(rewards_history)))
            reward_line.set_data(frames, rewards_history)
            distance_line.set_data(frames, distances_history)

            ax2.relim()
            ax2.autoscale_view()

        # Update info text
        info_text.set_text(
            f"Frame: {frame_count}\n"
            f"Reward: {cumulative_reward:.2f}\n"
            f"Min Distance: {min_distance:.2f}\n"
            f"{action_info}\n"
            f"Target Strategy: {target.last_strategy}\n"
            f"Epsilon: {rl_system.agent.epsilon:.3f}"
        )

        return (
            agent1_point,
            agent2_point,
            target_point,
            target_trail,
            reward_line,
            distance_line,
            info_text,
        )

    # Create animation
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=max_frames,
        init_func=init,
        interval=100,
        blit=True,
        repeat=False,
    )

    plt.tight_layout()
    plt.show()

    return ani


def main():
    """Main function to run the RL-powered animation"""
    # Create environment
    env = create_sample_neighborhood()
    start = (0.5, 0.5)
    target_position = (35, 35)

    # Try with trained model first, fall back to untrained if not available
    try:
        print("Attempting to use trained model...")
        ani = animate_rl_multi_agent_pursuit(
            env,
            start,
            target_position,
            model_path="qmix_model_final.pth",
            use_trained_model=True,
            max_frames=1000,
        )
    except:
        print("Trained model not found, using untrained model...")
        ani = animate_rl_multi_agent_pursuit(
            env, start, target_position, use_trained_model=False, max_frames=1000
        )

    return ani


if __name__ == "__main__":
    main()
