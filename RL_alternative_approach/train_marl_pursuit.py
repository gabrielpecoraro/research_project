import numpy as np
import random
import matplotlib.pyplot as plt
from pathfinding import Environment, AStar, VehicleTarget
from marl_system import MARLPursuitSystem
import torch
import os
from datetime import datetime
import pickle
import json


# Add MPS device configuration at the top
def get_device():
    """Get the best available device (MPS, CUDA, or CPU)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using MPS (Apple Silicon GPU) device: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using CPU device: {device}")
    return device


# Global device variable
DEVICE = get_device()


class MapGenerator:
    """Generate diverse map configurations for training"""

    def __init__(self, width=50, height=50):
        self.width = width
        self.height = height

    def create_random_urban_layout(self, density=0.3):
        """Create a random urban-like environment"""
        env = Environment(self.width, self.height)

        # Number of buildings based on density
        num_buildings = int(density * 30)  # Base number scaled by density

        # Generate random buildings
        for _ in range(num_buildings):
            # Random building size (3-12 units wide/tall)
            building_width = random.uniform(3, 12)
            building_height = random.uniform(3, 12)

            # Random position (ensure it's not too close to edges)
            x = random.uniform(2, self.width - building_width - 2)
            y = random.uniform(2, self.height - building_height - 2)

            # Add building if it doesn't create impossible navigation
            env.add_block(x, y, building_width, building_height)

        return env

    def create_corridor_maze(self):
        """Create a maze-like environment with corridors"""
        env = Environment(self.width, self.height)

        # Create a grid of buildings with corridors
        corridor_width = 4
        building_size = 6

        for row in range(0, self.height, building_size + corridor_width):
            for col in range(0, self.width, building_size + corridor_width):
                if (
                    row + building_size < self.height
                    and col + building_size < self.width
                ):
                    # Randomly skip some buildings to create variety
                    if random.random() > 0.2:  # 80% chance to place building
                        env.add_block(col, row, building_size, building_size)

        return env

    def create_sparse_environment(self):
        """Create environment with few, large obstacles"""
        env = Environment(self.width, self.height)

        # Add 5-8 large buildings
        num_large_buildings = random.randint(5, 8)

        for _ in range(num_large_buildings):
            building_width = random.uniform(8, 15)
            building_height = random.uniform(8, 15)

            x = random.uniform(5, self.width - building_width - 5)
            y = random.uniform(5, self.height - building_height - 5)

            env.add_block(x, y, building_width, building_height)

        return env

    def create_dense_urban(self):
        """Create a dense urban environment"""
        env = Environment(self.width, self.height)

        # Many small to medium buildings
        num_buildings = random.randint(25, 40)

        for _ in range(num_buildings):
            building_width = random.uniform(2, 8)
            building_height = random.uniform(2, 8)

            x = random.uniform(1, self.width - building_width - 1)
            y = random.uniform(1, self.height - building_height - 1)

            env.add_block(x, y, building_width, building_height)

        return env

    def create_perimeter_fortress(self):
        """Create environment with buildings around the perimeter"""
        env = Environment(self.width, self.height)

        # Buildings along edges
        edge_thickness = 8

        # Top and bottom edges
        for i in range(0, self.width, 12):
            if i + 8 < self.width:
                env.add_block(i, 0, 8, edge_thickness)
                env.add_block(i, self.height - edge_thickness, 8, edge_thickness)

        # Left and right edges
        for i in range(0, self.height, 12):
            if i + 8 < self.height:
                env.add_block(0, i, edge_thickness, 8)
                env.add_block(self.width - edge_thickness, i, edge_thickness, 8)

        # Some random interior buildings
        for _ in range(random.randint(5, 10)):
            building_width = random.uniform(4, 10)
            building_height = random.uniform(4, 10)

            x = random.uniform(
                edge_thickness + 2, self.width - building_width - edge_thickness - 2
            )
            y = random.uniform(
                edge_thickness + 2, self.height - building_height - edge_thickness - 2
            )

            env.add_block(x, y, building_width, building_height)

        return env

    def get_random_environment(self):
        """Get a random environment type"""
        env_types = [
            self.create_random_urban_layout,
            self.create_corridor_maze,
            self.create_sparse_environment,
            self.create_dense_urban,
            self.create_perimeter_fortress,
        ]

        # Sometimes use different densities for urban layout
        if random.random() < 0.4:
            density = random.uniform(0.15, 0.6)
            return self.create_random_urban_layout(density)
        else:
            return random.choice(env_types)()


class TrainingLogger:
    """Log training progress and statistics"""

    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(log_dir, f"marl_training_{timestamp}")
        os.makedirs(self.session_dir, exist_ok=True)

        self.episode_logs = []
        self.performance_metrics = {
            "success_rates": [],
            "episode_lengths": [],
            "total_rewards": [],
            "capture_times": [],
            "cooperation_scores": [],
        }

        print(f"Training session started. Logs will be saved to: {self.session_dir}")

    def log_episode(self, episode_num, episode_data):
        """Log data from a single episode"""
        log_entry = {
            "episode": episode_num,
            "success": episode_data["success"],
            "steps": episode_data["steps"],
            "total_reward": episode_data["total_reward"],
            "map_type": episode_data.get("map_type", "unknown"),
            "agents_deployed": episode_data.get("agents_deployed", 0),
            "cooperation_events": episode_data.get("cooperation_events", 0),
        }

        self.episode_logs.append(log_entry)

        # Update performance metrics
        self.performance_metrics["success_rates"].append(
            1.0 if episode_data["success"] else 0.0
        )
        self.performance_metrics["episode_lengths"].append(episode_data["steps"])
        self.performance_metrics["total_rewards"].append(episode_data["total_reward"])

        if episode_data["success"]:
            self.performance_metrics["capture_times"].append(episode_data["steps"])

    def print_progress(self, episode_num, window_size=100):
        """Print training progress"""
        if episode_num % window_size == 0 and episode_num > 0:
            recent_success = np.mean(
                self.performance_metrics["success_rates"][-window_size:]
            )
            recent_reward = np.mean(
                self.performance_metrics["total_rewards"][-window_size:]
            )
            recent_steps = np.mean(
                self.performance_metrics["episode_lengths"][-window_size:]
            )

            print(f"\n=== Episode {episode_num} Progress ===")
            print(f"Success Rate (last {window_size}): {recent_success:.2%}")
            print(f"Average Reward (last {window_size}): {recent_reward:.2f}")
            print(f"Average Episode Length (last {window_size}): {recent_steps:.1f}")

            if self.performance_metrics["capture_times"]:
                avg_capture_time = np.mean(
                    self.performance_metrics["capture_times"][-50:]
                )
                print(f"Average Capture Time: {avg_capture_time:.1f} steps")

    def save_training_data(self):
        """Save training data and create plots"""
        # Save raw data
        with open(os.path.join(self.session_dir, "episode_logs.json"), "w") as f:
            json.dump(self.episode_logs, f, indent=2)

        with open(os.path.join(self.session_dir, "performance_metrics.pkl"), "wb") as f:
            pickle.dump(self.performance_metrics, f)

        # Create performance plots
        self.create_performance_plots()

        print(f"Training data saved to: {self.session_dir}")

    def create_performance_plots(self):
        """Create and save performance visualization plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("MARL Training Performance", fontsize=16)

        # Success rate over time (rolling average)
        window_size = 100
        if len(self.performance_metrics["success_rates"]) >= window_size:
            success_smooth = np.convolve(
                self.performance_metrics["success_rates"],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            axes[0, 0].plot(
                range(window_size - 1, len(self.performance_metrics["success_rates"])),
                success_smooth,
            )
            axes[0, 0].set_title("Success Rate (100-episode rolling average)")
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Success Rate")
            axes[0, 0].grid(True)

        # Reward over time
        if len(self.performance_metrics["total_rewards"]) >= window_size:
            reward_smooth = np.convolve(
                self.performance_metrics["total_rewards"],
                np.ones(window_size) / window_size,
                mode="valid",
            )
            axes[0, 1].plot(
                range(window_size - 1, len(self.performance_metrics["total_rewards"])),
                reward_smooth,
            )
            axes[0, 1].set_title("Total Reward (100-episode rolling average)")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Total Reward")
            axes[0, 1].grid(True)

        # Episode length distribution
        axes[1, 0].hist(self.performance_metrics["episode_lengths"], bins=50, alpha=0.7)
        axes[1, 0].set_title("Episode Length Distribution")
        axes[1, 0].set_xlabel("Episode Length (steps)")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].grid(True)

        # Capture time for successful episodes
        if self.performance_metrics["capture_times"]:
            axes[1, 1].hist(
                self.performance_metrics["capture_times"],
                bins=30,
                alpha=0.7,
                color="green",
            )
            axes[1, 1].set_title("Capture Time Distribution (Successful Episodes)")
            axes[1, 1].set_xlabel("Capture Time (steps)")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "No Successful Captures Yet",
                transform=axes[1, 1].transAxes,
                ha="center",
                va="center",
            )
            axes[1, 1].set_title("Capture Time Distribution")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.session_dir, "training_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


class MARLTrainingEnvironment:
    """Enhanced MARL system specifically for pursuit training"""

    def __init__(self, marl_system, map_generator, logger):
        self.marl_system = marl_system
        self.map_generator = map_generator
        self.logger = logger

        # Training statistics
        self.map_type_performance = {}
        self.cooperation_events = []

    def run_training_episode(self, episode_num, max_steps=800):
        """Run a single training episode with a random map"""
        # Generate random environment
        env = self.map_generator.get_random_environment()
        pathfinder = AStar(env)

        # Determine map type for logging
        map_type = self._classify_map_type(env)

        # Random start and target positions
        start_pos = self._get_random_valid_position(env, border_margin=3)
        target_pos = self._get_random_valid_position(
            env, border_margin=5, min_distance_from=start_pos, min_dist=20
        )

        # Initialize agents
        active_agents = 1
        agent_positions = [start_pos] + [None] * (self.marl_system.max_agents - 1)

        # Initialize target with random speed
        target_speed = random.uniform(0.2, 0.4)
        target = VehicleTarget(env, target_pos, speed=target_speed)

        # Episode tracking
        episode_data = []
        total_rewards = np.zeros(self.marl_system.max_agents)
        step_count = 0
        success = False
        cooperation_events = 0

        # Deployment timers with some randomness
        base_delay1 = random.randint(20, 40)
        base_delay2 = random.randint(60, 120)
        agent_deploy_times = [0, base_delay1, base_delay2]

        # Store previous states for experience replay
        prev_observations = None
        prev_actions = None

        for step in range(max_steps):
            step_count = step

            # Deploy additional agents
            if step >= agent_deploy_times[1] and active_agents == 1:
                agent_positions[1] = self._get_deployment_position(
                    env, agent_positions[0], target.position
                )
                active_agents = 2

            if step >= agent_deploy_times[2] and active_agents == 2:
                agent_positions[2] = self._get_deployment_position(
                    env, agent_positions[:2], target.position
                )
                active_agents = 3

            # Get observations and actions for active agents
            observations = []
            actions = []
            valid_positions = []

            for i in range(active_agents):
                if agent_positions[i] is not None:
                    obs = self.marl_system.get_local_observation(
                        agent_positions[i],
                        env,
                        [pos for j, pos in enumerate(agent_positions) if j != i],
                        target.position,
                    )
                    action = self.marl_system.select_action(obs, i, training=True)

                    observations.append(obs)
                    actions.append(action)
                    valid_positions.append(i)

                    # Move agent
                    dx, dy = self.marl_system.actions[action]
                    new_x = agent_positions[i][0] + dx * 0.5
                    new_y = agent_positions[i][1] + dy * 0.5

                    if env.is_valid_position(new_x, new_y):
                        agent_positions[i] = (new_x, new_y)

            # Move target away from closest agent
            if active_agents > 0 and any(
                pos is not None for pos in agent_positions[:active_agents]
            ):
                valid_agent_positions = [
                    pos for pos in agent_positions[:active_agents] if pos is not None
                ]
                closest_agent = min(
                    valid_agent_positions,
                    key=lambda pos: np.sqrt(
                        (pos[0] - target.position[0]) ** 2
                        + (pos[1] - target.position[1]) ** 2
                    ),
                )
                target.update_position(closest_agent)

            # Calculate rewards
            rewards = self._calculate_rewards(
                agent_positions, target.position, active_agents, env
            )

            # Check for cooperation events
            if active_agents >= 2:
                coop_score = self._detect_cooperation(
                    agent_positions[:active_agents], target.position
                )
                if coop_score > 0.7:  # High cooperation threshold
                    cooperation_events += 1
                    # Bonus reward for good cooperation
                    for i in range(active_agents):
                        rewards[i] += 3.0

            total_rewards += rewards

            # Store experience for replay buffer
            if prev_observations is not None and len(prev_observations) == len(
                observations
            ):
                # Create global state for QMIX
                global_state = self.marl_system.get_global_state(
                    env, agent_positions, target.position
                )
                prev_global_state = self.marl_system.get_global_state(
                    env, agent_positions, target.position
                )  # Approximation

                # Pad observations and actions to max_agents size
                padded_obs = self._pad_to_max_agents(observations)
                padded_prev_obs = self._pad_to_max_agents(prev_observations)
                padded_actions = self._pad_actions(prev_actions)
                padded_rewards = rewards[: self.marl_system.max_agents]

                # Store experience
                experience = (
                    np.concatenate([padded_prev_obs.flatten(), prev_global_state]),
                    padded_actions,
                    padded_rewards,
                    np.concatenate([padded_obs.flatten(), global_state]),
                    [False] * self.marl_system.max_agents,  # Not done yet
                )
                self.marl_system.store_experience(*experience)

            # Store step data for sequence learning
            step_data = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards[:active_agents].tolist(),
                "agent_positions": [
                    pos for pos in agent_positions[:active_agents] if pos is not None
                ],
                "target_position": target.position,
                "active_agents": active_agents,
            }
            episode_data.append(step_data)

            # Check for capture
            if self._check_capture(agent_positions[:active_agents], target.position):
                success = True
                # Final experience with done=True
                if prev_observations is not None:
                    final_rewards = rewards.copy()
                    final_rewards[:active_agents] += 50  # Big capture bonus

                    global_state = self.marl_system.get_global_state(
                        env, agent_positions, target.position
                    )
                    padded_obs = self._pad_to_max_agents(observations)
                    padded_actions = self._pad_actions(actions)

                    final_experience = (
                        np.concatenate([padded_prev_obs.flatten(), prev_global_state]),
                        padded_actions,
                        final_rewards[: self.marl_system.max_agents],
                        np.concatenate([padded_obs.flatten(), global_state]),
                        [True] * self.marl_system.max_agents,  # Episode done
                    )
                    self.marl_system.store_experience(*final_experience)
                break

            prev_observations = observations
            prev_actions = actions

        # Store sequence for transformer training
        self.marl_system.store_sequence(episode_data)

        # Update performance tracking
        if map_type not in self.map_type_performance:
            self.map_type_performance[map_type] = {"successes": 0, "total": 0}
        self.map_type_performance[map_type]["total"] += 1
        if success:
            self.map_type_performance[map_type]["successes"] += 1

        return {
            "success": success,
            "steps": step_count,
            "total_reward": total_rewards.sum(),
            "episode_data": episode_data,
            "map_type": map_type,
            "agents_deployed": active_agents,
            "cooperation_events": cooperation_events,
            "target_speed": target_speed,
        }

    def _get_random_valid_position(
        self, env, border_margin=2, min_distance_from=None, min_dist=10
    ):
        """Get a random valid position in the environment"""
        max_attempts = 100
        for _ in range(max_attempts):
            x = random.uniform(border_margin, env.width - border_margin)
            y = random.uniform(border_margin, env.height - border_margin)

            if env.is_valid_position(x, y):
                if min_distance_from is None:
                    return (x, y)
                else:
                    dist = np.sqrt(
                        (x - min_distance_from[0]) ** 2
                        + (y - min_distance_from[1]) ** 2
                    )
                    if dist >= min_dist:
                        return (x, y)

        # Fallback to corners if random selection fails
        corners = [
            (border_margin, border_margin),
            (env.width - border_margin, border_margin),
            (border_margin, env.height - border_margin),
            (env.width - border_margin, env.height - border_margin),
        ]

        for corner in corners:
            if env.is_valid_position(corner[0], corner[1]):
                return corner

        return (env.width / 2, env.height / 2)  # Last resort

    def _get_deployment_position(self, env, existing_positions, target_pos):
        """Get strategic deployment position for new agent"""
        if isinstance(existing_positions, tuple):
            existing_positions = [existing_positions]

        # Try to deploy agents to surround the target
        angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]  # 4 cardinal directions

        for angle in angles:
            # Position at distance from target
            deploy_distance = random.uniform(15, 25)
            x = target_pos[0] + np.cos(angle) * deploy_distance
            y = target_pos[1] + np.sin(angle) * deploy_distance

            # Clamp to environment bounds
            x = max(2, min(env.width - 2, x))
            y = max(2, min(env.height - 2, y))

            if env.is_valid_position(x, y):
                # Check it's not too close to existing agents
                min_dist_to_agents = min(
                    [
                        np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2)
                        for pos in existing_positions
                        if pos is not None
                    ],
                    default=float("inf"),
                )

                if min_dist_to_agents > 5:  # Minimum separation
                    return (x, y)

        # Fallback to random position
        return self._get_random_valid_position(env)

    def _calculate_rewards(self, agent_positions, target_pos, active_agents, env):
        """Calculate rewards for all agents"""
        rewards = np.zeros(self.marl_system.max_agents)

        for i in range(active_agents):
            if agent_positions[i] is not None:
                # Distance-based reward
                dist = np.sqrt(
                    (agent_positions[i][0] - target_pos[0]) ** 2
                    + (agent_positions[i][1] - target_pos[1]) ** 2
                )

                # Closer = better reward
                rewards[i] += max(0, 15 - dist)

                # Penalty for being too close to other agents (avoid clustering)
                for j in range(i + 1, active_agents):
                    if agent_positions[j] is not None:
                        agent_dist = np.sqrt(
                            (agent_positions[i][0] - agent_positions[j][0]) ** 2
                            + (agent_positions[i][1] - agent_positions[j][1]) ** 2
                        )
                        if agent_dist < 2.0:
                            rewards[i] -= 2
                            rewards[j] -= 2

        return rewards

    def _detect_cooperation(self, agent_positions, target_pos):
        """Detect and score cooperation between agents"""
        if len(agent_positions) < 2:
            return 0.0

        # Calculate if agents are positioning to surround target
        target_to_agents = []
        for pos in agent_positions:
            if pos is not None:
                angle = np.arctan2(pos[1] - target_pos[1], pos[0] - target_pos[0])
                target_to_agents.append(angle)

        if len(target_to_agents) < 2:
            return 0.0

        # Sort angles
        target_to_agents.sort()

        # Calculate angular separation between consecutive agents
        separations = []
        for i in range(len(target_to_agents)):
            next_i = (i + 1) % len(target_to_agents)
            sep = abs(target_to_agents[next_i] - target_to_agents[i])
            if sep > np.pi:
                sep = 2 * np.pi - sep
            separations.append(sep)

        # Good cooperation = agents roughly evenly spaced around target
        ideal_separation = 2 * np.pi / len(target_to_agents)
        cooperation_score = 1.0 - np.std(separations) / ideal_separation

        return max(0.0, min(1.0, cooperation_score))

    def _check_capture(self, agent_positions, target_pos):
        """Check if target is captured"""
        valid_positions = [pos for pos in agent_positions if pos is not None]

        if len(valid_positions) < 2:
            return False

        # At least 2 agents within capture radius
        close_agents = 0
        for pos in valid_positions:
            dist = np.sqrt(
                (pos[0] - target_pos[0]) ** 2 + (pos[1] - target_pos[1]) ** 2
            )
            if dist <= 2.0:  # Capture radius
                close_agents += 1

        return close_agents >= 2

    def _classify_map_type(self, env):
        """Classify the type of map for performance tracking"""
        total_obstacle_area = sum(
            width * height for _, _, width, height in env.obstacles
        )
        density = total_obstacle_area / (env.width * env.height)

        if density < 0.15:
            return "sparse"
        elif density < 0.3:
            return "medium"
        else:
            return "dense"

    def _pad_to_max_agents(self, observations):
        """Pad observations to max_agents size"""
        padded = np.zeros((self.marl_system.max_agents, self.marl_system.obs_dim))
        for i, obs in enumerate(observations[: self.marl_system.max_agents]):
            padded[i] = obs
        return padded

    def _pad_actions(self, actions):
        """Pad actions to max_agents size"""
        padded = np.zeros(self.marl_system.max_agents, dtype=int)
        for i, action in enumerate(actions[: self.marl_system.max_agents]):
            padded[i] = action
        return padded


def main():
    """Main training function"""
    print("Starting MARL Pursuit Training")
    print("=" * 50)
    print(f"Device: {DEVICE}")

    # Training parameters
    NUM_EPISODES = 5000
    MAP_WIDTH = 50
    MAP_HEIGHT = 50
    MAX_AGENTS = 3

    # Initialize components
    map_generator = MapGenerator(MAP_WIDTH, MAP_HEIGHT)
    logger = TrainingLogger()

    # Create MARL system with device specification
    marl_system = MARLPursuitSystem(
        env_width=MAP_WIDTH,
        env_height=MAP_HEIGHT,
        max_agents=MAX_AGENTS,
        obs_radius=5,
        lr=0.0005,  # Slightly lower learning rate for stability
        device=DEVICE,  # Pass device to MARL system
    )

    print(f"Training Configuration:")
    print(f"- Episodes: {NUM_EPISODES}")
    print(f"- Map Size: {MAP_WIDTH}x{MAP_HEIGHT}")
    print(f"- Max Agents: {MAX_AGENTS}")
    print(f"- Learning Rate: {marl_system.agent_optimizers[0].param_groups[0]['lr']}")
    print(f"- Replay Buffer Size: {marl_system.replay_buffer.maxlen}")
    print()

    # Initialize training environment
    training_env = MARLTrainingEnvironment(marl_system, map_generator, logger)

    # Training loop
    try:
        for episode in range(NUM_EPISODES):
            # Run episode
            episode_result = training_env.run_training_episode(episode)

            # Log episode
            logger.log_episode(episode, episode_result)

            # Train networks
            if episode >= 50:  # Start training after some experience
                if episode % 2 == 0:  # Train every 2 episodes
                    for _ in range(3):  # Multiple training steps per update
                        marl_system.train_step()

                    # Train sequence model less frequently
                    if episode % 10 == 0:
                        marl_system.train_sequence_model()

            # Print progress
            logger.print_progress(episode)

            # Save intermediate models
            if episode % 500 == 0 and episode > 0:
                model_path = os.path.join(
                    logger.session_dir, f"models_episode_{episode}"
                )
                marl_system.save_models(model_path)
                print(f"Intermediate models saved at episode {episode}")

            # Print episode summary
            if episode % 50 == 0:
                success_rate = (
                    np.mean(logger.performance_metrics["success_rates"][-50:])
                    if len(logger.performance_metrics["success_rates"]) >= 50
                    else 0
                )
                print(
                    f"Episode {episode}: Success={'✓' if episode_result['success'] else '✗'}, "
                    f"Steps={episode_result['steps']}, "
                    f"Reward={episode_result['total_reward']:.1f}, "
                    f"Map={episode_result['map_type']}, "
                    f"Success Rate (50)={success_rate:.2%}"
                )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Save final results
        print("\nSaving training results...")

        # Save final models
        final_model_path = os.path.join(logger.session_dir, "final_models")
        marl_system.save_models(final_model_path)

        # Save training data and plots
        logger.save_training_data()

        # Print final statistics
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)

        if logger.performance_metrics["success_rates"]:
            overall_success = np.mean(logger.performance_metrics["success_rates"])
            recent_success = (
                np.mean(logger.performance_metrics["success_rates"][-500:])
                if len(logger.performance_metrics["success_rates"]) >= 500
                else overall_success
            )

            print(f"Overall Success Rate: {overall_success:.2%}")
            print(f"Recent Success Rate (last 500): {recent_success:.2%}")

            if logger.performance_metrics["capture_times"]:
                avg_capture_time = np.mean(logger.performance_metrics["capture_times"])
                print(f"Average Capture Time: {avg_capture_time:.1f} steps")

        # Map type performance
        print("\nPerformance by Map Type:")
        for map_type, stats in training_env.map_type_performance.items():
            success_rate = (
                stats["successes"] / stats["total"] if stats["total"] > 0 else 0
            )
            print(
                f"  {map_type.capitalize()}: {success_rate:.2%} ({stats['successes']}/{stats['total']})"
            )

        print(f"\nAll results saved to: {logger.session_dir}")


if __name__ == "__main__":
    main()
