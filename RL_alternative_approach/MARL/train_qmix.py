import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random
import time
from collections import deque
import pickle
from adaptive_expert_system import AdaptivePathfindingExpert
from qmix_marl import QMIX
from pursuit_environment import PursuitEnvironment, create_test_environments

# Add pathfinding integration imports
try:
    from expert_pathfinding_system import PathfindingFlowExpertSystem
    from pathfinding import AStar

    PATHFINDING_AVAILABLE = True
    print("Pathfinding integration available")
except ImportError as e:
    PATHFINDING_AVAILABLE = False
    print(f"Pathfinding integration not available: {e}")


class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.recent_captures = deque(maxlen=window_size)
        self.recent_rewards = deque(maxlen=window_size)
        self.recent_episode_lengths = deque(maxlen=window_size)

    def add_episode(self, captured, reward, length):
        self.recent_captures.append(captured)
        self.recent_rewards.append(reward)
        self.recent_episode_lengths.append(length)

    def is_struggling(self):
        """Detect if the algorithm is struggling based on multiple metrics"""
        if len(self.recent_captures) < self.window_size:
            return False

        # Low capture rate
        capture_rate = np.mean(self.recent_captures)

        # Declining reward trend
        if len(self.recent_rewards) >= 50:
            early_rewards = np.mean(list(self.recent_rewards)[:25])
            late_rewards = np.mean(list(self.recent_rewards)[-25:])
            reward_declining = late_rewards < early_rewards * 0.9
        else:
            reward_declining = False

        # Very long episodes (failing to capture)
        avg_length = np.mean(self.recent_episode_lengths)
        max_reasonable_length = 300  # Adjust based on your environment

        # Define struggling conditions
        struggling = (
            capture_rate < 0.3  # Less than 30% capture rate
            or reward_declining  # Rewards getting worse
            or avg_length > max_reasonable_length  # Episodes too long
        )

        return struggling, {
            "capture_rate": capture_rate,
            "reward_declining": reward_declining,
            "avg_length": avg_length,
            "reason": self._get_struggle_reason(
                capture_rate, reward_declining, avg_length, max_reasonable_length
            ),
        }

    def _get_struggle_reason(
        self, capture_rate, reward_declining, avg_length, max_length
    ):
        reasons = []
        if capture_rate < 0.3:
            reasons.append(f"low capture rate ({capture_rate:.2f})")
        if reward_declining:
            reasons.append("declining rewards")
        if avg_length > max_length:
            reasons.append(f"long episodes ({avg_length:.1f})")
        return ", ".join(reasons)


class QMixTrainer:
    def __init__(
        self, device="mps", use_pathfinding_guidance=False, guidance_weight=0.3
    ):
        self.device = device
        self.use_pathfinding_guidance = (
            use_pathfinding_guidance and PATHFINDING_AVAILABLE
        )
        self.guidance_weight = guidance_weight

        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "capture_rates": [],
            "losses": [],
            "environment_sizes": [],
            "episode_times": [],
            "expert_match_rates": [] if self.use_pathfinding_guidance else [],
        }

        # Visualization setup
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        title = "QMIX Training Progress"
        if self.use_pathfinding_guidance:
            title += " (with Pathfinding Guidance)"
        self.fig.suptitle(title, fontsize=16)

        (self.reward_line,) = self.axes[0, 0].plot([], [], "b-", alpha=0.7)
        (self.reward_avg_line,) = self.axes[0, 0].plot([], [], "r-", linewidth=2)
        self.axes[0, 0].set_title("Episode Rewards")
        self.axes[0, 0].set_xlabel("Episode")
        self.axes[0, 0].set_ylabel("Reward")
        self.axes[0, 0].grid(True)

        (self.length_line,) = self.axes[0, 1].plot([], [], "g-", alpha=0.7)
        (self.length_avg_line,) = self.axes[0, 1].plot([], [], "r-", linewidth=2)
        self.axes[0, 1].set_title("Episode Length")
        self.axes[0, 1].set_xlabel("Episode")
        self.axes[0, 1].set_ylabel("Steps")
        self.axes[0, 1].grid(True)

        (self.capture_line,) = self.axes[0, 2].plot([], [], "purple", linewidth=2)
        self.axes[0, 2].set_title("Capture Rate (last 100 episodes)")
        self.axes[0, 2].set_xlabel("Episode")
        self.axes[0, 2].set_ylabel("Capture Rate")
        self.axes[0, 2].grid(True)
        self.axes[0, 2].set_ylim(0, 1)

        (self.loss_line,) = self.axes[1, 0].plot([], [], "orange", alpha=0.7)
        self.axes[1, 0].set_title("Training Loss")
        self.axes[1, 0].set_xlabel("Training Step")
        self.axes[1, 0].set_ylabel("Loss")
        self.axes[1, 0].grid(True)
        self.axes[1, 0].set_yscale("log")

        # Environment size performance
        self.axes[1, 1].set_title("Performance by Environment Size")
        self.axes[1, 1].set_xlabel("Environment Size")
        self.axes[1, 1].set_ylabel("Average Reward (last 50 episodes)")
        self.axes[1, 1].grid(True)

        # Real-time episode visualization or expert match rate
        if self.use_pathfinding_guidance:
            (self.expert_match_line,) = self.axes[1, 2].plot(
                [], [], "orange", linewidth=2
            )
            self.axes[1, 2].set_title("Expert Action Match Rate")
            self.axes[1, 2].set_xlabel("Episode")
            self.axes[1, 2].set_ylabel("Match Rate")
            self.axes[1, 2].grid(True)
            self.axes[1, 2].set_ylim(0, 1)
        else:
            self.axes[1, 2].set_title("Current Episode Progress")
            self.axes[1, 2].set_xlabel("X")
            self.axes[1, 2].set_ylabel("Y")
            self.axes[1, 2].grid(True)

        plt.tight_layout()
        plt.ion()
        plt.show()

        # Add flow-specific tracking
        self.flow_stats = {
            "agent1_deployment_times": [],
            "agent2_deployment_times": [],
            "pursuit_start_times": [],
            "coordination_scores": [],
            "adaptive_demos_generated": 0,
            "struggle_episodes": [],
        }

        # Add performance monitoring and adaptive expert
        self.performance_monitor = PerformanceMonitor(window_size=100)
        self.adaptive_expert = None
        self.last_struggle_check = 0
        self.struggle_demo_count = 0

    def update_plots(self, episode):
        """Update all training visualization plots"""
        if not self.training_stats["episode_rewards"]:
            return

        # Update episode rewards
        episodes = list(range(len(self.training_stats["episode_rewards"])))
        rewards = self.training_stats["episode_rewards"]

        self.reward_line.set_data(episodes, rewards)

        # Moving average for rewards
        if len(rewards) >= 50:
            window_size = min(50, len(rewards))
            moving_avg = []
            for i in range(len(rewards)):
                start_idx = max(0, i - window_size + 1)
                avg = np.mean(rewards[start_idx : i + 1])
                moving_avg.append(avg)
            self.reward_avg_line.set_data(episodes, moving_avg)

        # Auto-scale rewards plot
        if rewards:
            self.axes[0, 0].set_xlim(0, max(10, len(rewards)))
            self.axes[0, 0].set_ylim(min(rewards) - 1, max(rewards) + 1)

        # Update episode lengths
        if self.training_stats["episode_lengths"]:
            lengths = self.training_stats["episode_lengths"]
            self.length_line.set_data(episodes, lengths)

            # Moving average for lengths
            if len(lengths) >= 50:
                window_size = min(50, len(lengths))
                moving_avg = []
                for i in range(len(lengths)):
                    start_idx = max(0, i - window_size + 1)
                    avg = np.mean(lengths[start_idx : i + 1])
                    moving_avg.append(avg)
                self.length_avg_line.set_data(episodes, moving_avg)

            # Auto-scale lengths plot
            self.axes[0, 1].set_xlim(0, max(10, len(lengths)))
            self.axes[0, 1].set_ylim(0, max(lengths) + 10)

        # Update capture rates
        if self.training_stats["capture_rates"]:
            capture_rates = self.training_stats["capture_rates"]
            self.capture_line.set_data(episodes, capture_rates)
            self.axes[0, 2].set_xlim(0, max(10, len(capture_rates)))

        # Update training loss
        if self.training_stats["losses"]:
            losses = self.training_stats["losses"]
            loss_episodes = list(range(len(losses)))
            self.loss_line.set_data(loss_episodes, losses)
            self.axes[1, 0].set_xlim(0, max(10, len(losses)))
            if losses:
                self.axes[1, 0].set_ylim(min(losses) * 0.1, max(losses) * 10)

        # Update performance by environment size
        if len(self.training_stats["environment_sizes"]) >= 50:
            self.axes[1, 1].clear()
            self.axes[1, 1].set_title("Performance by Environment Size")
            self.axes[1, 1].set_xlabel("Environment Size")
            self.axes[1, 1].set_ylabel("Average Reward (last 50 episodes)")
            self.axes[1, 1].grid(True)

            # Group by environment size
            env_performance = {}
            recent_data = list(
                zip(
                    self.training_stats["environment_sizes"][-50:],
                    self.training_stats["episode_rewards"][-50:],
                )
            )

            for env_size, reward in recent_data:
                if env_size not in env_performance:
                    env_performance[env_size] = []
                env_performance[env_size].append(reward)

            # Plot average performance for each environment size
            sizes = sorted(env_performance.keys())
            avg_rewards = [np.mean(env_performance[size]) for size in sizes]

            if sizes and avg_rewards:
                self.axes[1, 1].bar(sizes, avg_rewards, alpha=0.7)
                for i, (size, reward) in enumerate(zip(sizes, avg_rewards)):
                    self.axes[1, 1].text(
                        size, reward + 0.1, f"{reward:.1f}", ha="center", va="bottom"
                    )

        # Update expert match rate (if using pathfinding guidance)
        if self.use_pathfinding_guidance and self.training_stats["expert_match_rates"]:
            match_rates = self.training_stats["expert_match_rates"]
            match_episodes = list(range(len(match_rates)))
            self.expert_match_line.set_data(match_episodes, match_rates)
            self.axes[1, 2].set_xlim(0, max(10, len(match_rates)))

        # Refresh the plots
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except:
            pass  # In case of display issues

    def save_training_stats(self, filename):
        """Save training statistics to file"""
        stats_to_save = {
            "training_stats": self.training_stats,
            "flow_stats": self.flow_stats,
            "use_pathfinding_guidance": self.use_pathfinding_guidance,
            "guidance_weight": self.guidance_weight,
        }

        try:
            with open(filename, "wb") as f:
                pickle.dump(stats_to_save, f)
            print(f"Training statistics saved to {filename}")
        except Exception as e:
            print(f"Warning: Could not save training statistics: {e}")

    def load_training_stats(self, filename):
        """Load training statistics from file"""
        try:
            with open(filename, "rb") as f:
                data = pickle.load(f)
                self.training_stats = data["training_stats"]
                self.flow_stats = data["flow_stats"]
                print(f"Training statistics loaded from {filename}")
                return True
        except Exception as e:
            print(f"Warning: Could not load training statistics: {e}")
            return False

    def plot_final_training_summary(self):
        """Create a comprehensive training summary plot"""
        if not self.training_stats["episode_rewards"]:
            print("No training data to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("QMIX Training Summary", fontsize=16)

        # Episode rewards with moving average
        episodes = list(range(len(self.training_stats["episode_rewards"])))
        rewards = self.training_stats["episode_rewards"]

        axes[0, 0].plot(episodes, rewards, alpha=0.3, label="Episode Reward")
        if len(rewards) >= 10:
            # Smooth moving average
            window = min(100, len(rewards) // 10)
            smooth_rewards = []
            for i in range(len(rewards)):
                start = max(0, i - window + 1)
                smooth_rewards.append(np.mean(rewards[start : i + 1]))
            axes[0, 0].plot(
                episodes,
                smooth_rewards,
                linewidth=2,
                label=f"Moving Average ({window})",
            )

        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Capture rate progression
        if self.training_stats["capture_rates"]:
            axes[0, 1].plot(
                episodes, self.training_stats["capture_rates"], "g-", linewidth=2
            )
            axes[0, 1].set_title("Capture Rate (Rolling Average)")
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Capture Rate")
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].grid(True)

        # Episode length progression
        if self.training_stats["episode_lengths"]:
            lengths = self.training_stats["episode_lengths"]
            axes[0, 2].plot(episodes, lengths, alpha=0.3, label="Episode Length")
            if len(lengths) >= 10:
                window = min(100, len(lengths) // 10)
                smooth_lengths = []
                for i in range(len(lengths)):
                    start = max(0, i - window + 1)
                    smooth_lengths.append(np.mean(lengths[start : i + 1]))
                axes[0, 2].plot(
                    episodes,
                    smooth_lengths,
                    linewidth=2,
                    label=f"Moving Average ({window})",
                )

            axes[0, 2].set_title("Episode Lengths")
            axes[0, 2].set_xlabel("Episode")
            axes[0, 2].set_ylabel("Steps")
            axes[0, 2].legend()
            axes[0, 2].grid(True)

        # Training loss
        if self.training_stats["losses"]:
            losses = self.training_stats["losses"]
            loss_steps = list(range(len(losses)))
            axes[1, 0].plot(loss_steps, losses, alpha=0.7)
            axes[1, 0].set_title("Training Loss")
            axes[1, 0].set_xlabel("Training Step")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_yscale("log")
            axes[1, 0].grid(True)

        # Performance by environment size
        if self.training_stats["environment_sizes"]:
            env_performance = {}
            for env_size, reward in zip(
                self.training_stats["environment_sizes"],
                self.training_stats["episode_rewards"],
            ):
                if env_size not in env_performance:
                    env_performance[env_size] = []
                env_performance[env_size].append(reward)

            sizes = sorted(env_performance.keys())
            avg_rewards = [np.mean(env_performance[size]) for size in sizes]
            std_rewards = [np.std(env_performance[size]) for size in sizes]

            axes[1, 1].bar(sizes, avg_rewards, yerr=std_rewards, alpha=0.7, capsize=5)
            axes[1, 1].set_title("Performance by Environment Size")
            axes[1, 1].set_xlabel("Environment Size")
            axes[1, 1].set_ylabel("Average Reward")
            axes[1, 1].grid(True)

        # Expert match rate (if available)
        if self.use_pathfinding_guidance and self.training_stats["expert_match_rates"]:
            match_rates = self.training_stats["expert_match_rates"]
            match_episodes = list(range(len(match_rates)))
            axes[1, 2].plot(match_episodes, match_rates, "orange", linewidth=2)
            axes[1, 2].set_title("Expert Action Match Rate")
            axes[1, 2].set_xlabel("Episode")
            axes[1, 2].set_ylabel("Match Rate")
            axes[1, 2].set_ylim(0, 1)
            axes[1, 2].grid(True)
        else:
            # Flow statistics summary
            axes[1, 2].text(
                0.1,
                0.8,
                "Flow Statistics:",
                fontsize=14,
                weight="bold",
                transform=axes[1, 2].transAxes,
            )

            if self.flow_stats["agent1_deployment_times"]:
                avg_deploy1 = np.mean(
                    [t for t in self.flow_stats["agent1_deployment_times"] if t > 0]
                )
                axes[1, 2].text(
                    0.1,
                    0.6,
                    f"Avg Agent 1 Deploy: {avg_deploy1:.1f} steps",
                    transform=axes[1, 2].transAxes,
                )

            if self.flow_stats["agent2_deployment_times"]:
                avg_deploy2 = np.mean(
                    [t for t in self.flow_stats["agent2_deployment_times"] if t > 0]
                )
                axes[1, 2].text(
                    0.1,
                    0.4,
                    f"Avg Agent 2 Deploy: {avg_deploy2:.1f} steps",
                    transform=axes[1, 2].transAxes,
                )

            final_capture_rate = (
                np.mean(self.training_stats["capture_rates"][-100:])
                if len(self.training_stats["capture_rates"]) >= 100
                else 0
            )
            axes[1, 2].text(
                0.1,
                0.2,
                f"Final Capture Rate: {final_capture_rate:.3f}",
                transform=axes[1, 2].transAxes,
            )

            axes[1, 2].set_title("Training Summary")
            axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

        # Print final statistics
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED - FINAL STATISTICS")
        print("=" * 50)

        if self.training_stats["episode_rewards"]:
            final_avg_reward = np.mean(self.training_stats["episode_rewards"][-100:])
            print(f"Final Average Reward (last 100 episodes): {final_avg_reward:.2f}")

        if self.training_stats["capture_rates"]:
            final_capture_rate = self.training_stats["capture_rates"][-1]
            print(f"Final Capture Rate: {final_capture_rate:.3f}")

        if self.training_stats["episode_lengths"]:
            final_avg_length = np.mean(self.training_stats["episode_lengths"][-100:])
            print(f"Final Average Episode Length: {final_avg_length:.1f} steps")

        if self.use_pathfinding_guidance and self.training_stats["expert_match_rates"]:
            final_match_rate = np.mean(self.training_stats["expert_match_rates"][-100:])
            print(f"Final Expert Match Rate: {final_match_rate:.3f}")

        print("=" * 50)

    def train(
        self, num_episodes=5000, visualize_every=50, save_every=500, expert_ratio=0.4
    ):
        """Training with pathfinding flow imitation"""
        print("Starting QMIX training with pathfinding flow imitation...")
        print(f"Expert guidance ratio: {expert_ratio}")
        print(f"Both agents start from (0.5, 0.5) every episode")

        # Create environments
        environments = create_test_environments()

        # Create flow-based expert systems
        expert_systems = []
        if self.use_pathfinding_guidance:
            for env in environments:
                try:
                    expert_system = PathfindingFlowExpertSystem(env)
                    expert_systems.append(expert_system)
                    print(
                        f"Created flow expert for {env.width}x{env.height} environment"
                    )
                except Exception as e:
                    print(
                        f"Warning: Could not create expert for {env.width}x{env.height}: {e}"
                    )
                    expert_systems.append(None)

        # Create adaptive expert system
        if self.use_pathfinding_guidance:
            self.adaptive_expert = AdaptivePathfindingExpert(environments[0])

        # Initialize agent
        sample_env = environments[0]
        global_state, individual_obs = sample_env.reset()
        actual_state_dim = len(individual_obs[0])
        print(f"Actual state dimension: {actual_state_dim}")

        # Initialize agent with correct state dimension
        agent = QMIX(
            n_agents=2,
            state_dim=actual_state_dim,
            action_dim=8,
            lr=0.0002,  # Slightly higher learning rate
            gamma=0.95,  # Good discount factor
            epsilon=0.9,  # Start with high exploration
            epsilon_decay=0.999,  # Very slow decay
            epsilon_min=0.1,  # Higher minimum exploration
            device=self.device,
        )

        # Add learning rate scheduling
        scheduler = torch.optim.lr_scheduler.StepLR(
            agent.optimizer, step_size=1000, gamma=0.95
        )

        # Pre-populate with flow-based demonstrations
        if self.use_pathfinding_guidance:
            print("Generating flow-based expert demonstrations...")
            demo_count = 0

            for i, (env, expert_system) in enumerate(
                zip(environments[:3], expert_systems[:3])
            ):
                if expert_system is None:
                    continue

                print(f"Generating flow demos for environment {i + 1}")
                for demo_idx in range(3):  # 3 demonstrations per environment
                    try:
                        demo_data = expert_system.generate_demonstration_episode()

                        # Add to replay buffer
                        for j in range(len(demo_data["states"]) - 1):
                            global_state, individual_obs = demo_data["states"][j]
                            next_global_state, next_individual_obs = demo_data[
                                "next_states"
                            ][j]

                            # Combine individual states for storage
                            combined_state = np.concatenate(individual_obs)
                            next_combined_state = np.concatenate(next_individual_obs)

                            agent.memory.push(
                                combined_state,
                                demo_data["actions"][j],
                                demo_data["rewards"][j],
                                next_combined_state,
                                demo_data["dones"][j],
                            )
                            demo_count += 1

                        print(
                            f"  Flow demo {demo_idx + 1}/3 completed "
                            f"({len(demo_data['states'])} steps, "
                            f"captured: {demo_data['info'][-1].get('captured', False)})"
                        )

                    except Exception as e:
                        print(f"  Warning: Flow demo {demo_idx + 1} failed: {e}")

            print(f"Added {demo_count} flow-based demonstrations")

        # Training loop
        for episode in range(num_episodes):
            # Select environment
            env_idx = random.randint(0, len(environments) - 1)
            env = environments[env_idx]
            expert_system = (
                expert_systems[env_idx] if self.use_pathfinding_guidance else None
            )

            # Reset environment (agents start at (0.5, 0.5))
            global_state, individual_obs = env.reset()

            episode_reward = 0
            episode_start = time.time()
            done = False
            step = 0

            # Flow tracking
            episode_expert_matches = 0
            episode_total_actions = 0

            while not done and step < env.max_steps:
                # Decide whether to use expert guidance
                use_expert_this_step = (
                    self.use_pathfinding_guidance
                    and expert_system is not None
                    and random.random() < expert_ratio
                    and episode < num_episodes * 0.8  # Reduce expert guidance later
                )

                if use_expert_this_step:
                    # Get expert actions following the flow
                    try:
                        expert_actions = (
                            expert_system.generate_expert_actions_with_flow(env)
                        )
                        actions = expert_actions.copy()

                        # Count deployed agents for matching
                        deployed_count = sum([env.agent1_deployed, env.agent2_deployed])
                        episode_expert_matches += deployed_count
                    except Exception as e:
                        actions = agent.get_actions(individual_obs)
                        expert_actions = actions
                        print(f"Expert system failed: {e}")
                else:
                    # Use learned policy
                    actions = agent.get_actions(individual_obs)

                    # Get expert actions for comparison
                    if self.use_pathfinding_guidance and expert_system is not None:
                        try:
                            expert_actions = (
                                expert_system.generate_expert_actions_with_flow(env)
                            )
                            # Count matches for deployed agents only
                            for i, (action, expert_action) in enumerate(
                                zip(actions, expert_actions)
                            ):
                                if (i == 0 and env.agent1_deployed) or (
                                    i == 1 and env.agent2_deployed
                                ):
                                    if action == expert_action:
                                        episode_expert_matches += 1
                        except:
                            expert_actions = actions
                    else:
                        expert_actions = actions

                # Count total actions for deployed agents
                episode_total_actions += sum([env.agent1_deployed, env.agent2_deployed])

                # Take step
                (next_global_state, next_individual_obs), reward, done, info = env.step(
                    actions
                )

                # Add flow-based reward shaping
                if self.use_pathfinding_guidance and expert_system is not None:
                    flow_reward = self.calculate_flow_reward(
                        env, actions, expert_actions
                    )
                    enhanced_reward = reward + self.guidance_weight * flow_reward
                else:
                    enhanced_reward = reward

                # Store experience
                agent.store_experience(
                    individual_obs, actions, enhanced_reward, next_individual_obs, done
                )

                # Train
                loss = agent.train()
                if loss:
                    self.training_stats["losses"].append(loss)

                # Update state
                global_state = next_global_state
                individual_obs = next_individual_obs
                episode_reward += enhanced_reward
                step += 1

                # Store current scenario for potential expert demonstration
                current_scenario = {
                    "agent1_pos": env.agent_positions[0],
                    "agent2_pos": env.agent_positions[1],
                    "target_pos": env.target.position,
                    "target_speed": env.target.speed,
                    "agent1_deployed": env.agent1_deployed,
                    "agent2_deployed": env.agent2_deployed,
                    "pursuit_active": env.pursuit_active,
                    "step_count": env.step_count,
                    "episode": episode,
                    "step": step,
                }

                # Visualize occasionally
                if episode % visualize_every == 0 and step % 20 == 0:
                    self.visualize_episode_with_flow(env, episode, step)

            # Episode finished
            episode_time = time.time() - episode_start
            captured = info.get("captured", False)

            # Track flow statistics
            self.flow_stats["agent1_deployment_times"].append(
                info.get("agent1_deploy_step", -1)
            )
            self.flow_stats["agent2_deployment_times"].append(
                info.get("agent2_deploy_step", -1)
            )

            # Expert matching rate
            if self.use_pathfinding_guidance and episode_total_actions > 0:
                expert_match_rate = episode_expert_matches / episode_total_actions
                self.training_stats["expert_match_rates"].append(expert_match_rate)

            # Update statistics
            self.training_stats["episode_rewards"].append(episode_reward)
            self.training_stats["episode_lengths"].append(step)
            self.training_stats["capture_rates"].append(
                np.mean(self.training_stats["capture_rates"][-100:])
            )
            self.training_stats["environment_sizes"].append(env.width)
            self.training_stats["episode_times"].append(episode_time)

            # Update performance monitor
            self.performance_monitor.add_episode(captured, episode_reward, step)

            # Check if struggling
            if (
                episode > 200 and episode - self.last_struggle_check >= 50
            ):  # Check every 50 episodes
                struggling, struggle_info = self.performance_monitor.is_struggling()

                if struggling and self.adaptive_expert is not None:
                    print(f"\n🚨 ALGORITHM STRUGGLING at episode {episode}")
                    print(f"📊 Reason: {struggle_info['reason']}")
                    print(f"🎯 Capture rate: {struggle_info['capture_rate']:.3f}")
                    print(f"📏 Avg episode length: {struggle_info['avg_length']:.1f}")

                    # Generate expert demonstrations for recent struggling scenarios
                    self._generate_adaptive_demonstrations(env, episode, agent)

                    # Temporarily increase expert guidance
                    original_expert_ratio = expert_ratio
                    expert_ratio = min(0.9, expert_ratio + 0.3)
                    print(
                        f"🔧 Temporarily increased expert guidance to {expert_ratio:.2f}"
                    )

                    # Record struggle episode
                    self.flow_stats["struggle_episodes"].append(
                        {
                            "episode": episode,
                            "reason": struggle_info["reason"],
                            "capture_rate": struggle_info["capture_rate"],
                        }
                    )

                    self.last_struggle_check = episode

                    # Reduce expert ratio back after some episodes
                    if episode % 100 == 0 and expert_ratio > original_expert_ratio:
                        expert_ratio = max(original_expert_ratio, expert_ratio - 0.1)
                        print(f"🔄 Reduced expert guidance back to {expert_ratio:.2f}")

            # Update visualization
            if episode % 10 == 0:
                try:
                    self.update_plots(episode)
                except Exception as e:
                    print(f"Warning: Could not update plots: {e}")

            # Progress reporting
            if episode % 100 == 0:
                avg_reward = np.mean(self.training_stats["episode_rewards"][-100:])
                capture_rate = np.mean(self.performance_monitor.recent_captures)

                progress_text = (
                    f"Episode {episode:4d} | Avg Reward: {avg_reward:7.2f} | "
                    f"Capture Rate: {capture_rate:.3f} | Epsilon: {agent.epsilon:.3f} | "
                    f"Adaptive Demos: {self.flow_stats['adaptive_demos_generated']}"
                )

                if (
                    self.use_pathfinding_guidance
                    and self.training_stats["expert_match_rates"]
                ):
                    expert_match_rate_avg = np.mean(
                        self.training_stats["expert_match_rates"]
                    )
                    progress_text += f" | Expert Match: {expert_match_rate_avg:.3f}"

                print(progress_text)

                # Show struggle history
                if self.flow_stats["struggle_episodes"]:
                    recent_struggles = [
                        s
                        for s in self.flow_stats["struggle_episodes"]
                        if s["episode"] > episode - 500
                    ]
                    if recent_struggles:
                        print(f"📊 Recent struggles: {len(recent_struggles)} episodes")

                # Update learning rate
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                print(f"Learning rate updated to: {current_lr:.6f}")

            # Reduce expert guidance gradually
            if self.use_pathfinding_guidance and episode % 500 == 0 and episode > 0:
                expert_ratio = max(0.05, expert_ratio * 0.9)
                print(f"Reduced expert guidance to {expert_ratio:.3f}")

            # Save periodically
            if episode % save_every == 0 and episode > 0:
                model_name = (
                    "qmix_pathfinding_flow"
                    if self.use_pathfinding_guidance
                    else "qmix_model"
                )
                agent.save(f"{model_name}_episode_{episode}.pth")
                self.save_training_stats(f"flow_training_stats_episode_{episode}.pkl")

            # In the training loop, gradually increase difficulty:
            if episode > 0 and episode % 1000 == 0:
                # Gradually make capture conditions stricter
                if hasattr(env, "capture_radius"):
                    env.capture_radius = max(1.0, env.capture_radius * 0.95)
                    print(f"Reduced capture radius to {env.capture_radius:.2f}")

        # Final save
        model_name = (
            "qmix_pathfinding_flow_final"
            if self.use_pathfinding_guidance
            else "qmix_model_final"
        )
        agent.save(f"{model_name}.pth")
        self.save_training_stats("flow_training_stats_final.pkl")

        print("Training completed!")
        print(
            f"Final capture rate: {np.mean(self.training_stats['capture_rates'][-100:]):.3f}"
        )

        # Show final training summary
        self.plot_final_training_summary()

        return agent

    def calculate_flow_reward(self, env, actions, expert_actions):
        """Calculate reward for following the pathfinding flow"""
        reward = 0.0

        # Reward for correct deployment timing
        if env.step_count == env.agent1_deploy_step and env.agent1_deployed:
            reward += 1.0

        if env.step_count == env.agent2_deploy_step and env.agent2_deployed:
            reward += 1.0

        # Reward for matching expert actions when deployed
        if env.agent1_deployed and actions[0] == expert_actions[0]:
            reward += 0.5

        if env.agent2_deployed and actions[1] == expert_actions[1]:
            reward += 0.5

        return reward

    def visualize_episode_with_flow(self, env, episode, step):
        """Visualize episode with flow information"""
        if self.use_pathfinding_guidance:
            return  # Skip when using pathfinding guidance

        self.axes[1, 2].clear()

        # Plot environment
        for ox, oy, ow, oh in env.env.obstacles:
            rect = plt.Rectangle((ox, oy), ow, oh, color="gray", alpha=0.7)
            self.axes[1, 2].add_patch(rect)

        # Plot agents with deployment status
        colors = [
            "red" if env.agent1_deployed else "lightcoral",
            "blue" if env.agent2_deployed else "lightblue",
        ]

        for i, pos in enumerate(env.agent_positions):
            self.axes[1, 2].plot(
                pos[0],
                pos[1],
                "o",
                color=colors[i],
                markersize=8,
                label=f"Agent {i + 1} ({'active' if (i == 0 and env.agent1_deployed) or (i == 1 and env.agent2_deployed) else 'waiting'})",
            )

        # Plot target
        target_color = "green" if env.pursuit_active else "lightgreen"
        self.axes[1, 2].plot(
            env.target.position[0],
            env.target.position[1],
            "s",
            color=target_color,
            markersize=8,
            label=f"Target ({'fleeing' if env.pursuit_active else 'stationary'})",
        )

        self.axes[1, 2].set_xlim(0, env.width)
        self.axes[1, 2].set_ylim(0, env.height)
        self.axes[1, 2].set_title(f"Episode {episode}, Step {step}")
        self.axes[1, 2].legend()
        self.axes[1, 2].grid(True)

        plt.pause(0.01)

    def _generate_adaptive_demonstrations(self, current_env, current_episode, agent):
        """Generate expert demonstrations for the current struggling scenario"""
        try:
            # Create scenario config from current environment state
            scenario_config = {
                "agent1_pos": current_env.agent_positions[0],
                "agent2_pos": current_env.agent_positions[1],
                "target_pos": current_env.target.position,
                "target_speed": current_env.target.speed,
                "agent1_deployed": current_env.agent1_deployed,
                "agent2_deployed": current_env.agent2_deployed,
                "pursuit_active": current_env.pursuit_active,
                "step_count": current_env.step_count,
            }

            # Generate expert demonstration using pathfinding
            demo_data = self.adaptive_expert.generate_expert_demonstration_for_scenario(
                scenario_config
            )

            # Add demonstrations to replay buffer with high priority
            demo_count = 0
            for j in range(len(demo_data["states"]) - 1):
                global_state, individual_obs = demo_data["states"][j]
                next_global_state, next_individual_obs = demo_data["next_states"][j]

                # Combine individual states for storage
                combined_state = np.concatenate(individual_obs)
                next_combined_state = np.concatenate(next_individual_obs)

                # Add multiple copies for higher priority (pathfinding is very good!)
                for _ in range(3):  # Add each transition 3 times
                    agent.memory.push(
                        combined_state.copy(),
                        demo_data["actions"][j],
                        demo_data["rewards"][j],
                        next_combined_state.copy(),
                        demo_data["dones"][j],
                    )
                    demo_count += 1

            self.struggle_demo_count += demo_count
            self.flow_stats["adaptive_demos_generated"] += 1

            print(f"✅ Added {demo_count} adaptive expert transitions to replay buffer")
            print(
                f"📈 Total adaptive demos generated: {self.flow_stats['adaptive_demos_generated']}"
            )

        except Exception as e:
            print(f"❌ Failed to generate adaptive demonstration: {e}")


# Update the main section
if __name__ == "__main__":
    # Check device availability
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        device = "cpu"
        print("Using CPU")

    # Choose training mode
    use_pathfinding_guidance = True  # Set to False for standard QMIX training

    if use_pathfinding_guidance and PATHFINDING_AVAILABLE:
        print("=== PATHFINDING FLOW IMITATION LEARNING ===")
        trainer = QMixTrainer(
            device=device,
            use_pathfinding_guidance=True,
            guidance_weight=0.4,
        )

        # Use more forgiving training parameters:
        trained_agent = trainer.train(
            num_episodes=4000,  # More episodes
            visualize_every=200,  # Less frequent visualization
            save_every=500,
            expert_ratio=0.8,  # Higher expert guidance
        )
    else:
        if use_pathfinding_guidance and not PATHFINDING_AVAILABLE:
            print(
                "Pathfinding guidance requested but not available, falling back to standard training"
            )

        print("=== STANDARD QMIX TRAINING ===")
        trainer = QMixTrainer(device=device, use_pathfinding_guidance=False)
        trained_agent = trainer.train(num_episodes=3000, visualize_every=50)

    print("Training completed! Press Enter to close plots and exit...")
    input()
