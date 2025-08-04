import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation
import random
import time
from qmix_marl import QMIX
from pursuit_environment import create_test_environments


class TrainedAgentVisualizer:
    """Visualize trained QMIX agents in action"""

    def __init__(self, model_path=None):
        """
        Initialize the visualizer

        Args:
            model_path: Path to trained QMIX model
        """
        print("Initializing Trained Agent Visualizer")

        # Load environments
        self.environments = create_test_environments()
        self.current_env_idx = 0
        self.current_env = None
        self.trained_agent = None

        # Episode tracking
        self.episode_history = []
        self.current_episode_data = {
            "positions": {"agent1": [], "agent2": [], "target": []},
            "phases": [],
            "rewards": [],
            "actions": [],
        }

        # Animation controls
        self.is_running = False
        self.is_paused = False
        self.current_step = 0
        self.max_steps = 200
        self.step_delay = 0.1  # seconds between steps

        # Load trained model
        if model_path:
            self.load_trained_model(model_path)
        else:
            print("No model provided - will use random actions")

        # Setup visualization
        self.setup_visualization()
        self.reset_current_environment()

    def load_trained_model(self, model_path):
        """Load the trained QMIX model"""
        try:
            # Get state dimension from environment
            sample_env = self.environments[0]
            global_state, individual_obs = sample_env.reset()
            state_dim = len(individual_obs[0])

            self.trained_agent = QMIX(
                n_agents=2,
                state_dim=state_dim,
                action_dim=8,
                device="cpu",  # Use CPU for visualization
            )
            self.trained_agent.load(model_path)
            self.trained_agent.epsilon = 0.0  # No exploration for evaluation
            print(f"Loaded trained model from {model_path}")

        except Exception as e:
            print(f"Failed to load model: {e}")
            self.trained_agent = None

    def setup_visualization(self):
        """Setup the visualization interface"""
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))

        # Main environment plot (larger)
        self.ax_main = plt.subplot2grid((3, 4), (0, 0), colspan=3, rowspan=2)
        self.ax_main.set_title("Hybrid QMIX: Agent 1 (A*) + Agent 2 (MARL)")
        self.ax_main.set_xlabel("X Position")
        self.ax_main.set_ylabel("Y Position")
        self.ax_main.grid(True, alpha=0.3)

        # Performance metrics
        self.ax_metrics = plt.subplot2grid((3, 4), (0, 3))
        self.ax_metrics.set_title("Episode Metrics")
        self.ax_metrics.axis("off")

        # Phase tracking
        self.ax_phases = plt.subplot2grid((3, 4), (1, 3))
        self.ax_phases.set_title("Phase Progress")
        self.ax_phases.set_xlabel("Step")
        self.ax_phases.set_ylabel("Phase")

        # Controls and info
        self.ax_controls = plt.subplot2grid((3, 4), (2, 0), colspan=4)
        self.ax_controls.axis("off")

        # Create control buttons
        self.create_control_buttons()

        # Create sliders
        self.create_sliders()

        plt.tight_layout()

    def create_control_buttons(self):
        """Create control buttons for the visualization"""
        button_height = 0.04
        button_width = 0.08
        button_y = 0.02

        # Start/Stop button
        ax_start = plt.axes([0.05, button_y, button_width, button_height])
        self.btn_start = Button(ax_start, "Start")
        self.btn_start.on_clicked(self.toggle_simulation)

        # Pause/Resume button
        ax_pause = plt.axes([0.15, button_y, button_width, button_height])
        self.btn_pause = Button(ax_pause, "Pause")
        self.btn_pause.on_clicked(self.toggle_pause)

        # Step button
        ax_step = plt.axes([0.25, button_y, button_width, button_height])
        self.btn_step = Button(ax_step, "Step")
        self.btn_step.on_clicked(self.single_step)

        # Reset button
        ax_reset = plt.axes([0.35, button_y, button_width, button_height])
        self.btn_reset = Button(ax_reset, "Reset")
        self.btn_reset.on_clicked(self.reset_episode)

        # Next Environment button
        ax_next_env = plt.axes([0.45, button_y, button_width, button_height])
        self.btn_next_env = Button(ax_next_env, "Next Env")
        self.btn_next_env.on_clicked(self.next_environment)

        # Analyze button
        ax_analyze = plt.axes([0.55, button_y, button_width, button_height])
        self.btn_analyze = Button(ax_analyze, "Analyze")
        self.btn_analyze.on_clicked(self.analyze_performance)

        # Save Episode button
        ax_save = plt.axes([0.65, button_y, button_width, button_height])
        self.btn_save = Button(ax_save, "Save Ep")
        self.btn_save.on_clicked(self.save_episode)

    def create_sliders(self):
        """Create control sliders"""
        # Speed slider
        ax_speed = plt.axes([0.80, 0.02, 0.15, 0.04])
        self.speed_slider = Slider(ax_speed, "Speed", 0.01, 1.0, valinit=0.1)
        self.speed_slider.on_changed(self.update_speed)

        # Environment selector
        ax_env = plt.axes([0.80, 0.08, 0.15, 0.04])
        self.env_slider = Slider(
            ax_env, "Environment", 0, len(self.environments) - 1, valinit=0, valfmt="%d"
        )
        self.env_slider.on_changed(self.change_environment)

    def reset_current_environment(self):
        """Reset the current environment"""
        self.current_env = self.environments[self.current_env_idx]
        global_state, individual_obs = self.current_env.reset()
        self.current_step = 0
        self.is_running = False
        self.is_paused = False

        # Reset episode data
        self.current_episode_data = {
            "positions": {"agent1": [], "agent2": [], "target": []},
            "phases": [],
            "rewards": [],
            "actions": [],
            "info": [],
        }

        # Store initial positions
        self.current_episode_data["positions"]["agent1"].append(
            self.current_env.agent_positions[0]
        )
        self.current_episode_data["positions"]["agent2"].append(
            self.current_env.agent_positions[1]
        )
        self.current_episode_data["positions"]["target"].append(
            self.current_env.target.position
        )

        self.update_display()

    def get_agent_actions(self, individual_obs):
        """Get actions from trained model or random"""
        if self.trained_agent is not None:
            return self.trained_agent.get_actions(individual_obs)
        else:
            # Random actions for demo
            return [random.randint(0, 7), random.randint(0, 7)]

    def single_step(self, event=None):
        """Execute a single step"""
        if self.current_step >= self.max_steps:
            print("Episode completed")
            return

        # Get current state - FIX: Remove print statement
        _, individual_obs = (
            self.current_env._get_global_state(),
            self.current_env._get_individual_observations(),
        )
        # Get actions from trained model
        actions = self.get_agent_actions(individual_obs)

        # Take step
        (next_global_state, next_individual_obs), reward, done, info = (
            self.current_env.step(actions)
        )

        # Store step data
        self.current_episode_data["positions"]["agent1"].append(
            self.current_env.agent_positions[0]
        )
        self.current_episode_data["positions"]["agent2"].append(
            self.current_env.agent_positions[1]
        )
        self.current_episode_data["positions"]["target"].append(
            self.current_env.target.position
        )
        self.current_episode_data["phases"].append(info.get("phase", "unknown"))
        self.current_episode_data["rewards"].append(reward)
        self.current_episode_data["actions"].append(actions)
        self.current_episode_data["info"].append(info)

        self.current_step += 1

        # Update display
        self.update_display()

        # Print step information
        phase = info.get("phase", "unknown")
        training_phase = info.get("training_phase", False)
        print(
            f"Step {self.current_step}: {phase.upper()} | "
            f"Agent 1: {'A*' if not info.get('agent1_learning', False) else 'Learning'} | "
            f"Agent 2: {'Learning' if info.get('agent2_learning', False) else 'Waiting'} | "
            f"Reward: {reward:.3f} | Training: {training_phase}"
        )

        if done or self.current_step >= self.max_steps:
            self.is_running = False
            captured = info.get("captured", False)
            print(f"\n Episode finished! Captured: {captured}")
            self.episode_history.append(self.current_episode_data.copy())

    def update_display(self):
        """Update the visualization display"""
        # Clear main plot
        self.ax_main.clear()

        # Plot environment obstacles
        for ox, oy, ow, oh in self.current_env.env.obstacles:
            rect = patches.Rectangle(
                (ox, oy), ow, oh, facecolor="gray", alpha=0.7, edgecolor="black"
            )
            self.ax_main.add_patch(rect)

        # Plot agent trajectories
        if len(self.current_episode_data["positions"]["agent1"]) > 1:
            agent1_x = [
                pos[0] for pos in self.current_episode_data["positions"]["agent1"]
            ]
            agent1_y = [
                pos[1] for pos in self.current_episode_data["positions"]["agent1"]
            ]
            self.ax_main.plot(
                agent1_x,
                agent1_y,
                "r-",
                alpha=0.5,
                linewidth=2,
                label="Agent 1 (A*) Path",
            )

        if len(self.current_episode_data["positions"]["agent2"]) > 1:
            agent2_x = [
                pos[0] for pos in self.current_episode_data["positions"]["agent2"]
            ]
            agent2_y = [
                pos[1] for pos in self.current_episode_data["positions"]["agent2"]
            ]
            self.ax_main.plot(
                agent2_x,
                agent2_y,
                "b-",
                alpha=0.5,
                linewidth=2,
                label="Agent 2 (MARL) Path",
            )

        # Plot target trajectory
        if len(self.current_episode_data["positions"]["target"]) > 1:
            target_x = [
                pos[0] for pos in self.current_episode_data["positions"]["target"]
            ]
            target_y = [
                pos[1] for pos in self.current_episode_data["positions"]["target"]
            ]
            self.ax_main.plot(
                target_x, target_y, "g--", alpha=0.5, linewidth=2, label="Target Path"
            )

        # Plot current positions
        current_info = (
            self.current_episode_data["info"][-1]
            if self.current_episode_data["info"]
            else {}
        )

        # Agent 1 (changes color based on deployment)
        agent1_color = "red" if current_info.get("agent1_deployed", False) else "pink"
        agent1_size = 150 if current_info.get("agent1_deployed", False) else 100
        self.ax_main.scatter(
            *self.current_env.agent_positions[0],
            c=agent1_color,
            s=agent1_size,
            marker="o",
            edgecolors="black",
            linewidth=2,
            label=f"Agent 1 ({'A* Active' if current_info.get('agent1_deployed', False) else 'Waiting'})",
        )

        # Agent 2 (changes color based on deployment)
        agent2_color = (
            "blue" if current_info.get("agent2_deployed", False) else "lightblue"
        )
        agent2_size = 150 if current_info.get("agent2_deployed", False) else 100
        self.ax_main.scatter(
            *self.current_env.agent_positions[1],
            c=agent2_color,
            s=agent2_size,
            marker="s",
            edgecolors="black",
            linewidth=2,
            label=f"Agent 2 ({'MARL Active' if current_info.get('agent2_deployed', False) else 'Waiting'})",
        )

        # Target (changes based on pursuit status)
        target_color = (
            "green" if current_info.get("pursuit_active", False) else "lightgreen"
        )
        target_size = 120
        self.ax_main.scatter(
            *self.current_env.target.position,
            c=target_color,
            s=target_size,
            marker="^",
            edgecolors="black",
            linewidth=2,
            label=f"Target ({'Fleeing' if current_info.get('pursuit_active', False) else 'Stationary'})",
        )

        # Set plot properties
        self.ax_main.set_xlim(0, self.current_env.width)
        self.ax_main.set_ylim(0, self.current_env.height)
        self.ax_main.set_title(
            f"Environment {self.current_env_idx + 1} ({self.current_env.width}x{self.current_env.height}) - Step {self.current_step}"
        )
        self.ax_main.set_xlabel("X Position")
        self.ax_main.set_ylabel("Y Position")
        self.ax_main.grid(True, alpha=0.3)
        self.ax_main.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Update metrics display
        self.update_metrics_display()

        # Update phase tracking
        self.update_phase_display()

        # Refresh the plot
        self.fig.canvas.draw()

    def update_metrics_display(self):
        """Update the metrics display"""
        self.ax_metrics.clear()
        self.ax_metrics.axis("off")

        # Current episode metrics
        total_reward = sum(self.current_episode_data["rewards"])
        current_info = (
            self.current_episode_data["info"][-1]
            if self.current_episode_data["info"]
            else {}
        )

        metrics_text = f""" Episode Metrics:
        
Step: {self.current_step}
Total Reward: {total_reward:.2f}
Phase: {current_info.get("phase", "unknown").upper()}
Training: {current_info.get("training_phase", False)}

Agent Status:
Agent 1: {"A* Deployed" if current_info.get("agent1_deployed", False) else "Waiting"}
Agent 2: {"MARL Learning" if current_info.get("agent2_deployed", False) else "Waiting"}
Pursuit: {"Active" if current_info.get("pursuit_active", False) else "Inactive"}

Performance:
Episodes Completed: {len(self.episode_history)}
Model: {"Trained QMIX" if self.trained_agent else "Random Actions"}
Capture: {current_info.get("captured", False)}
"""

        self.ax_metrics.text(
            0.05,
            0.95,
            metrics_text,
            transform=self.ax_metrics.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )

    def update_phase_display(self):
        """Update the phase tracking display"""
        self.ax_phases.clear()

        if len(self.current_episode_data["phases"]) > 0:
            steps = list(range(len(self.current_episode_data["phases"])))

            # Convert phases to numeric for plotting
            phase_mapping = {
                "waiting": 0,
                "pathfinding": 1,
                "pursuit": 1,
                "coordination": 2,
                "hybrid_coordination": 2,
            }
            numeric_phases = [
                phase_mapping.get(phase, 0)
                for phase in self.current_episode_data["phases"]
            ]

            self.ax_phases.plot(steps, numeric_phases, "o-", linewidth=2, markersize=4)
            self.ax_phases.set_xlabel("Step")
            self.ax_phases.set_ylabel("Phase")
            self.ax_phases.set_yticks([0, 1, 2])
            self.ax_phases.set_yticklabels(
                ["Waiting", "A* Pathfinding", "MARL Coordination"]
            )
            self.ax_phases.grid(True, alpha=0.3)
            self.ax_phases.set_title("Training Phase Progress")

            # Highlight current step
            if len(steps) > 0:
                self.ax_phases.axvline(
                    x=len(steps) - 1, color="red", linestyle="--", alpha=0.7
                )

    def toggle_simulation(self, event):
        """Start/stop the automatic simulation"""
        self.is_running = not self.is_running
        if self.is_running:
            self.btn_start.label.set_text("Stop")
            self.run_simulation()
        else:
            self.btn_start.label.set_text("Start")

    def toggle_pause(self, event):
        """Pause/resume the simulation"""
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.btn_pause.label.set_text("Resume")
        else:
            self.btn_pause.label.set_text("Pause")

    def run_simulation(self):
        """Run the simulation automatically"""

        def animate(frame):
            if not self.is_running or self.is_paused:
                return

            if self.current_step < self.max_steps:
                self.single_step()

            if self.current_step >= self.max_steps:
                self.is_running = False
                self.btn_start.label.set_text("Start")

        # Create animation
        interval = max(50, int(self.step_delay * 1000))  # Convert to milliseconds
        self.animation = animation.FuncAnimation(
            self.fig, animate, interval=interval, repeat=False
        )

    def reset_episode(self, event):
        """Reset the current episode"""
        self.reset_current_environment()
        print("Episode reset")

    def next_environment(self, event):
        """Switch to next environment"""
        self.current_env_idx = (self.current_env_idx + 1) % len(self.environments)
        self.reset_current_environment()
        print(f" Switched to environment {self.current_env_idx + 1}")

    def change_environment(self, val):
        """Change environment via slider"""
        new_env_idx = int(self.env_slider.val)
        if new_env_idx != self.current_env_idx:
            self.current_env_idx = new_env_idx
            self.reset_current_environment()

    def update_speed(self, val):
        """Update simulation speed"""
        self.step_delay = self.speed_slider.val

    def analyze_performance(self, event):
        """Analyze agent performance across episodes"""
        if len(self.episode_history) == 0:
            print("No completed episodes to analyze")
            return

        self.show_performance_analysis()

    def show_performance_analysis(self):
        """Show detailed performance analysis"""
        # Create analysis window
        fig_analysis, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig_analysis.suptitle("Agent Performance Analysis", fontsize=16)

        # Extract data from episode history
        episode_rewards = [sum(ep["rewards"]) for ep in self.episode_history]
        episode_lengths = [len(ep["rewards"]) for ep in self.episode_history]
        capture_success = [
            ep["info"][-1].get("captured", False) for ep in self.episode_history
        ]

        # Plot 1: Episode rewards
        axes[0, 0].plot(episode_rewards, "b-o", markersize=4)
        axes[0, 0].set_title("Episode Rewards")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Total Reward")
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Episode lengths
        axes[0, 1].plot(episode_lengths, "g-o", markersize=4)
        axes[0, 1].set_title("Episode Lengths")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Steps")
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Capture success rate
        capture_rate = np.cumsum(capture_success) / np.arange(
            1, len(capture_success) + 1
        )
        axes[1, 0].plot(capture_rate, "r-", linewidth=2)
        axes[1, 0].set_title("Capture Success Rate")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Success Rate")
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Summary statistics
        axes[1, 1].axis("off")

        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        final_capture_rate = capture_rate[-1] if len(capture_rate) > 0 else 0

        summary_text = f""" Performance Summary:
        
Total Episodes: {len(self.episode_history)}
Average Reward: {avg_reward:.2f}
Average Length: {avg_length:.1f} steps
Final Capture Rate: {final_capture_rate:.2%}

 Best Episode:
Reward: {max(episode_rewards):.2f}
Shortest: {min(episode_lengths)} steps

 Agent Behavior:
Agent 1: A* Pathfinding
Agent 2: MARL Coordination
Model: {"Trained" if self.trained_agent else "Random"}
        """

        axes[1, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axes[1, 1].transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()
        plt.show()

        print(f" Analysis complete - {len(self.episode_history)} episodes analyzed")

    def save_episode(self, event):
        """Save current episode data"""
        if len(self.current_episode_data["rewards"]) == 0:
            print(" No episode data to save")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"episode_visualization_{timestamp}.pkl"

        import pickle

        try:
            with open(filename, "wb") as f:
                pickle.dump(
                    {
                        "episode_data": self.current_episode_data,
                        "environment_info": {
                            "width": self.current_env.width,
                            "height": self.current_env.height,
                            "obstacles": self.current_env.env.obstacles,
                        },
                        "model_info": {
                            "has_trained_model": self.trained_agent is not None,
                            "timestamp": timestamp,
                        },
                    },
                    f,
                )
            print(f" Episode saved to {filename}")
        except Exception as e:
            print(f" Failed to save episode: {e}")

    def show(self):
        """Show the visualization interface"""
        print("\nVisualization Controls:")
        print("  • Start/Stop: Run simulation automatically")
        print("  • Pause/Resume: Pause/resume automatic simulation")
        print("  • Step: Execute single step manually")
        print("  • Reset: Reset current episode")
        print("  • Next Env: Switch to next environment")
        print("  • Analyze: Show performance analysis")
        print("  • Save Ep: Save current episode data")
        print("  • Speed Slider: Control simulation speed")
        print("  • Environment Slider: Select environment")

        plt.show()


def main():
    """Main function to run the visualization"""
    print(" QMIX Hybrid Agent Visualization Tool")
    print("=" * 50)

    # Ask for model path
    model_path = input(
        "Enter path to trained model (or press Enter for random actions): "
    ).strip()
    if not model_path:
        model_path = None
        print(" Running with random actions for demonstration")
    else:
        print(f" Loading model from: {model_path}")

    try:
        # Create visualizer
        visualizer = TrainedAgentVisualizer(model_path)

        print("\n Visualization ready!")
        print("Close the window to exit.")

        # Show the visualization
        visualizer.show()

    except KeyboardInterrupt:
        print("\n Exiting visualization...")
    except Exception as e:
        print(f" Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
