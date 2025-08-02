import torch
import numpy as np
from qmix_marl import QMixAgent
from pursuit_environment import PursuitEnvironment


class RLPursuitSystem:
    def __init__(self, model_path=None, device="mps"):
        self.device = device

        # Initialize agent
        state_dim = 400 + 7  # Grid + additional features
        self.agent = QMixAgent(state_dim=state_dim, device=device)

        # Load trained model if provided
        if model_path:
            self.agent.load_model(model_path)
            self.agent.epsilon = 0.0  # No exploration during evaluation
            print(f"Loaded trained model from {model_path}")

        self.env = None

    def set_environment(self, env):
        """Set the environment for the RL system"""
        self.env = env

    def get_coordinated_actions(self, agent_positions, target_position, target_history):
        """Get RL-coordinated actions for agents"""
        if self.env is None:
            raise ValueError("Environment not set. Call set_environment() first.")

        # Create temporary environment state
        temp_env = PursuitEnvironment(self.env.width, self.env.height)
        temp_env.env = self.env
        temp_env.agent_positions = agent_positions
        temp_env.target_history = target_history[-10:]  # Keep recent history

        # Create mock target with current position
        class MockTarget:
            def __init__(self, position):
                self.position = position

        temp_env.target = MockTarget(target_position)

        # Get state representation
        global_state, individual_obs = temp_env._get_state()

        # Get actions from trained agent
        actions = self.agent.select_actions(individual_obs, training=False)

        return actions

    def convert_actions_to_movements(self, actions, current_positions, speed=0.5):
        """Convert RL actions to actual movement coordinates"""
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
        for action, current_pos in zip(actions, current_positions):
            dx, dy = directions[action]
            new_x = current_pos[0] + dx * speed
            new_y = current_pos[1] + dy * speed

            # Check if new position is valid
            if self.env.is_valid_position(new_x, new_y):
                new_positions.append((new_x, new_y))
            else:
                new_positions.append(current_pos)  # Stay in place if invalid

        return new_positions

    def evaluate_performance(self, num_episodes=100):
        """Evaluate the trained agent's performance"""
        if self.env is None:
            raise ValueError("Environment not set. Call set_environment() first.")

        print("Evaluating trained agent performance...")

        total_rewards = []
        capture_count = 0
        episode_lengths = []

        for episode in range(num_episodes):
            # Reset environment
            global_state, individual_obs = self.env.reset()

            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < 500:
                # Get actions
                actions = self.agent.select_actions(individual_obs, training=False)

                # Take step
                (next_global_state, next_individual_obs), reward, done, info = (
                    self.env.step(actions)
                )

                episode_reward += reward
                step_count += 1

                # Update observations
                global_state = next_global_state
                individual_obs = next_individual_obs

            total_rewards.append(episode_reward)
            episode_lengths.append(step_count)

            if info.get("captured", False):
                capture_count += 1

            if episode % 20 == 0:
                print(
                    f"Episode {episode}: Reward = {episode_reward:.2f}, "
                    f"Length = {step_count}, Captured = {info.get('captured', False)}"
                )

        # Print results
        avg_reward = np.mean(total_rewards)
        capture_rate = capture_count / num_episodes
        avg_length = np.mean(episode_lengths)

        print(f"\nEvaluation Results ({num_episodes} episodes):")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Capture Rate: {capture_rate:.3f}")
        print(f"Average Episode Length: {avg_length:.1f}")

        return {
            "avg_reward": avg_reward,
            "capture_rate": capture_rate,
            "avg_length": avg_length,
            "all_rewards": total_rewards,
        }


def integrate_rl_with_animation(
    env,
    pathfinder,
    start,
    target_position=None,
    model_path="qmix_model_final.pth",
    max_frames=500,
):
    """
    Integration function to use trained RL model with existing animation
    This replaces the deterministic coordination in your original code
    """

    # Initialize RL system
    rl_system = RLPursuitSystem(model_path=model_path)
    rl_system.set_environment(env)

    print("Using trained QMIX model for agent coordination")

    # Your existing animation code would go here, but instead of
    # deterministic pathfinding, you would call:

    def get_rl_coordinated_movements(agent_positions, target_position, target_history):
        """Get coordinated movements using trained RL model"""

        # Get actions from RL system
        actions = rl_system.get_coordinated_actions(
            agent_positions, target_position, target_history
        )

        # Convert actions to movements
        new_positions = rl_system.convert_actions_to_movements(
            actions, agent_positions, speed=0.5
        )

        return new_positions, actions

    return get_rl_coordinated_movements


# Example usage
if __name__ == "__main__":
    # Test the RL integration
    from pursuit_environment import create_test_environments

    # Create test environment
    env = create_test_environments()[2]  # 30x30 environment

    # Initialize RL system
    rl_system = RLPursuitSystem()
    rl_system.set_environment(env.env)

    # Evaluate untrained agent
    print("Evaluating untrained agent...")
    results = rl_system.evaluate_performance(num_episodes=50)

    print("\nRL Integration system ready!")
