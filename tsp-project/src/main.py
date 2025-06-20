from environment.delivery import Delivery
from agents.delivery_q_agent import DeliveryQAgent
import numpy as np
import logging
import torch
import matplotlib

# Force matplotlib to use a safe font family
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_coordinates(n_stops, max_box):
    """Generate random coordinates for the cities"""
    return np.random.rand(n_stops, 2) * max_box


def main():
    try:
        # Create directories for saving results
        Path("training_progress").mkdir(parents=True, exist_ok=True)
        Path("runs/tsp_training").mkdir(parents=True, exist_ok=True)

        # Process command line arguments first
        parser = argparse.ArgumentParser(description="TSP solver with Q-Learning")
        parser.add_argument(
            "--end", type=str, required=True, help="Goal coordinates in format x,y"
        )
        parser.add_argument(
            "--start",
            type=str,
            default="0,0",
            help="Start coordinates in format x,y (default: 0,0)",
        )
        parser.add_argument(
            "--max-steps",
            type=int,
            default=4,
            help="Maximum intermediate steps allowed (default: 4)",
        )
        args = parser.parse_args()

        # Parse end point
        try:
            end_x, end_y = map(int, args.end.split(","))
            logger.info(f"Target destination set to coordinates: ({end_x}, {end_y})")
        except ValueError:
            logger.error("Invalid end point format. Use format: x,y (e.g. 10,10)")
            return

        # Parse start point
        try:
            start_x, start_y = map(int, args.start.split(","))
            logger.info(f"Start position set to coordinates: ({start_x}, {start_y})")
        except ValueError:
            logger.error("Invalid start point format. Use format: x,y (e.g. 0,0)")
            start_x, start_y = 0, 0  # Default to bottom-left

        # Get max intermediate steps
        max_intermediate_steps = args.max_steps
        logger.info(f"Maximum intermediate stops set to: {max_intermediate_steps}")

        # Parameters
        n_stops = 5
        max_box = 20
        n_episodes = 10000  # Increase from 5000
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        print(device)
        writer = SummaryWriter("runs/tsp_training")

        # Generate random coordinates
        xy = generate_random_coordinates(n_stops, max_box)

        # Initialize environment and agent with end point
        delivery_env = Delivery(
            xy=xy,
            boundary_index=list(range(n_stops)),
            n_stops=n_stops,
            max_box=max_box,
            fixed=True,  # Use fixed start point
            end_point=(end_x, end_y),  # Pass end point to environment
        )

        # Find closest stop to the goal point
        goal_distances = np.sqrt((xy[:, 0] - end_x) ** 2 + (xy[:, 1] - end_y) ** 2)
        goal_stop = np.argmin(goal_distances)

        # Inform reward model of the goal
        delivery_env.reward_model.goal_state = goal_stop

        # Initialize agent
        logger.info(f"Using device: {device}")
        agent = DeliveryQAgent(
            xy=xy,
            max_box=max_box,
            method="cluster",
            boundary_index=list(range(n_stops)),
            boundary_points=xy,
            mydict={0: list(range(n_stops))},
            labels=np.zeros(n_stops),
            n_cluster=1,
            simplices=[],
            vertices=[],
            states_size=n_stops,
            actions_size=4,  # 4 actions representing the 4 directions
        )

        # Add exploration schedule
        log_interval = 10
        plot_interval = 500

        # Enable curriculum learning (start with easier tasks)
        curriculum_phases = [
            {"epsilon": 0.9, "episodes": 2000},
            {"epsilon": 0.5, "episodes": 3000},
            {"epsilon": 0.2, "episodes": 3000},
            {"epsilon": 0.1, "episodes": 2000},
        ]

        # Track improvement
        no_improvement_count = 0
        patience = 20  # Number of plot intervals without improvement before stopping

        # Training loop with visualization
        episode_rewards = []
        best_route = None
        best_reward = float("-inf")

        # Main training loop
        for phase_idx, phase in enumerate(curriculum_phases):
            phase_epsilon = phase["epsilon"]
            phase_episodes = phase["episodes"]

            logger.info(
                f"Starting curriculum phase {phase_idx + 1}: epsilon={phase_epsilon}, episodes={phase_episodes}"
            )

            for episode in tqdm(range(phase_episodes)):
                # Set exploration rate for this phase
                agent.epsilon = phase_epsilon

                state = delivery_env.reset()
                agent.reset_memory()
                done = False
                total_reward = 0
                route = [state]
                max_steps = 10
                invalid_moves = 0  # Track number of building collisions

                while not done:
                    action = agent.act(state)  # Now returns a direction (0-3)
                    new_state, reward = delivery_env.step(action)

                    # Make sure reward is CPU tensor or scalar before adding to total
                    if isinstance(reward, torch.Tensor):
                        reward = reward.cpu().item()

                    agent.train(state, action, reward)  # Pass action, not new_state
                    total_reward += reward
                    state = new_state
                    route.append(state)

                    # End when goal reached or max steps exceeded
                    done = (state == goal_stop) or (len(route) > max_steps)

                    # Also track if we hit buildings (for metrics)
                    if reward <= -500:  # Building collision threshold
                        invalid_moves += 1

                # Store reward as scalar, not tensor
                episode_rewards.append(float(total_reward))

                # Log additional metrics
                writer.add_scalar("Loss/train", np.mean(agent.episode_losses), episode)
                writer.add_scalar("Reward/train", total_reward, episode)
                writer.add_scalar("InvalidMoves/train", invalid_moves, episode)
                writer.add_scalar("Epsilon/train", agent.epsilon, episode)

                # Update best route
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_route = route

                # Visualize every 100 episodes
                if episode % 100 == 0:
                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))

                    # Plot current route
                    delivery_env.render(route=route, ax=ax1)
                    ax1.set_title(f"Episode {episode} Route")

                    # Plot learning curve (reward)
                    ax2.plot(episode_rewards)
                    ax2.set_xlabel("Episode")
                    ax2.set_ylabel("Total Reward")
                    ax2.set_title("Learning Curve (Reward)")

                    # Plot learning curve (loss)
                    losses = [
                        np.mean(agent.episode_losses[: i + 1])
                        if agent.episode_losses
                        else 0
                        for i in range(episode + 1)
                    ]
                    ax3.plot(losses)
                    ax3.set_xlabel("Episode")
                    ax3.set_ylabel("Loss")
                    ax3.set_title("Learning Curve (Loss)")

                    # Save figure and close it
                    save_path = os.path.join(
                        "training_progress", f"episode_{episode}.png"
                    )
                    # Wrap tight_layout in try-except
                    try:
                        plt.tight_layout()
                    except Exception:
                        pass
                    plt.savefig(save_path)
                    plt.close(fig)

                # Add early stopping check at plot intervals
                if episode % plot_interval == 0 and episode > 0:
                    avg_recent_reward = np.mean(episode_rewards[-plot_interval:])
                    if avg_recent_reward > best_reward:
                        best_reward = avg_recent_reward
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1

                    if no_improvement_count >= patience:
                        logger.info(
                            f"No improvement for {patience} intervals. Stopping training."
                        )
                        break

        # Final visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot best route
        delivery_env.render(route=best_route, ax=ax1)
        ax1.set_title("Best Route Found")

        # Plot final learning curve
        ax2.plot(episode_rewards)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Total Reward")
        ax2.set_title("Learning Curve")

        # Save final result
        # Wrap tight_layout in try-except
        try:
            plt.tight_layout()
        except Exception:
            pass
        final_path = os.path.join("training_progress", "final_result.png")
        plt.savefig(final_path)
        # plt.show()
        plt.close(fig)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
