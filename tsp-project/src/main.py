from environment.delivery import Delivery
from agents.delivery_q_agent import DeliveryQAgent
import numpy as np
import logging
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from pathlib import Path

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

        # Parameters
        n_stops = 5
        max_box = 20
        n_episodes = 5000  # Increase number of episodes
        epsilon_start = 1.0
        epsilon_end = 0.01
        epsilon_decay = 0.995
        device = torch.device("mps" if torch.mps.is_available() else "cpu")
        writer = SummaryWriter("runs/tsp_training")

        # Generate random coordinates
        xy = generate_random_coordinates(n_stops, max_box)

        # Initialize environment and agent
        logger.info(f"Using device: {device}")
        delivery_env = Delivery(
            xy=xy,
            boundary_index=list(range(n_stops)),
            n_stops=n_stops,
            max_box=max_box,
            fixed=False,
        )

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
            actions_size=n_stops,
        )

        # Training loop with visualization
        episode_rewards = []
        best_route = None
        best_reward = float("-inf")

        for episode in tqdm(range(n_episodes)):
            # Update exploration rate
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -episode / (n_episodes * 0.3)
            )
            agent.epsilon = epsilon

            state = delivery_env.reset()
            agent.reset_memory()
            done = False
            total_reward = 0
            route = [state]
            invalid_moves = 0  # Track number of building collisions

            while not done:
                action = agent.act(state)
                new_state, reward = delivery_env.step(action)

                # Track invalid moves
                if reward <= -1000:
                    invalid_moves += 1

                agent.train(state, new_state, reward)
                total_reward += reward
                state = new_state
                route.append(state)
                done = len(delivery_env.stops) == delivery_env.n_stops

            episode_rewards.append(total_reward)

            # Log additional metrics
            writer.add_scalar("Loss/train", np.mean(agent.episode_losses), episode)
            writer.add_scalar("Reward/train", total_reward, episode)
            writer.add_scalar("InvalidMoves/train", invalid_moves, episode)
            writer.add_scalar("Epsilon/train", epsilon, episode)

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
                save_path = os.path.join("training_progress", f"episode_{episode}.png")
                plt.tight_layout()  # Adjust subplot spacing
                plt.savefig(save_path)
                plt.close(fig)

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
        plt.tight_layout()
        final_path = os.path.join("training_progress", "final_result.png")
        plt.savefig(final_path)
        plt.show()
        plt.close(fig)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
