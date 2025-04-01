from environment.delivery import Deliveryutils/plotting.py
from agents.delivery_q_agent import DeliveryQAgent
import numpy as np
import logging
import torchivery_stops(xy, stops):
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriterl='Delivery Stops')
from tqdm import tqdmumerate(stops):
        plt.annotate(f'Stop {i+1}', (xy[stop, 0], xy[stop, 1]), textcoords="offset points", xytext=(0,10), ha='center')
# Configure loggingvery Stops')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
    plt.legend()
    plt.grid()
def generate_random_coordinates(n_stops, max_box):
    """Generate random coordinates for the cities"""
    return np.random.rand(n_stops, 2) * max_box
    plt.figure(figsize=(10, 6))
    plt.plot(xy[tour, 0], xy[tour, 1], marker='o', color='blue', label='Tour Path')
def main():le('Tour Path')
    try:xlabel('X Coordinate')
        # Parametersordinate')
        n_stops = 5
        max_box = 20
        n_episodes = 1000        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")        writer = SummaryWriter("runs/tsp_training")        # Generate random coordinates        xy = generate_random_coordinates(n_stops, max_box)        # Initialize environment and agent        logger.info(f"Using device: {device}")        delivery_env = Delivery(            xy=xy,            boundary_index=list(range(n_stops)),            n_stops=n_stops,            max_box=max_box,            fixed=False,        )        agent = DeliveryQAgent(            xy=xy,            max_box=max_box,            method="cluster",            boundary_index=list(range(n_stops)),            boundary_points=xy,            mydict={0: list(range(n_stops))},            labels=np.zeros(n_stops),            n_cluster=1,            simplices=[],            vertices=[],            states_size=n_stops,            actions_size=n_stops,        )        # Training loop with visualization        episode_rewards = []        best_route = None        best_reward = float("-inf")        for episode in tqdm(range(n_episodes)):            state = delivery_env.reset()            agent.reset_memory()            done = False            total_reward = 0            route = [state]            while not done:                action = agent.act(state)                new_state, reward = delivery_env.step(action)                agent.train(state, new_state, reward)                total_reward += reward                state = new_state                route.append(state)                done = len(delivery_env.stops) == delivery_env.n_stops            episode_rewards.append(total_reward)            # Log metrics            avg_loss = np.mean(agent.episode_losses) if agent.episode_losses else 0            writer.add_scalar("Loss/train", avg_loss, episode)            writer.add_scalar("Reward/train", total_reward, episode)            # Update best route            if total_reward > best_reward:                best_reward = total_reward                best_route = route            # Visualize every 100 episodes            if episode % 100 == 0:                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 5))                # Plot current route                delivery_env.render(ax=ax1)                ax1.set_title(f"Episode {episode} Route")                # Plot learning curve (reward)                ax2.plot(episode_rewards)                ax2.set_xlabel("Episode")                ax2.set_ylabel("Total Reward")                ax2.set_title("Learning Curve (Reward)")                # Plot learning curve (loss)                ax3.plot([np.mean(agent.episode_losses) for _ in range(episode + 1)])                ax3.set_xlabel("Episode")                ax3.set_ylabel("Loss")                ax3.set_title("Learning Curve (Loss)")                plt.savefig(f"training_progress/episode_{episode}.png")                plt.close()        # Final visualization        plt.figure(figsize=(15, 5))        plt.subplot(1, 2, 1)        delivery_env.render(best_route)        plt.title("Best Route Found")        plt.subplot(1, 2, 2)        plt.plot(episode_rewards)        plt.xlabel("Episode")        plt.ylabel("Total Reward")        plt.title("Learning Curve")        plt.savefig("training_progress/final_result.png")        plt.show()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
