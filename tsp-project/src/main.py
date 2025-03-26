from environment.delivery import Delivery
from agents.delivery_q_agent import DeliveryQAgent
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_random_coordinates(n_stops, max_box):
    """Generate random coordinates for the cities"""
    return np.random.rand(n_stops, 2) * max_box


def main():
    try:
        # Parameters
        n_stops = 5
        max_box = 20
        n_episodes = 1000

        # Generate random coordinates
        xy = generate_random_coordinates(n_stops, max_box)

        logger.info("Initializing environment...")
        delivery_env = Delivery(
            xy=xy,
            boundary_index=list(range(n_stops)),
            n_stops=n_stops,
            max_box=max_box,
            fixed=False,
        )

        logger.info("Initializing agent...")
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

        logger.info("Starting training...")
        for episode in range(n_episodes):
            state = delivery_env.reset()
            agent.reset_memory()
            done = False
            total_reward = 0

            while not done:
                action = agent.act(state)
                new_state, reward = delivery_env.step(action)
                agent.train(state, new_state, reward)
                total_reward += reward
                state = new_state
                done = len(delivery_env.stops) == delivery_env.n_stops

            if episode % 100 == 0:
                logger.info(
                    f"Episode {episode}/{n_episodes} completed - Total reward: {total_reward:.2f}"
                )

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
