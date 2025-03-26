from environment.delivery import Delivery
from agents.delivery_q_agent import DeliveryQAgent

def main():
    # Initialize the environment
    delivery_env = Delivery(xy=None, boundary_index=None, n_stops=5, max_box=20, fixed=False)
    
    # Initialize the agent
    agent = DeliveryQAgent(
        xy=None,
        max_box=20,
        method="cluster",
        boundary_index=None,
        boundary_points=None,
        mydict=None,
        labels=None,
        n_cluster=1,
        simplices=None,
        vertices=None,
        states_size=5,
        actions_size=5
    )
    
    # Run the episode
    for episode in range(1000):
        delivery_env.reset()
        agent.reset_memory()
        done = False
        
        while not done:
            state = agent.act(delivery_env._get_state())
            new_state, reward = delivery_env.step(state)
            agent.train(state, new_state, reward)
            done = len(delivery_env.stops) == delivery_env.n_stops

if __name__ == "__main__":
    main()