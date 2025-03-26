import torch
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


class DeliveryQAgent:
    def __init__(
        self,
        xy,
        max_box,
        method,
        boundary_index,
        boundary_points,
        mydict,
        labels,
        n_cluster,
        simplices,
        vertices,
        states_size,
        actions_size,
        alpha=1,
        beta=1,
        gamma=0.9,
        lr=0.6,
    ):
        self.xy = xy
        self.method = method
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 0.1  # exploration rate

        # Initialize Q-values
        self.Q, self.visits, self.update, self.rewards = self.build_model(
            states_size, actions_size
        )

        # Move Q-matrix to GPU if available
        self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        self.Q = torch.tensor(self.Q, device=self.device, dtype=torch.float32)

        # Initialize memory
        self.memory = []
        self.states_memory = []
        self.route = ""

        # Initialize metrics tracking
        self.writer = SummaryWriter("runs/tsp_training")
        self.episode_losses = []
        self.best_route = None
        self.best_distance = float("inf")

    def build_model(self, states_size, actions_size):
        Q = cdist(self.xy, self.xy)
        visits = {}
        update = {}
        rewards = {}
        for i in range(states_size):
            Q[i, i] = -np.inf
            for j in range(states_size):
                visits[(i, j)] = 0
                update[(i, j)] = 0
                rewards[(i, j)] = 0
        return Q, visits, update, rewards

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.Q.size(0))
        else:
            with torch.no_grad():
                return torch.argmax(self.Q[state]).item()

    def train(self, state, new_state, reward):
        if self.method == "cluster":
            with torch.no_grad():
                old_value = self.Q[state, new_state]
                next_max = torch.max(self.Q[new_state])
                new_value = (1 - self.lr) * old_value + self.lr * (
                    reward + self.gamma * next_max
                )
                self.Q[state, new_state] = new_value

                # Track loss
                loss = (new_value - old_value).pow(2)
                self.episode_losses.append(loss.item())

    def reset_memory(self):
        self.memory = []
        self.states_memory = []
        self.route = ""
        self.episode_losses = []

    def remember_state(self, s):
        self.states_memory.append(s)


if __name__ == "__main__":
    # Test parameters
    n_stops = 5
    max_box = 20
    grid_size = 3
    block_size = 4

    # Generate sample coordinates
    xy = np.random.rand(n_stops, 2) * max_box

    # Create test agent
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

    # Test basic functionality
    print("Initial Q-matrix shape:", agent.Q.shape)

    # Test state-action selection
    state = 0
    action = agent.act(state)
    print(f"Selected action for state {state}: {action}")

    # Test training
    new_state = action
    reward = -agent.Q[state, new_state]  # Use negative distance as reward
    agent.train(state, new_state, reward)
    print(
        f"Updated Q-value for state {state} and action {action}: {agent.Q[state, action]}"
    )

    # Test memory reset
    agent.reset_memory()
    print("Memory reset successfully")
