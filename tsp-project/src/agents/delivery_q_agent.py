import seaborn as sns
import torch
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random


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
        gamma=0.95,  # Increased from 0.9
        lr=0.3,  # Reduced from 0.6
    ):
        self.xy = xy
        self.method = method
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 0.1  # exploration rate
        self.grid_size = max_box

        # Define the four cardinal directions: up, down, left, right
        self.directions = [
            (0, -1),  # Up (decrease y)
            (0, 1),  # Down (increase y)
            (-1, 0),  # Left (decrease x)
            (1, 0),  # Right (increase x)
        ]
        self.direction_names = ["Up", "Down", "Left", "Right"]

        # Initialize Q-values for states and cardinal directions
        self.Q, self.visits, self.update, self.rewards = self.build_model(
            states_size, len(self.directions)
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

        # Experience replay buffer
        self.replay_buffer = []
        self.replay_buffer_size = 1000
        self.batch_size = 32

    def build_model(self, states_size, actions_size):
        # Instead of state-to-state Q-values, build state-to-action Q-values
        Q = np.random.uniform(0, 0.1, (states_size, actions_size))
        visits = {}
        update = {}
        rewards = {}
        for i in range(states_size):
            for j in range(actions_size):
                visits[(i, j)] = 0
                update[(i, j)] = 0
                rewards[(i, j)] = 0
        return Q, visits, update, rewards

    def act(self, state):
        """Select an action (direction) based on Q-values and epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            # Pure random exploration of directions
            return np.random.randint(len(self.directions))
        else:
            with torch.no_grad():
                # Get Q-values for current state and all directions
                q_values = self.Q[state].cpu().numpy()

                # Apply numerical stability fixes for softmax
                temperature = max(0.1, self.epsilon * 2)
                q_shifted = q_values - np.max(q_values)
                probs = np.exp(np.clip(q_shifted / temperature, -20, 20))
                sum_probs = np.sum(probs)

                if sum_probs <= 0 or np.isnan(sum_probs):
                    return np.random.randint(len(self.directions))

                probs = probs / sum_probs

                # Choose based on probability distribution
                if np.random.random() < 0.8:
                    return np.random.choice(len(self.directions), p=probs)
                else:
                    return np.argmax(q_values)

    def train(self, state, action, reward):
        """Train the Q-agent using the Q-learning update rule"""
        if self.method == "cluster":
            with torch.no_grad():
                # Get old value
                old_value = self.Q[state, action]
                # Calculate next max value
                next_max = torch.max(self.Q[state])
                # Calculate new value
                new_value = (1 - self.lr) * old_value + self.lr * (
                    reward + self.gamma * next_max
                )
                # Update Q-table
                self.Q[state, action] = new_value

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

    def visualize_q_table(self, filepath=None):
        """
        Visualize the Q-table as a heatmap using seaborn

        Args:
            filepath (str, optional): Path to save the visualization. If None, only displays.
        """
        plt.figure(figsize=(10, 8))

        # Convert Q-table to CPU numpy array for visualization
        q_values = self.Q.cpu().numpy()

        # Create a mask for -inf values
        mask = np.isinf(q_values)

        # Create heatmap
        ax = sns.heatmap(
            q_values,
            cmap="viridis",
            mask=mask,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "Q-value"},
            vmin=q_values[~mask].min(),
            vmax=q_values[~mask].max(),
        )

        # Enhance visualization
        plt.title("Q-table Heatmap")


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
