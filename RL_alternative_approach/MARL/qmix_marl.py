import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QMixAgent(nn.Module):
    """Individual agent Q-network for QMIX"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QMixAgent, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Agent network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, state):
        return self.network(state)


class MixingNetwork(nn.Module):
    """QMIX Mixing Network"""

    def __init__(self, n_agents, state_dim, embed_dim=32, hypernet_embed=64):
        super(MixingNetwork, self).__init__()

        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed

        # Hypernetwork that generates weights for mixing network
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim * n_agents),
        )

        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim),
        )

        # State-dependent bias
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)
        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, agent_qs, states):
        """
        Forward pass of mixing network

        Args:
            agent_qs: [batch_size, n_agents] - Individual agent Q-values
            states: [batch_size, state_dim] - Global state

        Returns:
            q_tot: [batch_size, 1] - Mixed Q-value
        """
        batch_size = agent_qs.size(0)

        # Generate weights and biases for first layer
        w1 = torch.abs(self.hyper_w_1(states))  # Ensure positive weights
        b1 = self.hyper_b_1(states)

        # Reshape for matrix multiplication
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)
        b1 = b1.view(batch_size, 1, self.embed_dim)

        # First layer: [batch_size, 1, n_agents] * [batch_size, n_agents, embed_dim]
        agent_qs = agent_qs.view(batch_size, 1, self.n_agents)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        # Generate weights for final layer
        w_final = torch.abs(self.hyper_w_final(states))  # Ensure positive weights
        w_final = w_final.view(batch_size, self.embed_dim, 1)

        # State value function
        v = self.V(states).view(batch_size, 1, 1)

        # Final layer: [batch_size, 1, embed_dim] * [batch_size, embed_dim, 1]
        y = torch.bmm(hidden, w_final) + v

        return y.view(batch_size, 1)


class QMixMemory:
    """Experience replay buffer for QMIX"""

    def __init__(self, capacity=50000):  # Increased capacity
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, actions, reward, next_state, done):
        """Store experience"""
        # Clip rewards to prevent explosion
        clipped_reward = np.clip(reward, -10.0, 10.0)
        self.buffer.append((state, actions, clipped_reward, next_state, done))

    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.BoolTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


class QMIX:
    """QMIX Multi-Agent Reinforcement Learning"""

    def __init__(
        self,
        n_agents=2,
        state_dim=20,  # Individual agent observation dimension
        global_state_dim=None,  # Global state dimension for mixing network
        action_dim=8,
        lr=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        device="cpu",
    ):
        self.n_agents = n_agents
        self.state_dim = state_dim  # Individual agent state dimension
        self.global_state_dim = global_state_dim or (
            state_dim * n_agents
        )  # Global state dimension
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.device = device

        # Networks - agent networks use individual state dim
        self.agent_networks = nn.ModuleList(
            [QMixAgent(state_dim, action_dim) for _ in range(n_agents)]
        ).to(device)

        self.target_agent_networks = nn.ModuleList(
            [QMixAgent(state_dim, action_dim) for _ in range(n_agents)]
        ).to(device)

        # Mixing networks use global state dim
        self.mixing_network = MixingNetwork(n_agents, self.global_state_dim).to(device)
        self.target_mixing_network = MixingNetwork(n_agents, self.global_state_dim).to(
            device
        )

        # Copy weights to target networks
        self.update_target_networks()

        # Optimizers
        params = list(self.agent_networks.parameters()) + list(
            self.mixing_network.parameters()
        )
        self.optimizer = optim.Adam(params, lr=lr)

        # Memory
        self.memory = QMixMemory()

        # Training parameters
        self.batch_size = 32
        self.target_update_freq = 100
        self.training_step = 0

    def get_actions(self, states, epsilon=None):
        """Get actions for all agents"""
        if epsilon is None:
            epsilon = self.epsilon

        actions = []

        with torch.no_grad():
            for i in range(self.n_agents):
                state = torch.FloatTensor(states[i]).unsqueeze(0).to(self.device)

                if random.random() < epsilon:
                    # Random action
                    action = random.randint(0, self.action_dim - 1)
                else:
                    # Greedy action
                    q_values = self.agent_networks[i](state)
                    action = q_values.argmax().item()

                actions.append(action)

        return actions

    def store_experience(self, states, actions, reward, next_states, done):
        """Store experience in replay buffer"""
        # Convert to global state representation for QMIX
        global_state = self._combine_states(states)
        next_global_state = self._combine_states(next_states)

        self.memory.push(global_state, actions, reward, next_global_state, done)

    def _combine_states(self, agent_states):
        """Combine individual agent states into global state"""
        return np.concatenate(agent_states)

    def train(self):
        """Train the QMIX networks"""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        agent_qs = []
        for i in range(self.n_agents):
            # Extract individual agent states from global state
            agent_state = self._extract_agent_state(states, i)
            q_values = self.agent_networks[i](agent_state)
            agent_q = q_values.gather(1, actions[:, i].unsqueeze(1))
            agent_qs.append(agent_q)

        agent_qs = torch.cat(agent_qs, dim=1)  # [batch_size, n_agents]

        # Mix current Q values
        q_tot = self.mixing_network(agent_qs, states)

        # Target Q values
        with torch.no_grad():
            next_agent_qs = []
            for i in range(self.n_agents):
                next_agent_state = self._extract_agent_state(next_states, i)
                next_q_values = self.target_agent_networks[i](next_agent_state)
                next_agent_q = next_q_values.max(1)[0].unsqueeze(1)
                next_agent_qs.append(next_agent_q)

            next_agent_qs = torch.cat(next_agent_qs, dim=1)
            next_q_tot = self.target_mixing_network(next_agent_qs, next_states)

            target_q_tot = rewards.unsqueeze(1) + self.gamma * next_q_tot * (
                ~dones
            ).unsqueeze(1)

        # Loss
        loss = F.mse_loss(q_tot, target_q_tot)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_networks.parameters())
            + list(self.mixing_network.parameters()),
            max_norm=10,
        )
        self.optimizer.step()

        # Update target networks
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.update_target_networks()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def _extract_agent_state(self, global_states, agent_idx):
        """Extract individual agent state from global state"""
        # Assuming equal state sizes for all agents
        agent_state_size = self.state_dim
        start_idx = agent_idx * agent_state_size
        end_idx = start_idx + agent_state_size
        return global_states[:, start_idx:end_idx]

    def update_target_networks(self):
        """Copy weights to target networks"""
        for i in range(self.n_agents):
            self.target_agent_networks[i].load_state_dict(
                self.agent_networks[i].state_dict()
            )

        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def save(self, path):
        """Save model"""
        torch.save(
            {
                "agent_networks": [net.state_dict() for net in self.agent_networks],
                "mixing_network": self.mixing_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "training_step": self.training_step,
            },
            path,
        )

    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)

        for i, state_dict in enumerate(checkpoint["agent_networks"]):
            self.agent_networks[i].load_state_dict(state_dict)

        self.mixing_network.load_state_dict(checkpoint["mixing_network"])
        self.update_target_networks()

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]
