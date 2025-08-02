import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random


class MultiAgentCoordinator:
    def __init__(self, state_dim, action_dim, num_agents=2):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared policy network for coordination
        self.coordination_network = CoordinationNetwork(state_dim, action_dim)

        # Individual agent networks
        self.agent_networks = [
            AgentNetwork(state_dim, action_dim) for _ in range(num_agents)
        ]

        # Experience replay buffer
        self.memory = deque(maxlen=10000)

    def get_coordinated_actions(self, global_state, agent_states):
        """Get coordinated actions for all agents"""
        # Global coordination decision
        coordination_action = self.coordination_network(global_state)

        # Individual agent decisions based on coordination
        agent_actions = []
        for i, agent_state in enumerate(agent_states):
            combined_input = torch.cat([agent_state, coordination_action], dim=-1)
            action = self.agent_networks[i](combined_input)
            agent_actions.append(action)

        return agent_actions, coordination_action


class CoordinationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        return torch.tanh(self.network(x))


class AgentNetwork(nn.Module):
    def __init__(self, state_dim, coordination_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + coordination_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),  # 8 directional actions
        )

    def forward(self, x):
        return torch.softmax(self.network(x), dim=-1)
