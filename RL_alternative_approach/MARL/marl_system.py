import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict, deque
import random
from typing import Dict, List, Tuple, Optional
import pickle
import os


class QMIXNetwork(nn.Module):
    """QMIX monotonic value function factorization network"""

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_embed: int = 64,
    ):
        super(QMIXNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork for generating mixing network weights
        self.hyper_w_1 = nn.Linear(state_dim, embed_dim * n_agents)
        self.hyper_w_final = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for generating mixing network biases
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)

        # Value state embedding
        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of QMIX network
        Args:
            agent_qs: Individual agent Q-values [batch_size, n_agents]
            states: Global state [batch_size, state_dim]
        Returns:
            Total Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)

        # Generate mixing network weights
        w1 = torch.abs(
            self.hyper_w_1(states)
        )  # Ensure positive weights for monotonicity
        w1 = w1.view(batch_size, self.n_agents, self.embed_dim)

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(batch_size, self.embed_dim, 1)

        # Generate mixing network biases
        b1 = self.hyper_b_1(states)
        b1 = b1.view(batch_size, 1, self.embed_dim)

        # Forward pass through mixing network
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)
        mixed = torch.bmm(hidden, w_final)  # [batch_size, 1, 1]

        # Get state value
        state_value = self.V(states)  # [batch_size, 1]

        # Combine mixed Q-values with state value
        y = mixed.squeeze(1) + state_value  # [batch_size, 1]

        return y


class AgentDQN(nn.Module):
    """Individual agent DQN network"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(AgentDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class SequenceModelingTransformer(nn.Module):
    """Transformer for sequence modeling as inspired by the 2022 paper"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 50,
    ):
        super(SequenceModelingTransformer, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Input embeddings
        self.obs_embedding = nn.Linear(obs_dim, d_model)
        self.action_embedding = nn.Embedding(action_dim, d_model)
        self.agent_embedding = nn.Embedding(n_agents, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads for each agent
        self.action_heads = nn.ModuleList(
            [nn.Linear(d_model, action_dim) for _ in range(n_agents)]
        )

    def forward(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        agent_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Forward pass for sequence modeling
        Args:
            obs_seq: Observation sequence [batch_size, seq_len, obs_dim]
            action_seq: Action sequence [batch_size, seq_len]
            agent_ids: Agent ID sequence [batch_size, seq_len]
            positions: Position in sequence [batch_size, seq_len]
        """
        batch_size, seq_len = obs_seq.shape[:2]

        # Create embeddings
        obs_emb = self.obs_embedding(obs_seq)
        action_emb = self.action_embedding(action_seq)
        agent_emb = self.agent_embedding(agent_ids)
        pos_emb = self.pos_embedding(positions)

        # Combine embeddings
        x = obs_emb + action_emb + agent_emb + pos_emb

        # Apply transformer
        x = self.transformer(x)

        # Get action predictions for each agent
        outputs = []
        for i, head in enumerate(self.action_heads):
            # Use last timestep output for predictions
            agent_output = head(x[:, -1, :])
            outputs.append(agent_output)

        return outputs


class MARLPursuitSystem:
    """Multi-Agent Reinforcement Learning system for pursuit scenarios"""

    def __init__(
        self,
        env_width: int = 50,
        env_height: int = 50,
        max_agents: int = 3,
        obs_radius: int = 5,
        lr: float = 0.001,
        device: torch.device = None,
    ):
        # Set device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(f"MARL System using device: {self.device}")

        # Environment parameters
        self.env_width = env_width
        self.env_height = env_height
        self.max_agents = max_agents
        self.obs_radius = obs_radius

        # Action space: 8 directions + wait
        self.action_dim = 9
        self.actions = [
            (0, 0),  # wait
            (-1, -1),
            (-1, 0),
            (-1, 1),  # up-left, up, up-right
            (0, -1),
            (0, 1),  # left, right
            (1, -1),
            (1, 0),
            (1, 1),  # down-left, down, down-right
        ]

        # Observation space: local grid + agent positions + target info
        grid_obs_size = (2 * obs_radius + 1) ** 2
        agent_info_size = max_agents * 3  # x, y, active
        target_info_size = 3  # x, y, visible
        self.obs_dim = grid_obs_size + agent_info_size + target_info_size

        # Global state for QMIX
        self.state_dim = (
            env_width * env_height + max_agents * 3 + 3
        )  # full map + agents + target

        # Neural networks
        self.agent_networks = [
            AgentDQN(self.obs_dim, self.action_dim).to(self.device)
            for _ in range(max_agents)
        ]
        self.target_networks = [
            AgentDQN(self.obs_dim, self.action_dim).to(self.device)
            for _ in range(max_agents)
        ]
        self.qmix_net = QMIXNetwork(max_agents, self.state_dim).to(self.device)
        self.target_qmix_net = QMIXNetwork(max_agents, self.state_dim).to(self.device)

        # Sequence modeling transformer
        self.seq_model = SequenceModelingTransformer(
            self.obs_dim, self.action_dim, max_agents
        ).to(self.device)

        # Optimizers
        self.agent_optimizers = [
            optim.Adam(net.parameters(), lr=lr) for net in self.agent_networks
        ]
        self.qmix_optimizer = optim.Adam(self.qmix_net.parameters(), lr=lr)
        self.seq_optimizer = optim.Adam(self.seq_model.parameters(), lr=lr)

        # Training parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.gamma = 0.99
        self.tau = 0.005  # Soft update rate

        # Experience replay
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32

        # Episode tracking
        self.episode_rewards = []
        self.episode_steps = []
        self.success_rate = deque(maxlen=100)

        # Sequence data for transformer training
        self.sequence_buffer = deque(maxlen=1000)
        self.max_seq_len = 50

    def get_local_observation(
        self,
        agent_pos: Tuple[float, float],
        env,
        other_agents: List[Tuple[float, float]],
        target_pos: Tuple[float, float],
    ) -> np.ndarray:
        """Get local observation for an agent"""
        x, y = int(agent_pos[0]), int(agent_pos[1])

        # Local grid observation
        local_grid = np.zeros((2 * self.obs_radius + 1, 2 * self.obs_radius + 1))

        for dy in range(-self.obs_radius, self.obs_radius + 1):
            for dx in range(-self.obs_radius, self.obs_radius + 1):
                world_x, world_y = x + dx, y + dy
                grid_x, grid_y = dx + self.obs_radius, dy + self.obs_radius

                if 0 <= world_x < self.env_width and 0 <= world_y < self.env_height:
                    if env.is_valid_position(world_x, world_y):
                        local_grid[grid_y, grid_x] = 0  # free space
                    else:
                        local_grid[grid_y, grid_x] = 1  # obstacle
                else:
                    local_grid[grid_y, grid_x] = 1  # out of bounds

        # Flatten grid
        grid_obs = local_grid.flatten()

        # Other agents info (relative positions)
        agent_obs = np.zeros(self.max_agents * 3)
        for i, other_pos in enumerate(other_agents[: self.max_agents]):
            if other_pos is not None and i < self.max_agents:
                rel_x = other_pos[0] - agent_pos[0]
                rel_y = other_pos[1] - agent_pos[1]
                agent_obs[i * 3 : (i + 1) * 3] = [rel_x, rel_y, 1.0]  # x, y, active

        # Target info (relative position and visibility)
        target_obs = np.zeros(3)
        if target_pos is not None:
            rel_x = target_pos[0] - agent_pos[0]
            rel_y = target_pos[1] - agent_pos[1]
            distance = np.sqrt(rel_x**2 + rel_y**2)
            visible = 1.0 if distance <= self.obs_radius * 2 else 0.0
            target_obs = [rel_x, rel_y, visible]

        return np.concatenate([grid_obs, agent_obs, target_obs])

    def get_global_state(
        self,
        env,
        agent_positions: List[Tuple[float, float]],
        target_pos: Tuple[float, float],
    ) -> np.ndarray:
        """Get global state for QMIX"""
        # Create full environment map
        env_map = np.zeros(self.env_width * self.env_height)
        for y in range(self.env_height):
            for x in range(self.env_width):
                idx = y * self.env_width + x
                env_map[idx] = 0.0 if env.is_valid_position(x, y) else 1.0

        # Agent positions
        agent_state = np.zeros(self.max_agents * 3)
        for i, pos in enumerate(agent_positions[: self.max_agents]):
            if pos is not None:
                agent_state[i * 3 : (i + 1) * 3] = [pos[0], pos[1], 1.0]

        # Target position
        target_state = np.zeros(3)
        if target_pos is not None:
            target_state = [target_pos[0], target_pos[1], 1.0]

        return np.concatenate([env_map, agent_state, target_state])

    def select_action(
        self, observation: np.ndarray, agent_id: int, training: bool = True
    ) -> int:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.agent_networks[agent_id](obs_tensor)
            return q_values.argmax().item()

    def store_experience(self, state, actions, rewards, next_state, dones):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, actions, rewards, next_state, dones))

    def store_sequence(self, episode_data):
        """Store episode sequence for transformer training"""
        if len(episode_data) > 2:
            self.sequence_buffer.append(episode_data)

    def train_step(self):
        """Perform one training step"""
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors and move to device - fix the numpy array warning
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.BoolTensor(np.array(dones)).to(self.device)

        # Since states are flattened (batch_size, flattened_obs + global_state),
        # we need to separate agent observations from global state
        obs_size = (
            self.max_agents * self.obs_dim
        )  # Size of flattened agent observations

        # Check if states have the expected size
        expected_total_size = obs_size + self.state_dim
        if states.shape[1] != expected_total_size:
            # Adjust obs_size if needed
            obs_size = states.shape[1] - self.state_dim

        # Extract agent observations and reshape
        agent_obs_flat = states[:, :obs_size]  # [batch_size, max_agents * obs_dim]
        agent_obs = agent_obs_flat.view(self.batch_size, self.max_agents, self.obs_dim)

        # Extract global state
        global_states = states[:, obs_size:]  # [batch_size, state_dim]

        # Same for next states
        next_agent_obs_flat = next_states[:, :obs_size]
        next_agent_obs = next_agent_obs_flat.view(
            self.batch_size, self.max_agents, self.obs_dim
        )
        next_global_states = next_states[:, obs_size:]

        # Get current Q-values for each agent
        current_q_values = []
        for i in range(self.max_agents):
            agent_obs_i = agent_obs[:, i, :]  # [batch_size, obs_dim]
            q_vals = self.agent_networks[i](agent_obs_i)
            current_q_values.append(q_vals.gather(1, actions[:, i].unsqueeze(1)))

        current_q_values = torch.cat(current_q_values, dim=1)

        # Get target Q-values
        with torch.no_grad():
            next_q_values = []
            for i in range(self.max_agents):
                agent_next_obs_i = next_agent_obs[:, i, :]
                next_q_vals = self.target_networks[i](agent_next_obs_i)
                next_q_values.append(next_q_vals.max(1)[0].unsqueeze(1))

            next_q_values = torch.cat(next_q_values, dim=1)

            # QMIX target values
            target_total_q = self.target_qmix_net(next_q_values, next_global_states)
            target_total_q = rewards.sum(
                dim=1, keepdim=True
            ) + self.gamma * target_total_q * (~dones.any(dim=1, keepdim=True))

        # Current total Q-value
        current_total_q = self.qmix_net(current_q_values, global_states)

        # QMIX loss
        qmix_loss = F.mse_loss(current_total_q, target_total_q)

        # Update QMIX network
        self.qmix_optimizer.zero_grad()
        qmix_loss.backward()
        self.qmix_optimizer.step()

        # Update individual agent networks separately to avoid graph conflicts
        # We need to recompute Q-values for each agent independently
        for i in range(self.max_agents):
            # Zero gradients for this agent
            self.agent_optimizers[i].zero_grad()

            # Recompute Q-values for this agent only
            agent_obs_i = agent_obs[:, i, :].detach()  # Detach to avoid graph issues
            q_vals = self.agent_networks[i](agent_obs_i)
            current_q_i = q_vals.gather(1, actions[:, i].unsqueeze(1))

            # Use detached target for individual agent loss
            agent_loss = F.mse_loss(current_q_i, target_total_q.detach())

            # Backward and step for this agent
            agent_loss.backward()
            self.agent_optimizers[i].step()

        # Soft update target networks
        self.soft_update_targets()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_sequence_model(self):
        """Train the sequence modeling transformer"""
        if len(self.sequence_buffer) < 10:
            return

        # Sample sequences
        sequences = random.sample(
            self.sequence_buffer, min(10, len(self.sequence_buffer))
        )

        for seq_data in sequences:
            if len(seq_data) < 3:
                continue

            # Prepare sequence data
            obs_seq = []
            action_seq = []
            agent_ids = []

            for step_data in seq_data[: self.max_seq_len]:
                obs_seq.append(step_data["observations"])
                action_seq.append(step_data["actions"])
                agent_ids.extend([i for i in range(len(step_data["actions"]))])

            if len(obs_seq) < 2:
                continue

            # Convert to tensors
            obs_tensor = torch.FloatTensor(obs_seq[:-1])  # All but last
            action_tensor = torch.LongTensor(action_seq[:-1])
            target_actions = torch.LongTensor(action_seq[1:])  # Shifted by 1

            positions = torch.arange(len(obs_seq[:-1])).unsqueeze(0).expand(1, -1)
            agent_id_tensor = torch.zeros_like(action_tensor)  # Simplified

            # Forward pass
            predictions = self.seq_model(
                obs_tensor.unsqueeze(0),
                action_tensor.unsqueeze(0),
                agent_id_tensor.unsqueeze(0),
                positions,
            )

            # Compute loss for each agent
            total_loss = 0
            for i, pred in enumerate(predictions):
                if i < target_actions.size(1):
                    loss = F.cross_entropy(pred, target_actions[:, i])
                    total_loss += loss

            # Update sequence model
            self.seq_optimizer.zero_grad()
            total_loss.backward()
            self.seq_optimizer.step()

    def soft_update_targets(self):
        """Soft update target networks"""
        for target_net, main_net in zip(self.target_networks, self.agent_networks):
            for target_param, main_param in zip(
                target_net.parameters(), main_net.parameters()
            ):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )

        for target_param, main_param in zip(
            self.target_qmix_net.parameters(), self.qmix_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * main_param.data + (1.0 - self.tau) * target_param.data
            )

    def run_episode(self, env, pathfinder, max_steps: int = 1000) -> Dict:
        """Run a single episode"""
        # Reset environment
        start_pos = (0.5, 0.5)
        target_start = (
            random.uniform(env.width * 0.7, env.width - 1),
            random.uniform(env.height * 0.7, env.height - 1),
        )

        # Initialize agents (start with 1, deploy more as needed)
        active_agents = 1
        agent_positions = [start_pos] + [None] * (self.max_agents - 1)

        # Initialize target
        from RL_alternative_approach.MARL.pathfinding_marl import (
            VehicleTarget,
        )  # Import your target class

        target = VehicleTarget(env, target_start, speed=0.25)

        episode_data = []
        total_rewards = np.zeros(self.max_agents)
        step_count = 0
        success = False

        # Deployment timers
        agent_deploy_times = [0, 30, 100]  # Frames when agents deploy

        for step in range(max_steps):
            step_count = step

            # Deploy additional agents based on time and performance
            if step >= agent_deploy_times[1] and active_agents == 1:
                # Deploy second agent if first agent hasn't caught target
                agent_positions[1] = start_pos
                active_agents = 2

            if step >= agent_deploy_times[2] and active_agents == 2:
                # Deploy third agent for coordination
                agent_positions[2] = start_pos
                active_agents = 3

            # Get observations for active agents
            observations = []
            actions = []

            for i in range(active_agents):
                if agent_positions[i] is not None:
                    obs = self.get_local_observation(
                        agent_positions[i],
                        env,
                        [pos for j, pos in enumerate(agent_positions) if j != i],
                        target.position,
                    )
                    action = self.select_action(obs, i, training=True)

                    observations.append(obs)
                    actions.append(action)

                    # Move agent
                    dx, dy = self.actions[action]
                    new_x = agent_positions[i][0] + dx * 0.5  # Step size
                    new_y = agent_positions[i][1] + dy * 0.5

                    if env.is_valid_position(new_x, new_y):
                        agent_positions[i] = (new_x, new_y)

            # Move target (flees from closest agent)
            if active_agents > 0:
                closest_agent = min(
                    [pos for pos in agent_positions[:active_agents] if pos is not None],
                    key=lambda pos: np.sqrt(
                        (pos[0] - target.position[0]) ** 2
                        + (pos[1] - target.position[1]) ** 2
                    ),
                )
                target.update_position(closest_agent)

            # Calculate rewards
            rewards = np.zeros(self.max_agents)

            # Check for capture
            captured = False
            for i in range(active_agents):
                if agent_positions[i] is not None:
                    dist = np.sqrt(
                        (agent_positions[i][0] - target.position[0]) ** 2
                        + (agent_positions[i][1] - target.position[1]) ** 2
                    )

                    if dist <= 1.5:  # Capture radius
                        rewards[i] += 100  # Large capture reward
                        captured = True
                        success = True
                    else:
                        # Reward for getting closer
                        rewards[i] += max(0, 10 - dist)

                    # Penalty for collisions with other agents
                    for j in range(i + 1, active_agents):
                        if agent_positions[j] is not None:
                            agent_dist = np.sqrt(
                                (agent_positions[i][0] - agent_positions[j][0]) ** 2
                                + (agent_positions[i][1] - agent_positions[j][1]) ** 2
                            )
                            if agent_dist < 1.0:
                                rewards[i] -= 5
                                rewards[j] -= 5

            # Collaboration bonus
            if active_agents >= 2:
                # Bonus for coordinated positioning around target
                distances_to_target = [
                    np.sqrt(
                        (pos[0] - target.position[0]) ** 2
                        + (pos[1] - target.position[1]) ** 2
                    )
                    for pos in agent_positions[:active_agents]
                    if pos is not None
                ]

                if len(distances_to_target) >= 2:
                    avg_dist = np.mean(distances_to_target)
                    if avg_dist < 5.0:  # Close coordination
                        coord_bonus = 5
                        for i in range(active_agents):
                            rewards[i] += coord_bonus

            total_rewards += rewards

            # Store step data
            step_data = {
                "observations": observations,
                "actions": actions,
                "rewards": rewards[:active_agents].tolist(),
                "agent_positions": agent_positions[:active_agents].copy(),
                "target_position": target.position,
                "active_agents": active_agents,
            }
            episode_data.append(step_data)

            if captured:
                break

        # Store episode results
        self.episode_rewards.append(total_rewards.sum())
        self.episode_steps.append(step_count)
        self.success_rate.append(1.0 if success else 0.0)

        # Store sequence for transformer training
        self.store_sequence(episode_data)

        return {
            "success": success,
            "steps": step_count,
            "total_reward": total_rewards.sum(),
            "episode_data": episode_data,
        }

    def train(self, env, pathfinder, num_episodes: int = 1000):
        """Train the MARL system"""
        print("Starting MARL training...")

        for episode in range(num_episodes):
            # Run episode
            result = self.run_episode(env, pathfinder)

            # Train networks
            if episode % 4 == 0:  # Train every 4 episodes
                for _ in range(5):  # Multiple training steps
                    self.train_step()

                if episode % 20 == 0:  # Train sequence model less frequently
                    self.train_sequence_model()

            # Print progress
            if episode % 100 == 0:
                avg_reward = (
                    np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
                )
                success_rate = np.mean(self.success_rate) if self.success_rate else 0
                print(
                    f"Episode {episode}: Avg Reward: {avg_reward:.2f}, "
                    f"Success Rate: {success_rate:.2%}, Epsilon: {self.epsilon:.3f}"
                )

        print("Training completed!")

    def save_models(self, path: str):
        """Save trained models"""
        os.makedirs(path, exist_ok=True)

        # Save agent networks
        for i, net in enumerate(self.agent_networks):
            torch.save(net.state_dict(), f"{path}/agent_{i}.pt")

        # Save QMIX network
        torch.save(self.qmix_net.state_dict(), f"{path}/qmix.pt")

        # Save sequence model
        torch.save(self.seq_model.state_dict(), f"{path}/sequence_model.pt")

        # Save training statistics
        stats = {
            "episode_rewards": self.episode_rewards,
            "episode_steps": self.episode_steps,
            "success_rate": list(self.success_rate),
        }
        with open(f"{path}/training_stats.pkl", "wb") as f:
            pickle.dump(stats, f)

    def load_models(self, path: str):
        """Load trained models"""
        # Load agent networks
        for i, net in enumerate(self.agent_networks):
            net.load_state_dict(torch.load(f"{path}/agent_{i}.pt"))

        # Load QMIX network
        self.qmix_net.load_state_dict(torch.load(f"{path}/qmix.pt"))

        # Load sequence model
        self.seq_model.load_state_dict(torch.load(f"{path}/sequence_model.pt"))

        # Load training statistics
        with open(f"{path}/training_stats.pkl", "rb") as f:
            stats = pickle.load(f)
            self.episode_rewards = stats["episode_rewards"]
            self.episode_steps = stats["episode_steps"]
            self.success_rate = deque(stats["success_rate"], maxlen=100)


# Integration function with your existing code
def integrate_marl_with_pathfinding():
    """Example of how to integrate MARL with your existing pathfinding code"""
    from RL_alternative_approach.MARL.pathfinding_marl import (
        create_sample_neighborhood,
        AStar,
    )

    # Create environment
    env = create_sample_neighborhood()
    pathfinder = AStar(env)

    # Create MARL system
    marl_system = MARLPursuitSystem(
        env_width=env.width, env_height=env.height, max_agents=3
    )

    # Train the system
    marl_system.train(env, pathfinder, num_episodes=2000)

    # Save trained models
    marl_system.save_models("./marl_models")

    return marl_system


if __name__ == "__main__":
    # Example usage
    marl_system = integrate_marl_with_pathfinding()
