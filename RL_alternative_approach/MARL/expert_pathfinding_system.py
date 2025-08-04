import numpy as np
from RL_alternative_approach.MARL.pathfinding_marl import (
    AStar,
    smooth_path_with_bezier,
)
from pursuit_environment import PursuitEnvironment


class PathfindingFlowExpertSystem:
    """
    Expert system that generates pathfinding-based demonstrations for multi-agent learning.

    This class implements sophisticated pathfinding strategies for coordinated pursuit scenarios,
    providing high-quality expert demonstrations to guide reinforcement learning algorithms.
    """

    def __init__(self, env):
        """
        Initialize the pathfinding expert system.

        Args:
            env: Environment instance used as template for pathfinding operations
        """
        self.env = env
        self.pathfinder = AStar(env.env)

    def generate_expert_actions_with_flow(self, env_state):
        """
        Generate expert actions with improved coordination strategies.

        This method implements sophisticated multi-agent coordination by combining
        direct pursuit and strategic interception behaviors based on current environment state.

        Args:
            env_state: Current environment state containing agent positions and target information

        Returns:
            list: Expert action indices for both agents [agent1_action, agent2_action]
        """
        agent_positions = env_state.agent_positions
        target_position = env_state.target.position
        agent1_deployed = env_state.agent1_deployed
        agent2_deployed = env_state.agent2_deployed
        pursuit_active = env_state.pursuit_active

        actions = [4, 4]  # Default: both agents stay in place

        # Agent 1: Direct pursuit when deployed
        if agent1_deployed:
            try:
                # More aggressive pursuit
                action1 = self.generate_pursuit_action(
                    agent_positions[0], target_position, aggressive=True
                )
                actions[0] = action1
            except Exception:
                actions[0] = 4

        # Agent 2: Strategic interception when deployed
        if agent2_deployed and pursuit_active:
            try:
                # Predict interception point more accurately
                intercept_position = self.predict_interception_point(
                    agent_positions, target_position, env_state.target_history
                )

                # Use different strategy if too close to agent 1
                agent1_pos = agent_positions[0]
                agent2_pos = agent_positions[1]
                distance_between_agents = np.sqrt(
                    (agent1_pos[0] - agent2_pos[0]) ** 2
                    + (agent1_pos[1] - agent2_pos[1]) ** 2
                )

                if distance_between_agents < 2.0:
                    # If too close, move to flanking position
                    flank_position = self.calculate_flanking_position(
                        agent_positions, target_position
                    )
                    action2 = self.generate_pursuit_action(
                        agent_positions[1], flank_position
                    )
                else:
                    action2 = self.generate_pursuit_action(
                        agent_positions[1], intercept_position
                    )

                actions[1] = action2
            except Exception:
                actions[1] = 4

        return actions

    def calculate_flanking_position(self, agent_positions, target_position):
        """
        Calculate optimal flanking position for agent coordination.

        This method computes a strategic flanking position to avoid agent clustering
        and improve capture probability through coordinated positioning.

        Args:
            agent_positions (list): Current positions of both agents
            target_position (tuple): Current target position

        Returns:
            tuple: Optimal flanking position coordinates (x, y)
        """
        agent1_pos = agent_positions[0]
        target_pos = target_position

        # Vector from agent 1 to target
        vec_to_target = np.array(
            [target_pos[0] - agent1_pos[0], target_pos[1] - agent1_pos[1]]
        )

        # Perpendicular vector for flanking
        perp_vec = np.array([-vec_to_target[1], vec_to_target[0]])

        # Normalize and scale
        if np.linalg.norm(perp_vec) > 0:
            perp_vec = perp_vec / np.linalg.norm(perp_vec) * 3.0

        # Try both sides for flanking
        flank1 = (target_pos[0] + perp_vec[0], target_pos[1] + perp_vec[1])
        flank2 = (target_pos[0] - perp_vec[0], target_pos[1] - perp_vec[1])

        # Choose the flank position that's valid and further from agent 1
        if self.env.env.is_valid_position(flank1[0], flank1[1]):
            return flank1
        elif self.env.env.is_valid_position(flank2[0], flank2[1]):
            return flank2
        else:
            return target_position

    def generate_pursuit_action(
        self, agent_pos, target_pos, aggressive=False, use_smooth=True
    ):
        """
        Generate optimal pursuit action using pathfinding algorithms.

        This method combines A* pathfinding with smooth trajectory generation
        to produce natural and effective pursuit behaviors.

        Args:
            agent_pos (tuple): Current agent position
            target_pos (tuple): Target position to pursue
            aggressive (bool): Whether to use more aggressive pursuit strategy
            use_smooth (bool): Whether to apply trajectory smoothing

        Returns:
            int: Action index for optimal pursuit movement
        """
        # Find path from agent to target
        path = self.pathfinder.find_path(agent_pos, target_pos)

        if not path or len(path) < 2:
            # If no path, move directly toward target
            return self.direct_movement_action(agent_pos, target_pos, aggressive)

        # Use smooth trajectories for more natural movement
        if use_smooth and len(path) >= 3:
            try:
                smooth_path = smooth_path_with_bezier(
                    path, num_points=min(20, len(path) * 2)
                )
                if smooth_path and len(smooth_path) >= 2:
                    # If aggressive, look further ahead in the path
                    next_idx = 2 if aggressive and len(smooth_path) > 2 else 1
                    next_pos = smooth_path[next_idx]
                else:
                    next_pos = path[1]
            except Exception:
                next_pos = path[1]
        else:
            # If aggressive, skip one step in the path
            next_idx = 2 if aggressive and len(path) > 2 else 1
            next_pos = path[next_idx] if next_idx < len(path) else path[-1]

        return self.position_to_action(agent_pos, next_pos)

    def direct_movement_action(self, agent_pos, target_pos, aggressive=False):
        """
        Generate direct movement toward target when pathfinding is unavailable.

        This fallback method provides direct movement calculations when A* pathfinding
        fails or is not applicable for the current scenario.

        Args:
            agent_pos (tuple): Current agent position
            target_pos (tuple): Target position
            aggressive (bool): Whether to scale movement for aggressive pursuit

        Returns:
            int: Action index for direct movement toward target
        """
        dx = target_pos[0] - agent_pos[0]
        dy = target_pos[1] - agent_pos[1]

        # Scale movement if aggressive
        if aggressive:
            scale = 1.5
            dx *= scale
            dy *= scale

        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return 4  # Stay in place

        # Map to 8 directions
        angle = np.arctan2(dy, dx)
        angle_degrees = (np.degrees(angle) + 360) % 360

        # Return action based on angle
        if angle_degrees < 22.5 or angle_degrees >= 337.5:
            return 2  # E
        elif angle_degrees < 67.5:
            return 1  # NE
        elif angle_degrees < 112.5:
            return 0  # N
        elif angle_degrees < 157.5:
            return 7  # NW
        elif angle_degrees < 202.5:
            return 6  # W
        elif angle_degrees < 247.5:
            return 5  # SW
        elif angle_degrees < 292.5:
            return 4  # S
        else:
            return 3  # SE

    def predict_interception_point(
        self, agent_positions, target_position, target_history
    ):
        """
        Predict optimal interception point based on target movement patterns.

        This method analyzes target movement history to predict future positions
        and calculate strategic interception points for coordinated capture.

        Args:
            agent_positions (list): Current positions of both agents
            target_position (tuple): Current target position
            target_history (list): Historical target positions for movement prediction

        Returns:
            tuple: Predicted interception coordinates (x, y)
        """
        if len(target_history) < 3:
            # If no history, try to position opposite to agent 1
            agent1_pos = agent_positions[0]
            dx = target_position[0] - agent1_pos[0]
            dy = target_position[1] - agent1_pos[1]

            # Position on opposite side
            intercept_x = target_position[0] + dx * 0.5
            intercept_y = target_position[1] + dy * 0.5

            # Ensure within bounds
            intercept_x = max(1, min(self.env.width - 1, intercept_x))
            intercept_y = max(1, min(self.env.height - 1, intercept_y))

            return (intercept_x, intercept_y)

        # Predict target movement direction from history
        recent_history = target_history[-5:]
        dx = recent_history[-1][0] - recent_history[0][0]
        dy = recent_history[-1][1] - recent_history[0][1]

        # Predict where target will be in 5-10 steps
        prediction_steps = 7
        predicted_x = target_position[0] + dx * prediction_steps
        predicted_y = target_position[1] + dy * prediction_steps

        # Ensure prediction is within bounds and valid
        predicted_x = max(1, min(self.env.width - 1, predicted_x))
        predicted_y = max(1, min(self.env.height - 1, predicted_y))

        if self.env.env.is_valid_position(predicted_x, predicted_y):
            return (predicted_x, predicted_y)
        else:
            # Fallback to current target position
            return target_position

    def position_to_action(self, current_pos, next_pos):
        """
        Convert position difference to discrete action index.

        Maps continuous position changes to discrete directional actions
        using angle-based direction mapping for consistent movement control.

        Args:
            current_pos (tuple): Current agent position
            next_pos (tuple): Target position for next step

        Returns:
            int: Discrete action index (0-7 for directions, 4 for stay)
        """
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return 4  # Stay in place

        # Map to 8 directions + stay
        angle = np.arctan2(dy, dx)
        angle_degrees = (np.degrees(angle) + 360) % 360

        # Map angles to actions (0-7 for 8 directions, 4 for stay)
        if angle_degrees < 22.5 or angle_degrees >= 337.5:
            return 2  # E
        elif angle_degrees < 67.5:
            return 1  # NE
        elif angle_degrees < 112.5:
            return 0  # N
        elif angle_degrees < 157.5:
            return 7  # NW
        elif angle_degrees < 202.5:
            return 6  # W
        elif angle_degrees < 247.5:
            return 5  # SW
        elif angle_degrees < 292.5:
            return 4  # S (could also be stay)
        else:
            return 3  # SE

    def generate_demonstration_episode(self, max_steps=500):
        """
        Generate complete expert demonstration episode following pathfinding flow.

        This method creates a full episode demonstration using expert pathfinding
        strategies, providing high-quality training data for imitation learning.

        Args:
            max_steps (int): Maximum number of steps for the demonstration episode

        Returns:
            dict: Complete episode data with states, actions, rewards, and transitions
        """
        # Create fresh environment
        env = PursuitEnvironment(self.env.width, self.env.height)
        global_state, individual_obs = env.reset()

        episode_data = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "info": [],
        }

        for step in range(max_steps):
            # Generate expert actions following the flow
            actions = self.generate_expert_actions_with_flow(env)

            # Store current state
            episode_data["states"].append(
                (global_state.copy(), [obs.copy() for obs in individual_obs])
            )
            episode_data["actions"].append(actions.copy())

            # Take step
            (next_global_state, next_individual_obs), reward, done, info = env.step(
                actions
            )

            # Store transition
            episode_data["rewards"].append(reward + 2.0)  # Expert bonus
            episode_data["next_states"].append(
                (next_global_state.copy(), [obs.copy() for obs in next_individual_obs])
            )
            episode_data["dones"].append(done)
            episode_data["info"].append(info.copy())

            if done:
                break

            # Update for next iteration
            global_state = next_global_state
            individual_obs = next_individual_obs

        return episode_data
