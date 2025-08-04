import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")

from pathfinding_marl import (
    AStar,
    VehicleTarget,
)
import numpy as np


class AdaptivePathfindingExpert:
    """
    Adaptive expert system that generates pathfinding demonstrations for struggling scenarios.

    This class provides expert guidance by running pathfinding algorithms on specific
    scenarios where the learning algorithm is performing poorly, generating high-quality
    demonstration data to improve learning performance.
    """

    def __init__(self, env):
        """
        Initialize the adaptive pathfinding expert system.

        Args:
            env: Base environment instance used as template for demonstrations
        """
        self.env = env
        self.pathfinder = AStar(env.env)  # Use the grid environment for pathfinding

    def generate_expert_demonstration_for_scenario(self, scenario_config):
        """
        Generate expert demonstration by running pathfinding algorithm on a specific scenario.

        This method creates a demonstration by executing optimal pathfinding strategies
        on the exact scenario configuration where the learning algorithm is struggling.

        Args:
            scenario_config (dict): Configuration dictionary containing agent positions,
                                  target state, and environment parameters

        Returns:
            dict: Demonstration data containing states, actions, rewards, and transitions
        """
        print("Generating pathfinding demonstration for struggling scenario...")

        # Create a copy of the environment in the exact same state
        demo_env = self._create_demo_environment(scenario_config)

        # Run pathfinding simulation
        demonstration = self._run_pathfinding_simulation(demo_env, scenario_config)

        print(f"Generated {len(demonstration['actions'])} expert action steps")
        return demonstration

    def _create_demo_environment(self, scenario_config):
        """
        Create demonstration environment matching the current struggling scenario.

        Args:
            scenario_config (dict): Scenario configuration parameters

        Returns:
            PursuitEnvironment: Environment instance configured to match scenario
        """
        from pursuit_environment import PursuitEnvironment

        # Create identical environment
        demo_env = PursuitEnvironment(
            self.env.width, self.env.height, training_mode=False
        )

        # Set exact same positions
        demo_env.agent_positions = [
            tuple(scenario_config["agent1_pos"]),
            tuple(scenario_config["agent2_pos"]),
        ]

        # Set exact same target
        demo_env.target = VehicleTarget(
            demo_env.env,
            tuple(scenario_config["target_pos"]),
            speed=scenario_config["target_speed"],
            inertia=0.8,
        )

        # Set deployment status
        demo_env.agent1_deployed = scenario_config["agent1_deployed"]
        demo_env.agent2_deployed = scenario_config["agent2_deployed"]
        demo_env.pursuit_active = scenario_config["pursuit_active"]
        demo_env.step_count = scenario_config["step_count"]

        return demo_env

    def _run_pathfinding_simulation(self, demo_env, scenario_config):
        """
        Execute pathfinding algorithm simulation and capture action sequence.

        Args:
            demo_env: Demonstration environment instance
            scenario_config (dict): Scenario configuration parameters

        Returns:
            dict: Complete demonstration data with state transitions and expert actions
        """
        # This simulates what your pathfinding animation does
        demonstration = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": [],
            "info": [],
        }

        max_demo_steps = 200

        for step in range(max_demo_steps):
            # Get current state
            global_state, individual_obs = demo_env._get_state()

            # Generate pathfinding actions using the exact same logic as your animation
            actions = self._generate_pathfinding_actions(demo_env)

            # Store current state
            demonstration["states"].append(
                (global_state.copy(), [obs.copy() for obs in individual_obs])
            )
            demonstration["actions"].append(actions.copy())

            # Take step in demo environment
            (next_global_state, next_individual_obs), reward, done, info = (
                demo_env.step(actions)
            )

            # Store transition with expert bonus
            demonstration["rewards"].append(reward + 5.0)  # Expert bonus
            demonstration["next_states"].append(
                (next_global_state.copy(), [obs.copy() for obs in next_individual_obs])
            )
            demonstration["dones"].append(done)
            demonstration["info"].append(info.copy())

            if done:
                print(
                    f"Pathfinding demo completed in {step + 1} steps - Captured: {info.get('captured', False)}"
                )
                break

        return demonstration

    def _generate_pathfinding_actions(self, env):
        """
        Generate optimal actions using pathfinding logic for both agents.

        Args:
            env: Environment instance for action generation

        Returns:
            list: Action indices for both agents [agent1_action, agent2_action]
        """
        actions = [4, 4]  # Default: stay in place

        # Agent 1: Direct pursuit when deployed (exactly like your animation)
        if env.agent1_deployed:
            try:
                # Find path to target
                agent1_path = self.pathfinder.find_path(
                    env.agent_positions[0], env.target.position
                )
                if agent1_path and len(agent1_path) >= 2:
                    next_pos = agent1_path[1]  # Next position in path
                    actions[0] = self._position_to_action(
                        env.agent_positions[0], next_pos
                    )
                else:
                    # Direct movement if no path
                    actions[0] = self._direct_movement_action(
                        env.agent_positions[0], env.target.position
                    )
            except Exception:
                actions[0] = 4

        # Agent 2: Interception when deployed (exactly like your animation)
        if env.agent2_deployed and env.pursuit_active:
            try:
                # Predict target movement and intercept (simplified version of your prediction)
                intercept_point = self._predict_intercept_point(env)

                # Find path to intercept point
                agent2_path = self.pathfinder.find_path(
                    env.agent_positions[1], intercept_point
                )
                if agent2_path and len(agent2_path) >= 2:
                    next_pos = agent2_path[1]
                    actions[1] = self._position_to_action(
                        env.agent_positions[1], next_pos
                    )
                else:
                    actions[1] = self._direct_movement_action(
                        env.agent_positions[1], intercept_point
                    )
            except Exception:
                actions[1] = 4

        return actions

    def _predict_intercept_point(self, env):
        """
        Predict optimal interception point for Agent 2 based on target movement.

        This method implements a simplified version of the interception prediction
        used in the pathfinding animation system.

        Args:
            env: Environment instance containing target and agent information

        Returns:
            tuple: Predicted interception coordinates (x, y)
        """
        target_pos = env.target.position

        # Simple prediction: where target will be in a few steps
        if hasattr(env, "target_history") and len(env.target_history) >= 3:
            recent = env.target_history[-3:]
            dx = recent[-1][0] - recent[0][0]
            dy = recent[-1][1] - recent[0][1]

            # Predict 5 steps ahead
            predicted_x = target_pos[0] + dx * 1.5
            predicted_y = target_pos[1] + dy * 1.5

            # Ensure within bounds
            predicted_x = max(1, min(env.width - 1, predicted_x))
            predicted_y = max(1, min(env.height - 1, predicted_y))

            if env.env.is_valid_position(predicted_x, predicted_y):
                return (predicted_x, predicted_y)

        # Fallback: position opposite to agent 1
        agent1_pos = env.agent_positions[0]
        dx = target_pos[0] - agent1_pos[0]
        dy = target_pos[1] - agent1_pos[1]

        opposite_x = target_pos[0] + dx * 0.5
        opposite_y = target_pos[1] + dy * 0.5

        opposite_x = max(1, min(env.width - 1, opposite_x))
        opposite_y = max(1, min(env.height - 1, opposite_y))

        return (opposite_x, opposite_y)

    def _position_to_action(self, current_pos, next_pos):
        """
        Convert position difference to discrete action index.

        Maps continuous position changes to one of 8 discrete directional actions
        or stay-in-place action based on movement angle.

        Args:
            current_pos (tuple): Current agent position (x, y)
            next_pos (tuple): Target position (x, y)

        Returns:
            int: Action index (0-7 for directions, 4 for stay)
        """
        dx = next_pos[0] - current_pos[0]
        dy = next_pos[1] - current_pos[1]

        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return 4  # Stay in place

        # Map to 8 directions
        angle = np.arctan2(dy, dx)
        angle_degrees = (np.degrees(angle) + 360) % 360

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

    def _direct_movement_action(self, agent_pos, target_pos):
        """
        Generate direct movement action when pathfinding fails.

        Args:
            agent_pos (tuple): Current agent position
            target_pos (tuple): Target position

        Returns:
            int: Action index for direct movement toward target
        """
        return self._position_to_action(agent_pos, target_pos)
