import numpy as np
import random
from pathfinding import Environment, VehicleTarget


class PursuitEnvironment:
    def __init__(self, width, height, training_mode=True):
        self.width = width
        self.height = height
        self.env = Environment(width, height)
        self.agent_positions = [(0.5, 0.5), (0.5, 0.5)]
        self.target = None
        self.target_history = []
        self.step_count = 0
        self.max_steps = 500
        self.training_mode = training_mode  # More lenient capture for training

        # Generate urban-style obstacles
        self._generate_urban_environment()

    def _generate_urban_environment(self):
        """Generate realistic urban environment with buildings and streets"""
        # Clear existing obstacles
        self.env.obstacles = []

        # Create urban grid with buildings and streets
        self._create_urban_blocks()

        # Add some scattered smaller buildings
        self._add_scattered_buildings()

        # Ensure there are clear paths (streets)
        self._ensure_street_network()

    def _create_urban_blocks(self):
        """Create main urban blocks with streets between them"""
        # Calculate block sizes based on environment size
        min_block_size = max(3, self.width // 10)
        max_block_size = max(5, self.width // 5)
        street_width = max(2, self.width // 20)

        # Ensure minimum sizes are reasonable
        if min_block_size >= max_block_size:
            max_block_size = min_block_size + 2

        # Create a grid of blocks
        current_x = street_width
        while current_x < self.width - max_block_size:
            current_y = street_width
            while current_y < self.height - max_block_size:
                # Random block dimensions with better validation
                block_width = random.randint(min_block_size, max_block_size)
                block_height = random.randint(min_block_size, max_block_size)

                # Ensure block fits in environment
                block_width = min(block_width, self.width - current_x - street_width)
                block_height = min(block_height, self.height - current_y - street_width)

                # Only create block if it's large enough and fits
                if block_width >= min_block_size and block_height >= min_block_size:
                    # Sometimes create L-shaped or partial blocks for variety
                    if (
                        random.random() < 0.3
                        and block_width >= min_block_size * 2
                        and block_height >= min_block_size * 2
                    ):
                        self._create_complex_block(
                            current_x, current_y, block_width, block_height
                        )
                    else:
                        self.env.add_block(
                            current_x, current_y, block_width, block_height
                        )

                # Move to next block position
                current_y += (
                    block_height + street_width + random.randint(0, street_width)
                )

            current_x += (
                random.randint(min_block_size, max_block_size)
                + street_width
                + random.randint(0, street_width)
            )

    def _create_complex_block(self, start_x, start_y, max_width, max_height):
        """Create L-shaped or complex building structures"""
        # Create main block
        main_width = random.randint(max_width // 2, max_width)
        main_height = random.randint(max_height // 2, max_height)
        self.env.add_block(start_x, start_y, main_width, main_height)

        # Add extension with some probability
        if random.random() < 0.6:
            if random.random() < 0.5:  # Horizontal extension
                min_ext_width = max_width // 3
                max_ext_width = max_width - main_width

                # Ensure valid range for extension width
                if min_ext_width < max_ext_width and max_ext_width > 0:
                    ext_width = random.randint(min_ext_width, max_ext_width)

                    min_ext_height = max_height // 3
                    max_ext_height = max_height // 2

                    # Ensure valid range for extension height
                    if min_ext_height < max_ext_height:
                        ext_height = random.randint(min_ext_height, max_ext_height)
                    else:
                        ext_height = min_ext_height

                    ext_x = start_x + main_width

                    # Ensure extension doesn't go outside bounds
                    max_ext_y = main_height - ext_height
                    if max_ext_y > 0:
                        ext_y = start_y + random.randint(0, max_ext_y)
                    else:
                        ext_y = start_y

                    # Check bounds before adding
                    if (
                        ext_x + ext_width < self.width
                        and ext_y + ext_height < self.height
                        and ext_x >= 0
                        and ext_y >= 0
                    ):
                        self.env.add_block(ext_x, ext_y, ext_width, ext_height)

            else:  # Vertical extension
                min_ext_width = max_width // 3
                max_ext_width = max_width // 2

                # Ensure valid range for extension width
                if min_ext_width < max_ext_width:
                    ext_width = random.randint(min_ext_width, max_ext_width)
                else:
                    ext_width = min_ext_width

                min_ext_height = max_height // 3
                max_ext_height = max_height - main_height

                # Ensure valid range for extension height
                if min_ext_height < max_ext_height and max_ext_height > 0:
                    ext_height = random.randint(min_ext_height, max_ext_height)

                    # Ensure extension doesn't go outside bounds
                    max_ext_x = main_width - ext_width
                    if max_ext_x > 0:
                        ext_x = start_x + random.randint(0, max_ext_x)
                    else:
                        ext_x = start_x

                    ext_y = start_y + main_height

                    # Check bounds before adding
                    if (
                        ext_x + ext_width < self.width
                        and ext_y + ext_height < self.height
                        and ext_x >= 0
                        and ext_y >= 0
                    ):
                        self.env.add_block(ext_x, ext_y, ext_width, ext_height)

    def _add_scattered_buildings(self):
        """Add smaller scattered buildings for variety"""
        num_scattered = random.randint(2, max(3, self.width // 15))

        for _ in range(num_scattered):
            attempts = 0
            while attempts < 50:  # Limit attempts to prevent infinite loop
                # Random small building with minimum size constraints
                min_building_size = 2
                max_building_width = max(3, self.width // 12)
                max_building_height = max(3, self.height // 12)

                building_width = random.randint(min_building_size, max_building_width)
                building_height = random.randint(min_building_size, max_building_height)

                x = random.randint(1, max(2, self.width - building_width - 1))
                y = random.randint(1, max(2, self.height - building_height - 1))

                # Check if this position is valid (not overlapping with existing buildings)
                if self._is_area_clear(x, y, building_width, building_height):
                    self.env.add_block(x, y, building_width, building_height)
                    break

                attempts += 1

    def _ensure_street_network(self):
        """Ensure there are clear horizontal and vertical streets"""
        street_width = 1

        # Add a main horizontal street
        main_street_y = self.height // 2
        for x in range(self.width):
            # Remove any obstacles in the main street area
            self._clear_area(x, main_street_y - street_width, 1, street_width * 2 + 1)

        # Add a main vertical street
        main_street_x = self.width // 2
        for y in range(self.height):
            # Remove any obstacles in the main street area
            self._clear_area(main_street_x - street_width, y, street_width * 2 + 1, 1)

    def _is_area_clear(self, x, y, width, height, margin=1):
        """Check if an area is clear of obstacles"""
        for check_x in range(max(0, x - margin), min(self.width, x + width + margin)):
            for check_y in range(
                max(0, y - margin), min(self.height, y + height + margin)
            ):
                if not self.env.is_valid_position(check_x, check_y):
                    return False
        return True

    def _clear_area(self, x, y, width, height):
        """Clear an area of obstacles (for creating streets)"""
        # This is a simplified version - you might need to implement
        # obstacle removal in your Environment class
        pass

    def _get_random_valid_position(self, margin=2.0):
        """Get a random valid position that's not in buildings with margin"""
        max_attempts = 200
        for _ in range(max_attempts):
            x = random.uniform(margin, self.width - margin)
            y = random.uniform(margin, self.height - margin)

            # Check if position is valid and has some clearance
            if self.env.is_valid_position(x, y) and self._has_clearance(x, y, margin):
                return (x, y)

        # Fallback: find any valid position
        for _ in range(max_attempts):
            x = random.uniform(1, self.width - 1)
            y = random.uniform(1, self.height - 1)
            if self.env.is_valid_position(x, y):
                return (x, y)

        # Final fallback
        return (1.0, 1.0)

    def _has_clearance(self, x, y, clearance):
        """Check if a position has sufficient clearance from obstacles"""
        for check_x in np.arange(x - clearance, x + clearance, 0.5):
            for check_y in np.arange(y - clearance, y + clearance, 0.5):
                if not self.env.is_valid_position(check_x, check_y):
                    return False
        return True

    def _get_random_valid_target_position(self):
        """Get a random valid position for the target (not in buildings)"""
        return self._get_random_valid_position(margin=1.5)

    def _get_random_valid_agent_positions(self):
        """Get random valid starting positions for agents"""
        positions = []
        min_distance = max(
            3.0, min(self.width, self.height) / 10
        )  # Minimum distance between agents

        for i in range(2):  # Two agents
            attempts = 0
            while attempts < 100:
                pos = self._get_random_valid_position(margin=1.0)

                # Check distance from other agents
                if i == 0 or all(
                    np.sqrt((pos[0] - p[0]) ** 2 + (pos[1] - p[1]) ** 2) >= min_distance
                    for p in positions
                ):
                    positions.append(pos)
                    break
                attempts += 1

            # Fallback if no good position found
            if len(positions) <= i:
                positions.append(self._get_random_valid_position(margin=0.5))

        return positions

    def reset(self):
        """Reset environment for new episode with fixed agent positions"""
        # Both agents ALWAYS start at (0.5, 0.5) - matching your pathfinding flow
        self.agent_positions = [(0.5, 0.5), (0.5, 0.5)]

        # Get random valid position for target only
        target_pos = self._get_random_valid_target_position()

        # Target speed scaled to environment size (matching your pathfinding speeds)
        target_speed = min(0.5, max(0.2, self.width / 100.0))
        self.target = VehicleTarget(
            self.env, target_pos, speed=target_speed, inertia=0.8
        )

        self.target_history = [self.target.position]
        self.step_count = 0

        # Initialize deployment timing (matching your pathfinding flow)
        self.agent1_deployed = False
        self.agent2_deployed = False
        self.pursuit_active = False
        self.agent1_deploy_step = 30  # Agent 1 deploys after 30 steps
        self.agent2_deploy_step = 130  # Agent 2 deploys after 130 steps (30 + 100)

        return self._get_state()

    def step(self, actions):
        """Take a step in the environment following the pathfinding flow"""
        self.step_count += 1

        # Handle agent deployment timing (matching your pathfinding flow)
        if self.step_count >= self.agent1_deploy_step and not self.agent1_deployed:
            self.agent1_deployed = True
            print(f"Agent 1 deployed at step {self.step_count}")

        # Check if target starts fleeing (when agent 1 gets close)
        if self.agent1_deployed and not self.pursuit_active:
            distance_to_target = np.sqrt(
                (self.agent_positions[0][0] - self.target.position[0]) ** 2
                + (self.agent_positions[0][1] - self.target.position[1]) ** 2
            )
            if (
                distance_to_target < 3.0
            ):  # Target starts fleeing when agent 1 gets close
                self.pursuit_active = True
                print(f"Target starts fleeing at step {self.step_count}")

        # Deploy agent 2 when pursuit is active and timing is right
        if (
            self.pursuit_active
            and self.step_count >= self.agent2_deploy_step
            and not self.agent2_deployed
        ):
            self.agent2_deployed = True
            print(f"Agent 2 deployed for interception at step {self.step_count}")

        # Apply actions based on deployment status
        if self.agent1_deployed:
            # Agent 1 is deployed and can move
            self._apply_action(0, actions[0])
        # Agent 1 stays at (0.5, 0.5) until deployed

        if self.agent2_deployed:
            # Agent 2 is deployed and can move
            self._apply_action(1, actions[1])
        # Agent 2 stays at (0.5, 0.5) until deployed

        # Update target behavior based on pursuit status
        if self.pursuit_active:
            # Target flees from the closest deployed agent
            closest_agent_pos = self._get_closest_deployed_agent_position()
            if closest_agent_pos is not None:
                self.target.update_position(closest_agent_pos)
        # Target stays stationary until pursuit begins

        # Update target history
        self.target_history.append(self.target.position)
        if len(self.target_history) > 50:  # Keep only recent history
            self.target_history = self.target_history[-50:]

        # Calculate reward
        reward = self._calculate_reward()

        # Check if episode is done
        done = self._check_done()

        # Additional info
        info = {
            "captured": done and self._is_target_captured(),
            "agent1_deployed": self.agent1_deployed,
            "agent2_deployed": self.agent2_deployed,
            "pursuit_active": self.pursuit_active,
            "step": self.step_count,
        }

        return self._get_state(), reward, done, info

    def _get_closest_deployed_agent_position(self):
        """Get position of closest deployed agent to target"""
        deployed_agents = []

        if self.agent1_deployed:
            deployed_agents.append(self.agent_positions[0])
        if self.agent2_deployed:
            deployed_agents.append(self.agent_positions[1])

        if not deployed_agents:
            return None

        # Find closest deployed agent
        distances = [
            np.sqrt(
                (pos[0] - self.target.position[0]) ** 2
                + (pos[1] - self.target.position[1]) ** 2
            )
            for pos in deployed_agents
        ]

        closest_idx = np.argmin(distances)
        return deployed_agents[closest_idx]

    def _calculate_reward(self):
        """Enhanced reward structure for better coordination learning"""
        reward = 0.0

        if not self.agent1_deployed:
            return -0.01

        # Calculate distances
        dist1 = np.sqrt(
            (self.agent_positions[0][0] - self.target.position[0]) ** 2
            + (self.agent_positions[0][1] - self.target.position[1]) ** 2
        )

        max_distance = np.sqrt(self.width**2 + self.height**2)
        normalized_dist1 = dist1 / max_distance

        # Progressive proximity rewards with higher bonuses
        if dist1 <= 1.0:
            reward += 15.0  # Very close (increased)
        elif dist1 <= 1.5:
            reward += 12.0  # Close (increased)
        elif dist1 <= 2.5:
            reward += 8.0  # Approaching (increased)
        elif dist1 <= 4.0:
            reward += 4.0  # Getting closer (increased)

        # Base distance reward
        reward += (1.0 - normalized_dist1) * 5.0  # Increased

        # Two-agent coordination with enhanced teamwork
        if self.agent2_deployed:
            dist2 = np.sqrt(
                (self.agent_positions[1][0] - self.target.position[0]) ** 2
                + (self.agent_positions[1][1] - self.target.position[1]) ** 2
            )

            normalized_dist2 = dist2 / max_distance

            # Progressive proximity rewards for agent 2
            if dist2 <= 1.0:
                reward += 15.0
            elif dist2 <= 1.5:
                reward += 12.0
            elif dist2 <= 2.5:
                reward += 8.0
            elif dist2 <= 4.0:
                reward += 4.0

            reward += (1.0 - normalized_dist2) * 5.0

            # ENHANCED COORDINATION BONUSES
            # 1. Both agents in capture zone
            if dist1 <= 3.0 and dist2 <= 3.0:
                reward += 12.0  # Increased teamwork bonus

                # 2. Pincer movement bonus (agents on opposite sides)
                target_pos = np.array(self.target.position)
                agent1_vec = np.array(self.agent_positions[0]) - target_pos
                agent2_vec = np.array(self.agent_positions[1]) - target_pos

                if np.linalg.norm(agent1_vec) > 0 and np.linalg.norm(agent2_vec) > 0:
                    agent1_vec_norm = agent1_vec / np.linalg.norm(agent1_vec)
                    agent2_vec_norm = agent2_vec / np.linalg.norm(agent2_vec)

                    dot_product = np.dot(agent1_vec_norm, agent2_vec_norm)
                    # Reward when agents are on opposite sides (dot product < 0)
                    if dot_product < -0.3:  # Agents are more opposite
                        coordination_bonus = (1.0 + abs(dot_product)) * 8.0  # Increased
                        reward += coordination_bonus

                # 3. Optimal distance between agents
                agent_distance = np.sqrt(
                    (self.agent_positions[0][0] - self.agent_positions[1][0]) ** 2
                    + (self.agent_positions[0][1] - self.agent_positions[1][1]) ** 2
                )
                optimal_distance = 3.0  # Ideal distance between agents
                distance_bonus = max(0, 5.0 - abs(agent_distance - optimal_distance))
                reward += distance_bonus

            # 4. Escape route reduction bonus
            escape_routes = self._count_target_escape_routes()
            if escape_routes <= 4:
                reward += (
                    5 - escape_routes
                ) * 3.0  # More reward for fewer escape routes

            # 5. Speed coordination bonus (both agents moving toward target)
            if self._are_both_agents_advancing():
                reward += 5.0

        # MASSIVE capture reward with difficulty scaling
        if self._is_target_captured():
            capture_bonus = 200.0  # Increased from 150

            # Bonus for early capture
            if self.step_count < 200:
                capture_bonus += 50.0
            elif self.step_count < 300:
                capture_bonus += 25.0

            reward += capture_bonus

        # Deployment timing rewards
        if self.step_count == self.agent1_deploy_step and self.agent1_deployed:
            reward += 8.0  # Increased
        if self.step_count == self.agent2_deploy_step and self.agent2_deployed:
            reward += 8.0  # Increased

        # Reduced time penalty
        reward -= 0.01  # Reduced penalty

        return reward

    def _count_target_escape_routes(self):
        """Count available escape routes for target"""
        escape_routes = 0

        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            dx = np.cos(angle) * 2.0
            dy = np.sin(angle) * 2.0

            escape_x = self.target.position[0] + dx
            escape_y = self.target.position[1] + dy

            if self.env.is_valid_position(escape_x, escape_y):
                # Check if this escape leads away from both agents
                if self.agent2_deployed:
                    escape_dist1 = np.sqrt(
                        (escape_x - self.agent_positions[0][0]) ** 2
                        + (escape_y - self.agent_positions[0][1]) ** 2
                    )
                    escape_dist2 = np.sqrt(
                        (escape_x - self.agent_positions[1][0]) ** 2
                        + (escape_y - self.agent_positions[1][1]) ** 2
                    )

                    curr_dist1 = np.sqrt(
                        (self.agent_positions[0][0] - self.target.position[0]) ** 2
                        + (self.agent_positions[0][1] - self.target.position[1]) ** 2
                    )
                    curr_dist2 = np.sqrt(
                        (self.agent_positions[1][0] - self.target.position[0]) ** 2
                        + (self.agent_positions[1][1] - self.target.position[1]) ** 2
                    )

                    if (
                        escape_dist1 > curr_dist1 * 1.2
                        and escape_dist2 > curr_dist2 * 1.2
                    ):
                        escape_routes += 1
                else:
                    escape_routes += 1

        return escape_routes

    def _are_both_agents_advancing(self):
        """Check if both agents are moving toward the target"""
        if not hasattr(self, "prev_distances"):
            return False

        if not self.agent2_deployed:
            return False

        # Current distances
        curr_dist1 = np.sqrt(
            (self.agent_positions[0][0] - self.target.position[0]) ** 2
            + (self.agent_positions[0][1] - self.target.position[1]) ** 2
        )
        curr_dist2 = np.sqrt(
            (self.agent_positions[1][0] - self.target.position[0]) ** 2
            + (self.agent_positions[1][1] - self.target.position[1]) ** 2
        )

        # Check if both are getting closer
        advancing = (
            curr_dist1 < self.prev_distances[0] and curr_dist2 < self.prev_distances[1]
        )

        # Update previous distances
        self.prev_distances = [curr_dist1, curr_dist2]

        return advancing

    def _is_target_captured(self):
        """Adaptive capture conditions based on training mode"""
        if self.training_mode:
            return self._is_target_captured_training()
        else:
            return self._is_target_captured_strict()

    def _is_target_captured_training(self):
        """Lenient capture for training"""
        # Use the lenient conditions from above
        if not self.agent1_deployed:
            return False

        # Calculate distances
        dist1 = np.sqrt(
            (self.agent_positions[0][0] - self.target.position[0]) ** 2
            + (self.agent_positions[0][1] - self.target.position[1]) ** 2
        )

        # Single agent capture (close proximity)
        if dist1 <= 1.5:  # Increased from 1.0
            return True

        # Two-agent coordination capture
        if self.agent2_deployed:
            dist2 = np.sqrt(
                (self.agent_positions[1][0] - self.target.position[0]) ** 2
                + (self.agent_positions[1][1] - self.target.position[1]) ** 2
            )

            # Both agents reasonably close (increased capture area)
            if dist1 <= 2.5 and dist2 <= 2.5:  # Increased from 2.0
                # Count available escape routes (more lenient)
                escape_routes = 0

                # Test 8 directions (reduced from 32 for performance)
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    dx = np.cos(angle) * 2.0  # Reduced escape distance requirement
                    dy = np.sin(angle) * 2.0

                    escape_x = self.target.position[0] + dx
                    escape_y = self.target.position[1] + dy

                    if self.env.is_valid_position(escape_x, escape_y):
                        # Check if this escape leads away from agents (more lenient)
                        escape_dist1 = np.sqrt(
                            (escape_x - self.agent_positions[0][0]) ** 2
                            + (escape_y - self.agent_positions[0][1]) ** 2
                        )
                        escape_dist2 = np.sqrt(
                            (escape_x - self.agent_positions[1][0]) ** 2
                            + (escape_y - self.agent_positions[1][1]) ** 2
                        )

                        # More lenient escape criteria (1.3x instead of 2.0x)
                        if escape_dist1 > dist1 * 1.3 or escape_dist2 > dist2 * 1.3:
                            escape_routes += 1

                # Captured if limited escape routes (more lenient)
                if escape_routes <= 3:  # Increased from 2
                    return True

            # Additional teamwork capture: both agents very close
            if dist1 <= 1.8 and dist2 <= 1.8:  # New condition
                return True

        return False

    def _is_target_captured_strict(self):
        """Strict capture for final evaluation (like your pathfinding animation)"""
        if not self.agent1_deployed:
            return False

        # Calculate distances
        dist1 = np.sqrt(
            (self.agent_positions[0][0] - self.target.position[0]) ** 2
            + (self.agent_positions[0][1] - self.target.position[1]) ** 2
        )

        # Single agent capture (close proximity)
        if dist1 <= 1.0:
            return True

        # Two-agent coordination capture
        if self.agent2_deployed:
            dist2 = np.sqrt(
                (self.agent_positions[1][0] - self.target.position[0]) ** 2
                + (self.agent_positions[1][1] - self.target.position[1]) ** 2
            )

            # Both agents reasonably close
            if dist1 <= 2.0 and dist2 <= 2.0:
                # Count available escape routes
                escape_routes = 0

                # Test 32 directions
                for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                    dx = np.cos(angle) * 3.0
                    dy = np.sin(angle) * 3.0

                    escape_x = self.target.position[0] + dx
                    escape_y = self.target.position[1] + dy

                    if self.env.is_valid_position(escape_x, escape_y):
                        # Check if this escape leads away from agents
                        escape_dist1 = np.sqrt(
                            (escape_x - self.agent_positions[0][0]) ** 2
                            + (escape_y - self.agent_positions[0][1]) ** 2
                        )
                        escape_dist2 = np.sqrt(
                            (escape_x - self.agent_positions[1][0]) ** 2
                            + (escape_y - self.agent_positions[1][1]) ** 2
                        )

                        if escape_dist1 > dist1 * 2.0 or escape_dist2 > dist2 * 2.0:
                            escape_routes += 1

                # Captured if limited escape routes
                if escape_routes <= 2:
                    return True

        return False

    def _check_done(self):
        """Check if episode should end"""
        # Episode ends on capture
        if self._is_target_captured():
            return True

        # Episode ends if too many steps
        if self.step_count >= self.max_steps:
            return True

        # Episode ends if target escapes to edge
        margin = 2.0
        if (
            self.target.position[0] < margin
            or self.target.position[0] > self.width - margin
            or self.target.position[1] < margin
            or self.target.position[1] > self.height - margin
        ):
            return True

        return False

    def _get_state(self):
        """Get current state representation"""
        # Global state for mixing network
        global_state = self._get_global_state()
        # Individual observations for agent networks
        individual_obs = self._get_individual_observations()

        return global_state, individual_obs

    def _get_global_state(self):
        """Get global state representation"""
        grid_size = 20
        grid = np.zeros((grid_size, grid_size))

        # Scale positions to grid
        scale_x = grid_size / self.width
        scale_y = grid_size / self.height

        # Mark obstacles
        for ox, oy, ow, oh in self.env.obstacles:
            start_x = int(ox * scale_x)
            start_y = int(oy * scale_y)
            end_x = min(grid_size, int((ox + ow) * scale_x))
            end_y = min(grid_size, int((oy + oh) * scale_y))

            for x in range(start_x, end_x):
                for y in range(start_y, end_y):
                    if 0 <= x < grid_size and 0 <= y < grid_size:
                        grid[y, x] = -1  # Obstacle marker

        # Mark agents
        for i, (ax, ay) in enumerate(self.agent_positions):
            grid_x = int(ax * scale_x)
            grid_y = int(ay * scale_y)
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                grid[grid_y, grid_x] = i + 1  # Agent markers

        # Mark target
        tx, ty = (
            int(self.target.position[0] * scale_x),
            int(self.target.position[1] * scale_y),
        )
        if 0 <= tx < grid_size and 0 <= ty < grid_size:
            grid[ty, tx] = 9  # Target marker

        # Additional global features
        features = []

        # Distances from agents to target
        for agent_pos in self.agent_positions:
            dist = np.sqrt(
                (agent_pos[0] - self.target.position[0]) ** 2
                + (agent_pos[1] - self.target.position[1]) ** 2
            )
            features.append(dist / max(self.width, self.height))  # Normalized

        # Distance between agents
        agent_dist = np.sqrt(
            (self.agent_positions[0][0] - self.agent_positions[1][0]) ** 2
            + (self.agent_positions[0][1] - self.agent_positions[1][1]) ** 2
        )
        features.append(agent_dist / max(self.width, self.height))  # Normalized

        return np.concatenate([grid.flatten(), features])

    def _get_individual_observations(self):
        """Get individual agent observations"""
        observations = []

        for i, agent_pos in enumerate(self.agent_positions):
            obs = []

            # Agent's own position (normalized)
            obs.extend([agent_pos[0] / self.width, agent_pos[1] / self.height])

            # Target position relative to agent
            rel_target_x = (self.target.position[0] - agent_pos[0]) / self.width
            rel_target_y = (self.target.position[1] - agent_pos[1]) / self.height
            obs.extend([rel_target_x, rel_target_y])

            # Other agent's position relative to this agent
            other_agent_idx = 1 - i
            other_pos = self.agent_positions[other_agent_idx]
            rel_other_x = (other_pos[0] - agent_pos[0]) / self.width
            rel_other_y = (other_pos[1] - agent_pos[1]) / self.height
            obs.extend([rel_other_x, rel_other_y])

            # Distance to target
            target_dist = np.sqrt(
                (agent_pos[0] - self.target.position[0]) ** 2
                + (agent_pos[1] - self.target.position[1]) ** 2
            )
            obs.append(target_dist / max(self.width, self.height))

            # Local obstacle information (simplified)
            local_obstacles = self._get_local_obstacles(agent_pos, radius=3.0)
            obs.extend(local_obstacles)

            observations.append(np.array(obs))

        return observations

    def _get_local_obstacles(self, agent_pos, radius=3.0):
        """Get local obstacle information around agent"""
        # Check 8 directions around agent
        directions = [
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
            (-1, 0),
            (-1, 1),
        ]

        obstacle_info = []
        for dx, dy in directions:
            # Check if there's an obstacle in this direction
            check_x = agent_pos[0] + dx * radius
            check_y = agent_pos[1] + dy * radius

            if self.env.is_valid_position(check_x, check_y):
                obstacle_info.append(0.0)  # No obstacle
            else:
                obstacle_info.append(1.0)  # Obstacle present

        return obstacle_info

    def _apply_action(self, agent_idx, action):
        """Apply action to specific agent"""
        if agent_idx >= len(self.agent_positions):
            return

        current_pos = self.agent_positions[agent_idx]

        # Action mapping: 0=N, 1=NE, 2=E, 3=SE, 4=S/Stay, 5=SW, 6=W, 7=NW
        action_to_delta = {
            0: (0, 1),  # North
            1: (1, 1),  # Northeast
            2: (1, 0),  # East
            3: (1, -1),  # Southeast
            4: (0, 0),  # Stay/South
            5: (-1, -1),  # Southwest
            6: (-1, 0),  # West
            7: (-1, 1),  # Northwest
        }

        if action in action_to_delta:
            dx, dy = action_to_delta[action]

            # Calculate new position
            new_x = current_pos[0] + dx * 0.5  # Movement speed
            new_y = current_pos[1] + dy * 0.5

            # Check bounds and validity
            if (
                0 <= new_x <= self.width
                and 0 <= new_y <= self.height
                and self.env.is_valid_position(new_x, new_y)
            ):
                self.agent_positions[agent_idx] = (new_x, new_y)

    def _apply_actions(self, actions):
        """Apply actions to both agents (existing method)"""
        for i, action in enumerate(actions):
            self._apply_action(i, action)


def create_test_environments():
    """Create varied urban test environments of different sizes"""
    configurations = [
        (15, 15),  # Small urban area
        (25, 25),  # Medium urban area
        (35, 35),  # Large urban area
        (45, 45),  # Very large urban area
        (20, 30),  # Rectangular urban area
    ]

    environments = []
    for width, height in configurations:
        env = PursuitEnvironment(width, height)
        environments.append(env)
        print(f"Created urban environment: {width}x{height}")

    return environments
