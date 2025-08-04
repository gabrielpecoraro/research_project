import numpy as np
import random
from pathfinding_marl import Environment


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
        """Reset environment for hybrid training"""
        print(f"\n🔄 ENVIRONMENT RESET")

        # Reset positions
        self.agent_positions = [(0.5, 0.5), (0.5, 0.5)]

        # Reset deployment status
        self.agent1_deployed = False
        self.agent2_deployed = False
        self.pursuit_active = False

        # Reset timing
        self.agent1_deploy_step = -1
        self.agent2_deploy_step = -1

        # Create new target
        target_pos = self._get_random_valid_target_position()

        # Create target with proper fallback
        try:
            from pathfinding_marl import VehicleTarget

            self.target = VehicleTarget(self.env, target_pos, speed=0.5, inertia=0.8)
        except ImportError:
            try:
                from pathfinding import VehicleTarget

                self.target = VehicleTarget(
                    self.env, target_pos, speed=0.5, inertia=0.8
                )
            except ImportError:
                # Fallback: create a simple target
                class SimpleTarget:
                    def __init__(self, env, position, speed=0.5):
                        self.env = env
                        self.position = position
                        self.speed = speed

                    def update_position(self, agent_position):
                        """Simple fleeing behavior"""
                        current_pos = np.array(self.position)
                        agent_pos = np.array(agent_position)

                        # Move away from agent
                        direction = current_pos - agent_pos
                        distance = np.linalg.norm(direction)

                        if distance > 0:
                            direction = direction / distance
                            new_pos = current_pos + direction * self.speed

                            # Keep within bounds
                            new_pos[0] = np.clip(new_pos[0], 1.0, self.env.width - 1.0)
                            new_pos[1] = np.clip(new_pos[1], 1.0, self.env.height - 1.0)

                            # Check if position is valid
                            if self.env.is_valid_position(new_pos[0], new_pos[1]):
                                self.position = tuple(new_pos)

                self.target = SimpleTarget(self.env, target_pos, speed=0.5)

        self.target_history = [self.target.position]
        self.step_count = 0

        print(f"🎯 Target at: {self.target.position}")
        print(f"🤖 Agents start at: {self.agent_positions}")
        print(f"📋 Hybrid training ready:")
        print(f"   Steps 1-20: Agent 1 uses A* pathfinding")
        print(f"   Steps 21+: Agent 1 (A*) + Agent 2 (MARL coordination)")

        return self._get_global_state(), self._get_individual_observations()

    def step(self, actions):
        """HYBRID APPROACH: Agent 1 uses A*, Agent 2 learns coordination"""
        self.step_count += 1

        # Phase 1: Agent 1 pathfinding (Steps 1-20) - Agent 1 uses A* pathfinding
        if self.step_count <= 20:
            # Deploy Agent 1 immediately on first step
            if not self.agent1_deployed:
                self.agent1_deployed = True
                self.agent1_deploy_step = self.step_count
                self.pursuit_active = True
                print(
                    f"🚀 Agent 1 deployed with A* pathfinding at step {self.step_count}"
                )

            # Agent 1 uses deterministic A* pathfinding (NO LEARNING)
            self._move_agent1_with_astar()

            # Target flees from Agent 1 - FIX: Use update_position instead of move
            if len(self.agent_positions) > 0:
                self.target.update_position(self.agent_positions[0])

            training_phase = False
            reward = 0.0

        # Phase 2: Hybrid coordination (Steps 21+) - Agent 1 continues A*, Agent 2 learns
        else:
            # Deploy Agent 2 for learned interception
            if not self.agent2_deployed:
                self.agent2_deployed = True
                self.agent2_deploy_step = self.step_count
                print(
                    f"🤖 Agent 2 deployed for LEARNED interception at step {self.step_count}"
                )
                print(f"🎮 HYBRID COORDINATION: Agent 1 (A*) + Agent 2 (MARL)")

            # Agent 1 continues using deterministic A* pathfinding
            self._move_agent1_with_astar()

            # Agent 2 uses LEARNED action from QMIX (ONLY Agent 2 trains)
            if len(actions) > 1:
                agent2_action = actions[1]
                self._apply_action_with_smoothing(1, agent2_action)

            # Target flees from both agents - FIX: Use update_position with closest agent
            if len(self.agent_positions) >= 2:
                # Find closest agent to target for fleeing behavior
                target_pos = np.array(self.target.position)
                agent1_pos = np.array(self.agent_positions[0])
                agent2_pos = np.array(self.agent_positions[1])

                dist1 = np.linalg.norm(agent1_pos - target_pos)
                dist2 = np.linalg.norm(agent2_pos - target_pos)

                # Target flees from closest agent
                closest_agent_pos = (
                    self.agent_positions[0]
                    if dist1 < dist2
                    else self.agent_positions[1]
                )
                self.target.update_position(closest_agent_pos)
            elif len(self.agent_positions) >= 1:
                self.target.update_position(self.agent_positions[0])

            # This is the training phase - but ONLY for Agent 2
            training_phase = True
            reward = self._calculate_interception_reward()

        # Check if done
        done = self._check_done()

        # Track target history
        self.target_history.append(self.target.position)

        # Create info dict with hybrid training information
        info = {
            "captured": self._is_target_captured(),
            "agent1_deployed": self.agent1_deployed,
            "agent2_deployed": self.agent2_deployed,
            "pursuit_active": self.pursuit_active,
            "agent1_deploy_step": self.agent1_deploy_step,
            "agent2_deploy_step": self.agent2_deploy_step,
            "step_count": self.step_count,
            "training_phase": training_phase,
            "agent1_learning": False,  # Agent 1 never learns (uses A*)
            "agent2_learning": training_phase,  # Only Agent 2 learns
            "phase": self._get_current_phase(),
            "coordination_difficulty": self._get_coordination_difficulty(),
        }

        return (
            (self._get_global_state(), self._get_individual_observations()),
            reward,
            done,
            info,
        )

    def _move_agent1_with_astar(self):
        """Agent 1 uses deterministic A* pathfinding (NO LEARNING)"""
        try:
            from pathfinding_marl import AStar

            # Create pathfinder
            pathfinder = AStar(self.env)

            # Find path from current position to target
            start = self.agent_positions[0]
            goal = self.target.position

            path = pathfinder.find_path(start, goal)

            if path and len(path) > 1:
                # Move to next position in path
                next_pos = path[1]

                # Validate and apply movement with Bezier smoothing
                if (
                    0 <= next_pos[0] <= self.width
                    and 0 <= next_pos[1] <= self.height
                    and self.env.is_valid_position(next_pos[0], next_pos[1])
                ):
                    self._smooth_agent_movement(0, next_pos)
                else:
                    self._fallback_agent_movement(0)
            else:
                self._fallback_agent_movement(0)

        except Exception as e:
            print(f"A* pathfinding failed: {e}, using fallback")
            self._fallback_agent_movement(0)

    def _smooth_agent_movement(self, agent_idx, target_pos):
        """Apply Bezier smoothing to agent movement"""
        try:
            from pathfinding_marl import smooth_path_with_bezier

            current_pos = self.agent_positions[agent_idx]

            # Create mini-path for smoothing
            mini_path = [current_pos, target_pos]

            # Apply Bezier smoothing
            smoothed_path = smooth_path_with_bezier(
                mini_path, smoothing_factor=0.1, num_points=5
            )

            if len(smoothed_path) > 1:
                # Move to next position in smoothed path
                next_pos = smoothed_path[1]
                self.agent_positions[agent_idx] = next_pos
            else:
                # Fallback to direct movement
                self.agent_positions[agent_idx] = target_pos

        except Exception as e:
            print(f"Bezier smoothing failed: {e}, using direct movement")
            self.agent_positions[agent_idx] = target_pos

    def _fallback_agent_movement(self, agent_idx):
        """Fallback movement when A* fails"""
        current_pos = np.array(self.agent_positions[agent_idx])
        target_pos = np.array(self.target.position)

        # Move directly towards target
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance > 0:
            # Normalize and apply speed
            direction = direction / distance
            speed = 0.8  # Agent 1 speed
            new_pos = current_pos + direction * speed

            # Validate new position
            if (
                0 <= new_pos[0] <= self.width
                and 0 <= new_pos[1] <= self.height
                and self.env.is_valid_position(new_pos[0], new_pos[1])
            ):
                # Apply smoothing even for fallback
                self._smooth_agent_movement(agent_idx, tuple(new_pos))

    def _apply_action_with_smoothing(self, agent_idx, action):
        """Apply MARL action with Bezier smoothing (for Agent 2)"""
        current_pos = self.agent_positions[agent_idx]

        # Action mapping
        action_to_delta = {
            0: (0, 1),  # N
            1: (1, 1),  # NE
            2: (1, 0),  # E
            3: (1, -1),  # SE
            4: (0, 0),  # Stay
            5: (-1, -1),  # SW
            6: (-1, 0),  # W
            7: (-1, 1),  # NW
        }

        if action in action_to_delta:
            dx, dy = action_to_delta[action]

            # Agent 2 is faster for interception
            speed = 1.2 if agent_idx == 1 else 0.8

            new_x = current_pos[0] + dx * speed
            new_y = current_pos[1] + dy * speed

            # Validate position
            if (
                0 <= new_x <= self.width
                and 0 <= new_y <= self.height
                and self.env.is_valid_position(new_x, new_y)
            ):
                # Apply Bezier smoothing for realistic movement
                self._smooth_agent_movement(agent_idx, (new_x, new_y))
            else:
                print(f"Agent {agent_idx + 1} movement blocked")

    def _calculate_interception_reward(self):
        """Reward specifically for Agent 2's interception learning"""
        reward = 0.0

        # Base reward for target capture
        if self._is_target_captured():
            reward += 20.0
            return reward

        # Coordination rewards between A* Agent 1 and learning Agent 2
        target_pos = np.array(self.target.position)
        agent1_pos = np.array(self.agent_positions[0])
        agent2_pos = np.array(self.agent_positions[1])

        # Distance rewards (closer is better)
        dist1 = np.linalg.norm(agent1_pos - target_pos)
        dist2 = np.linalg.norm(agent2_pos - target_pos)

        # Reward Agent 2 for getting closer to target
        reward += max(0, 5.0 - dist2) * 0.2

        # Coordination bonus: Agent 2 learns to work with deterministic Agent 1
        # Reward pincer movement (agents on opposite sides)
        agent1_to_target = target_pos - agent1_pos
        agent2_to_target = target_pos - agent2_pos

        if (
            np.linalg.norm(agent1_to_target) > 0
            and np.linalg.norm(agent2_to_target) > 0
        ):
            agent1_norm = agent1_to_target / np.linalg.norm(agent1_to_target)
            agent2_norm = agent2_to_target / np.linalg.norm(agent2_to_target)

            dot_product = np.dot(agent1_norm, agent2_norm)
            if dot_product < 0:  # Opposite sides
                reward += 1.0

        # Reward Agent 2 for staying at optimal interception distance from Agent 1
        agent_distance = np.linalg.norm(agent1_pos - agent2_pos)
        optimal_distance = 4.0  # Ideal coordination distance
        distance_bonus = max(0, 3.0 - abs(agent_distance - optimal_distance))
        reward += distance_bonus * 0.3

        # Penalty for being too far from action
        if dist2 > 8.0:
            reward -= 0.2

        # Small step penalty to encourage efficiency
        reward -= 0.02

        return reward

    def _get_coordination_difficulty(self):
        """Calculate how difficult the current coordination scenario is"""
        if not self.agent2_deployed:
            return 0.0

        target_pos = np.array(self.target.position)
        agent1_pos = np.array(self.agent_positions[0])
        agent2_pos = np.array(self.agent_positions[1])

        # Distance factors
        dist1 = np.linalg.norm(agent1_pos - target_pos)
        dist2 = np.linalg.norm(agent2_pos - target_pos)

        # Environment complexity (number of nearby obstacles)
        obstacle_density = len(
            [
                obs
                for obs in self.env.obstacles
                if self._obstacle_near_point(target_pos, obs, radius=5.0)
            ]
        )

        # Target escape options
        escape_routes = self._count_target_escape_routes()

        # Normalize and combine factors
        difficulty = (
            min(dist1, dist2) / max(self.width, self.height) * 0.3  # Closer = harder
            + obstacle_density
            / max(1, len(self.env.obstacles))
            * 0.4  # More obstacles = harder
            + escape_routes / 8.0 * 0.3  # More escapes = harder
        )

        return min(1.0, difficulty)

    def _obstacle_near_point(self, point, obstacle, radius):
        """Check if obstacle is within radius of point"""
        ox, oy, ow, oh = obstacle
        # Distance from point to closest edge of obstacle
        closest_x = max(ox, min(point[0], ox + ow))
        closest_y = max(oy, min(point[1], oy + oh))

        distance = np.linalg.norm([point[0] - closest_x, point[1] - closest_y])
        return distance <= radius

    def _get_current_phase(self):
        """Get current phase name"""
        if self.step_count <= 20:
            return "pathfinding"  # Agent 1 A* pathfinding
        else:
            return "hybrid_coordination"  # Agent 1 (A*) + Agent 2 (MARL)

    def _get_global_state(self):
        """Get global state for QMIX mixing network"""
        # Combine all agent positions, target position, and environment info
        global_state = []

        # Agent positions (flattened)
        for pos in self.agent_positions:
            global_state.extend([pos[0], pos[1]])

        # Target position
        global_state.extend([self.target.position[0], self.target.position[1]])

        # Environment context
        global_state.extend(
            [
                self.width / 20.0,  # Normalized environment width
                self.height / 20.0,  # Normalized environment height
                len(self.env.obstacles) / 10.0,  # Normalized obstacle count
                self.step_count / 200.0,  # Normalized step count
            ]
        )

        # Deployment status
        global_state.extend(
            [
                1.0 if self.agent1_deployed else 0.0,
                1.0 if self.agent2_deployed else 0.0,
                1.0 if self.pursuit_active else 0.0,
            ]
        )

        return np.array(global_state, dtype=np.float32)

    def _get_individual_observations(self):
        """Get individual observations for each agent"""
        observations = []

        for i, agent_pos in enumerate(self.agent_positions):
            obs = []

            # Agent's own position
            obs.extend([agent_pos[0], agent_pos[1]])

            # Target position (relative to agent)
            target_pos = self.target.position
            obs.extend(
                [
                    target_pos[0] - agent_pos[0],  # Relative x
                    target_pos[1] - agent_pos[1],  # Relative y
                ]
            )

            # Other agent position (relative)
            other_agent_idx = 1 - i  # 0 -> 1, 1 -> 0
            if other_agent_idx < len(self.agent_positions):
                other_pos = self.agent_positions[other_agent_idx]
                obs.extend(
                    [
                        other_pos[0] - agent_pos[0],  # Relative x
                        other_pos[1] - agent_pos[1],  # Relative y
                    ]
                )
            else:
                obs.extend([0.0, 0.0])  # No other agent

            # Distance to target
            target_distance = np.linalg.norm(np.array(target_pos) - np.array(agent_pos))
            obs.append(target_distance)

            # Distance to other agent
            if other_agent_idx < len(self.agent_positions):
                other_distance = np.linalg.norm(
                    np.array(self.agent_positions[other_agent_idx])
                    - np.array(agent_pos)
                )
            else:
                other_distance = 0.0
            obs.append(other_distance)

            # Environment context
            obs.extend(
                [
                    self.width / 20.0,  # Normalized environment width
                    self.height / 20.0,  # Normalized environment height
                    len(self.env.obstacles) / 10.0,  # Normalized obstacle count
                ]
            )

            # Agent-specific deployment status
            obs.extend(
                [
                    1.0
                    if (i == 0 and self.agent1_deployed)
                    else 0.0,  # This agent deployed
                    1.0
                    if (i == 1 and self.agent2_deployed)
                    else 0.0,  # Other agent deployed
                    1.0 if self.pursuit_active else 0.0,  # Pursuit active
                ]
            )

            # Phase information
            if self.step_count <= 20:
                phase_encoding = [1.0, 0.0]  # Pathfinding phase
            else:
                phase_encoding = [0.0, 1.0]  # Coordination phase
            obs.extend(phase_encoding)

            # Nearby obstacles (simplified - count within radius)
            obstacle_count = 0
            for ox, oy, ow, oh in self.env.obstacles:
                obstacle_center = (ox + ow / 2, oy + oh / 2)
                distance_to_obstacle = np.linalg.norm(
                    np.array(obstacle_center) - np.array(agent_pos)
                )
                if distance_to_obstacle < 5.0:  # Within 5 units
                    obstacle_count += 1

            obs.append(obstacle_count / 5.0)  # Normalized

            # Step count (normalized)
            obs.append(self.step_count / 200.0)

            observations.append(np.array(obs, dtype=np.float32))

        return observations

    def _get_random_valid_target_position(self):
        """Get a random valid position for the target"""
        max_attempts = 50
        for _ in range(max_attempts):
            x = random.uniform(2.0, self.width - 2.0)
            y = random.uniform(2.0, self.height - 2.0)

            # Check if position is valid and not too close to agents
            if self.env.is_valid_position(x, y):
                # Ensure target is not too close to starting agent positions
                min_distance_to_agents = min(
                    [
                        np.linalg.norm(np.array([x, y]) - np.array(agent_pos))
                        for agent_pos in self.agent_positions
                    ]
                )

                if min_distance_to_agents > 3.0:  # Minimum distance from agents
                    return (x, y)

        # Fallback position if no valid position found
        return (self.width / 2.0, self.height / 2.0)

    def _is_target_captured(self):
        """Simple capture detection"""
        if not (self.agent1_deployed and self.agent2_deployed):
            return False

        target_pos = np.array(self.target.position)
        agent1_pos = np.array(self.agent_positions[0])
        agent2_pos = np.array(self.agent_positions[1])

        # Calculate distances
        dist1 = np.linalg.norm(agent1_pos - target_pos)
        dist2 = np.linalg.norm(agent2_pos - target_pos)

        # Simple proximity capture
        capture_radius = 2.5
        if dist1 <= capture_radius and dist2 <= capture_radius:
            print(f"🏆 TARGET CAPTURED!")
            print(f"   Agent 1 distance: {dist1:.2f}")
            print(f"   Agent 2 distance: {dist2:.2f}")
            return True

        return False

    def _check_done(self):
        """Check if episode is done"""
        # Done if captured or max steps reached
        return self._is_target_captured() or self.step_count >= self.max_steps

    def _count_target_escape_routes(self):
        """Simple escape route counting"""
        if not hasattr(self, "target") or self.target is None:
            return 4

        try:
            target_pos = self.target.position
            escape_count = 0

            # Check 4 main directions
            directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]  # N, E, S, W

            for dx, dy in directions:
                check_x = target_pos[0] + dx
                check_y = target_pos[1] + dy

                # Check if this escape route is clear
                if (
                    0 <= check_x <= self.width
                    and 0 <= check_y <= self.height
                    and self.env.is_valid_position(check_x, check_y)
                ):
                    escape_count += 1

            return escape_count

        except Exception as e:
            return 4  # Default


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
