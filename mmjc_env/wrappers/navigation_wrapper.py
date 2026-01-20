"""Navigation wrapper for MMJCENV.

This wrapper transforms the MMJCENV observations and rewards for navigation tasks:
- Observations: heading (12-dim one-hot), goal position (one-hot x,y),
  current position (one-hot x,y), and proprioception (all sensors)
- Actions: motor control (passthrough from base env)
- Goal: randomly generated position avoiding walls
- Reward: configurable (progress, sparse, exponential, etc.)

Reward types:
- "progress": Distance reduction (pure progress, no bonus)
- "progress_bonus": Distance reduction + goal bonus (+1.0 at goal)
- "sparse": 0 at goal, -1 elsewhere
- "neg_dist": Negative distance each step
- "exponential": exp(-0.5 * distance)
- "potential": Potential-based reward shaping
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2


class NavigationWrapper(gym.Wrapper):
    """Wrapper for MMJCENV that provides navigation-specific observations and rewards.

    Observation space:
        - heading: 12-dimensional one-hot encoding of heading direction
        - goal_pos_x: maze_size-dimensional one-hot encoding of goal x coordinate
        - goal_pos_y: maze_size-dimensional one-hot encoding of goal y coordinate
        - current_pos_x: maze_size-dimensional one-hot encoding of current x coordinate
        - current_pos_y: maze_size-dimensional one-hot encoding of current y coordinate
        - proprioception: all sensor information from base environment

    Reward types:
        - "progress": Distance reduction (pure progress, no bonus)
        - "progress_bonus": Distance reduction + goal bonus (+1.0 at goal)
        - "sparse": 0 at goal, -1 elsewhere
        - "neg_dist": Negative distance each step
        - "exponential": exp(-0.5 * distance)
        - "potential": Potential-based reward shaping
    """

    def __init__(self, env, heading_bins=12, reward_type="progress_bonus",
                 min_goal_distance=1.0, max_goal_distance=None,
                 forward_reward_scale=0.0):
        """Initialize the navigation wrapper.

        Args:
            env: The base MMJCENV environment
            heading_bins: Number of bins for heading discretization (default: 12)
            reward_type: Type of reward function. Options:
                - "progress": Distance reduction (default)
                - "progress_bonus": Distance reduction + goal bonus
                - "sparse": 0 at goal, -1 elsewhere
                - "neg_dist": Negative distance each step
                - "exponential": exp(-0.5 * distance)
                - "potential": Potential-based reward shaping
            min_goal_distance: Minimum distance from agent to goal (default: 1.0)
            max_goal_distance: Maximum distance from agent to goal (None = no limit)
            forward_reward_scale: Scale for forward velocity bonus (0 = disabled)
        """
        super().__init__(env)

        self.heading_bins = heading_bins
        self.reward_type = reward_type
        self.min_goal_distance = min_goal_distance
        self.max_goal_distance = max_goal_distance
        self.forward_reward_scale = forward_reward_scale
        self.maze_size = env.unwrapped.mm_env._task._maze_arena._maze.width - 2  # Exclude outer walls

        # Goal position (continuous, for reward calculation)
        self.goal_pos = None

        # Track previous distance for progress-based reward
        self._prev_distance = None

        # Track total reward for display
        self._total_reward = 0.0
        self._last_reward = 0.0

        # Get proprioception dimension from base environment's sensors
        self.proprioception_dim = env.observation_space["sensors"].shape[0]

        # Define new observation space keys
        obs_spaces = {
            "heading": spaces.Box(0, 1, shape=(heading_bins,), dtype=np.float32),
            "goal_pos_x": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "goal_pos_y": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "current_pos_x": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "current_pos_y": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "proprioception": spaces.Box(-np.inf, np.inf, shape=(self.proprioception_dim,), dtype=np.float32),
        }

        # Pass through image observation if available
        if "image" in env.observation_space.spaces:
            obs_spaces["image"] = env.observation_space["image"]

        self.observation_space = spaces.Dict(obs_spaces)

        # Action space is passed through from base env
        self.action_space = env.action_space

    def _get_heading_one_hot(self, agent_dir):
        """Convert agent direction to one-hot heading representation.

        Args:
            agent_dir: 2D direction vector [dx, dy]

        Returns:
            One-hot vector of size heading_bins
        """
        # Calculate angle from direction vector (range: -pi to pi)
        angle = np.arctan2(agent_dir[1], agent_dir[0])

        # Convert to range [0, 2*pi)
        if angle < 0:
            angle += 2 * np.pi

        # Discretize into bins
        bin_size = 2 * np.pi / self.heading_bins
        bin_idx = int(angle / bin_size) % self.heading_bins

        # Create one-hot vector
        one_hot = np.zeros(self.heading_bins, dtype=np.float32)
        one_hot[bin_idx] = 1.0

        return one_hot

    def _get_position_one_hot(self, pos, axis):
        """Convert continuous position to one-hot representation.

        Args:
            pos: Continuous position value (in grid coordinates)
            axis: 0 for x, 1 for y

        Returns:
            One-hot vector of size maze_size
        """
        # pos is in grid coordinates (0 to maze_size-1 range for inner maze)
        # Clamp to valid range
        idx = int(np.clip(pos, 0, self.maze_size - 1))

        one_hot = np.zeros(self.maze_size, dtype=np.float32)
        one_hot[idx] = 1.0

        return one_hot

    def _generate_random_goal(self, maze_map, agent_pos):
        """Generate a random goal position avoiding walls and within distance constraints.

        Args:
            maze_map: 2D character array of the maze
            agent_pos: Current agent position

        Returns:
            Tuple (x, y) of goal position in grid coordinates
        """
        # Find all floor cells (non-wall positions) within distance constraints
        valid_positions = []
        all_floor_positions = []
        height, width = maze_map.shape

        for i in range(height):
            for j in range(width):
                char = maze_map[i, j]
                # Floor cells are ' ' (empty), '.', 'P' (spawn), or 'G' (existing goals)
                # Note: maze_map can be string or int array depending on source
                if char in [' ', '.', 'P', 'G', ord(' '), ord('.'), ord('P'), ord('G')]:
                    # Convert to inner maze coordinates (subtract 1 for outer wall)
                    # i is row index (0 at top), but world y=0 is at bottom
                    inner_x = j - 1
                    inner_y = (height - 2) - i  # Flip y-axis: row 1 -> inner_y = maze_size-1
                    if 0 <= inner_x < self.maze_size and 0 <= inner_y < self.maze_size:
                        all_floor_positions.append((inner_x, inner_y))
                        # Check distance constraints
                        dist = np.sqrt((inner_x + 0.5 - agent_pos[0])**2 +
                                       (inner_y + 0.5 - agent_pos[1])**2)
                        if dist >= self.min_goal_distance:
                            if self.max_goal_distance is None or dist <= self.max_goal_distance:
                                valid_positions.append((inner_x, inner_y))

        # Use valid positions if available, otherwise fall back to all floor positions
        floor_positions = valid_positions if valid_positions else all_floor_positions

        if not floor_positions:
            # Final fallback: return center of maze
            return (self.maze_size // 2, self.maze_size // 2)

        # Randomly select a floor position
        idx = self.np_random.integers(len(floor_positions))
        goal_x, goal_y = floor_positions[idx]

        # Add 0.5 to center the goal in the cell
        return (float(goal_x) + 0.5, float(goal_y) + 0.5)

    def _transform_observation(self, obs, info):
        """Transform base observation to navigation observation.

        Args:
            obs: Observation from base environment
            info: Info dict from base environment

        Returns:
            Transformed observation dict
        """
        # Get agent direction and position from info
        agent_dir = info.get("agent_dir", np.array([1.0, 0.0]))
        agent_pos = info.get("agent_pos", np.array([0.0, 0.0]))

        # Create new observation
        new_obs = {
            "heading": self._get_heading_one_hot(agent_dir),
            "goal_pos_x": self._get_position_one_hot(self.goal_pos[0], 0),
            "goal_pos_y": self._get_position_one_hot(self.goal_pos[1], 1),
            "current_pos_x": self._get_position_one_hot(agent_pos[0], 0),
            "current_pos_y": self._get_position_one_hot(agent_pos[1], 1),
            "proprioception": obs["sensors"].astype(np.float32),
        }

        if "image" in obs:
            new_obs["image"] = obs["image"]

        return new_obs

    def _get_forward_velocity(self, obs, info):
        """Calculate forward velocity from velocimeter sensor.

        Forward velocity is the component of torso velocity in the agent's
        facing direction. Positive = moving forward, negative = moving backward.

        Note: The MuJoCo velocimeter sensor outputs velocity in the site's LOCAL
        frame, not world frame. For the ant agent, the forward direction corresponds
        to the local X-axis (since agent_dir is derived from absolute_orientation[:2, 0]).
        Therefore, forward velocity is simply velocimeter[0].

        Args:
            obs: Observation dict (not used, kept for API consistency)
            info: Info dict containing agent_dir and velocimeter data

        Returns:
            Forward velocity (scalar)
        """
        try:
            if "velocimeter" in info:
                velocimeter = info["velocimeter"]
                # velocimeter is [vx, vy, vz] in LOCAL site frame (not world frame!)
                # For ant: forward direction is local X-axis, so forward velocity = vx_local
                forward_vel = velocimeter[0]
                return float(forward_vel)
        except (AttributeError, KeyError, IndexError):
            pass
        return 0.0

    def _reward_progress(self, agent_pos, distance):
        """Pure progress reward: distance reduction."""
        reward = self._prev_distance - distance
        self._prev_distance = distance
        return reward

    def _reward_progress_bonus(self, agent_pos, distance):
        """Progress reward with goal bonus."""
        reward = self._prev_distance - distance
        self._prev_distance = distance
        goal_bonus = 1.0 if distance < 0.5 else 0.0
        return reward + goal_bonus

    def _reward_sparse(self, agent_pos, distance):
        """Sparse reward: 0 at goal, -1 elsewhere."""
        agent_grid_x = int(agent_pos[0])
        agent_grid_y = int(agent_pos[1])
        goal_grid_x = int(self.goal_pos[0])
        goal_grid_y = int(self.goal_pos[1])
        if agent_grid_x == goal_grid_x and agent_grid_y == goal_grid_y:
            return 0.0
        return -1.0

    def _reward_neg_dist(self, agent_pos, distance):
        """Negative distance reward: -d each step."""
        return -distance

    def _reward_exponential(self, agent_pos, distance):
        """Exponential reward: exp(-0.5 * distance), peaks at goal."""
        return np.exp(-0.5 * distance)

    def _reward_potential(self, agent_pos, distance):
        """Potential-based reward shaping: gamma * phi(s') - phi(s)."""
        gamma = 0.99
        reward = gamma * (-distance) - (-self._prev_distance)
        self._prev_distance = distance
        return reward

    def _calculate_reward(self, info, obs=None):
        """Calculate reward based on reward_type.

        Args:
            info: Info dict containing agent_pos
            obs: Observation dict (needed for forward velocity)

        Returns:
            Reward value based on selected reward type
        """
        agent_pos = info.get("agent_pos", np.array([0.0, 0.0]))
        distance = np.sqrt(
            (agent_pos[0] - self.goal_pos[0])**2 +
            (agent_pos[1] - self.goal_pos[1])**2
        )

        # Dispatch to appropriate reward function
        if self.reward_type == "progress":
            base_reward = self._reward_progress(agent_pos, distance)
        elif self.reward_type == "progress_bonus":
            base_reward = self._reward_progress_bonus(agent_pos, distance)
        elif self.reward_type == "sparse":
            base_reward = self._reward_sparse(agent_pos, distance)
        elif self.reward_type == "neg_dist":
            base_reward = self._reward_neg_dist(agent_pos, distance)
        elif self.reward_type == "exponential":
            base_reward = self._reward_exponential(agent_pos, distance)
        elif self.reward_type == "potential":
            base_reward = self._reward_potential(agent_pos, distance)
        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # Add forward velocity bonus if enabled AND making progress toward goal
        # This prevents rewarding the agent for just running in circles
        if self.forward_reward_scale > 0 and obs is not None and base_reward > 0:
            forward_vel = self._get_forward_velocity(obs, info)
            forward_reward = self.forward_reward_scale * max(0, forward_vel)
            self.forward_reward = forward_reward
            return base_reward + forward_reward

        return base_reward

    def reset(self, seed=None, options=None):
        """Reset the environment and generate a new random goal.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        base_env = self.env.unwrapped

        # Temporarily disable base env rendering
        original_render_mode = base_env.render_mode
        base_env.render_mode = None

        # Reset base environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Restore render mode
        base_env.render_mode = original_render_mode

        # Seed our random number generator
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        elif not hasattr(self, 'np_random'):
            self.np_random = np.random.default_rng()

        # Get agent position for goal generation and distance calculation
        agent_pos = info.get("agent_pos", np.array([0.0, 0.0]))

        # Generate random goal position using maze map, avoiding agent location
        maze_map = info.get("maze_map")
        if maze_map is not None:
            self.goal_pos = self._generate_random_goal(maze_map, agent_pos)
        else:
            # Fallback: center of maze
            self.goal_pos = (self.maze_size / 2, self.maze_size / 2)

        # Initialize previous distance for progress-based reward
        self._prev_distance = np.sqrt(
            (agent_pos[0] - self.goal_pos[0])**2 +
            (agent_pos[1] - self.goal_pos[1])**2
        )

        # Transform observation
        new_obs = self._transform_observation(obs, info)

        # Add goal position to info
        info["goal_pos"] = self.goal_pos

        # Reset reward tracking
        self._total_reward = 0.0
        self._last_reward = 0.0
        self.forward_reward = 0.0

        # Render with our custom goal visualization
        if original_render_mode == "human":
            self._render_with_goal()

        return new_obs, info

    def step(self, action):
        """Take a step in the environment.

        Args:
            action: Motor control action

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        base_env = self.env.unwrapped

        # Temporarily disable base env rendering
        original_render_mode = base_env.render_mode
        base_env.render_mode = None

        # Step base environment
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Restore render mode
        base_env.render_mode = original_render_mode

        # Calculate navigation reward
        reward = self._calculate_reward(info, obs)

        # Check if goal reached and update if necessary
        agent_pos = info.get("agent_pos", np.array([0.0, 0.0]))
        distance = np.sqrt(
            (agent_pos[0] - self.goal_pos[0])**2 +
            (agent_pos[1] - self.goal_pos[1])**2
        )

        if distance < 0.5:
            # Generate new goal
            maze_map = info.get("maze_map")
            if maze_map is not None:
                self.goal_pos = self._generate_random_goal(maze_map, agent_pos)
            else:
                self.goal_pos = (self.maze_size / 2, self.maze_size / 2)

            # Reset previous distance for the new goal
            self._prev_distance = np.sqrt(
                (agent_pos[0] - self.goal_pos[0])**2 +
                (agent_pos[1] - self.goal_pos[1])**2
            )

        # Transform observation (using updated goal)
        new_obs = self._transform_observation(obs, info)

        # Track reward for display
        self._last_reward = reward
        self._total_reward += reward

        # Add goal position to info
        info["goal_pos"] = self.goal_pos
        info["base_reward"] = base_reward

        # Render with our custom goal visualization
        if original_render_mode == "human":
            self._render_with_goal()

        return new_obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment with navigation goal visualization."""
        base_env = self.env.unwrapped

        if base_env.render_mode == "human":
            self._render_with_goal()
        elif base_env.render_mode == "rgb_array":
            return self._render_with_goal()

    def _render_with_goal(self):
        """Render frame with navigation goal marker and text."""
        import pygame

        base_env = self.env.unwrapped

        # Initialize pygame window if needed
        if base_env.window is None and base_env.render_mode == "human":
            pygame.init()
            pygame.display.init()
            base_env.window = pygame.display.set_mode(
                (base_env.window_size * 3, base_env.window_size)
            )
            pygame.display.set_caption(base_env.model_name)
        if base_env.clock is None and base_env.render_mode == "human":
            base_env.clock = pygame.time.Clock()

        if base_env.render_mode == "human":
            # Create maze map visualization with navigation goal
            maze_map_vis = self._create_maze_map_with_goal()

            # Horizontally stack: ego-centric camera, top down view, integrated maze map
            if base_env.top_camera:
                combined_image = np.concatenate(
                    [base_env._ego_centric_camera, base_env._top_camera, maze_map_vis],
                    axis=1,
                )
            else:
                combined_image = np.concatenate(
                    [base_env._ego_centric_camera, maze_map_vis, maze_map_vis], axis=1
                )
            combined_surface = pygame.surfarray.make_surface(
                combined_image.transpose(1, 0, 2)
            )

            # Scale the surface to the window size
            scaled_surface = pygame.transform.scale(
                combined_surface, base_env.window.get_size()
            )

            # Blit the scaled surface onto the display window
            base_env.window.blit(scaled_surface, (0, 0))

            # Write text info on the window
            font = pygame.font.SysFont("Arial", 24)

            # Display "Nav Goal" text with position
            goal_text = font.render(
                f"Nav Goal: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f})",
                True, (0, 0, 0)
            )
            goal_rect = goal_text.get_rect(topleft=(10, 10))
            base_env.window.blit(goal_text, goal_rect)

            # Display distance to goal
            agent_pos = getattr(base_env, '_agent_pos', np.array([0.0, 0.0]))
            distance = np.sqrt((agent_pos[0] - self.goal_pos[0])**2 + (agent_pos[1] - self.goal_pos[1])**2)
            dist_text = font.render(
                f"Dist: {distance:.2f}", True, (0, 0, 0)
            )
            dist_rect = dist_text.get_rect(topleft=(goal_rect.right + 20, 10))
            base_env.window.blit(dist_text, dist_rect)

            # Display return (cumulative reward for the episode)
            reward_text = font.render(
                f"Return: {self._total_reward:.2f}", True, (0, 0, 0)
            )
            reward_rect = reward_text.get_rect(topleft=(dist_rect.right + 20, 10))
            base_env.window.blit(reward_text, reward_rect)

            # Display time step
            time_text = font.render(
                f"Step: {base_env._time_step//40}/{base_env.time_limit}", True, (0, 0, 0)
            )
            time_rect = time_text.get_rect(topleft=(reward_rect.right + 20, 10))
            base_env.window.blit(time_text, time_rect)

            # Display forward reward 
            forward_reward_text = font.render(
                f"Forward Reward: {self.forward_reward}", True, (0, 0, 0)
            )
            forward_reward_text_rect = forward_reward_text.get_rect(topleft=(time_rect.right + 20, 10))
            base_env.window.blit(forward_reward_text, forward_reward_text_rect)

            # Display coverage over the third panel
            coverage_text = font.render(
                f"Coverage: {base_env._calculate_coverage():.1f}%",
                True, (0, 0, 0),
            )
            coverage_x = 2 * base_env.window_size + 10
            coverage_rect = coverage_text.get_rect(topleft=(coverage_x, 10))
            base_env.window.blit(coverage_text, coverage_rect)

            # Update the display
            pygame.event.pump()
            pygame.display.flip()

        elif base_env.render_mode == "rgb_array":
            maze_map_vis = self._create_maze_map_with_goal()
            if base_env.top_camera:
                return np.concatenate(
                    [base_env._ego_centric_camera, base_env._top_camera, maze_map_vis],
                    axis=1,
                )
            else:
                return np.concatenate(
                    [base_env._ego_centric_camera, maze_map_vis, maze_map_vis], axis=1
                )

    def _create_maze_map_with_goal(self):
        """Create maze map visualization with navigation goal marker.

        Returns:
            np.ndarray: RGB image of the maze map with goal marker.
        """
        base_env = self.env.unwrapped

        # Get base visualization
        rgb_image = base_env._create_maze_map_visualization()

        # Draw navigation goal as a green square marker
        if self.goal_pos is not None:
            maze_map = base_env._get_maze_map()
            height, width = maze_map.shape
            scale_factor = base_env.camera_resolution / height

            # Convert goal position to image coordinates
            # goal_pos is in inner maze coordinates, add 1 for outer wall
            goal_img_x = int((self.goal_pos[0] + 1) * scale_factor)
            goal_img_y = int((height - 1 - self.goal_pos[1]) * scale_factor)

            # Draw a green square marker for the goal
            marker_size = max(4, int(scale_factor * 0.5))
            cv2.rectangle(
                rgb_image,
                (goal_img_x - marker_size, goal_img_y - marker_size),
                (goal_img_x + marker_size, goal_img_y + marker_size),
                (0, 255, 0),  # Green color
                2  # Line thickness
            )

        return rgb_image
