"""Navigation wrapper for MMJCENV.

This wrapper transforms the MMJCENV observations and rewards for navigation tasks:
- Observations: heading (12-dim one-hot), goal position (one-hot x,y),
  current position (one-hot x,y), and proprioception (all sensors)
- Actions: motor control (passthrough from base env)
- Goal: randomly generated position avoiding walls
- Reward: dense (negative MSE distance) or sparse (0 at goal, -1 elsewhere)
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

    Reward:
        - Dense: Negative MSE distance between current position and goal position
        - Sparse: 0 if agent is at goal grid location, -1 otherwise
    """

    def __init__(self, env, heading_bins=12, dense_reward=True):
        """Initialize the navigation wrapper.

        Args:
            env: The base MMJCENV environment
            heading_bins: Number of bins for heading discretization (default: 12)
            dense_reward: If True, use negative MSE distance as reward.
                         If False, use sparse reward (0 at goal, -1 elsewhere).
        """
        super().__init__(env)

        self.heading_bins = heading_bins
        self.dense_reward = dense_reward
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

        # Define new observation space
        self.observation_space = spaces.Dict({
            "heading": spaces.Box(0, 1, shape=(heading_bins,), dtype=np.float32),
            "goal_pos_x": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "goal_pos_y": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "current_pos_x": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "current_pos_y": spaces.Box(0, 1, shape=(self.maze_size,), dtype=np.float32),
            "proprioception": spaces.Box(-np.inf, np.inf, shape=(self.proprioception_dim,), dtype=np.float32),
        })

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

    def _generate_random_goal(self, maze_map, agent_pos, min_distance=2.0):
        """Generate a random goal position avoiding walls and agent location.

        Args:
            maze_map: 2D character array of the maze
            agent_pos: Current agent position to avoid
            min_distance: Minimum distance from agent (default: 2.0 grid units)

        Returns:
            Tuple (x, y) of goal position in grid coordinates
        """
        # Find all floor cells (non-wall positions) that are far enough from agent
        floor_positions = []
        height, width = maze_map.shape

        for i in range(height):
            for j in range(width):
                char = maze_map[i, j]
                # Floor cells are '.', 'P' (spawn), or 'G' (existing goals)
                if char in ['.', 'P', 'G', ord('.'), ord('P'), ord('G')]:
                    # Convert to inner maze coordinates (subtract 1 for outer wall)
                    # i is row index (0 at top), but world y=0 is at bottom
                    inner_x = j - 1
                    inner_y = (height - 2) - i  # Flip y-axis: row 1 -> inner_y = maze_size-1
                    if 0 <= inner_x < self.maze_size and 0 <= inner_y < self.maze_size:
                        # Check distance from agent
                        dist = np.sqrt((inner_x + 0.5 - agent_pos[0])**2 +
                                       (inner_y + 0.5 - agent_pos[1])**2)
                        if dist >= min_distance:
                            floor_positions.append((inner_x, inner_y))

        if not floor_positions:
            # Fallback: use any floor position if none far enough
            for i in range(height):
                for j in range(width):
                    char = maze_map[i, j]
                    if char in ['.', 'P', 'G', ord('.'), ord('P'), ord('G')]:
                        inner_x = j - 1
                        inner_y = (height - 2) - i
                        if 0 <= inner_x < self.maze_size and 0 <= inner_y < self.maze_size:
                            floor_positions.append((inner_x, inner_y))

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

        return new_obs

    def _calculate_reward(self, info):
        """Calculate reward based on progress toward goal.

        Args:
            info: Info dict containing agent_pos

        Returns:
            Dense reward: Change in distance (positive = getting closer)
            Sparse reward: 0 if at goal grid location, -1 otherwise
        """
        agent_pos = info.get("agent_pos", np.array([0.0, 0.0]))

        if self.dense_reward:
            # Calculate current distance
            current_distance = np.sqrt(
                (agent_pos[0] - self.goal_pos[0])**2 +
                (agent_pos[1] - self.goal_pos[1])**2
            )

            # Reward is progress (reduction in distance)
            # Positive when getting closer, negative when getting further
            progress_reward = self._prev_distance - current_distance

            # Update previous distance for next step
            self._prev_distance = current_distance

            # Add goal bonus when very close
            goal_bonus = 1.0 if current_distance < 0.5 else 0.0

            return progress_reward + goal_bonus
        else:
            # Sparse reward: 0 if at goal grid location, -1 otherwise
            agent_grid_x = int(agent_pos[0])
            agent_grid_y = int(agent_pos[1])
            goal_grid_x = int(self.goal_pos[0])
            goal_grid_y = int(self.goal_pos[1])

            if agent_grid_x == goal_grid_x and agent_grid_y == goal_grid_y:
                return 0.0
            else:
                return -1.0

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

        # Transform observation
        new_obs = self._transform_observation(obs, info)

        # Calculate navigation reward (MSE distance)
        reward = self._calculate_reward(info)

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

            # Display reward (total reward for the episode)
            reward_text = font.render(
                f"Reward: {self._total_reward:.2f}", True, (0, 0, 0)
            )
            reward_rect = reward_text.get_rect(topleft=(dist_rect.right + 20, 10))
            base_env.window.blit(reward_text, reward_rect)

            # Display time step
            time_text = font.render(
                f"Step: {base_env._time_step//40}/{base_env.time_limit}", True, (0, 0, 0)
            )
            time_rect = time_text.get_rect(topleft=(reward_rect.right + 20, 10))
            base_env.window.blit(time_text, time_rect)

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
