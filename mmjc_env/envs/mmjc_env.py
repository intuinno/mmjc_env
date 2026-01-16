import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from mmjc_env.wrappers.gymnasium_wrappers import _convert_to_space

from mmjc_env.memory_maze import tasks
import cv2

TARGET_COLORS = np.array(
    [
        np.array([170, 38, 30]) / 220,  # red
        np.array([99, 170, 88]) / 220,  # green
        np.array([39, 140, 217]) / 220,  # blue
        np.array([93, 105, 199]) / 220,  # purple
        np.array([220, 193, 59]) / 220,  # yellow
        np.array([220, 128, 107]) / 220,  # salmon
    ]
)

# RGB colors extracted from labmaze style_01 wall textures
STYLE_01_TEXTURE_COLORS = {
    "blue": (32, 119, 182),
    "cerise": (174, 67, 106),
    "green": (88, 125, 78),
    "green_bright": (169, 200, 159),
    "purple": (71, 83, 181),
    "red": (147, 33, 26),
    "red_bright": (218, 126, 105),
    "yellow": (202, 174, 39),
}


class MMJCENV(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1000}

    def __init__(
        self,
        render_mode=None,
        maze_size=9,
        top_camera=True,
        optional_reward=False,
        seed=42,
        model_name="mmjc_model",
        targets_per_room=1,
        time_limit=250,
        num_targets=3,
        exploration_reward=False,
        target_reward=True,
        room_min_size=3,
        room_max_size=5,
    ):
        # self.maze_size = maze_size  # The size of the maze (maze_size x maze_size)
        self.window_size = 512  # The size of the PyGame window
        self.top_camera = top_camera
        self.optional_reward = optional_reward
        self._seed = seed
        self.time_limit = time_limit
        self.exploration_reward = exploration_reward
        self.target_reward = target_reward

        if render_mode == "human":
            self.camera_resolution = 256
            # self.camera_resolution = 64
        else:
            self.camera_resolution = 64

        global_observables = self.optional_reward or self.exploration_reward

        self.mm_env = tasks._memory_maze(
            maze_size,
            num_targets,
            time_limit=time_limit,
            room_min_size=room_min_size,
            room_max_size=room_max_size,
            global_observables=global_observables,
            image_only_obs=False,
            top_camera=top_camera,
            camera_resolution=self.camera_resolution,
            control_freq=40.0,
            discrete_actions=False,
            target_color_in_image=False,
            walker_str="ant",
            remap_obs=True,
            targets_per_room=targets_per_room,
        )

        # Initialize exploration history as 2D integer array
        self.exploration_history = np.zeros((maze_size, maze_size), dtype=int)
        # Initialize trajectory history to store agent positions for line drawing
        self.trajectory_history = []

        action_space = _convert_to_space(self.mm_env.action_spec())
        original_obs_space = _convert_to_space(self.mm_env.observation_spec())

        # Create new observation space by removing 'top_camera' and change 'walker/egocentric_camer' to 'image'
        new_obs_spaces = {}
        total_sensor_dim = 0
        for key, value in original_obs_space.spaces.items():
            if key not in [
                "top_camera",
                "walker/egocentric_camera",
                "target_color",
                "target_pos",
                "target_vec",
                "agent_dir",
                "agent_pos",
            ]:  # Exclude these keys
                total_sensor_dim += np.prod(value.shape)

        new_obs_space = {
            "image": spaces.Box(0, 255, shape=(64, 64, 3), dtype=np.uint8),
            "sensors": spaces.Box(
                -np.inf, np.inf, shape=(int(total_sensor_dim),), dtype=np.float64
            ),
            "target_color": spaces.Box(0, 1, shape=(3,), dtype=np.float64),
        }

        # Create new Dict space
        self.observation_space = spaces.Dict(new_obs_space)
        self.action_space = action_space
        self.model_name = model_name

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self._cached_wall_colors = None

    def _calculate_coverage(self):
        """Calculate exploration coverage percentage."""
        return (self.exploration_history > 0).sum() / self.exploration_history.size * 100

    # 1. We will move 'top_camera' to info
    # 2. We rename 'walker/egocentric_camera' to 'image'
    def _get_obs_and_info(self, obs):
        observation = {}
        new_info = {}
        sensors = []
        for key, value in obs.items():
            if key == "walker/egocentric_camera":
                self._ego_centric_camera = value
                down_sampled_image = cv2.resize(
                    value, (64, 64), interpolation=cv2.INTER_AREA
                )
                observation["image"] = down_sampled_image
            elif key == "top_camera":
                new_info["top_camera"] = value
                self._top_camera = value
            elif key == "target_color":
                observation["target_color"] = value
                self._target_color = value
            elif key == "agent_pos":
                new_info[key] = value
                self._agent_pos = value
            elif key == "agent_dir":
                new_info[key] = value
                self._agent_dir = value
            elif key in ["target_pos", "target_vec"]:
                new_info[key] = value
            elif np.prod(value.shape) != 0:
                sensors.append(value.flatten())

        observation["sensors"] = np.concatenate(sensors)
        new_info["coverage"] = self._calculate_coverage()
        new_info["maze_map"] = self._get_maze_map()
        return observation, new_info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._total_reward = 0.0
        self._time_step = 0
        # Reset exploration history and trajectory
        self.exploration_history.fill(0)
        self.trajectory_history = []

        time_step = self.mm_env.reset()
        # Cache wall colors after maze reset
        self._cached_wall_colors = self._get_wall_colors()
        observation, info = self._get_obs_and_info(
            time_step.observation,
        )

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def calculate_optional_reward(self, time_step):
        optional_reward = 0
        exploration_reward = 0
        if self.optional_reward:
            observation = time_step.observation

            # Calculate angle to target
            angle = self.calculate_angle_from_obs(observation)

            # Calculate distance to target
            target_vec = observation["target_vec"]
            distance = np.linalg.norm(target_vec)

            # Reward for facing the target, scaled by distance.
            # The reward is higher when the agent is facing the target (cos(angle) is close to 1)
            # and is closer to the target (distance is small).
            # We add a small constant to distance to avoid division by zero.
            alignment_reward = np.cos(angle) / (1.0 + distance)

            # Scale the reward to be small compared to the main task reward
            reward_scale = 0.00001
            optional_reward = reward_scale * alignment_reward
        if self.exploration_reward:
            agent_pos = time_step.observation["agent_pos"]
            x, y = int(agent_pos[0]), int(agent_pos[1])
            if self.exploration_history[x, y] == 0:
                exploration_reward = 0.02
            self.exploration_history[x, y] += 1
            # Store normalized agent position for trajectory drawing
            self.trajectory_history.append((agent_pos[0], agent_pos[1]))

        total_reward = optional_reward + exploration_reward
        return total_reward

    def calculate_angle_from_obs(self, observation):
        """
        Calculates the relative angle to the target from the agent's perspective.

        Args:
            observation (dict): The observation dictionary from the wrapped environment,
                                containing 'agent_dir' and 'target_vec'.

        Returns:
            float: The angle in radians, in the range [-pi, pi].
                A positive value means the target is to the left.
                A negative value means the target is to the right.
        """
        agent_dir = observation["agent_dir"]
        target_vec = observation["target_vec"]

        # Angle of the agent's forward direction
        angle_agent = np.arctan2(agent_dir[1], agent_dir[0])

        # Angle of the vector to the target
        angle_target = np.arctan2(target_vec[1], target_vec[0])

        # The relative angle, normalized to [-pi, pi]
        angle = angle_target - angle_agent
        if angle > np.pi:
            angle -= 2 * np.pi
        if angle < -np.pi:
            angle += 2 * np.pi

        return angle

    def step(self, action):
        self._time_step += 1
        time_step = self.mm_env.step(action)
        observation, info = self._get_obs_and_info(time_step.observation)

        terminated = False
        truncated = False

        optional_reward = self.calculate_optional_reward(time_step)

        if self.target_reward:
            reward = time_step.reward or 0.0
        else:
            reward = 0.0

        reward += optional_reward

        if time_step.last():
            if time_step.discount == 1.0:
                terminated = False
                truncated = True
                # print("Truncation of Episode")
            else:
                terminated = True
                truncated = False
                # print("Termination of Episode")

        # Update total reward
        self._total_reward += reward

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            # Update window size to accommodate three images
            self.window = pygame.display.set_mode(
                (self.window_size * 3, self.window_size)
            )
            pygame.display.set_caption(self.model_name)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            # Create integrated maze map visualization (walls + exploration + targets)
            maze_map_vis = self._create_maze_map_visualization()

            # Horizontally stack: ego-centric camera, top down view, integrated maze map
            if self.top_camera:
                combined_image = np.concatenate(
                    [self._ego_centric_camera, self._top_camera, maze_map_vis],
                    axis=1,
                )
            else:
                combined_image = np.concatenate(
                    [self._ego_centric_camera, maze_map_vis, maze_map_vis], axis=1
                )
            combined_surface = pygame.surfarray.make_surface(
                combined_image.transpose(1, 0, 2)
            )

            # Scale the surface to the window size
            scaled_surface = pygame.transform.scale(
                combined_surface, self.window.get_size()
            )

            # Blit the scaled surface onto the display window
            self.window.blit(scaled_surface, (0, 0))

            # Write the action text on the window
            font = pygame.font.SysFont("Arial", 24)
            action_text = font.render(f"Target", True, (0, 0, 0))
            text_rect = action_text.get_rect(topleft=(10, 10))
            self.window.blit(action_text, text_rect)

            # Draw the target color rectangle
            target_color_rgb = tuple((self._target_color * 255).astype(np.uint8))
            pygame.draw.rect(
                self.window,
                target_color_rgb,
                (text_rect.right + 10, 10, 24, 24),
            )

            # Display total reward next to the target color
            reward_text = font.render(
                f"Reward: {self._total_reward:.2f}", True, (0, 0, 0)
            )
            reward_rect = reward_text.get_rect(topleft=(text_rect.right + 44, 10))
            self.window.blit(reward_text, reward_rect)

            # Display time step next to the reward_text
            time_text = font.render(
                f"Timestep: {self._time_step//40}/{self.time_limit}", True, (0, 0, 0)
            )
            time_rect = time_text.get_rect(topleft=(reward_rect.right + 44, 10))
            self.window.blit(time_text, time_rect)

            # Display coverage over the exploration map (third panel)
            coverage_text = font.render(
                f"Coverage: {self._calculate_coverage():.1f}%",
                True,
                (0, 0, 0),
            )
            # Position over the third panel (exploration map)
            coverage_x = 2 * self.window_size + 10
            coverage_rect = coverage_text.get_rect(topleft=(coverage_x, 10))
            self.window.blit(coverage_text, coverage_rect)

            # Update the display
            pygame.event.pump()
            pygame.display.flip()  # Use flip to update the entire screen

            # Tick the clock to maintain FPS
            # self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":  # rgb_array
            maze_map_vis = self._create_maze_map_visualization()
            if self.top_camera:
                return np.concatenate(
                    [self._ego_centric_camera, self._top_camera, maze_map_vis],
                    axis=1,
                )  # shape (192, 64, 3)
            else:
                return np.concatenate(
                    [self._ego_centric_camera, maze_map_vis, maze_map_vis], axis=1
                )  # shape (192, 64, 3)

    def _get_maze_map(self):
        """Get the maze layout as a 2D character array.

        Returns:
            np.ndarray: 2D array of characters where:
                - '*' = default wall (yellow)
                - '0'-'9' = wall variations with different textures
                - '.' = floor
                - 'P' = spawn point
                - 'G' = target/goal position
        """
        return self.mm_env._task._maze_arena._maze.entity_layer.copy()

    def _get_wall_colors(self):
        """Get the current wall colors from the arena's texture mapping.

        Extracts the actual texture colors used in the 3D view by reading
        the arena's _current_wall_texture dictionary.

        Returns:
            dict: Mapping from wall characters to RGB tuples (0-255).
        """
        wall_colors = {}
        arena = self.mm_env._task._maze_arena

        # Get the current wall texture mapping from the arena
        if hasattr(arena, "_current_wall_texture"):
            for wall_char, texture in arena._current_wall_texture.items():
                # Extract texture name from the texture object
                texture_name = texture.name if hasattr(texture, "name") else None
                if texture_name and texture_name in STYLE_01_TEXTURE_COLORS:
                    wall_colors[wall_char] = np.array(
                        STYLE_01_TEXTURE_COLORS[texture_name], dtype=np.uint8
                    )
                else:
                    # Fallback to gray if texture not found
                    wall_colors[wall_char] = np.array([128, 128, 128], dtype=np.uint8)

        return wall_colors

    def _create_maze_map_visualization(self):
        """Create a visualization of the maze with wall colors, targets, and exploration coverage.

        This integrates the maze structure with exploration history:
        - Walls are shown with their color textures
        - Unvisited floor cells are shown in white
        - Visited floor cells are shown in grayscale based on visit frequency
        - Targets are shown with their assigned colors

        Returns:
            np.ndarray: RGB image of the integrated maze/exploration map.
        """
        maze_map = self._get_maze_map()
        height, width = maze_map.shape

        # Use cached wall colors (computed in reset)
        wall_colors = self._cached_wall_colors or self._get_wall_colors()

        # Create RGB image - white background for unvisited floor
        rgb_image = np.full((height, width, 3), 255, dtype=np.uint8)

        # Fill in walls with their colors
        for i in range(height):
            for j in range(width):
                char = maze_map[i, j]
                if char in wall_colors:
                    rgb_image[i, j] = wall_colors[char]

        # Draw targets as colored circle markers
        from dm_control import mjcf
        task = self.mm_env._task
        arena = task._maze_arena

        # Calculate scale factor for resizing (used for circle radius)
        scale_factor = self.camera_resolution / height
        target_radius = max(2, int(scale_factor * 0.4))  # Small circle radius in pixels

        # Resize before drawing targets so circles are properly sized
        rgb_image = cv2.resize(
            rgb_image,
            (self.camera_resolution, self.camera_resolution),
            interpolation=cv2.INTER_NEAREST,
        )

        # Draw trajectory as a thin black line
        # agent_pos is in grid coordinates (e.g., 0.1 to maze_size-0.1)
        # Need to add 1 to account for outer wall, then scale to image
        if len(self.trajectory_history) > 1:
            trajectory_pts = []
            for pos_x, pos_y in self.trajectory_history:
                # agent_pos is already in grid coords, add 1 for outer wall offset
                img_x = int((pos_x + 1) * scale_factor)
                img_y = int((height - 1 - pos_y) * scale_factor)  # flip y-axis
                trajectory_pts.append((img_x, img_y))

            # Draw polyline connecting all trajectory points
            pts_array = np.array(trajectory_pts, dtype=np.int32)
            cv2.polylines(rgb_image, [pts_array], isClosed=False, color=(180, 180, 180), thickness=1)

        for idx, target in enumerate(task._targets):
            # Get target position from the attachment frame
            frame = mjcf.get_attachment_frame(target.mjcf_model)
            if frame is not None and frame.pos is not None:
                pos = frame.pos
                # Convert world position to grid coordinates then scale to image coordinates
                # Add 0.5 to center on the cell
                grid_x = int((pos[0] / arena._xy_scale + arena._x_offset + 0.5) * scale_factor)
                grid_y = int((-pos[1] / arena._xy_scale + arena._y_offset + 0.5) * scale_factor)
                # Ensure within bounds
                if 0 <= grid_x < self.camera_resolution and 0 <= grid_y < self.camera_resolution:
                    # Use target color
                    target_color = tuple(int(c) for c in (task._target_colors[idx] * 255))
                    cv2.circle(rgb_image, (grid_x, grid_y), target_radius, target_color, -1)

        # Draw agent as a triangle pointing in heading direction
        if hasattr(self, "_agent_pos") and hasattr(self, "_agent_dir"):
            # agent_pos is in grid coordinates (0.1 to maze_size-0.1)
            # Add 1 for outer wall offset, then scale to image
            agent_x = int((self._agent_pos[0] + 1) * scale_factor)
            agent_y = int((height - 1 - self._agent_pos[1]) * scale_factor)  # flip y-axis

            # agent_dir is a 2D direction vector
            # Flip y because image y-axis is inverted, add pi/2 to correct 90 degree offset
            dir_x, dir_y = self._agent_dir[0], -self._agent_dir[1]
            angle = np.arctan2(dir_y, dir_x) + np.pi / 2

            # Triangle size
            triangle_size = max(3, int(scale_factor * 0.6))

            # Calculate triangle vertices (pointing in heading direction)
            # Front vertex (in direction of heading)
            front_x = int(agent_x + triangle_size * 1.5 * np.cos(angle))
            front_y = int(agent_y + triangle_size * 1.5 * np.sin(angle))

            # Back-left vertex
            back_left_x = int(agent_x + triangle_size * np.cos(angle + 2.5))
            back_left_y = int(agent_y + triangle_size * np.sin(angle + 2.5))

            # Back-right vertex
            back_right_x = int(agent_x + triangle_size * np.cos(angle - 2.5))
            back_right_y = int(agent_y + triangle_size * np.sin(angle - 2.5))

            triangle_pts = np.array(
                [[front_x, front_y], [back_left_x, back_left_y], [back_right_x, back_right_y]],
                dtype=np.int32,
            )

            # Draw filled triangle in cyan color
            cv2.fillPoly(rgb_image, [triangle_pts], (0, 255, 255))

        return rgb_image

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self._seed = seed
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
        if hasattr(self, "mm_env") and hasattr(self.mm_env, "close"):
            self.mm_env.close()
