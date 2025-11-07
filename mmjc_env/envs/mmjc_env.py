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


class MMJCENV(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        render_mode=None,
        maze_size=9,
        top_camera=True,
        optional_reward=False,
        seed=42,
        model_name="mmjc_model",
        targets_per_room=1,
    ):
        # self.maze_size = maze_size  # The size of the maze (maze_size x maze_size)
        self.window_size = 512  # The size of the PyGame window
        self.top_camera = top_camera
        self.optional_reward = optional_reward
        self.seed = seed

        if render_mode == "human":
            self.camera_resolution = 256
            # self.camera_resolution = 64
        else:
            self.camera_resolution = 256

        self.mm_env = tasks._memory_maze(
            maze_size,
            3,
            room_min_size=3,
            room_max_size=5,
            global_observables=self.optional_reward,
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
            elif key in ["target_pos", "target_vec", "agent_dir"]:
                new_info[key] = value
            elif np.prod(value.shape) != 0:
                sensors.append(value.flatten())

        observation["sensors"] = np.concatenate(sensors)
        return observation, new_info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self._total_reward = 0.0

        time_step = self.mm_env.reset()
        observation, info = self._get_obs_and_info(
            time_step.observation,
        )

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def calculate_optional_reward(self, time_step):
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
            return reward_scale * alignment_reward

        return 0

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

        time_step = self.mm_env.step(action)
        observation, info = self._get_obs_and_info(time_step.observation)
        reward = time_step.reward
        terminated = False
        truncated = False

        optional_reward = self.calculate_optional_reward(time_step)
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
            self.window = pygame.display.set_mode(
                (self.window_size * 2, self.window_size)
            )
            pygame.display.set_caption("Memory Maze Environment")
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size * 2, self.window_size))
        canvas.fill((255, 255, 255))
        pygame.display.flip()

        # self._ego_centric_camera has shape (64, 64, 3)
        # We will render the frame

        if self.render_mode == "human":
            # First horizontally stack the ego-centric camera views and top down views
            # to build a single image at time t
            if self.top_camera:
                combined_image = np.concatenate(
                    [self._ego_centric_camera, self._top_camera], axis=1
                )
            else:
                combined_image = self._ego_centric_camera
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
                f"Reward: {self._total_reward:.7f}", True, (0, 0, 0)
            )
            reward_rect = reward_text.get_rect(topleft=(text_rect.right + 44, 10))
            self.window.blit(reward_text, reward_rect)

            # Update the display
            pygame.event.pump()
            pygame.display.flip()  # Use flip to update the entire screen

            # Tick the clock to maintain FPS
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":  # rgb_array
            return np.concatenate(
                [self._ego_centric_camera, self._top_camera], axis=1
            )  # shape (128, 64, 3)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
