from collections import OrderedDict
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from importlib import resources

from mmjc_env.memory_maze import tasks

from mmjc_env.memory_maze.gym_wrappers import _convert_to_space

# from bilaterian.ant_floor_steering import Agent
import torch
from torch.distributions.normal import Normal

from torch import nn

from .helper import render_maze_with_agent_and_targets


class Actions(Enum):
    forward = 0
    left = 1
    right = 2


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, single_obs_space, single_action_space):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(single_obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(single_obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(
            torch.zeros(1, np.prod(single_action_space.shape))
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            self.critic(x),
        )


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


class VertebrateEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 40}

    def __init__(
        self,
        render_mode=None,
        maze_size=7,
        temp_k=8,
        top_camera=True,
        device="cpu",
        optional_reward=True,
        model_name="model_name",
    ):
        # self.maze_size = maze_size  # The size of the maze (maze_size x maze_size)
        self.window_size = 512  # The size of the PyGame window
        self.temp_k = temp_k  # Temporal abstraction parameter the number of steps per action taken
        self.top_camera = top_camera
        self.optional_reward = optional_reward
        self.model_name = model_name

        if render_mode == "human":
            # self.camera_resolution = 256
            self.camera_resolution = 64
        else:
            self.camera_resolution = 64

        self.mm_env = tasks._memory_maze(
            maze_size,
            3,
            room_min_size=maze_size,
            room_max_size=maze_size,
            global_observables=True,
            image_only_obs=False,
            top_camera=top_camera,
            camera_resolution=self.camera_resolution,
            control_freq=40.0,
            discrete_actions=False,
            target_color_in_image=False,
            walker_str="ant",
            remap_obs=True,
        )

        lower_observation_space = spaces.Box(
            -np.inf, np.inf, shape=(153,), dtype=np.float32
        )
        lower_action_space = _convert_to_space(self.mm_env.action_spec())

        forward_policy = Agent(lower_observation_space, lower_action_space)
        forward_policy_path = resources.files("vertebrate_env.envs").joinpath(
            "steering_policy/maz-001-rev3-forward.pt"
        )
        forward_policy.load_state_dict(
            torch.load(forward_policy_path, map_location=device)
        )

        left_policy = Agent(lower_observation_space, lower_action_space)
        left_policy_path = resources.files("vertebrate_env.envs").joinpath(
            "steering_policy/maz-001-rev3-left.pt"
        )
        left_policy.load_state_dict(torch.load(left_policy_path, map_location=device))
        right_policy = Agent(lower_observation_space, lower_action_space)
        right_policy_path = resources.files("vertebrate_env.envs").joinpath(
            "steering_policy/maz-001-rev3-right.pt"
        )
        right_policy.load_state_dict(torch.load(right_policy_path, map_location=device))

        self.steering_policy = {
            Actions.forward: forward_policy,
            Actions.left: left_policy,
            Actions.right: right_policy,
        }

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2,
        # i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "egocentric_camera": spaces.Box(
                    0, 255, shape=(self.temp_k, 64, 64, 3), dtype=np.uint8
                ),
                "target_color": spaces.Box(0, 1, shape=(3,), dtype=np.float64),
            }
        )

        # We have 3 actions, corresponding to "forward", "left", "right"
        self.action_space = spaces.Discrete(3)

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

    def _get_obs(self):
        return {
            "egocentric_camera": self._ego_centric_camera,
            "target_color": self._target_color,
        }

    def _get_info(self):
        return {
            "top_camera": self._top_camera,
        }

    def _get_steering_obs(self, time_step):
        steering_obs = OrderedDict(
            (k, v)
            for k, v in time_step.observation.items()
            if k
            not in [
                "walker/egocentric_camera",
                "target_color",
                "target_index",
                "top_camera",
                "agent_dir",
                "target_vec",
                "target_pos",
            ]
        )
        flat_obs = np.concatenate([v.flatten() for v in steering_obs.values()], axis=0)
        return flat_obs

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        time_step = self.mm_env.reset()
        ego_cameras = []
        top_cameras = []
        ego_cameras.append(time_step.observation["walker/egocentric_camera"])
        top_cameras.append(time_step.observation["top_camera"])
        self._target_color = time_step.observation["target_color"]
        steering_obs = self._get_steering_obs(time_step)

        # Initialize total reward
        self._total_reward = 0.0

        for i in range(self.temp_k - 1):
            flat_obs = torch.Tensor(steering_obs).unsqueeze(0)
            action, _, _, _ = self.steering_policy[
                Actions.forward
            ].get_action_and_value(flat_obs)

            time_step = self.mm_env.step(action.squeeze())
            ego_cameras.append(time_step.observation["walker/egocentric_camera"])
            top_cameras.append(time_step.observation["top_camera"])
            steering_obs = self._get_steering_obs(time_step)

            if time_step.last():
                return self.reset()

        self._last_steering_obs = steering_obs
        self._ego_centric_camera = np.stack(ego_cameras, axis=0)
        self._top_camera = np.stack(top_cameras, axis=0)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(0)

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

        ego_cameras = []
        top_cameras = []
        terminated = False
        truncated = False
        reward = 0

        for i in range(self.temp_k):
            flat_obs = torch.Tensor(self._last_steering_obs).unsqueeze(0)
            steering_action, _, _, _ = self.steering_policy[
                Actions(action)
            ].get_action_and_value(flat_obs)

            time_step = self.mm_env.step(steering_action.squeeze())
            ego_cameras.append(time_step.observation["walker/egocentric_camera"])
            top_cameras.append(time_step.observation["top_camera"])
            self._last_steering_obs = self._get_steering_obs(time_step)
            reward += time_step.reward
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
                for j in range(i + 1, self.temp_k):
                    ego_cameras.append(
                        time_step.observation["walker/egocentric_camera"]
                    )
                    top_cameras.append(time_step.observation["top_camera"])
                break

        # Update total reward
        self._total_reward += reward

        self._ego_centric_camera = np.stack(ego_cameras, axis=0)
        self._top_camera = np.stack(top_cameras, axis=0)
        self._target_color = time_step.observation["target_color"]
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame(action)

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self, action=0):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size * 2, self.window_size)
            )
            pygame.display.set_caption(self.model_name)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size * 2, self.window_size))
        canvas.fill((255, 255, 255))
        pygame.display.flip()

        # self._ego_centric_camera has shape (temp_k, 64, 64, 3)
        # We will render each of the temp_k frames sequentially

        if self.render_mode == "human":
            for t in range(self.temp_k):
                # First horizontally stack the ego-centric camera views and top down views
                # to build a single image at time t
                if self.top_camera:
                    combined_image = np.concatenate(
                        [self._ego_centric_camera[t], self._top_camera[t]], axis=1
                    )
                else:
                    combined_image = self._ego_centric_camera[t]
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
                action_text = font.render(f"{Actions(action).name}", True, (0, 0, 0))
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
                    f"Reward: {self._total_reward:.5f}", True, (0, 0, 0)
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
                [self._ego_centric_camera, self._top_camera], axis=2
            )  # shape (temp_k, 128, 64, 3)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
