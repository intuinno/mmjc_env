"""Taxi Navigation Environment using dm_control locomotion.

This environment trains an Ant agent to follow dynamically changing goals:
- FORWARD: Move forward at a target velocity
- ROTATE_CW: Rotate clockwise at a target angular velocity
- ROTATE_CCW: Rotate counter-clockwise at a target angular velocity

The goal switches every k steps. Rewards are Gaussian, peaking at target
velocities to encourage "reasonable" locomotion (not maximum speed).
"""

from enum import IntEnum
import gymnasium
from gymnasium import spaces
import numpy as np
import pygame

from dm_control import composer
from dm_control.locomotion.walkers import ant
from dm_control.locomotion.arenas import floors


class GoalType(IntEnum):
    """Goal types for taxi navigation."""
    FORWARD = 0
    ROTATE_CW = 1      # Clockwise (negative yaw rate)
    ROTATE_CCW = 2     # Counter-clockwise (positive yaw rate)


class TaxiNavigationTask(composer.Task):
    """dm_control task for taxi-style goal following."""

    def __init__(
        self,
        walker,
        arena,
        goal_switch_interval: int = 100,
        target_forward_velocity: float = 1.0,
        target_angular_velocity: float = 0.5,
        velocity_tolerance: float = 0.3,
        physics_timestep: float = 0.005,
        control_timestep: float = 0.025,
    ):
        """Initialize the taxi navigation task.

        Args:
            walker: Ant walker instance
            arena: Floor arena instance
            goal_switch_interval: Steps between goal switches
            target_forward_velocity: Target forward speed (m/s)
            target_angular_velocity: Target rotation speed (rad/s)
            velocity_tolerance: Gaussian reward width
            physics_timestep: Physics simulation timestep
            control_timestep: Control input timestep
        """
        self._walker = walker
        self._arena = arena
        self._walker.create_root_joints(self._arena.attach(self._walker))

        self.goal_switch_interval = goal_switch_interval
        self.target_forward_velocity = target_forward_velocity
        self.target_angular_velocity = target_angular_velocity
        self.velocity_tolerance = velocity_tolerance

        # State
        self.current_goal = GoalType.FORWARD
        self._step_count = 0

        # Enable observables
        self._configure_observables()

        self.set_timesteps(
            physics_timestep=physics_timestep,
            control_timestep=control_timestep
        )

    def _configure_observables(self):
        """Enable walker observables for proprioception and sensors."""
        enabled_observables = []
        enabled_observables += self._walker.observables.proprioception
        enabled_observables += self._walker.observables.kinematic_sensors
        enabled_observables += self._walker.observables.dynamic_sensors
        for obs in enabled_observables:
            obs.enabled = True

    @property
    def root_entity(self):
        return self._arena

    def initialize_episode_mjcf(self, random_state):
        """Called before physics compilation."""
        self._arena.regenerate(random_state=random_state)

    def initialize_episode(self, physics, random_state):
        """Reset episode state."""
        self._walker.reinitialize_pose(physics, random_state)
        self._step_count = 0
        self._switch_goal(random_state)

    def _switch_goal(self, random_state):
        """Randomly select a new goal."""
        self.current_goal = GoalType(random_state.randint(0, 3))

    def before_step(self, physics, action, random_state):
        """Apply action before physics step."""
        self._walker.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        """Called after physics step - handle goal switching."""
        self._step_count += 1
        if self._step_count % self.goal_switch_interval == 0:
            self._switch_goal(random_state)

    def get_forward_velocity(self, physics):
        """Get forward velocity from velocimeter sensor."""
        # velocimeter[0] is forward velocity in local frame
        velocimeter = physics.bind(
            self._walker.mjcf_model.sensor.velocimeter
        ).sensordata
        return float(velocimeter[0])

    def get_angular_velocity(self, physics):
        """Get yaw angular velocity from gyro sensor."""
        # gyro[2] is angular velocity about Z-axis (yaw)
        gyro = physics.bind(
            self._walker.mjcf_model.sensor.gyro
        ).sensordata
        return float(gyro[2])

    def get_reward(self, physics):
        """Calculate reward based on current goal."""
        forward_vel = self.get_forward_velocity(physics)
        angular_vel = self.get_angular_velocity(physics)

        if self.current_goal == GoalType.FORWARD:
            return self._reward_forward(forward_vel)
        elif self.current_goal == GoalType.ROTATE_CW:
            return self._reward_rotation(angular_vel, clockwise=True)
        else:  # ROTATE_CCW
            return self._reward_rotation(angular_vel, clockwise=False)

    def _reward_forward(self, forward_vel: float) -> float:
        """Gaussian reward for forward movement at target velocity."""
        error = abs(forward_vel - self.target_forward_velocity)
        reward = np.exp(-0.5 * (error / self.velocity_tolerance) ** 2)
        # Penalize backward movement
        if forward_vel < 0:
            reward -= 0.5 * abs(forward_vel)
        return reward

    def _reward_rotation(self, angular_vel: float, clockwise: bool) -> float:
        """Gaussian reward for rotation at target angular velocity.

        Clockwise = negative angular velocity (looking down)
        Counter-clockwise = positive angular velocity
        """
        if clockwise:
            expected = -self.target_angular_velocity
        else:
            expected = self.target_angular_velocity

        error = abs(angular_vel - expected)
        return np.exp(-0.5 * (error / self.velocity_tolerance) ** 2)

    def get_discount(self, physics):
        """Return discount factor."""
        return 1.0

    def should_terminate_episode(self, physics):
        """Check if episode should terminate (e.g., if ant falls over)."""
        # Terminate if ant is no longer upright
        aliveness = self._walker.aliveness(physics)
        return aliveness < -0.5  # Threshold for "fallen over"


class TaxiNavigationEnv(gymnasium.Env):
    """Gymnasium environment for taxi-style navigation.

    The agent (Ant) follows dynamically changing goals on a flat arena.
    Goals are: FORWARD, ROTATE_CW, ROTATE_CCW.

    Observations:
        - goal: One-hot encoding of current goal (3,)
        - proprioception: Sensor data from the walker

    Actions:
        - 8D continuous motor control for Ant's joints
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        goal_switch_interval: int = 100,
        target_forward_velocity: float = 1.0,
        target_angular_velocity: float = 0.5,
        velocity_tolerance: float = 0.3,
        time_limit: float = 30.0,
        render_mode: str = None,
    ):
        """Initialize the taxi navigation environment.

        Args:
            goal_switch_interval: Steps between goal switches
            target_forward_velocity: Target forward speed (m/s)
            target_angular_velocity: Target rotation speed (rad/s)
            velocity_tolerance: Gaussian reward tolerance
            time_limit: Episode time limit in seconds
            render_mode: "human" or "rgb_array"
        """
        super().__init__()

        self.render_mode = render_mode
        self.goal_switch_interval = goal_switch_interval
        self.target_forward_velocity = target_forward_velocity
        self.target_angular_velocity = target_angular_velocity

        # Create dm_control components
        # marker_rgba marks the front legs red for visual orientation
        self._walker = ant.Ant(marker_rgba=(255, 0, 0, 1.0))
        self._arena = floors.Floor(size=(20, 20))
        self._task = TaxiNavigationTask(
            walker=self._walker,
            arena=self._arena,
            goal_switch_interval=goal_switch_interval,
            target_forward_velocity=target_forward_velocity,
            target_angular_velocity=target_angular_velocity,
            velocity_tolerance=velocity_tolerance,
        )

        # Create dm_control environment
        self._env = composer.Environment(
            task=self._task,
            time_limit=time_limit,
            strip_singleton_obs_buffer_dim=True,
        )

        # Determine proprioception dimension by doing a test reset
        self._random_state = np.random.RandomState()
        time_step = self._env.reset()
        self._proprioception_dim = self._get_proprioception(time_step.observation).shape[0]

        # Define observation space
        self.observation_space = spaces.Dict({
            "goal": spaces.Box(0, 1, shape=(3,), dtype=np.float32),
            "proprioception": spaces.Box(
                -np.inf, np.inf,
                shape=(self._proprioception_dim,),
                dtype=np.float32
            ),
        })

        # Action space: 8 motors for Ant
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )

        # Rendering
        self._window = None
        self._clock = None
        self._total_reward = 0.0

    def _get_proprioception(self, observation):
        """Extract proprioception from dm_control observation."""
        # Collect all sensor data (excluding cameras)
        sensors = []
        for key, value in observation.items():
            # Skip camera observations
            if 'camera' in key.lower() or 'image' in key.lower():
                continue
            if isinstance(value, np.ndarray) and value.size > 0:
                sensors.append(value.flatten())
        return np.concatenate(sensors).astype(np.float32)

    def _get_goal_one_hot(self) -> np.ndarray:
        """Return one-hot encoding of current goal."""
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[self._task.current_goal.value] = 1.0
        return one_hot

    def _transform_observation(self, dm_obs):
        """Transform dm_control observation to gymnasium format."""
        return {
            "goal": self._get_goal_one_hot(),
            "proprioception": self._get_proprioception(dm_obs),
        }

    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            self._random_state = np.random.RandomState(seed)

        time_step = self._env.reset()
        self._total_reward = 0.0

        obs = self._transform_observation(time_step.observation)
        info = {
            "current_goal": self._task.current_goal.name,
            "forward_velocity": self._task.get_forward_velocity(self._env.physics),
            "angular_velocity": self._task.get_angular_velocity(self._env.physics),
        }

        if self.render_mode == "human":
            self._render_human()

        return obs, info

    def step(self, action):
        """Take a step in the environment."""
        time_step = self._env.step(action)

        obs = self._transform_observation(time_step.observation)
        reward = time_step.reward or 0.0
        self._total_reward += reward

        terminated = time_step.last() and time_step.discount == 0.0
        truncated = time_step.last() and time_step.discount != 0.0

        info = {
            "current_goal": self._task.current_goal.name,
            "forward_velocity": self._task.get_forward_velocity(self._env.physics),
            "angular_velocity": self._task.get_angular_velocity(self._env.physics),
            "step_count": self._task._step_count,
        }

        if self.render_mode == "human":
            self._render_human()

        return obs, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            self._render_human()

    def _render_rgb_array(self):
        """Return RGB array of the top-down view."""
        return self._env.physics.render(
            camera_id="top_camera",
            height=480,
            width=480
        )

    def _render_human(self):
        """Render to pygame window with overlays."""
        if self._window is None:
            pygame.init()
            pygame.display.init()
            self._window = pygame.display.set_mode((480, 480))
            pygame.display.set_caption("TaxiNavigationEnv")
        if self._clock is None:
            self._clock = pygame.time.Clock()

        # Get top-down camera image
        frame = self._env.physics.render(
            camera_id="top_camera",
            height=480,
            width=480
        )

        # Create pygame surface
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))
        self._window.blit(surface, (0, 0))

        # Text overlays
        font = pygame.font.SysFont("Arial", 24)

        # Goal with color coding
        goal_colors = {
            GoalType.FORWARD: (0, 255, 0),      # Green
            GoalType.ROTATE_CW: (255, 165, 0),  # Orange
            GoalType.ROTATE_CCW: (0, 191, 255), # Blue
        }
        goal_names = {
            GoalType.FORWARD: "FORWARD",
            GoalType.ROTATE_CW: "ROTATE CW",
            GoalType.ROTATE_CCW: "ROTATE CCW",
        }
        goal = self._task.current_goal
        goal_text = font.render(
            f"Goal: {goal_names[goal]}", True, goal_colors[goal]
        )
        self._window.blit(goal_text, (10, 10))

        # Velocity info
        forward_vel = self._task.get_forward_velocity(self._env.physics)
        angular_vel = self._task.get_angular_velocity(self._env.physics)

        vel_text = font.render(
            f"Forward: {forward_vel:.2f} m/s", True, (255, 255, 255)
        )
        self._window.blit(vel_text, (10, 40))

        angular_text = font.render(
            f"Angular: {angular_vel:.2f} rad/s", True, (255, 255, 255)
        )
        self._window.blit(angular_text, (10, 70))

        # Steps until switch
        steps_left = self.goal_switch_interval - (
            self._task._step_count % self.goal_switch_interval
        )
        switch_text = font.render(
            f"Switch in: {steps_left}", True, (255, 255, 255)
        )
        self._window.blit(switch_text, (10, 100))

        # Total reward
        reward_text = font.render(
            f"Return: {self._total_reward:.2f}", True, (255, 255, 255)
        )
        self._window.blit(reward_text, (10, 130))

        # Step count
        step_text = font.render(
            f"Step: {self._task._step_count}", True, (255, 255, 255)
        )
        self._window.blit(step_text, (10, 160))

        pygame.event.pump()
        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        """Clean up resources."""
        if self._window is not None:
            pygame.display.quit()
            pygame.quit()
            self._window = None
            self._clock = None
