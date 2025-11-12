import numpy as np
from dm_control import composer
from dm_control.locomotion.arenas import labmaze_textures

from mmjc_env.memory_maze.maze import *
from mmjc_env.memory_maze.oracle import DrawMinimapWrapper, PathToTargetWrapper
from mmjc_env.memory_maze.wrappers import *
from dm_control.locomotion.walkers.ant import Ant
from mmjc_env.memory_maze.joe import Joe


# Slow control (4Hz), so that agent without HRL has a chance.
# Native control would be ~20Hz, so this corresponds roughly to action_repeat=5.
DEFAULT_CONTROL_FREQ = 4.0


def memory_maze_9x9(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_small
    {
        mazeHeight = 11,  # with outer walls
        mazeWidth = 11,
        roomCount = 4,
        roomMaxSize = 5,
        roomMinSize = 3,
    }
    """
    return _memory_maze(9, 3, time_limit=250, **kwargs)


def memory_maze_11x11(**kwargs):
    return _memory_maze(11, 4, time_limit=500, **kwargs)


def memory_maze_13x13(**kwargs):
    return _memory_maze(13, 5, time_limit=750, **kwargs)


def memory_maze_15x15(**kwargs):
    """
    Maze based on DMLab30-explore_goal_locations_large
    {
        mazeHeight = 17,  # with outer walls
        mazeWidth = 17,
        roomCount = 9,
        roomMaxSize = 3,
        roomMaxSize = 3,
    }
    """
    return _memory_maze(15, 6, time_limit=1000, max_rooms=9, room_max_size=3, **kwargs)


def _memory_maze(
    maze_size,  # measured without exterior walls
    n_targets,
    time_limit=250,
    max_rooms=6,
    room_min_size=3,
    room_max_size=5,
    control_freq=DEFAULT_CONTROL_FREQ,
    discrete_actions=True,
    image_only_obs=False,
    target_color_in_image=True,
    global_observables=False,
    top_camera=False,
    good_visibility=False,
    show_path=False,
    camera_resolution=64,
    seed=None,
    randomize_colors=False,
    walker_str="ball",
    remap_obs=True,
    bonus_time_limit=0,
    targets_per_room=10,
):
    random_state = np.random.RandomState(seed)
    if walker_str == "ball":
        walker = RollingBallWithFriction(camera_height=0.3, add_ears=top_camera)
    elif walker_str == "ant":
        # walker = Ant(observable_option={"egocentric_camera": dict(enabled=True)})
        walker = Ant(marker_rgba=(255, 0, 0, 1.0))
    elif walker_str == "joe":
        # walker = Joe(observable_option={"egocentric_camera": dict(enabled=True)})
        walker = Joe()
    else:
        raise NotImplementedError
    arena = MazeWithTargetsArena(
        x_cells=maze_size + 2,  # inner size => outer size
        y_cells=maze_size + 2,
        xy_scale=2.0,
        z_height=1.5 if not good_visibility else 0.4,
        max_rooms=max_rooms,
        room_min_size=room_min_size,
        room_max_size=room_max_size,
        spawns_per_room=10,
        targets_per_room=targets_per_room,
        floor_textures=FixedFloorTexture("style_01", ["blue", "blue_bright"]),
        wall_textures=dict(
            {
                "*": FixedWallTexture("style_01", "yellow"),  # default wall
            },
            **{
                str(i): labmaze_textures.WallTextures("style_01") for i in range(10)
            }  # variations
        ),
        skybox_texture=None,
        random_seed=random_state.randint(2147483648),
    )

    task = MemoryMazeTask(
        walker=walker,
        maze_arena=arena,
        n_targets=n_targets,
        target_radius=0.6,
        target_height_above_ground=0.5 if good_visibility else -0.6,
        enable_global_task_observables=global_observables,  # Removed due to maze_layout
        control_timestep=1.0 / control_freq,
        camera_resolution=camera_resolution,
        target_randomize_colors=randomize_colors,
    )

    if top_camera:
        task.observables["top_camera"].enabled = True

    total_time_limit = time_limit + bonus_time_limit

    env = composer.Environment(
        time_limit=total_time_limit
        - 1e-3,  # subtract epsilon to make sure ep_length=time_limit*fps
        task=task,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True,
    )

    if remap_obs:
        obs_mapping = {
            "walker/egocentric_camera": "walker/egocentric_camera",
            "top_camera": "top_camera",
            "target_color": "target_color",
            "walker/appendages_pos": "walker/appendages_pos",
            "walker/bodies_pos": "walker/bodies_pos",
            "walker/bodies_quats": "walker/bodies_quats",
            "walker/body_height": "walker/body_height",
            "walker/end_effectors_pos": "walker/end_effectors_pos",
            "walker/joints_pos": "walker/joints_pos",
            "walker/joints_vel": "walker/joints_vel",
            "walker/sensors_accelerometer": "walker/sensors_accelerometer",
            "walker/sensors_force": "walker/sensors_force",
            "walker/sensors_gyro": "walker/sensors_gyro",
            "walker/sensors_torque": "walker/sensors_torque",
            "walker/sensors_touch": "walker/sensors_touch",
            "walker/sensors_velocimeter": "walker/sensors_velocimeter",
            "walker/world_zaxis": "walker/world_zaxis",
        }
        if global_observables:
            env = TargetsPositionWrapper(
                env,
                task._maze_arena.xy_scale,
                task._maze_arena.maze.width,
                task._maze_arena.maze.height,
            )
            env = AgentPositionWrapper(
                env,
                task._maze_arena.xy_scale,
                task._maze_arena.maze.width,
                task._maze_arena.maze.height,
            )
            env = MazeLayoutWrapper(env)
            obs_mapping = dict(
                obs_mapping,
                **{
                    "agent_pos": "agent_pos",
                    "agent_dir": "agent_dir",
                    # "targets_vec": "targets_vec",
                    # "targets_pos": "targets_pos",
                    "target_vec": "target_vec",
                    "target_pos": "target_pos",
                    # "maze_layout": "maze_layout",
                }
            )

        env = RemapObservationWrapper(env, obs_mapping)

    if target_color_in_image:
        env = TargetColorAsBorderWrapper(env)

    if show_path:
        env = PathToTargetWrapper(env)
        env = DrawMinimapWrapper(env)

    if image_only_obs:
        assert (
            target_color_in_image
        ), "Image-only observation only makes sense with target_color_in_image"
        env = ImageOnlyObservationWrapper(env)

    if discrete_actions:
        env = DiscreteActionSetWrapper(
            env,
            [
                np.array([0.0, 0.0]),  # noop
                np.array([-1.0, 0.0]),  # forward
                np.array([0.0, -1.0]),  # left
                np.array([0.0, +1.0]),  # right
                np.array([-1.0, -1.0]),  # forward + left
                np.array([-1.0, +1.0]),  # forward + right
            ],
        )

    return env
