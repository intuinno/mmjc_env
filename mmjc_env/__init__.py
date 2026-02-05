from gymnasium.envs.registration import register

# register(
#     id="vertebrate_env/VertebrateEnv-v0",
#     entry_point="mmjc_env.envs:VertebrateEnv",
# )

# register(
#     id="vertebrate_env/MMJCENV-v0",
#     entry_point="mmjc_env.envs:MMJCENV",
# )

register(
    id="MMJC-v0",
    entry_point="mmjc_env.envs:MMJCENV",
)

register(
    id="MMJC-13x13-v0",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "maze_size": 13,
        "num_targets": 5,
        "time_limit": 750,
    },
)

register(
    id="mmjc-easy",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "optional_reward": True,
        "targets_per_room": 10,
    },
)

register(
    id="mmjc-explore",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "maze_size": 13,
        "num_targets": 5,
        "time_limit": 750,
        "exploration_reward": True,
        "optional_reward": True,
    },
)

register(
    id="mmjc-explore-only",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "maze_size": 13,
        "num_targets": 5,
        "time_limit": 750,
        "exploration_reward": True,
        "optional_reward": False,
        "target_reward": False,
    },
)

register(
    id="mmjc-13",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "maze_size": 13,
        "num_targets": 5,
        "time_limit": 750,
        "exploration_reward": False,
        "optional_reward": False,
        "target_reward": True,
    },
)


register(
    id="mmjc-9",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "maze_size": 9,
        "num_targets": 3,
        "time_limit": 250,
        "exploration_reward": False,
        "optional_reward": False,
        "target_reward": True,
    },
)

register(
    id="mmjc-low-navigation",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "maze_size": 13,
        "num_targets": 5,
        "time_limit": 50,
        "exploration_reward": True,
        "optional_reward": False,
        "target_reward": False,
    },
)


register(
    id="mmjc-low-navigation-no-wall",
    entry_point="mmjc_env.envs:MMJCENV",
    kwargs={
        "maze_size": 13,
        "num_targets": 5,
        "time_limit": 50,
        "exploration_reward": True,
        "optional_reward": False,
        "target_reward": False,
        "targets_per_room": 10,
        "room_min_size": 13,
        "room_max_size": 13,
    },
)

# Taxi Navigation Environment - Ant on flat ground with goal switching (3 goals)
register(
    id="TaxiNavigation-v0",
    entry_point="mmjc_env.envs:TaxiNavigationEnv",
)

# Taxi Navigation Environment with 4 goals (FORWARD, ROTATE_CW, ROTATE_CCW, BACKWARD)
register(
    id="TaxiNavigation-v1",
    entry_point="mmjc_env.envs:TaxiNavigation4GoalEnv",
)

# Taxi Navigation with 4 goals and windowed distance/angle reward (40-step window)
register(
    id="TaxiNavigation-v2",
    entry_point="mmjc_env.envs:TaxiNavigation4GoalDistanceEnv",
)

# Taxi Navigation with 4 goals, 2D goal vector encoding, windowed distance/angle reward
register(
    id="TaxiNavigation-v3",
    entry_point="mmjc_env.envs:TaxiNavigation4GoalVectorEnv",
)