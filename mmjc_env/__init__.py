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
