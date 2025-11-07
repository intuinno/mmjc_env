import gymnasium as gym

from gymnasium import spaces
from dm_env import specs
import numpy as np
from typing import Any


def _convert_to_space(spec: Any) -> gym.Space:
    # Inverse of acme.gym_wrappers._convert_to_spec

    if isinstance(spec, specs.DiscreteArray):
        return spaces.Discrete(spec.num_values)

    if isinstance(spec, specs.BoundedArray):
        return spaces.Box(
            shape=spec.shape,
            dtype=spec.dtype,
            low=spec.minimum.item() if len(spec.minimum.shape) == 0 else spec.minimum,
            high=spec.maximum.item() if len(spec.maximum.shape) == 0 else spec.maximum,
        )

    if isinstance(spec, specs.Array):
        return spaces.Box(shape=spec.shape, dtype=spec.dtype, low=-np.inf, high=np.inf)

    if isinstance(spec, tuple):
        return spaces.Tuple(_convert_to_space(s) for s in spec)

    if isinstance(spec, dict):
        return spaces.Dict(
            {key: _convert_to_space(value) for key, value in spec.items() if value.shape != (0,)}
        )

    raise ValueError(f"Unexpected spec: {spec}")
