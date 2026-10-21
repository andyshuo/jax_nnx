"""Rollout buffer shared by both PPO variants.

The two variants used to carry byte-identical copies of this module.
"""

import jax.numpy as jnp
import numpy as np
from chex import dataclass

f32 = jnp.float32
u8 = jnp.uint8


@dataclass
class Episode:
    obs: np.array
    actions: np.array
    logprobs: np.array
    rewards: np.array
    dones: np.array
    values: np.array


def make_episode(
    obs_shape: tuple[int, int, int],
    act_dim: int,
    num_steps: int,
    num_envs: int,
):
    """Allocate a zeroed rollout buffer of ``num_steps`` x ``num_envs``."""
    return Episode(
        obs=np.zeros((num_steps, num_envs) + obs_shape, dtype=u8),
        actions=np.zeros((num_steps, num_envs) + act_dim, dtype=f32),
        logprobs=np.zeros((num_steps, num_envs), dtype=f32),
        rewards=np.zeros((num_steps, num_envs), dtype=f32),
        dones=np.zeros((num_steps, num_envs), dtype=f32),
        values=np.zeros((num_steps, num_envs), dtype=f32),
    )
