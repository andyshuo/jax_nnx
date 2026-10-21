"""Greedy policy evaluation on a dedicated set of environments."""

import jax
import jax.numpy as jnp
import numpy as np
from jax import NamedSharding
from jax.sharding import PartitionSpec as P


def evaluate(envs, mesh, graphdef, state, greedy_action_fn, max_steps: int = 4096) -> dict:
    """Run the deterministic policy for ``max_steps`` and summarise episode returns.

    ``envs`` is a vectorised ProcGen environment with ``RecordEpisodeStatistics``
    already applied, so finished episodes arrive in ``info``. The policy and the
    environment are untouched: only ``obs`` is pushed to the device.
    """
    obs = np.array(envs.reset())
    returns = []

    for _ in range(max_steps):
        obs_device = jax.device_put(
            jnp.asarray(obs, dtype=jnp.uint8),
            NamedSharding(mesh, P('batch', None, None, None)),
        )
        action = greedy_action_fn(graphdef, state, obs_device)
        obs, _, _, info = envs.step(np.asarray(action))
        for item in info:
            if 'episode' in item:
                returns.append(float(item['episode']['r']))

    return {
        'Eval/return': float(np.mean(returns)) if returns else float('nan'),
        'Eval/episodes': float(len(returns)),
        'Eval/steps': float(max_steps),
    }
