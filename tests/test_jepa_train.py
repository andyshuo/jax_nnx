"""One full JEPA PPO update on a synthetic rollout (slow: builds and compiles the agent)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

pytestmark = pytest.mark.slow

ACTION_DIM = 6
OBS_SHAPE = (16, 16, 3)


@pytest.fixture
def jepa_config():
    """Tiny 16x16 config (``tests/conftest.py`` cannot be extended for a new variant)."""
    from ppo_jepa.config import Config

    config = Config()
    resnet = config.agent_config.resnet_config
    resnet.obs_shape = OBS_SHAPE
    resnet.hidden_dims = [4, 8]

    config.agent_config.proj_config.out_features = 16
    config.agent_config.actor_config.in_features = 16
    config.agent_config.actor_config.out_features = ACTION_DIM
    config.agent_config.critic_config.in_features = 16

    tc = config.training_config
    tc.num_envs = 2
    tc.num_steps = 8
    tc.num_minibatches = 1
    tc.batch_size = tc.num_envs * tc.num_steps
    tc.minibatch_size = tc.batch_size
    tc.update_epochs = 1
    return config


def build_agent(config, rngs):
    """Mirror of ``main.build_model`` for this variant."""
    from ppo_jepa.agent import PPOAgent
    from ppo_jepa.utils import get_optimizer

    agent = PPOAgent(config.agent_config, rngs, shift_px=config.shift_px)
    optimizer = get_optimizer(config, agent)
    graphdef, state = nnx.split((agent, optimizer))
    return agent, optimizer, graphdef, state


def build_rollout(config, action_dim=ACTION_DIM, seed=0):
    """Same synthetic (num_steps, num_envs) rollout the shared smoke tests use."""
    tc = config.training_config
    num_steps, num_envs = tc.num_steps, tc.num_envs
    rng = jax.random.PRNGKey(seed)

    obs = jax.random.randint(rng, (num_steps, num_envs) + OBS_SHAPE, 0, 256, dtype=jnp.uint8)
    actions = jax.random.randint(rng, (num_steps, num_envs), 0, action_dim, dtype=jnp.int32)

    return {
        'obs': obs,
        'actions': actions.astype(jnp.float32),
        'logprobs': -jnp.ones((num_steps, num_envs)),
        'values': jax.random.normal(rng, (num_steps, num_envs)),
        'rewards': jax.random.normal(rng, (num_steps, num_envs)),
        'dones': jnp.zeros((num_steps, num_envs)),
        'next_obs': jax.random.randint(rng, (num_envs,) + OBS_SHAPE, 0, 256, dtype=jnp.uint8),
        'next_done': jnp.zeros((num_envs,)),
    }


def test_train_runs_one_update_and_reports_finite_metrics(jepa_config, rngs_factory, single_device_mesh):
    from ppo_jepa.utils import train

    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)

    _, _, graphdef, state = build_agent(jepa_config, rngs_factory(0))
    before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(state)]

    loss, new_state, metrics = train(graphdef, state, build_rollout(jepa_config), mesh, jepa_config, rng)

    after = [np.asarray(leaf) for leaf in jax.tree.leaves(new_state)]
    assert jnp.isfinite(loss)
    assert 'Losses/aux_loss' in metrics
    for key, value in metrics.items():
        assert np.isfinite(np.asarray(value)), key
    # The auxiliary term must actually take part in the update.
    assert np.asarray(metrics['Losses/aux_loss']) > 0.0
    assert any(not np.array_equal(b, a) for b, a in zip(before, after, strict=True))


def test_aux_coefficient_reaches_the_loss(jepa_config, rngs_factory):
    from ppo_jepa.agent import PPOAgent
    from ppo_jepa.utils import loss_fn

    agent = PPOAgent(jepa_config.agent_config, rngs_factory(0), shift_px=jepa_config.shift_px)
    batch = {
        'mb_obs': jax.random.randint(jax.random.PRNGKey(0), (4,) + OBS_SHAPE, 0, 256, dtype=jnp.uint8),
        'mb_actions': jnp.zeros((4,), dtype=jnp.int32),
        'mb_logprobs': -jnp.ones((4,)),
        'mb_advantages': jax.random.normal(jax.random.PRNGKey(1), (4,)),
        'mb_returns': jax.random.normal(jax.random.PRNGKey(2), (4,)),
        'mb_values': jnp.zeros((4,)),
        'rng': jax.random.PRNGKey(3),
    }

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent, batch, aux_coef=jepa_config.aux_coef)
    off, _ = loss_fn(agent, batch, aux_coef=0.0)

    pg_loss, v_loss, entropy_loss, aux_loss, logratio, ratio = aux
    assert len(aux) == 6
    assert jnp.isfinite(aux_loss) and aux_loss > 0.0
    assert jnp.allclose(loss - off, jepa_config.aux_coef * aux_loss, rtol=1e-2, atol=1e-3)
    assert all(jnp.isfinite(leaf).all() for leaf in jax.tree.leaves(grads))
