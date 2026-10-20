"""End-to-end checks on a real update step and on rollout collection.

Everything here runs on a single-device mesh so it is independent of how many GPUs
the host has.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

pytestmark = pytest.mark.slow

VARIANTS = {
    'cleanrl': {'fixture': 'cleanrl_config', 'obs_shape': (16, 16, 3)},
    'vae': {'fixture': 'vae_config', 'obs_shape': (16, 16, 3)},
}


def build_agent(kind, config, rngs):
    agent_module = __import__(f'ppo_{kind}.agent', fromlist=['PPOAgent'])
    utils = __import__(f'ppo_{kind}.utils', fromlist=['*'])
    agent = agent_module.PPOAgent(config.agent_config, rngs)
    optimizer = utils.get_optimizer(config, agent)
    graphdef, state = nnx.split((agent, optimizer))
    return agent, optimizer, graphdef, state, utils


def build_rollout(config, obs_shape, action_dim=6, seed=0):
    tc = config.training_config
    num_steps, num_envs = tc.num_steps, tc.num_envs
    rng = jax.random.PRNGKey(seed)

    obs = jax.random.randint(rng, (num_steps, num_envs) + obs_shape, 0, 256, dtype=jnp.uint8)
    actions = jax.random.randint(rng, (num_steps, num_envs), 0, action_dim, dtype=jnp.int32)

    return {
        'obs': obs,
        'actions': actions.astype(jnp.float32),
        'logprobs': -jnp.ones((num_steps, num_envs)),
        'values': jax.random.normal(rng, (num_steps, num_envs)),
        'rewards': jax.random.normal(rng, (num_steps, num_envs)),
        'dones': jnp.zeros((num_steps, num_envs)),
        'next_obs': jax.random.randint(rng, (num_envs,) + obs_shape, 0, 256, dtype=jnp.uint8),
        'next_done': jnp.zeros((num_envs,)),
    }


@pytest.mark.parametrize('kind', list(VARIANTS))
def test_train_updates_parameters(kind, request, rngs_factory, single_device_mesh):
    config = request.getfixturevalue(VARIANTS[kind]['fixture'])
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)
    rngs = rngs_factory(0)

    _, _, graphdef, state, utils = build_agent(kind, config, rngs)
    before = [np.asarray(leaf).copy() for leaf in jax.tree.leaves(state)]

    loss, new_state, metrics = utils.train(
        graphdef, state, build_rollout(config, VARIANTS[kind]['obs_shape']), mesh, config, rng
    )

    after = [np.asarray(leaf) for leaf in jax.tree.leaves(new_state)]
    assert jnp.isfinite(loss)
    assert any(not np.array_equal(b, a) for b, a in zip(before, after, strict=True))
    assert isinstance(metrics, dict)


@pytest.mark.parametrize('kind', list(VARIANTS))
def test_train_metrics_are_finite(kind, request, rngs_factory, single_device_mesh):
    config = request.getfixturevalue(VARIANTS[kind]['fixture'])
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)

    _, _, graphdef, state, utils = build_agent(kind, config, rngs_factory(0))
    _, _, metrics = utils.train(graphdef, state, build_rollout(config, VARIANTS[kind]['obs_shape']), mesh, config, rng)

    for key, value in metrics.items():
        if 'Images' in key:
            continue
        assert np.isfinite(np.asarray(value)), key


def test_vae_train_produces_reconstruction_image(vae_config, rngs_factory, single_device_mesh):
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)

    _, _, graphdef, state, utils = build_agent('vae', vae_config, rngs_factory(0))
    _, _, metrics = utils.train(graphdef, state, build_rollout(vae_config, (16, 16, 3)), mesh, vae_config, rng)

    image = metrics['Images/recon_x']
    assert image.shape == (16, 32, 3)  # obs | reconstruction, side by side
    assert np.isfinite(image).all()


def test_cleanrl_train_has_no_reconstruction_image(cleanrl_config, rngs_factory, single_device_mesh):
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)

    _, _, graphdef, state, utils = build_agent('cleanrl', cleanrl_config, rngs_factory(0))
    _, _, metrics = utils.train(graphdef, state, build_rollout(cleanrl_config, (16, 16, 3)), mesh, cleanrl_config, rng)

    assert 'Images/recon_x' not in metrics


def test_train_handles_rollout_lengths_other_than_256(cleanrl_config, rngs_factory, single_device_mesh):
    """Regression: GAE used to hard-code ``num_steps=256``."""
    cleanrl_config.training_config.num_steps = 5
    cleanrl_config.training_config.num_envs = 2
    cleanrl_config.training_config.batch_size = 10
    cleanrl_config.training_config.minibatch_size = 10
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)

    _, _, graphdef, state, utils = build_agent('cleanrl', cleanrl_config, rngs_factory(0))
    loss, _, metrics = utils.train(
        graphdef, state, build_rollout(cleanrl_config, (16, 16, 3)), mesh, cleanrl_config, rng
    )

    assert jnp.isfinite(loss)
    assert np.isfinite(np.asarray(metrics['Losses/explained_var']))


def test_train_raises_without_any_minibatch(cleanrl_config, rngs_factory, single_device_mesh):
    cleanrl_config.training_config.batch_size = 0
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)

    _, _, graphdef, state, utils = build_agent('cleanrl', cleanrl_config, rngs_factory(0))

    with pytest.raises(ValueError, match='no minibatch'):
        utils.train(graphdef, state, build_rollout(cleanrl_config, (16, 16, 3), seed=1), mesh, cleanrl_config, rng)


# --------------------------------------------------------------------------- #
# env_step
# --------------------------------------------------------------------------- #
class FakeVecEnv:
    """Minimal vectorised env: enough for ``env_step`` to consume."""

    def __init__(self, obs_shape, num_envs):
        self.obs_shape = obs_shape
        self.num_envs = num_envs
        self.stepped = 0

    def reset(self):
        return np.zeros((self.num_envs,) + self.obs_shape, dtype=np.uint8)

    def step(self, actions):
        assert actions.shape == (self.num_envs,)
        self.stepped += 1
        obs = np.full((self.num_envs,) + self.obs_shape, 7, dtype=np.uint8)
        rewards = np.ones(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        info = [{'episode': {'r': 1.0, 'l': 10}}] + [{} for _ in range(self.num_envs - 1)]
        return obs, rewards, dones, info


class RecordingLogger:
    enabled = True

    def __init__(self):
        self.records = []

    def log(self, data, step=None):
        self.records.append((data, step))

    def image(self, path):
        return str(path)

    def finish(self):
        pass


@pytest.fixture
def fake_episode(cleanrl_config):
    from ppo_cleanrl.episode import make_episode

    return make_episode(obs_shape=(16, 16, 3), act_dim=(), num_steps=3, num_envs=2)


def test_env_step_fills_the_episode_buffer(cleanrl_config, rngs_factory, single_device_mesh, fake_episode):
    cleanrl_config.training_config.num_steps = 3
    cleanrl_config.training_config.num_envs = 2
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)
    envs = FakeVecEnv((16, 16, 3), 2)

    _, _, graphdef, state, utils = build_agent('cleanrl', cleanrl_config, rngs_factory(0))
    global_step, episode = utils.env_step(
        0, envs, fake_episode, envs.reset(), np.zeros(2), mesh, graphdef, state, rng, cleanrl_config
    )

    assert global_step == cleanrl_config.training_config.num_steps * cleanrl_config.training_config.num_envs
    assert envs.stepped == 3
    assert np.all(episode.rewards[:] == 1.0)
    assert np.all(episode.obs[0] == 0)
    assert np.all(episode.obs[1] == 7)
    assert np.all(np.isfinite(episode.values[:]))


def test_env_step_logs_episode_statistics(cleanrl_config, rngs_factory, single_device_mesh, fake_episode):
    cleanrl_config.training_config.num_steps = 2
    cleanrl_config.training_config.num_envs = 2
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)
    envs = FakeVecEnv((16, 16, 3), 2)
    logger = RecordingLogger()

    _, _, graphdef, state, utils = build_agent('cleanrl', cleanrl_config, rngs_factory(0))
    utils.env_step(
        0, envs, fake_episode, envs.reset(), np.zeros(2), mesh, graphdef, state, rng, cleanrl_config, logger=logger
    )

    assert len(logger.records) == 2
    payload, _ = logger.records[0]
    assert payload['charts/episodic_return'] == 1.0
    assert payload['charts/episodic_length'] == 10


def test_env_step_works_without_a_logger(cleanrl_config, rngs_factory, single_device_mesh, fake_episode):
    cleanrl_config.training_config.num_steps = 1
    cleanrl_config.training_config.num_envs = 2
    mesh, _ = single_device_mesh
    rng = jax.random.PRNGKey(0)
    envs = FakeVecEnv((16, 16, 3), 2)

    _, _, graphdef, state, utils = build_agent('cleanrl', cleanrl_config, rngs_factory(0))
    utils.env_step(0, envs, fake_episode, envs.reset(), np.zeros(2), mesh, graphdef, state, rng, cleanrl_config)
