"""Rollout buffers: shapes and dtypes for both variants."""

import numpy as np
import pytest


@pytest.mark.parametrize('package', ['ppo_cleanrl', 'ppo_vae'])
def test_make_episode_shapes(package):
    module = __import__(f'{package}.episode', fromlist=['make_episode'])
    episode = module.make_episode(obs_shape=(16, 16, 3), act_dim=(), num_steps=8, num_envs=2)

    assert episode.obs.shape == (8, 2, 16, 16, 3)
    assert episode.actions.shape == (8, 2)
    assert episode.logprobs.shape == (8, 2)
    assert episode.rewards.shape == (8, 2)
    assert episode.dones.shape == (8, 2)
    assert episode.values.shape == (8, 2)


@pytest.mark.parametrize('package', ['ppo_cleanrl', 'ppo_vae'])
def test_make_episode_supports_vector_action_dims(package):
    module = __import__(f'{package}.episode', fromlist=['make_episode'])
    episode = module.make_episode(obs_shape=(8, 8, 3), act_dim=(4,), num_steps=3, num_envs=5)

    assert episode.actions.shape == (3, 5, 4)


@pytest.mark.parametrize('package', ['ppo_cleanrl', 'ppo_vae'])
def test_make_episode_dtypes(package):
    module = __import__(f'{package}.episode', fromlist=['make_episode'])
    episode = module.make_episode(obs_shape=(8, 8, 3), act_dim=(), num_steps=2, num_envs=2)

    assert episode.obs.dtype == np.uint8
    assert episode.actions.dtype == np.float32
    assert episode.rewards.dtype == np.float32
    assert episode.dones.dtype == np.float32
    assert episode.values.dtype == np.float32
    assert episode.logprobs.dtype == np.float32


@pytest.mark.parametrize('package', ['ppo_cleanrl', 'ppo_vae'])
def test_make_episode_is_zero_initialised(package):
    module = __import__(f'{package}.episode', fromlist=['make_episode'])
    episode = module.make_episode(obs_shape=(8, 8, 3), act_dim=(), num_steps=2, num_envs=2)

    for field in ('obs', 'actions', 'logprobs', 'rewards', 'dones', 'values'):
        assert not np.any(getattr(episode, field)), field


@pytest.mark.parametrize('package', ['ppo_cleanrl', 'ppo_vae'])
def test_episode_buffer_is_writable(package):
    module = __import__(f'{package}.episode', fromlist=['make_episode'])
    episode = module.make_episode(obs_shape=(4, 4, 3), act_dim=(), num_steps=1, num_envs=1)

    episode.rewards[0] = 1.5
    episode.dones[0] = 1.0

    assert episode.rewards[0] == pytest.approx(1.5)
    assert episode.dones[0] == pytest.approx(1.0)
