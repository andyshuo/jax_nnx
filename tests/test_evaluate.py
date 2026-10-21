"""Greedy policy evaluation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from shared.evaluate import evaluate


class FakeVecEnv:
    """Vectorised env that finishes an episode on every step."""

    def __init__(self, num_envs=2, obs_shape=(4, 4, 3)):
        self.num_envs = num_envs
        self.obs_shape = obs_shape

    def reset(self):
        return np.zeros((self.num_envs,) + self.obs_shape, dtype=np.uint8)

    def step(self, actions):
        assert actions.shape == (self.num_envs,)
        obs = np.ones((self.num_envs,) + self.obs_shape, dtype=np.uint8)
        rewards = np.ones(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        info = [{'episode': {'r': 1.0, 'l': 10}}] + [{} for _ in range(self.num_envs - 1)]
        return obs, rewards, dones, info


def greedy(graphdef, state, obs):
    del graphdef, state
    return jnp.zeros(obs.shape[0], dtype=jnp.int32)


def test_evaluate_reports_episode_returns(single_device_mesh):
    mesh, _ = single_device_mesh

    metrics = evaluate(FakeVecEnv(), mesh, None, None, greedy, max_steps=5)

    assert metrics['Eval/return'] == pytest.approx(1.0)
    assert metrics['Eval/episodes'] == 5.0
    assert metrics['Eval/steps'] == 5.0


def test_evaluate_is_nan_without_finished_episodes(single_device_mesh):
    class NoEpisodes(FakeVecEnv):
        def step(self, actions):
            obs, rewards, dones, _ = super().step(actions)
            return obs, rewards, dones, [{} for _ in actions]

    mesh, _ = single_device_mesh

    metrics = evaluate(NoEpisodes(), mesh, None, None, greedy, max_steps=3)

    assert np.isnan(metrics['Eval/return'])
    assert metrics['Eval/episodes'] == 0.0


@pytest.mark.parametrize('kind, fixture', [('cleanrl', 'cleanrl_config'), ('vae', 'vae_config')])
def test_act_greedy_is_in_range_and_deterministic(kind, fixture, request, rngs_factory):
    config = request.getfixturevalue(fixture)
    agent_module = __import__(f'ppo_{kind}.agent', fromlist=['PPOAgent'])
    agent = agent_module.PPOAgent(config.agent_config, rngs_factory(0))
    obs = jax.random.randint(jax.random.PRNGKey(0), (2, 16, 16, 3), 0, 256, dtype=jnp.uint8)

    first = agent.act_greedy(obs)
    second = agent.act_greedy(obs)

    assert np.array_equal(np.asarray(first), np.asarray(second))
    assert int(jnp.max(first)) < 6
    assert int(jnp.min(first)) >= 0
