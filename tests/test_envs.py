"""ProcGen env wrappers (self-contained, no gym wrappers involved)."""

import sys
import types

import numpy as np
import pytest

from shared.make_envs import ClipReward, EpisodeStats, ExtractRGB, NormalizeReward, make_procgen


class Box:
    def __init__(self, shape):
        self.shape = shape


class Discrete:
    def __init__(self, n):
        self.n = n


class FakeProcgenEnv:
    def __init__(self, num_envs=2, env_name='jumper', **kwargs):
        self.num_envs = num_envs
        self.kwargs = {'env_name': env_name, **kwargs}
        self.action_space = Discrete(15)
        self.observation_space = {'rgb': Box((64, 64, 3))}
        self.closed = False

    def reset(self):
        return {'rgb': np.zeros((self.num_envs, 64, 64, 3), dtype=np.uint8)}

    def step(self, actions):
        assert actions.shape == (self.num_envs,)
        obs = {'rgb': np.ones((self.num_envs, 64, 64, 3), dtype=np.uint8)}
        return obs, np.ones(self.num_envs, np.float32), np.zeros(self.num_envs, bool), [{}] * self.num_envs

    def close(self):
        self.closed = True


@pytest.fixture
def fake_env():
    return FakeProcgenEnv(num_envs=2)


@pytest.fixture
def strconfig():
    from ppo_cleanrl.config import StrConfig

    config = StrConfig()
    config.env_config.num_envs = 2
    return config


def test_extract_rgb_returns_images(fake_env):
    env = ExtractRGB(fake_env)

    obs = env.reset()
    assert obs.shape == (2, 64, 64, 3)

    stepped, _, _, _ = env.step(np.zeros(2, np.int32))
    assert stepped.shape == (2, 64, 64, 3)


def test_episode_stats_reports_return_and_length(fake_env):
    env = EpisodeStats(fake_env)
    env._returns[:] = 4.0
    env._lengths[:] = 7

    class DoneFirst(FakeProcgenEnv):
        def step(self, actions):
            obs, reward, _, info = super().step(actions)
            done = np.array([True, False])
            return obs, reward, done, info

    env.env = DoneFirst(num_envs=2)
    _, _, _, info = env.step(np.zeros(2, np.int32))

    assert info[0]['episode']['l'] == 8
    assert info[1] == {}
    assert env._returns[0] == 0.0


def test_normalize_reward_scales_and_resets(fake_env):
    env = NormalizeReward(fake_env, gamma=0.99)

    _, reward, done, _ = env.step(np.zeros(2, np.int32))

    assert reward.shape == (2,)
    assert np.isfinite(reward).all()
    assert np.all(env._discounted[done] == 0.0)


def test_clip_reward_bounds(fake_env):
    env = ClipReward(fake_env)

    class BigReward(FakeProcgenEnv):
        def step(self, actions):
            obs, _, _, info = super().step(actions)
            return obs, np.array([100.0, -100.0], np.float32), np.zeros(2, bool), info

    env.env = BigReward(num_envs=2)
    _, reward, _, _ = env.step(np.zeros(2, np.int32))

    assert np.allclose(reward, [10.0, -10.0])


def test_make_procgen_builds_the_chain(strconfig, monkeypatch):
    procgen = types.ModuleType('procgen')
    procgen.ProcgenEnv = FakeProcgenEnv
    monkeypatch.setitem(sys.modules, 'procgen', procgen)

    envs = make_procgen(strconfig)

    assert isinstance(envs, ClipReward)
    assert envs.num_envs == 2
    assert envs.single_action_space.n == 15
    assert envs.single_observation_space.shape == (64, 64, 3)
    assert envs.is_vector_env is True
    assert envs.reset().shape == (2, 64, 64, 3)
    envs.close()
    assert envs.env.env.env.env.closed is True
