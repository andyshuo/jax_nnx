"""ProcGen environment construction.

``procgen`` is imported inside :func:`make_procgen` so this module stays importable
without the C++ extension. The wrappers are implemented here rather than reused from
``gym``: ProcGen speaks the pre-0.26 vector API (``reset()`` returns observations,
``step()`` returns a list of info dicts), while ``gym >= 0.26`` wrappers assume the
new single-env API and fail on it.
"""

import numpy as np


class _Wrapper:
    """Minimal vector-env wrapper base that copies the spaces through."""

    def __init__(self, env):
        self.env = env
        for attr in (
            'action_space',
            'observation_space',
            'single_action_space',
            'single_observation_space',
            'is_vector_env',
            'num_envs',
        ):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        return self.env.close()


class ExtractRGB(_Wrapper):
    """ProcGen observations are dicts; keep the ``rgb`` image only."""

    def reset(self):
        return self.env.reset()['rgb']

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs['rgb'], reward, done, info


class EpisodeStats(_Wrapper):
    """Accumulate episode return/length and expose them as ``info[i]['episode']``."""

    def __init__(self, env):
        super().__init__(env)
        self._returns = np.zeros(env.num_envs, dtype=np.float64)
        self._lengths = np.zeros(env.num_envs, dtype=np.int64)

    def reset(self):
        self._returns[:] = 0.0
        self._lengths[:] = 0
        return self.env.reset()

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        self._returns += reward
        self._lengths += 1

        info = list(info)
        for index, finished in enumerate(done):
            if not finished:
                continue
            entry = dict(info[index]) if index < len(info) else {}
            entry['episode'] = {'r': float(self._returns[index]), 'l': int(self._lengths[index])}
            if index < len(info):
                info[index] = entry
            else:
                info.append(entry)
            self._returns[index] = 0.0
            self._lengths[index] = 0
        return obs, reward, done, info


class NormalizeReward(_Wrapper):
    """Scale rewards by the running standard deviation of the discounted return."""

    def __init__(self, env, gamma=0.999, epsilon=1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self._discounted = np.zeros(env.num_envs, dtype=np.float64)
        self._count = 0.0
        self._mean = 0.0
        self._var = 1.0

    def _update(self, values):
        count = values.size
        batch_mean = float(values.mean())
        batch_var = float(values.var())
        total = self._count + count
        delta = batch_mean - self._mean
        self._mean = self._mean + delta * count / total
        self._var = (self._var * self._count + batch_var * count + delta**2 * self._count * count / total) / total
        self._count = total

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        self._discounted = self._discounted * self.gamma + reward
        self._update(self._discounted)
        self._discounted[done.astype(bool)] = 0.0
        return obs, reward / np.sqrt(self._var + self.epsilon), done, info


class ClipReward(_Wrapper):
    def __init__(self, env, low=-10.0, high=10.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, np.clip(reward, self.low, self.high), done, info


def make_procgen(config):
    """Build a vectorised, reward-normalised ProcGen environment."""
    import procgen

    envs = procgen.ProcgenEnv(**config.env_config)
    envs = ExtractRGB(envs)

    envs.single_action_space = envs.action_space
    # ProcGen exposes a mapping observation space, but ``gym.spaces.Dict`` is not a
    # dict subclass, so try the key instead of type-checking.
    try:
        envs.single_observation_space = envs.observation_space['rgb']
    except (TypeError, KeyError):
        envs.single_observation_space = envs.observation_space
    envs.is_vector_env = True
    envs.num_envs = config.env_config.num_envs

    envs = EpisodeStats(envs)
    envs = NormalizeReward(envs, gamma=config.gamma)
    envs = ClipReward(envs)

    assert hasattr(envs.action_space, 'n'), 'only a discrete action space is supported'
    return envs
