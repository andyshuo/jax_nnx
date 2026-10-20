"""ProcGen wrapper construction, using fake ``gym``/``procgen`` modules."""

import sys
import types

import numpy as np
import pytest

from shared.make_envs import make_procgen

RECORD: list[str] = []


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeBox:
    def __init__(self, shape, low=0, high=255, dtype='uint8'):
        self.shape = shape
        self.low = low
        self.high = high
        self.dtype = dtype


class FakeDiscrete:
    def __init__(self, n):
        self.n = n
        self.shape = ()


class _Wrapper:
    name = 'wrapper'

    def __init__(self, env, *args, **kwargs):
        RECORD.append(self.name)
        self.env = env
        self.args = args
        self.kwargs = kwargs
        for attr in (
            'action_space',
            'observation_space',
            'single_action_space',
            'single_observation_space',
            'is_vector_env',
        ):
            if hasattr(env, attr):
                setattr(self, attr, getattr(env, attr))

    def reset(self):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)


class TransformObservation(_Wrapper):
    name = 'TransformObservation'

    def __init__(self, env, func):
        super().__init__(env, func)
        self.func = func

    def reset(self):
        return self.func(self.env.reset())

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return self.func(obs), reward, done, info


class RecordEpisodeStatistics(_Wrapper):
    name = 'RecordEpisodeStatistics'


class RecordVideo(_Wrapper):
    name = 'RecordVideo'


class NormalizeReward(_Wrapper):
    name = 'NormalizeReward'


class TransformReward(_Wrapper):
    name = 'TransformReward'

    def __init__(self, env, func):
        super().__init__(env, func)
        self.func = func

    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        return obs, self.func(reward), done, info


class FakeProcgenEnv:
    instances: list['FakeProcgenEnv'] = []

    def __init__(self, action_space=None, **kwargs):
        RECORD.append('ProcgenEnv')
        self.kwargs = kwargs
        self._action_space = action_space or FakeDiscrete(15)
        self.observation_space = {'rgb': FakeBox((64, 64, 3))}
        FakeProcgenEnv.instances.append(self)

    @property
    def action_space(self):
        return self._action_space

    def reset(self):
        num_envs = self.kwargs.get('num_envs', 1)
        return {'rgb': np.zeros((num_envs, 64, 64, 3), dtype=np.uint8)}

    def step(self, actions):
        num_envs = len(actions)
        obs = {'rgb': np.ones((num_envs, 64, 64, 3), dtype=np.uint8)}
        return obs, np.zeros(num_envs, np.float32), np.zeros(num_envs, np.float32), [{} for _ in range(num_envs)]


def install_fakes(monkeypatch, action_space=None):
    gym = types.ModuleType('gym')
    wrappers = types.ModuleType('gym.wrappers')
    spaces = types.ModuleType('gym.spaces')

    for cls in (TransformObservation, RecordEpisodeStatistics, RecordVideo, NormalizeReward, TransformReward):
        setattr(wrappers, cls.__name__, cls)

    spaces.Discrete = FakeDiscrete
    spaces.Box = FakeBox
    gym.wrappers = wrappers
    gym.spaces = spaces
    gym.Env = object

    procgen = types.ModuleType('procgen')

    def _procgen_env(**kwargs):
        return FakeProcgenEnv(action_space=action_space, **kwargs)

    procgen.ProcgenEnv = _procgen_env

    monkeypatch.setitem(sys.modules, 'gym', gym)
    monkeypatch.setitem(sys.modules, 'gym.wrappers', wrappers)
    monkeypatch.setitem(sys.modules, 'gym.spaces', spaces)
    monkeypatch.setitem(sys.modules, 'procgen', procgen)


@pytest.fixture(autouse=True)
def _reset_record():
    RECORD.clear()
    FakeProcgenEnv.instances.clear()
    yield
    RECORD.clear()
    FakeProcgenEnv.instances.clear()


@pytest.fixture
def strconfig():
    from ppo_cleanrl.config import StrConfig

    return StrConfig()


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_wrapper_order(monkeypatch, strconfig):
    install_fakes(monkeypatch)

    make_procgen(strconfig)

    assert RECORD == [
        'ProcgenEnv',
        'TransformObservation',
        'RecordEpisodeStatistics',
        'NormalizeReward',
        'TransformReward',
    ]


def test_env_config_is_forwarded(monkeypatch, strconfig):
    install_fakes(monkeypatch)
    strconfig.env_config.env_name = 'coinrun'
    strconfig.env_config.num_envs = 3

    make_procgen(strconfig)

    assert FakeProcgenEnv.instances[0].kwargs['env_name'] == 'coinrun'
    assert FakeProcgenEnv.instances[0].kwargs['num_envs'] == 3


def test_single_spaces_are_set(monkeypatch, strconfig):
    install_fakes(monkeypatch)

    envs = make_procgen(strconfig)

    assert envs.single_action_space is envs.action_space
    assert envs.single_observation_space.shape == (64, 64, 3)
    assert envs.is_vector_env is True


def test_observation_transform_extracts_rgb(monkeypatch, strconfig):
    install_fakes(monkeypatch)
    envs = make_procgen(strconfig)

    obs = envs.reset()
    stepped, *_ = envs.step(np.zeros(strconfig.env_config.num_envs, dtype=np.int32))

    assert obs.shape == (strconfig.env_config.num_envs, 64, 64, 3)
    assert stepped.shape == obs.shape


def test_rewards_are_clipped(monkeypatch, strconfig):
    install_fakes(monkeypatch)
    envs = make_procgen(strconfig)

    # Push the TransformReward clip function to its limits directly.
    clip = envs.func
    assert np.allclose(clip(np.array([-100.0, 0.0, 100.0])), [-10.0, 0.0, 10.0])


def test_non_discrete_action_space_is_rejected(monkeypatch, strconfig):
    install_fakes(monkeypatch, action_space=FakeBox((4,)))

    with pytest.raises(AssertionError):
        make_procgen(strconfig)


def test_capture_video_records_under_the_run_name(monkeypatch, strconfig):
    install_fakes(monkeypatch)
    strconfig.capture_video = True
    strconfig.wandb_config.name = 'my-run'

    envs = make_procgen(strconfig)

    assert 'RecordVideo' in RECORD
    video_wrapper = find_wrapper(envs, RecordVideo)
    assert video_wrapper is not None
    assert video_wrapper.args == ('videos/my-run',)


def test_capture_video_is_opt_in(monkeypatch, strconfig):
    install_fakes(monkeypatch)
    strconfig.capture_video = False

    make_procgen(strconfig)

    assert 'RecordVideo' not in RECORD


def test_make_envs_stays_importable_without_procgen():
    """The module must import even when ``procgen``/``gym`` are absent."""
    assert callable(make_procgen)


def find_wrapper(env, wrapper_type):
    while env is not None:
        if isinstance(env, wrapper_type):
            return env
        env = getattr(env, 'env', None)
    return None
