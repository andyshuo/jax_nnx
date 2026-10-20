"""ProcGen environment construction.

``procgen`` and ``gym`` are imported *inside* :func:`make_procgen` so that this
module stays importable on machines (and in CI) where those packages are not
installed. Only actually building an environment requires them.
"""

import numpy as np


def make_procgen(config):
    """Build a vectorised, reward-normalised ProcGen environment.

    Parameters
    ----------
    config:
        A ``StrConfig`` instance from one of the ``ppo_*`` packages. It must expose
        ``env_config``, ``capture_video``, ``gamma`` and ``wandb_config.name``.
    """
    import gym
    import procgen  # noqa: F401  (imported for its env registration side effects)

    envs = procgen.ProcgenEnv(**config.env_config)
    envs = gym.wrappers.TransformObservation(envs, lambda obs: obs['rgb'])

    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space['rgb']
    envs.is_vector_env = True

    envs = gym.wrappers.RecordEpisodeStatistics(envs)
    if config.capture_video:
        envs = gym.wrappers.RecordVideo(envs, f'videos/{config.wandb_config.name}')

    envs = gym.wrappers.NormalizeReward(envs, gamma=config.gamma)
    envs = gym.wrappers.TransformReward(envs, lambda reward: np.clip(reward, -10, 10))

    assert isinstance(envs.single_action_space, gym.spaces.Discrete), 'only discrete action space is supported'
    return envs
