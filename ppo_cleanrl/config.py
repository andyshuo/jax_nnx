"""Typed configuration for the ResNet PPO variant.

Every nested dataclass uses ``field(default_factory=...)`` so that separate
``Config`` instances never share the same mutable sub-config object. This matters
because the agents write derived values (``in_features``, ``rngs``) back into the
config objects during construction.
"""

from dataclasses import field

import jax.numpy as jnp
from chex import dataclass
from flax import nnx

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8


@dataclass
class EnvConfig:
    env_name: str = 'jumper'
    num_envs: int = 64
    num_levels: int = 200
    start_level: int = 0
    distribution_mode: str = 'easy'


@dataclass
class WandbConfig:
    name: str = 'ppo_jumper'
    entity: str | None = None
    project: str = 'jax_nnx'
    tags: list[str] = field(default_factory=lambda: ['cleanrl', 'ppo', 'procgen'])


@dataclass
class StrConfig:
    env_config: EnvConfig = field(default_factory=EnvConfig)
    wandb_config: WandbConfig = field(default_factory=WandbConfig)
    capture_video: bool = False
    gamma: float = 0.999


@dataclass
class ResNetConfig:
    obs_shape: tuple[int, int, int] = (64, 64, 3)
    hidden_dims: list[int] = field(default_factory=lambda: [16, 32, 32])
    rngs: nnx.Rngs | None = None


@dataclass
class ProjConfig:
    in_features: int = 2048
    out_features: int = 256
    rngs: nnx.Rngs | None = None
    dtype: jnp.dtype = bf16
    param_dtype: jnp.dtype = f32


@dataclass
class ActorConfig:
    in_features: int = 256
    out_features: int = 15
    rngs: nnx.Rngs | None = None
    dtype: jnp.dtype = bf16
    param_dtype: jnp.dtype = f32


@dataclass
class CriticConfig:
    in_features: int = 256
    out_features: int = 1
    rngs: nnx.Rngs | None = None
    dtype: jnp.dtype = bf16
    param_dtype: jnp.dtype = f32


@dataclass
class AgentConfig:
    resnet_config: ResNetConfig = field(default_factory=ResNetConfig)
    act_dim: int = 16
    critic_dim: int = 1
    proj_dim: int = 256
    proj_config: ProjConfig = field(default_factory=ProjConfig)
    actor_config: ActorConfig = field(default_factory=ActorConfig)
    critic_config: CriticConfig = field(default_factory=CriticConfig)


@dataclass
class TrainingConfig:
    total_timesteps: int = int(25e6)
    learning_rate: float = 5e-4
    num_steps: int = 256
    num_envs: int = 64
    num_minibatches: int = 8
    update_epochs: int = 3
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    anneal_lr: bool = False
    gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0


@dataclass
class Config:
    seed: int = 1
    capture_video: bool = False
    sampling_rng: int = 42
    categorical_rng: int = 43
    agent_config: AgentConfig = field(default_factory=AgentConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
