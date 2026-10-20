"""Typed configuration for the VAE-encoder PPO variant.

Every nested dataclass uses ``field(default_factory=...)`` so that separate
``Config`` instances never share the same mutable sub-config object. This matters
because the models write derived values (``in_features``/``out_features``/``rngs``)
back into the config objects while they are being constructed.
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
    name: str = 'ppo_vae_jumper'
    entity: str | None = None
    project: str = 'jax_nnx'
    tags: list[str] = field(default_factory=lambda: ['vae', 'ppo', 'procgen'])


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
class VAEProjConfig:
    in_features: int = 0
    out_features: int = 0
    rngs: nnx.Rngs | None = None
    dtype: jnp.dtype = bf16
    param_dtype: jnp.dtype = f32


@dataclass
class ConvConfig:
    in_features: int = 0
    out_features: int = 0
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (1, 1)
    padding: str = 'SAME'
    rngs: nnx.Rngs | None = None
    dtype: jnp.dtype = bf16
    param_dtype: jnp.dtype = f32


@dataclass
class ConvTransposeConfig:
    in_features: int = 0
    out_features: int = 0
    kernel_size: tuple[int, int] = (3, 3)
    strides: tuple[int, int] = (2, 2)
    padding: str = 'SAME'
    rngs: nnx.Rngs | None = None
    dtype: jnp.dtype = bf16
    param_dtype: jnp.dtype = f32


@dataclass
class ResidualBlockConfig:
    conv1_config: ConvConfig = field(default_factory=ConvConfig)
    conv2_config: ConvConfig = field(default_factory=ConvConfig)


@dataclass
class ConvSeqConfig:
    shape: tuple[int, int, int] = (64, 64, 3)
    out_features: int = 0
    rngs: nnx.Rngs | None = None
    conv_config: ConvConfig = field(default_factory=ConvConfig)
    res_block_config: ResidualBlockConfig = field(default_factory=ResidualBlockConfig)


@dataclass
class ResidualBlockTransposeConfig:
    conv1_config: ConvTransposeConfig = field(default_factory=lambda: ConvTransposeConfig(strides=(1, 1)))
    conv2_config: ConvTransposeConfig = field(default_factory=lambda: ConvTransposeConfig(strides=(1, 1)))


@dataclass
class ConvTransposeSeqConfig:
    shape: tuple[int, int, int] = (64, 64, 3)
    out_features: int = 0
    rngs: nnx.Rngs | None = None
    conv_transpose_config: ConvTransposeConfig = field(default_factory=ConvTransposeConfig)
    res_block_transpose_config: ResidualBlockTransposeConfig = field(default_factory=ResidualBlockTransposeConfig)


@dataclass
class VAEConfig:
    shape: tuple[int, int, int] = (64, 64, 3)
    enc_out_features: list[int] = field(default_factory=lambda: [16, 32, 32])
    dec_out_features: list[int] = field(default_factory=lambda: [32, 16, 3])
    num_categories: int = 16
    categorical_dim: int = 32
    temperature: float = 1.0
    vae_proj_config: VAEProjConfig = field(default_factory=VAEProjConfig)
    conv_seq_config: ConvSeqConfig = field(default_factory=ConvSeqConfig)
    conv_transpose_seq_config: ConvTransposeSeqConfig = field(default_factory=ConvTransposeSeqConfig)
    rngs: nnx.Rngs | None = None


@dataclass
class AgentConfig:
    vae_config: VAEConfig = field(default_factory=VAEConfig)
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
    num_minibatches: int = 1
    update_epochs: int = 16
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    anneal_lr: bool = False
    gamma: float = 0.999
    gae_lambda: float = 0.95
    max_grad_norm: float = 1.0
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
    # VAE loss weights: the original code hard-coded both coefficients to 1.0
    # inside the update step, which made them impossible to tune.
    recon_coef: float = 1.0
    kl_coef: float = 1.0
