"""Modules for the JEPA-encoder PPO variant.

The encoder is the plain ResNet from the cleanrl variant; the only new piece is
the predictor head that regresses the latent of a shifted view of the same
observation. ``ResNet`` is re-exported so ``ppo_jepa`` is usable on its own.
"""

import jax.numpy as jnp
from chex import dataclass
from flax import nnx

from ppo_cleanrl.modules import ResNet

__all__ = ['Predictor', 'PredictorConfig', 'ResNet']

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8


@dataclass
class PredictorConfig:
    """Shape/dtype of the latent predictor (``in_features == out_features``)."""

    in_features: int = 0
    out_features: int = 0
    #: Bottleneck width; 0 falls back to in_features.
    hidden_features: int = 0
    rngs: nnx.Rngs | None = None
    dtype: jnp.dtype = bf16
    param_dtype: jnp.dtype = f32


class Predictor(nnx.Module):
    """``Linear -> relu -> Linear`` head mapping a latent onto the same latent size.

    Both layers are square because the target lives in the encoder's own output
    space; the agent fills ``PredictorConfig`` in from ``ResNet.output_dim``.
    """

    def __init__(self, predictor_config):
        cfg = dict(**predictor_config)
        cfg.pop('hidden_features', None)
        hidden = predictor_config.hidden_features or predictor_config.in_features
        self.fc0 = nnx.Linear(**{**cfg, 'out_features': hidden})
        self.fc1 = nnx.Linear(**{**cfg, 'in_features': hidden, 'out_features': predictor_config.out_features})

    def __call__(self, x):
        x = nnx.relu(self.fc0(x))
        return self.fc1(x)
