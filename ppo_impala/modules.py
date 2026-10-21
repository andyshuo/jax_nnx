"""Deep IMPALA-style CNN encoder for the IMPALA PPO variant.

The building blocks are the ones already used by the VAE encoder
(``ppo_vae.modules``): a ``ConvSequence`` is a convolution, a stride-2 max-pool and
two ``ResidualBlock``s, which is exactly one IMPALA stage. Only the plumbing is new
here: one stage per entry of ``hidden_dims``, then a flatten to ``output_dim``.
"""

import einops
import jax.numpy as jnp
from flax import nnx

from ppo_impala.config import ConvSeqConfig
from ppo_vae.modules import ConvSequence, ResidualBlock

__all__ = ['ConvSequence', 'ImpalaCNN', 'ResidualBlock']

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8


class ImpalaCNN(nnx.Module):
    """Deep IMPALA CNN: one conv sequence per entry of ``hidden_dims``.

    Every stage halves the spatial dimensions and keeps the channel count of its
    entry, so a ``(64, 64, 3)`` observation with ``hidden_dims=(16, 32, 32)`` ends
    up as an ``8 x 8 x 32`` feature map (``output_dim == 2048``).
    """

    def __init__(self, obs_shape, hidden_dims, rngs):
        self._input_shape = obs_shape
        conv_seqs = []
        for out_features in hidden_dims:
            # A fresh config per stage: ``ConvSequence`` writes the derived shapes
            # back into it (including its own ``res_block_config``), so the stages
            # must not share one mutable object.
            conv_seq = ConvSequence(
                **ConvSeqConfig(
                    shape=self._input_shape,
                    out_features=out_features,
                    rngs=rngs,
                )
            )
            self._input_shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.net = nnx.Sequential(*conv_seqs)
        self.output_dim = self._input_shape[0] * self._input_shape[1] * self._input_shape[2]

    def __call__(self, x):
        x = self.net(x)
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nnx.relu(x)
        return x
