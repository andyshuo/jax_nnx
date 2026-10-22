"""Deliberately small convolutional encoder for fast experiments.

Three 3x3 stride-2 convolutions with ``SAME`` padding halve the spatial grid at every
stage, so a 64x64 ProcGen frame becomes 8x8 with ``[8, 16, 16]`` channels. The tower
holds only a few thousand parameters, which is the point of this variant: it is the
parameter-count floor of the repository and a cheap way to sanity-check a change to
the shared trainer before paying for a full ResNet or Nature-CNN run.

``SAME`` padding keeps every stage non-empty for any input of side >= 1, so unlike the
``VALID`` Nature stack there is no minimum observation size to pad up to and a tiny test
observation shrinks to a small but still valid feature map.
"""

import einops
import jax.numpy as jnp
from flax import nnx

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8

# (kernel size, stride) of the three SAME convolutions, in order.
_KERNEL_STRIDES = ((3, 2), (3, 2), (3, 2))


def _conv_output_dim(size, stride):
    """Spatial size after a ``SAME`` stride-``stride`` convolution (ceil rounding)."""
    return -(-size // stride)


def _conv_stack_shape(spatial):
    """Spatial shape after every conv stage of the tiny encoder."""
    for _, stride in _KERNEL_STRIDES:
        spatial = tuple(_conv_output_dim(dim, stride) for dim in spatial)
    return spatial


class TinyCNN(nnx.Module):
    """Encoder of the tiny variant: conv 3x3/2, conv 3x3/2, conv 3x3/2, flatten."""

    def __init__(
        self,
        obs_shape,
        hidden_dims,
        rngs,
    ):
        self._obs_shape = tuple(obs_shape)
        convs = []
        in_features = self._obs_shape[-1]
        spatial = self._obs_shape[:2]
        for out_features, (kernel, stride) in zip(hidden_dims, _KERNEL_STRIDES, strict=False):
            convs.append(
                nnx.Conv(
                    in_features=in_features,
                    out_features=out_features,
                    kernel_size=(kernel, kernel),
                    strides=(stride, stride),
                    padding='SAME',
                    rngs=rngs,
                    dtype=bf16,
                    param_dtype=f32,
                )
            )
            in_features = out_features
            spatial = tuple(_conv_output_dim(dim, stride) for dim in spatial)
            assert all(dim > 0 for dim in spatial), (
                f'TinyCNN: a {kernel}x{kernel} stride {stride} SAME convolution collapses '
                f'observation {self._obs_shape} to {spatial}'
            )
        self._output_spatial = spatial
        self.convs = convs
        self.output_dim = spatial[0] * spatial[1] * in_features

    def __call__(self, x):
        for conv in self.convs:
            x = nnx.relu(conv(x))
        return einops.rearrange(x, 'b h w c -> b (h w c)')
