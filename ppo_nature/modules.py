"""Classic DQN "Nature" convolutional encoder.

The stack is the one from the original DQN paper (Mnih et al., 2015): an 8x8 kernel
with stride 4, then a 4x4 kernel with stride 2, then a 3x3 kernel with stride 1, each
followed by a ReLU, and finally a flatten to a feature vector. The original uses
``VALID`` padding (no padding), so the spatial dimensions shrink at every stage and a
too-small observation collapses the feature map to an empty tensor.
"""

import einops
import jax.numpy as jnp
from flax import nnx

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8

# (kernel size, stride) of the three VALID convolutions, in order.
_KERNEL_STRIDES = ((8, 4), (4, 2), (3, 1))


def _conv_output_dim(size, kernel, stride):
    """Spatial size after a ``VALID`` convolution (no padding, floor rounding)."""
    return (size - kernel) // stride + 1


def _conv_stack_shape(spatial):
    """Spatial shape after every conv stage of the Nature encoder."""
    for kernel, stride in _KERNEL_STRIDES:
        spatial = tuple(_conv_output_dim(dim, kernel, stride) for dim in spatial)
    return spatial


def _minimum_obs_size(limit=1024):
    """Smallest square side the three VALID convolutions can consume (36 here).

    ``36 -> 8 -> 3 -> 1``; anything smaller reaches an empty feature map. Tiny test
    observations are zero-padded up to this size, which keeps the encoder usable at
    every shape without changing the geometry of real ProcGen frames.
    """
    for size in range(1, limit):
        if all(dim > 0 for dim in _conv_stack_shape((size, size))):
            return size
    raise ValueError(f'no positive observation size found below {limit}')


MIN_OBS_SIZE = _minimum_obs_size()


class NatureCNN(nnx.Module):
    """Encoder of the classic DQN agent: conv 8x8/4, conv 4x4/2, conv 3x3/1, flatten."""

    def __init__(
        self,
        obs_shape,
        hidden_dims,
        rngs,
    ):
        self._obs_shape = tuple(obs_shape)
        self._spatial = tuple(max(dim, MIN_OBS_SIZE) for dim in self._obs_shape[:2])
        convs = []
        in_features = self._obs_shape[-1]
        spatial = self._spatial
        for out_features, (kernel, stride) in zip(hidden_dims, _KERNEL_STRIDES, strict=False):
            convs.append(
                nnx.Conv(
                    in_features=in_features,
                    out_features=out_features,
                    kernel_size=(kernel, kernel),
                    strides=(stride, stride),
                    padding='VALID',
                    rngs=rngs,
                    dtype=bf16,
                    param_dtype=f32,
                )
            )
            in_features = out_features
            spatial = tuple(_conv_output_dim(dim, kernel, stride) for dim in spatial)
            assert all(dim > 0 for dim in spatial), (
                f'NatureCNN: a {kernel}x{kernel} stride {stride} VALID convolution collapses '
                f'observation {self._obs_shape} to {spatial}'
            )
        self._output_spatial = spatial
        self.convs = convs
        self.output_dim = spatial[0] * spatial[1] * in_features

    def _pad(self, x):
        """Zero-pad the spatial axes up to the receptive field, left/top aligned."""
        pads = (
            [(0, 0)]
            + [(0, target - size) for target, size in zip(self._spatial, x.shape[1:3], strict=False)]
            + [(0, 0)]
        )
        if any(pad for axis in pads for pad in axis):
            x = jnp.pad(x, pads)
        return x

    def __call__(self, x):
        x = self._pad(x)
        for conv in self.convs:
            x = nnx.relu(conv(x))
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        return x
