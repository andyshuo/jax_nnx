import einops
import jax.numpy as jnp
from flax import nnx


class ResidualBlock(nnx.Module):
    def __init__(self, out_features, rngs):
        self.conv0 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        self.conv1 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )

    def __call__(self, x):
        inputs = x
        x = nnx.relu(x)
        x = self.conv0(x)
        x = nnx.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nnx.Module):
    def __init__(self, input_shape, out_features, rngs):
        self._input_shape = input_shape
        self._out_features = out_features
        self.conv = nnx.Conv(
            in_features=self._input_shape[-1],
            out_features=self._out_features,
            kernel_size=(3, 3),
            strides=1,
            padding='SAME',
            rngs=rngs,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
        )
        self.res0 = ResidualBlock(out_features, rngs)
        self.res1 = ResidualBlock(out_features, rngs)

    def __call__(self, x):
        x = self.conv(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = self.res0(x)
        x = self.res1(x)
        assert x.shape[1:] == self.get_output_shape(), f'Expected input shape {self._input_shape}, got {x.shape[1:]}'
        return x

    def get_output_shape(self):
        return (self._input_shape[0] // 2, self._input_shape[1] // 2, self._out_features)


class ResNet(nnx.Module):
    def __init__(
        self,
        obs_shape,
        hidden_dims,
        rngs,
    ):
        self._input_shape = obs_shape
        conv_seqs = []
        for out_features in hidden_dims:
            conv_seq = ConvSequence(self._input_shape, out_features, rngs)
            self._input_shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.net = nnx.Sequential(*conv_seqs)
        self.output_dim = self._input_shape[0] * self._input_shape[1] * self._input_shape[2]

    def __call__(self, x):
        x = self.net(x)
        x = einops.rearrange(x, 'b h w c -> b (h w c)')
        x = nnx.relu(x)
        return x
