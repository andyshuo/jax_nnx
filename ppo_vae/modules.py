import einops
import jax
import jax.numpy as jnp
from flax import nnx

from ppo_vae.config import ConvSeqConfig, ConvTransposeSeqConfig, VAEProjConfig

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8
sg = jax.lax.stop_gradient


class ResidualBlock(nnx.Module):
    def __init__(self, conv1_config, conv2_config):
        self.conv0 = nnx.Conv(**conv1_config)
        self.conv1 = nnx.Conv(**conv2_config)

    def __call__(self, x):
        inputs = x
        x = nnx.relu(x)
        x = self.conv0(x)
        x = nnx.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvSequence(nnx.Module):
    def __init__(self, shape, out_features, rngs, conv_config, res_block_config):
        self._input_shape = shape
        self._out_features = out_features
        conv_config.in_features = self._input_shape[-1]
        conv_config.out_features = self._out_features
        conv_config.rngs = rngs
        self.conv = nnx.Conv(**conv_config)
        res_block_config.conv1_config.in_features = self._out_features
        res_block_config.conv2_config.in_features = self._out_features
        res_block_config.conv1_config.out_features = self._out_features
        res_block_config.conv2_config.out_features = self._out_features
        res_block_config.conv1_config.rngs = rngs
        res_block_config.conv2_config.rngs = rngs
        self.res0 = ResidualBlock(**res_block_config)
        self.res1 = ResidualBlock(**res_block_config)

    def __call__(self, x):
        x = self.conv(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        x = self.res0(x)
        x = self.res1(x)
        assert x.shape[1:] == self.get_output_shape(), f'Expected input shape {self._input_shape}, got {x.shape[1:]}'
        return x

    def get_output_shape(self):
        return (self._input_shape[0] // 2, self._input_shape[1] // 2, self._out_features)


class ResidualBlockTranspose(nnx.Module):
    def __init__(self, conv1_config, conv2_config):

        self.conv0 = nnx.ConvTranspose(**conv1_config)
        self.conv1 = nnx.ConvTranspose(**conv2_config)

    def __call__(self, x):
        inputs = x
        x = nnx.relu(x)
        x = self.conv0(x)
        x = nnx.relu(x)
        x = self.conv1(x)
        return x + inputs


class ConvTransposeSequence(nnx.Module):
    def __init__(self, shape, out_features, rngs, conv_transpose_config, res_block_transpose_config):
        self._input_shape = shape
        self._out_features = out_features
        conv_transpose_config.in_features = self._input_shape[-1]
        conv_transpose_config.out_features = self._out_features
        conv_transpose_config.rngs = rngs
        self.conv = nnx.ConvTranspose(**conv_transpose_config)
        res_block_transpose_config.conv1_config.in_features = self._out_features
        res_block_transpose_config.conv2_config.in_features = self._out_features
        res_block_transpose_config.conv1_config.out_features = self._out_features
        res_block_transpose_config.conv2_config.out_features = self._out_features
        res_block_transpose_config.conv1_config.rngs = rngs
        res_block_transpose_config.conv2_config.rngs = rngs
        self.res0 = ResidualBlockTranspose(**res_block_transpose_config)
        self.res1 = ResidualBlockTranspose(**res_block_transpose_config)

    def __call__(self, x):

        x = self.conv(x)
        x = self.res0(x)
        x = self.res1(x)
        assert x.shape[1:] == self.get_output_shape(), f'Expected input shape {self._input_shape}, got {x.shape[1:]}'
        return x

    def get_output_shape(self):
        return (self._input_shape[0] * 2, self._input_shape[1] * 2, self._out_features)


class VAE(nnx.Module):
    def __init__(
        self,
        shape,
        enc_out_features,
        dec_out_features,
        num_categories=64,
        categorical_dim=32,
        temperature=1.0,
        vae_proj_config=VAEProjConfig(),
        conv_seq_config=ConvSeqConfig(),
        conv_transpose_seq_config=ConvTransposeSeqConfig(),
        rngs=nnx.Rngs(),
    ):
        self._shape = shape
        conv_seqs = []
        output_dim = 0
        self._num_categories = num_categories
        self._categorical_dim = categorical_dim
        self._temperature = temperature
        for out_features in enc_out_features:
            conv_seq_config.shape = shape
            conv_seq_config.out_features = out_features
            conv_seq_config.rngs = rngs
            conv_seq = ConvSequence(**conv_seq_config)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
            output_dim = shape[0] * shape[1] * shape[2]
        self.encoder = nnx.Sequential(*conv_seqs)
        vae_proj_config.in_features = output_dim
        vae_proj_config.out_features = num_categories * categorical_dim
        self.proj1 = nnx.Linear(**vae_proj_config)
        self.output_dim = num_categories * categorical_dim

        self._encoder_output_shape = shape

        vae_proj_config.in_features = self._categorical_dim * self._num_categories
        vae_proj_config.out_features = output_dim
        self.proj2 = nnx.Linear(**vae_proj_config)

        conv_seqs = []
        for out_features in dec_out_features:
            conv_transpose_seq_config.shape = shape
            conv_transpose_seq_config.out_features = out_features
            conv_transpose_seq_config.rngs = rngs
            conv_seq = ConvTransposeSequence(**conv_transpose_seq_config)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        self.decoder = nnx.Sequential(*conv_seqs)

    def encode(self, x):
        h1 = nnx.relu(self.encoder(x))
        h1 = einops.rearrange(h1, 'b h w c -> b (h w c)')
        logits = self.proj1(h1)
        logits = nnx.relu(logits)
        logits = einops.rearrange(logits, 'b (c d) -> b c d', c=self._num_categories, d=self._categorical_dim)
        return logits  # b, c, d

    def straight_through_sample(self, logits, rng):
        # print(logits.shape)
        probs = nnx.softmax(logits, axis=-1)
        # print(probs.shape)
        indices = jnp.argmax(probs, axis=-1)
        # print(indices.shape)
        one_hot = jax.nn.one_hot(indices, self._categorical_dim)

        # print(one_hot.shape, probs.shape)
        sample = one_hot + probs - sg(probs)
        # print(sample.shape)
        return sample

    def decode(self, z):
        z = nnx.relu(self.proj2(z))
        z = einops.rearrange(
            z, 'b (h w c) -> b h w c', h=self._encoder_output_shape[0], w=self._encoder_output_shape[1]
        )
        z = self.decoder(z)
        assert z.shape[1:] == self._shape, f'Expected input shape {self._shape}, got {z.shape[1:]}'
        return z
