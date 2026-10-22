"""Vision Transformer (ViT) encoder for PPO.

The observation is cut into non-overlapping ``patch_size x patch_size`` patches by a
convolution whose kernel and stride are both the patch size, so every patch becomes
one token of ``embed_dim`` features. A learned positional embedding is added, the
tokens go through ``depth`` pre-norm transformer blocks, and a final LayerNorm feeds
a mean-pool over the token axis. The pooled vector is the encoder output, so
``output_dim`` is exactly ``embed_dim`` regardless of the observation size.

Spatial dimensions that are not a multiple of the patch size are zero-padded up to
the next multiple, which keeps the encoder usable at every shape (tiny test
observations included) without changing the geometry of real ProcGen frames.
"""

import einops
import jax
import jax.numpy as jnp
from flax import nnx

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8

# Hidden width of each block's MLP, as a multiple of the embedding width.
_MLP_RATIO = 4


class TransformerBlock(nnx.Module):
    """Pre-norm transformer block: self-attention, then an MLP, both residual."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        rngs,
    ):
        self.norm0 = nnx.LayerNorm(embed_dim, dtype=bf16, param_dtype=f32, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dtype=bf16,
            param_dtype=f32,
            dropout_rate=0.0,
            deterministic=True,
            decode=False,
            rngs=rngs,
        )
        self.norm1 = nnx.LayerNorm(embed_dim, dtype=bf16, param_dtype=f32, rngs=rngs)
        hidden_features = _MLP_RATIO * embed_dim
        self.fc0 = nnx.Linear(embed_dim, hidden_features, dtype=bf16, param_dtype=f32, rngs=rngs)
        self.fc1 = nnx.Linear(hidden_features, embed_dim, dtype=bf16, param_dtype=f32, rngs=rngs)

    def __call__(self, x):
        x = x + self.attn(self.norm0(x))
        x = x + self.fc1(nnx.gelu(self.fc0(self.norm1(x))))
        return x


class VitCNN(nnx.Module):
    """ViT encoder: patch embedding, positional embedding, transformer blocks, pool.

    ``hidden_dims`` is ``[embed_dim, depth, num_heads]``.
    """

    def __init__(
        self,
        obs_shape,
        hidden_dims,
        rngs,
        patch_size=8,
    ):
        if len(hidden_dims) != 3:
            raise ValueError(f'VitCNN expects hidden_dims=[embed_dim, depth, num_heads], got {hidden_dims}')
        self._obs_shape = tuple(obs_shape)
        self.patch_size = patch_size
        embed_dim, depth, num_heads = hidden_dims
        if embed_dim % num_heads:
            raise ValueError(f'embed_dim {embed_dim} is not divisible by num_heads {num_heads}')

        # Tokens per side, after padding the spatial axes up to a multiple of the patch.
        self._grid = tuple(-(-dim // patch_size) for dim in self._obs_shape[:2])
        self._spatial = tuple(side * patch_size for side in self._grid)
        self.num_tokens = self._grid[0] * self._grid[1]
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.patch_embed = nnx.Conv(
            in_features=self._obs_shape[-1],
            out_features=embed_dim,
            kernel_size=(patch_size, patch_size),
            strides=(patch_size, patch_size),
            padding='VALID',
            dtype=bf16,
            param_dtype=f32,
            rngs=rngs,
        )
        self.pos_embed = nnx.Param(0.02 * jax.random.normal(rngs.params(), (1, self.num_tokens, embed_dim), dtype=f32))
        self.blocks = [TransformerBlock(embed_dim, num_heads, rngs) for _ in range(depth)]
        self.norm = nnx.LayerNorm(embed_dim, dtype=bf16, param_dtype=f32, rngs=rngs)
        self.output_dim = embed_dim

    def _pad(self, x):
        """Zero-pad the spatial axes up to a whole number of patches, left/top aligned."""
        pads = (
            [(0, 0)]
            + [(0, target - size) for target, size in zip(self._spatial, x.shape[1:3], strict=False)]
            + [(0, 0)]
        )
        if any(pad for axis in pads for pad in axis):
            x = jnp.pad(x, pads)
        return x

    def __call__(self, x):
        x = self.patch_embed(self._pad(x))
        x = einops.rearrange(x, 'b h w c -> b (h w) c')
        x = x + self.pos_embed[...]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return einops.reduce(x, 'b t c -> b c', 'mean')
