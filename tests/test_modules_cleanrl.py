"""ResNet encoder building blocks (ppo_cleanrl)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from ppo_cleanrl.modules import ConvSequence, ResidualBlock, ResNet


def test_resnet_output_dim_matches_default_config():
    from ppo_cleanrl.config import ResNetConfig

    cfg = ResNetConfig()
    net = ResNet(cfg.obs_shape, cfg.hidden_dims, nnx.Rngs(0))

    # 64 -> 32 -> 16 -> 8, then flatten the 8x8x32 feature map.
    assert net.output_dim == 8 * 8 * 32


def test_resnet_forward_shape(rngs):
    net = ResNet((16, 16, 3), [4, 8], rngs)

    out = net(jnp.zeros((2, 16, 16, 3), dtype=jnp.uint8))

    assert out.shape == (2, net.output_dim)
    # Convolutions run in bfloat16, so the flattened features come back as bf16.
    assert out.dtype == jnp.bfloat16


def test_resnet_is_permutation_sensitive_across_pixels(rngs):
    """A conv net must not collapse two different observations to one vector."""
    net = ResNet((16, 16, 3), [4, 8], rngs)
    obs = jnp.zeros((1, 16, 16, 3), dtype=jnp.uint8)
    other = obs.at[:, :8].set(255)

    assert not jnp.allclose(net(obs), net(other))


def test_conv_sequence_output_shape_halves_spatial_dims(rngs):
    seq = ConvSequence((16, 16, 3), 8, rngs)

    assert seq.get_output_shape() == (8, 8, 8)
    assert seq(jnp.zeros((1, 16, 16, 3), dtype=jnp.uint8)).shape == (1, 8, 8, 8)


def test_conv_sequence_rejects_wrong_input_shape(rngs):
    seq = ConvSequence((16, 16, 3), 8, rngs)

    with pytest.raises(AssertionError):
        seq(jnp.zeros((1, 32, 32, 3), dtype=jnp.uint8))


def test_residual_block_preserves_shape(rngs):
    block = ResidualBlock(out_features=8, rngs=rngs)
    x = jnp.ones((1, 8, 8, 8))

    out = block(x)

    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


def test_residual_block_has_skip_connection(rngs):
    """With zeroed conv weights the block must reduce to the identity."""
    from flax import nnx

    block = ResidualBlock(out_features=4, rngs=rngs)
    # Zero every kernel/bias so both convs output zeros.
    graphdef, state = nnx.split(block)
    state = jax.tree.map(lambda leaf: jnp.zeros_like(leaf), state)
    block = nnx.merge(graphdef, state)

    x = jnp.ones((1, 4, 4, 4))
    assert jnp.allclose(block(x), x)


@pytest.mark.parametrize(
    'hidden_dims, expected',
    [
        ([4], 8 * 8 * 4),
        ([4, 8], 4 * 4 * 8),
        ([4, 8, 16], 2 * 2 * 16),
    ],
)
def test_resnet_output_dim_geometry(rngs, hidden_dims, expected):
    net = ResNet((16, 16, 3), hidden_dims, rngs)

    assert net.output_dim == expected
