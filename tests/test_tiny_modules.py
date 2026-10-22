"""The tiny CNN encoder (ppo_tiny)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from ppo_tiny.modules import TinyCNN


def test_tiny_output_dim_matches_default_config():
    from ppo_tiny.config import TinyConfig

    cfg = TinyConfig()
    net = TinyCNN(cfg.obs_shape, cfg.hidden_dims, nnx.Rngs(0))

    # SAME padding with stride 2 halves every side: 64 -> 32 -> 16 -> 8, then
    # flatten the 8x8x16 feature map.
    assert net.output_dim == 8 * 8 * 16


def test_tiny_output_dim_for_a_tiny_observation(rngs):
    net = TinyCNN((16, 16, 3), [4, 8], rngs)

    # 16 -> 8 -> 4, then flatten the 4x4x8 feature map.
    assert net.output_dim == 4 * 4 * 8


def test_tiny_forward_shape(rngs):
    net = TinyCNN((16, 16, 3), [4, 8], rngs)

    out = net(jnp.zeros((2, 16, 16, 3), dtype=jnp.uint8))

    assert out.shape == (2, net.output_dim)
    # Convolutions run in bfloat16, so the flattened features come back as bf16.
    assert out.dtype == jnp.bfloat16


def test_tiny_forward_shape_at_the_default_size(rngs):
    net = TinyCNN((64, 64, 3), [8, 16, 16], rngs)

    out = net(jnp.zeros((1, 64, 64, 3), dtype=jnp.uint8))

    assert out.shape == (1, net.output_dim)
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize(
    'hidden_dims, expected',
    [
        # Each stage halves both sides: 64 -> 32 -> 16 -> 8.
        ([8], 32 * 32 * 8),
        ([8, 16], 16 * 16 * 16),
        ([8, 16, 16], 8 * 8 * 16),
    ],
)
def test_tiny_output_dim_geometry(rngs, hidden_dims, expected):
    net = TinyCNN((64, 64, 3), hidden_dims, rngs)

    assert net.output_dim == expected


def test_tiny_parameter_count_is_below_100k(rngs):
    net = TinyCNN((64, 64, 3), [8, 16, 16], rngs)
    total = int(sum(leaf.size for leaf in jax.tree.leaves(nnx.state(net, nnx.Param))))

    # Three conv layers and nothing else: 3*3*3*8+8, 3*3*8*16+16, 3*3*16*16+16.
    assert total == 224 + 1168 + 2320
    assert total < 100_000


def test_tiny_is_deterministic_for_the_same_rngs(rngs_factory):
    obs = jax.random.randint(jax.random.PRNGKey(0), (2, 16, 16, 3), 0, 256, dtype=jnp.uint8)
    first = TinyCNN((16, 16, 3), [4, 8], rngs_factory(0))(obs)
    second = TinyCNN((16, 16, 3), [4, 8], rngs_factory(0))(obs)

    assert jnp.array_equal(first, second)
