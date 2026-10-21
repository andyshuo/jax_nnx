"""IMPALA CNN encoder (ppo_impala)."""

import jax.numpy as jnp
import pytest
from flax import nnx

from ppo_impala.modules import ImpalaCNN
from ppo_vae.modules import ConvSequence


def test_impala_output_dim_matches_default_config():
    from ppo_impala.config import ImpalaConfig

    cfg = ImpalaConfig()
    net = ImpalaCNN(cfg.obs_shape, cfg.hidden_dims, nnx.Rngs(0))

    # 64 -> 32 -> 16 -> 8, then flatten the 8x8x32 feature map.
    assert net.output_dim == 8 * 8 * 32


def test_impala_output_dim_small_config(rngs):
    net = ImpalaCNN((16, 16, 3), [8, 16], rngs)

    # 16 -> 8 -> 4, then flatten the 4x4x16 feature map.
    assert net.output_dim == 4 * 4 * 16


def test_impala_forward_shape(rngs):
    net = ImpalaCNN((16, 16, 3), [8, 16], rngs)

    out = net(jnp.zeros((2, 16, 16, 3), dtype=jnp.uint8))

    assert out.shape == (2, net.output_dim)
    # Convolutions run in bfloat16, so the flattened features come back as bf16.
    assert out.dtype == jnp.bfloat16


def test_impala_stacks_one_stage_per_hidden_dim(rngs):
    net = ImpalaCNN((16, 16, 3), [8, 16], rngs)

    assert len(net.net.layers) == 2
    # The stages are the shared VAE building blocks, not a re-implementation.
    assert all(isinstance(layer, ConvSequence) for layer in net.net.layers)


def test_impala_rejects_wrong_input_shape(rngs):
    net = ImpalaCNN((16, 16, 3), [8, 16], rngs)

    with pytest.raises(AssertionError):
        net(jnp.zeros((1, 32, 32, 3), dtype=jnp.uint8))


def test_impala_is_permutation_sensitive_across_pixels(rngs):
    """A conv net must not collapse two different observations to one vector."""
    net = ImpalaCNN((16, 16, 3), [8, 16], rngs)
    obs = jnp.zeros((1, 16, 16, 3), dtype=jnp.uint8)
    other = obs.at[:, :8].set(255)

    assert not jnp.allclose(net(obs), net(other))


@pytest.mark.parametrize(
    'hidden_dims, expected',
    [
        ([4], 8 * 8 * 4),
        ([4, 8], 4 * 4 * 8),
        ([4, 8, 16], 2 * 2 * 16),
    ],
)
def test_impala_output_dim_geometry(rngs, hidden_dims, expected):
    net = ImpalaCNN((16, 16, 3), hidden_dims, rngs)

    assert net.output_dim == expected
