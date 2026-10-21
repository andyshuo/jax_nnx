"""The classic DQN "Nature" CNN encoder (ppo_nature)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from ppo_nature.modules import MIN_OBS_SIZE, NatureCNN


def test_nature_output_dim_matches_default_config():
    from ppo_nature.config import NatureConfig

    cfg = NatureConfig()
    net = NatureCNN(cfg.obs_shape, cfg.hidden_dims, nnx.Rngs(0))

    # VALID padding: 64 -> 15 -> 6 -> 4, then flatten the 4x4x64 feature map.
    assert net.output_dim == 4 * 4 * 64


def test_nature_output_dim_for_a_tiny_observation(rngs):
    net = NatureCNN((16, 16, 3), [8, 16, 16], rngs)

    # 16x16 is below the receptive field of the stack, so it is padded up to
    # MIN_OBS_SIZE first: 36 -> 8 -> 3 -> 1.
    assert MIN_OBS_SIZE == 36
    assert net.output_dim == 1 * 1 * 16


def test_nature_forward_shape(rngs):
    net = NatureCNN((16, 16, 3), [8, 16, 16], rngs)

    out = net(jnp.zeros((2, 16, 16, 3), dtype=jnp.uint8))

    assert out.shape == (2, net.output_dim)
    # Convolutions run in bfloat16, so the flattened features come back as bf16.
    assert out.dtype == jnp.bfloat16


def test_nature_forward_shape_at_the_default_size(rngs):
    net = NatureCNN((64, 64, 3), [32, 64, 64], rngs)

    out = net(jnp.zeros((1, 64, 64, 3), dtype=jnp.uint8))

    assert out.shape == (1, net.output_dim)
    assert jnp.isfinite(out).all()


@pytest.mark.parametrize(
    'hidden_dims, expected',
    [
        # Each stage shrinks the padded 36x36 input: 36 -> 8 -> 3 -> 1.
        ([8], 8 * 8 * 8),
        ([8, 16], 3 * 3 * 16),
        ([8, 16, 16], 1 * 1 * 16),
    ],
)
def test_nature_output_dim_geometry(rngs, hidden_dims, expected):
    net = NatureCNN((16, 16, 3), hidden_dims, rngs)

    assert net.output_dim == expected


def test_nature_is_deterministic_for_the_same_rngs(rngs_factory):
    obs = jax.random.randint(jax.random.PRNGKey(0), (2, 16, 16, 3), 0, 256, dtype=jnp.uint8)
    first = NatureCNN((16, 16, 3), [8, 16, 16], rngs_factory(0))(obs)
    second = NatureCNN((16, 16, 3), [8, 16, 16], rngs_factory(0))(obs)

    assert jnp.array_equal(first, second)


def test_nature_is_permutation_sensitive_across_pixels(rngs):
    """A conv net must not collapse two different observations to one vector."""
    net = NatureCNN((16, 16, 3), [8, 16, 16], rngs)
    obs = jnp.zeros((1, 16, 16, 3), dtype=jnp.uint8)
    other = obs.at[:, :8].set(255)

    assert not jnp.allclose(net(obs), net(other))


def test_nature_weights_round_trip_through_split_and_merge(rngs):
    net = NatureCNN((16, 16, 3), [8, 16, 16], rngs)
    obs = jnp.full((1, 16, 16, 3), 17, dtype=jnp.uint8)

    graphdef, state = nnx.split(net)
    restored = nnx.merge(graphdef, state)

    assert jnp.array_equal(net(obs), restored(obs))
