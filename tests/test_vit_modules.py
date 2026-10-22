"""The Vision Transformer encoder (ppo_vit)."""

import jax
import jax.numpy as jnp
from flax import nnx

from ppo_vit.modules import VitCNN

TINY = ((16, 16, 3), [16, 2, 2], 8)


def tiny_net(rngs):
    obs_shape, hidden_dims, patch_size = TINY
    return VitCNN(obs_shape, hidden_dims, rngs, patch_size)


def test_vit_output_dim_is_the_embed_dim():
    from ppo_vit.config import VitConfig

    cfg = VitConfig()
    net = VitCNN(cfg.obs_shape, cfg.hidden_dims, nnx.Rngs(0), cfg.patch_size)

    # [embed_dim, depth, num_heads] = [128, 4, 4], so the pooled vector is 128 wide.
    assert net.output_dim == 128
    assert net.output_dim == cfg.hidden_dims[0]


def test_vit_output_dim_matches_the_hidden_dims(rngs):
    net = tiny_net(rngs)

    assert net.output_dim == 16


def test_vit_forward_shape(rngs):
    net = tiny_net(rngs)

    out = net(jnp.zeros((2, 16, 16, 3), dtype=jnp.uint8))

    assert out.shape == (2, net.output_dim)
    # The transformer blocks run in bfloat16, so the pooled features come back as bf16.
    assert out.dtype == jnp.bfloat16


def test_vit_forward_shape_at_the_default_size(rngs):
    from ppo_vit.config import VitConfig

    cfg = VitConfig()
    net = VitCNN(cfg.obs_shape, cfg.hidden_dims, rngs, cfg.patch_size)

    out = net(jnp.zeros((1, 64, 64, 3), dtype=jnp.uint8))

    # 64 / 8 = 8 patches per side, so 64 tokens are pooled down to one vector.
    assert net.num_tokens == 64
    assert out.shape == (1, net.output_dim)
    assert jnp.isfinite(out).all()


def test_vit_forward_shape_for_a_padded_observation(rngs):
    # 20x12 is not a multiple of the patch size, so it is padded up to 24x16.
    net = VitCNN((20, 12, 3), [16, 1, 2], rngs, 8)

    out = net(jnp.zeros((1, 20, 12, 3), dtype=jnp.uint8))

    assert net.num_tokens == 3 * 2
    assert out.shape == (1, 16)


def test_vit_is_deterministic_for_the_same_rngs(rngs_factory):
    obs = jax.random.randint(jax.random.PRNGKey(0), (2, 16, 16, 3), 0, 256, dtype=jnp.uint8)
    first = tiny_net(rngs_factory(0))(obs)
    second = tiny_net(rngs_factory(0))(obs)

    assert jnp.array_equal(first, second)


def test_vit_is_permutation_sensitive_across_pixels(rngs):
    """The encoder must not collapse two different observations to one vector."""
    net = tiny_net(rngs)
    obs = jnp.zeros((1, 16, 16, 3), dtype=jnp.uint8)
    other = obs.at[:, :8].set(255)

    assert not jnp.allclose(net(obs), net(other))


def test_vit_weights_round_trip_through_split_and_merge(rngs):
    net = tiny_net(rngs)
    obs = jnp.full((1, 16, 16, 3), 17, dtype=jnp.uint8)

    graphdef, state = nnx.split(net)
    restored = nnx.merge(graphdef, state)

    assert jnp.array_equal(net(obs), restored(obs))
