"""VAE encoder/decoder building blocks (ppo_vae)."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from ppo_vae.modules import (
    VAE,
    ConvSequence,
    ConvTransposeSequence,
    ResidualBlock,
    ResidualBlockTranspose,
)


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
def conv_config(rngs=None, in_features=4, out_features=4):
    from ppo_vae.config import ConvConfig

    cfg = ConvConfig()
    cfg.in_features = in_features
    cfg.out_features = out_features
    cfg.rngs = rngs
    return cfg


def conv_transpose_config(rngs=None, strides=(2, 2), in_features=4, out_features=4):
    from ppo_vae.config import ConvTransposeConfig

    cfg = ConvTransposeConfig(strides=strides)
    cfg.in_features = in_features
    cfg.out_features = out_features
    cfg.rngs = rngs
    return cfg


def res_block_config(rngs=None):
    from ppo_vae.config import ResidualBlockConfig

    cfg = ResidualBlockConfig()
    cfg.conv1_config.in_features = cfg.conv1_config.out_features = 4
    cfg.conv2_config.in_features = cfg.conv2_config.out_features = 4
    cfg.conv1_config.rngs = rngs
    cfg.conv2_config.rngs = rngs
    return cfg


def res_block_transpose_config(rngs=None):
    from ppo_vae.config import ResidualBlockTransposeConfig

    cfg = ResidualBlockTransposeConfig()
    cfg.conv1_config.strides = (1, 1)
    cfg.conv2_config.strides = (1, 1)
    cfg.conv1_config.in_features = cfg.conv1_config.out_features = 4
    cfg.conv2_config.in_features = cfg.conv2_config.out_features = 4
    cfg.conv1_config.rngs = rngs
    cfg.conv2_config.rngs = rngs
    return cfg


# --------------------------------------------------------------------------- #
# conv building blocks
# --------------------------------------------------------------------------- #
def test_conv_sequence_halves_spatial_dims(rngs):
    seq = ConvSequence((16, 16, 3), 8, rngs, conv_config(), res_block_config(rngs))

    assert seq.get_output_shape() == (8, 8, 8)
    assert seq(jnp.zeros((1, 16, 16, 3), dtype=jnp.uint8)).shape == (1, 8, 8, 8)


def test_conv_transpose_sequence_doubles_spatial_dims(rngs):
    seq = ConvTransposeSequence((4, 4, 8), 3, rngs, conv_transpose_config(), res_block_transpose_config(rngs))

    assert seq.get_output_shape() == (8, 8, 3)
    assert seq(jnp.zeros((1, 4, 4, 8))).shape == (1, 8, 8, 3)


def test_conv_sequence_rejects_wrong_input_shape(rngs):
    seq = ConvSequence((16, 16, 3), 8, rngs, conv_config(), res_block_config(rngs))

    with pytest.raises(AssertionError):
        seq(jnp.zeros((1, 32, 32, 3), dtype=jnp.uint8))


def test_residual_block_preserves_shape(rngs):
    block = ResidualBlock(conv_config(rngs), conv_config(rngs))
    x = jnp.ones((1, 4, 4, 4))

    out = block(x)

    assert out.shape == x.shape
    assert jnp.isfinite(out).all()


def test_residual_block_transpose_preserves_shape(rngs):
    block = ResidualBlockTranspose(
        conv_transpose_config(rngs, strides=(1, 1)),
        conv_transpose_config(rngs, strides=(1, 1)),
    )
    x = jnp.ones((1, 4, 4, 4))

    assert block(x).shape == x.shape


def test_residual_block_has_skip_connection(rngs):
    """With zeroed kernels the block must reduce to the identity."""
    from flax import nnx

    block = ResidualBlock(conv_config(rngs), conv_config(rngs))
    graphdef, state = nnx.split(block)
    state = jax.tree.map(jnp.zeros_like, state)
    block = nnx.merge(graphdef, state)

    x = jnp.ones((1, 4, 4, 4))
    assert jnp.allclose(block(x), x)


# --------------------------------------------------------------------------- #
# VAE
# --------------------------------------------------------------------------- #
def test_vae_encode_shape(vae_module):
    obs = jnp.zeros((2, 16, 16, 3), dtype=jnp.bfloat16)

    logits = vae_module.encode(obs)

    assert logits.shape == (2, 4, 8)


def test_vae_decode_round_trips_shape(vae_module):
    z = jnp.zeros((2, 4 * 8), dtype=jnp.float32)

    recon = vae_module.decode(z)

    assert recon.shape == (2, 16, 16, 3)


def test_vae_output_dim(vae_module):
    assert vae_module.output_dim == 4 * 8


def test_vae_encode_decode_end_to_end(vae_module):
    obs = jnp.full((1, 16, 16, 3), 128, dtype=jnp.bfloat16) / 255.0

    logits = vae_module.encode(obs)
    z = vae_module.straight_through_sample(logits, jax.random.PRNGKey(0))
    recon = vae_module.decode(z.reshape(1, -1))

    assert recon.shape == obs.shape
    assert jnp.isfinite(recon).all()


def test_vae_encoder_produces_finite_activations(vae_module):
    obs = jax.random.uniform(jax.random.PRNGKey(0), (2, 16, 16, 3), dtype=jnp.bfloat16)

    logits = vae_module.encode(obs)

    assert jnp.isfinite(logits).all()


def test_straight_through_sample_shape(vae_module):
    logits = jnp.zeros((2, 4, 8))

    sample = vae_module.straight_through_sample(logits, jax.random.PRNGKey(0))

    assert sample.shape == (2, 4, 8)


def test_straight_through_sample_forward_pass_is_one_hot(vae_module):
    """``one_hot + probs - stop_gradient(probs)`` must evaluate to exactly one_hot."""
    logits = jax.random.normal(jax.random.PRNGKey(0), (2, 4, 8))

    sample = vae_module.straight_through_sample(logits, jax.random.PRNGKey(0))
    probs = jax.nn.softmax(logits, axis=-1)
    one_hot = jax.nn.one_hot(jnp.argmax(probs, axis=-1), 8)

    assert jnp.allclose(sample, one_hot)


def test_straight_through_sample_is_one_hot_per_category(vae_module):
    logits = jax.random.normal(jax.random.PRNGKey(1), (3, 4, 8))

    sample = vae_module.straight_through_sample(logits, jax.random.PRNGKey(0))

    assert jnp.allclose(sample.sum(-1), 1.0)
    # ``(one_hot + p) - p`` is only exactly 0/1 up to float rounding.
    assert jnp.all((sample > -1e-3) & (sample < 1 + 1e-3))


def test_straight_through_sample_passes_gradients(vae_module):
    """The estimator must backprop through the softmax term, not the argmax."""
    weights = jnp.arange(8, dtype=jnp.float32)

    def objective(logits):
        sample = vae_module.straight_through_sample(logits, jax.random.PRNGKey(0))
        return jnp.sum(sample * weights)

    grads = jax.grad(objective)(jnp.zeros((1, 4, 8)))

    assert jnp.isfinite(grads).all()
    assert not jnp.allclose(grads, 0.0)


def test_stop_gradient_term_does_not_contribute(vae_module):
    """``sum(softmax(logits))`` is constant, so its gradient must vanish."""

    def objective(logits):
        sample = vae_module.straight_through_sample(logits, jax.random.PRNGKey(0))
        return sample.sum()

    grads = jax.grad(objective)(jnp.zeros((1, 4, 8)))

    assert jnp.allclose(grads, 0.0, atol=1e-6)


def test_vae_requires_matching_encoder_decoder_depth():
    """A decoder that does not undo the encoder's downsampling must fail loudly."""
    from ppo_vae.config import VAEConfig

    cfg = VAEConfig()
    cfg.shape = (16, 16, 3)
    cfg.enc_out_features = [4, 8]
    cfg.dec_out_features = [8]  # one stage short
    cfg.num_categories = 4
    cfg.categorical_dim = 8
    cfg.rngs = nnx.Rngs(0)
    cfg.vae_proj_config.rngs = nnx.Rngs(0)

    with pytest.raises(AssertionError):
        vae = VAE(**cfg)
        vae.decode(jnp.zeros((1, 4 * 8)))
