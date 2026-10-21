"""The JEPA predictor head and its composition with the ResNet encoder (ppo_jepa)."""

import jax
import jax.numpy as jnp
from flax import nnx

from ppo_cleanrl.modules import ResNet
from ppo_jepa.modules import Predictor, PredictorConfig


def make_predictor(rngs, dim=8):
    return Predictor(PredictorConfig(in_features=dim, out_features=dim, rngs=rngs))


def make_obs(batch=2):
    return jax.random.randint(jax.random.PRNGKey(0), (batch, 16, 16, 3), 0, 256, dtype=jnp.uint8)


def test_predictor_output_shape(rngs):
    predictor = make_predictor(rngs, dim=8)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 8))

    out = predictor(x)

    assert out.shape == (2, 8)
    assert jnp.isfinite(out).all()


def test_predictor_maps_the_latent_onto_the_same_latent_size(rngs):
    """Both layers are square, so the head never changes the latent width."""
    predictor = make_predictor(rngs, dim=5)

    assert predictor(jnp.ones((3, 5))).shape == (3, 5)
    assert predictor.fc0.kernel.shape == (5, 5)
    assert predictor.fc1.kernel.shape == (5, 5)


def test_predictor_is_sensitive_to_its_input(rngs):
    """A predictor that ignores its input would make the auxiliary task vacuous."""
    predictor = make_predictor(rngs, dim=8)
    x = jnp.ones((1, 8))
    other = jnp.zeros((1, 8))

    assert not jnp.allclose(predictor(x), predictor(other))


def test_encoder_and_predictor_compose(rngs_factory):
    enc = ResNet((16, 16, 3), [4, 8], rngs_factory(0))
    predictor = Predictor(
        PredictorConfig(in_features=enc.output_dim, out_features=enc.output_dim, rngs=rngs_factory(1))
    )
    obs = make_obs()

    latent = enc(obs)
    pred = predictor(latent)

    assert latent.shape == (2, enc.output_dim)
    assert pred.shape == latent.shape
    assert jnp.isfinite(pred).all()


def test_composed_head_is_differentiable(rngs_factory):
    enc = ResNet((16, 16, 3), [4, 8], rngs_factory(0))
    predictor = Predictor(
        PredictorConfig(in_features=enc.output_dim, out_features=enc.output_dim, rngs=rngs_factory(1))
    )
    obs = make_obs()

    def head_sum(obs):
        return predictor(enc(obs)).sum()

    grads = jax.grad(head_sum)(obs.astype(jnp.float32))

    assert jnp.isfinite(grads).all()


def test_encoder_module_is_reexported(rngs):
    """``ppo_jepa`` must be usable without reaching into ``ppo_cleanrl``."""
    from ppo_jepa import modules

    net = modules.ResNet((16, 16, 3), [4, 8], rngs)

    assert isinstance(net, ResNet)
    assert isinstance(make_predictor(rngs), nnx.Module)
