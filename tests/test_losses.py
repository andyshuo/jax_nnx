"""Loss-function pieces, including the PPO clipping behaviour."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

MODULES = ['ppo_cleanrl.utils', 'ppo_vae.utils']


def module(name):
    return __import__(name, fromlist=['*'])


def make_batch(obs_shape=(16, 16, 3), batch=4, actions=6):
    rng = jax.random.PRNGKey(0)
    return {
        'mb_obs': jax.random.randint(rng, (batch,) + obs_shape, 0, 256, dtype=jnp.uint8),
        'mb_actions': jax.random.randint(rng, (batch,), 0, actions, dtype=jnp.int32),
        'mb_logprobs': -jnp.ones((batch,)),
        'mb_advantages': jax.random.normal(rng, (batch,)),
        'mb_returns': jax.random.normal(rng, (batch,)),
        'mb_values': jnp.zeros((batch,)),
        'rng': rng,
    }


# --------------------------------------------------------------------------- #
# ratio
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('name', MODULES)
def test_compute_ratio_identity(name):
    ratio, logratio = module(name).compute_ratio(jnp.array([0.0, 1.0]), jnp.array([0.0, 1.0]))

    assert jnp.allclose(ratio, 1.0)
    assert jnp.allclose(logratio, 0.0)


@pytest.mark.parametrize('name', MODULES)
def test_compute_ratio_is_exponential(name):
    newlogprob = jnp.array([0.0, 0.5])
    logprob = jnp.array([0.0, 0.0])

    ratio, logratio = module(name).compute_ratio(newlogprob, logprob)

    assert jnp.allclose(logratio, newlogprob - logprob)
    assert jnp.allclose(ratio, jnp.exp(newlogprob - logprob))


# --------------------------------------------------------------------------- #
# policy loss
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('name', MODULES)
def test_policy_loss_clips_positive_advantage(name):
    advantage = jnp.array([1.0, -1.0])
    ratio = jnp.array([2.0, 2.0])

    loss = module(name).compute_policy_loss(advantage, advantage.mean(), advantage.std(), ratio, clip_coef=0.2)

    # +adv wants ratio up: clipping caps it at 1.2. -adv wants ratio down, and
    # pushing the ratio further up is penalised without clipping.
    assert jnp.allclose(loss, jnp.array([-1.2, 2.0]))


@pytest.mark.parametrize('name', MODULES)
def test_policy_loss_is_unclipped_near_ratio_one(name):
    advantage = jnp.array([1.0, -1.0])
    ratio = jnp.array([1.0, 1.0])

    loss = module(name).compute_policy_loss(advantage, advantage.mean(), advantage.std(), ratio, clip_coef=0.2)

    assert jnp.allclose(loss, jnp.array([-1.0, 1.0]))


@pytest.mark.parametrize('name', MODULES)
def test_policy_loss_uses_max_not_min(name):
    """PPO takes the pessimistic (maximum) of the clipped/unclipped surrogates."""
    advantage = jnp.array([1.0, -1.0])
    ratio = jnp.array([0.5, 0.5])

    loss = module(name).compute_policy_loss(advantage, advantage.mean(), advantage.std(), ratio, clip_coef=0.2)

    # pg1 = [-0.5, 0.5]; pg2 (clipped to 0.8) = [-0.8, 0.8] -> max = [-0.5, 0.8]
    assert jnp.allclose(loss, jnp.array([-0.5, 0.8]))


# --------------------------------------------------------------------------- #
# value loss
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('name', MODULES)
def test_value_loss_zero_when_prediction_matches(name):
    value = jnp.array([1.0])

    loss = module(name).compute_value_loss(value, jnp.array([1.0]), value, clip_coef=0.2)

    assert jnp.allclose(loss, 0.0)


@pytest.mark.parametrize('name', MODULES)
def test_value_loss_is_half_squared_error_without_clipping(name):
    loss = module(name).compute_value_loss(jnp.array([1.0]), jnp.array([1.5]), jnp.array([1.0]), clip_coef=10.0)

    assert jnp.allclose(loss, 0.5 * 0.25)


@pytest.mark.parametrize('name', MODULES)
def test_value_loss_is_pessimistic_when_clipped(name):
    """PPO takes the max of clipped/unclipped, so clipping can only raise the loss."""
    loss = module(name).compute_value_loss(jnp.array([0.5]), jnp.array([1.0]), jnp.array([0.0]), clip_coef=0.2)

    # unclipped: 0.5 * (0.5 - 1.0) ** 2 = 0.125
    # clipped to 0.2: 0.5 * (0.2 - 1.0) ** 2 = 0.32 -> the max wins
    assert jnp.allclose(loss, 0.5 * (0.2 - 1.0) ** 2)


@pytest.mark.parametrize('name', MODULES)
def test_value_loss_rejects_a_large_jump(name):
    """An over-large jump must not be rewarded: the clipped term dominates."""
    far = module(name).compute_value_loss(jnp.array([5.0]), jnp.array([1.0]), jnp.array([0.0]), clip_coef=0.2)
    near = module(name).compute_value_loss(jnp.array([1.0]), jnp.array([1.0]), jnp.array([0.0]), clip_coef=0.2)

    assert far > near


# --------------------------------------------------------------------------- #
# clip fractions / explained variance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('name', MODULES)
def test_clipfracs_zero_for_unclipped_ratio(name):
    ratio = jnp.ones((4,))
    logratio = jnp.zeros((4,))

    old_kl, approx_kl, cf = module(name).compute_clipfracs(ratio, logratio, clip_coef=0.2)

    assert jnp.allclose(old_kl, 0.0)
    assert jnp.allclose(approx_kl, 0.0)
    assert jnp.allclose(cf, 0.0)


@pytest.mark.parametrize('name', MODULES)
def test_clipfracs_one_when_everything_is_clipped(name):
    ratio = jnp.full((4,), 2.0)
    logratio = jnp.log(ratio)

    _, _, cf = module(name).compute_clipfracs(ratio, logratio, clip_coef=0.2)

    assert jnp.allclose(cf, 1.0)


@pytest.mark.parametrize('name', MODULES)
def test_explained_variance_perfect_prediction(name):
    y = jnp.array([1.0, 2.0, 3.0])

    assert module(name).compute_explained_variance(y, y) == pytest.approx(1.0)


@pytest.mark.parametrize('name', MODULES)
def test_explained_variance_is_nan_for_constant_targets(name):
    y_true = jnp.ones((3,))
    y_pred = jnp.zeros((3,))

    assert jnp.isnan(module(name).compute_explained_variance(y_pred, y_true))


# --------------------------------------------------------------------------- #
# VAE-specific losses
# --------------------------------------------------------------------------- #
def test_vae_reconstruction_loss_zero_for_identical_inputs():
    from ppo_vae.utils import compute_vae_loss

    obs = jnp.ones((2, 4, 4, 3))

    assert jnp.allclose(compute_vae_loss(obs, obs), 0.0)


def test_vae_reconstruction_loss_is_mse():
    from ppo_vae.utils import compute_vae_loss

    recon = jnp.zeros((1,))
    obs = jnp.full((1,), 2.0)

    assert jnp.allclose(compute_vae_loss(recon, obs), 4.0)


def test_kl_is_zero_for_uniform_logits():
    from ppo_vae.utils import compute_kl_loss

    assert jnp.allclose(compute_kl_loss(jnp.zeros((2, 3, 4))), 0.0)


def test_kl_is_positive_for_non_uniform_logits():
    from ppo_vae.utils import compute_kl_loss

    logits = jnp.zeros((1, 2, 4)).at[..., 0].set(5.0)

    assert compute_kl_loss(logits) > 0.0


# --------------------------------------------------------------------------- #
# full objective
# --------------------------------------------------------------------------- #
def test_cleanrl_loss_is_finite_and_differentiable(cleanrl_config, rngs):
    from ppo_cleanrl.agent import PPOAgent
    from ppo_cleanrl.utils import loss_fn

    agent = PPOAgent(cleanrl_config.agent_config, rngs)
    batch = make_batch()

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent, batch)

    assert jnp.isfinite(loss)
    assert len(aux) == 7
    leaves = jax.tree.leaves(grads)
    assert leaves
    assert all(jnp.isfinite(leaf).all() for leaf in leaves)


def test_vae_loss_is_finite_and_differentiable(vae_config, rngs):
    from ppo_vae.agent import PPOAgent
    from ppo_vae.utils import loss_fn

    agent = PPOAgent(vae_config.agent_config, rngs)
    batch = make_batch()

    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(agent, batch)

    assert jnp.isfinite(loss)
    assert len(aux) == 9
    leaves = jax.tree.leaves(grads)
    assert leaves
    assert all(jnp.isfinite(leaf).all() for leaf in leaves)


def test_vae_loss_respects_coefficients(vae_config, rngs):
    """Changing the recon/KL weights must change the total loss."""
    from ppo_vae.agent import PPOAgent
    from ppo_vae.utils import loss_fn

    agent = PPOAgent(vae_config.agent_config, rngs)
    batch = make_batch()

    base, _ = loss_fn(agent, batch, recon_coef=1.0, kl_coef=1.0)
    scaled, _ = loss_fn(agent, batch, recon_coef=4.0, kl_coef=4.0)

    assert not jnp.allclose(base, scaled)
