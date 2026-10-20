"""The VAE PPO agent (ppo_vae)."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def agent(vae_config, rngs):
    from ppo_vae.agent import PPOAgent

    return PPOAgent(vae_config.agent_config, rngs)


@pytest.fixture
def obs():
    return jax.random.randint(jax.random.PRNGKey(0), (2, 16, 16, 3), 0, 256, dtype=jnp.uint8)


def test_get_value_shape(agent, obs):
    value = agent.get_value(obs)

    assert value.shape == (2, 1)
    assert jnp.isfinite(value).all()


def test_get_action_and_value_arity(agent, obs):
    out = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert len(out) == 7


def test_get_action_and_value_shapes(agent, obs):
    action, logprob, entropy, value, recon_x, latent_logits, z = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert action.shape == (2,)
    assert logprob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)
    assert recon_x.shape == obs.shape
    # Latent categorical logits (b, c, d) feed the KL term, not the policy head.
    assert latent_logits.shape == (2, 4, 8)
    assert z.shape == (2, 4, 8)


def test_actions_are_within_the_action_space(agent, obs):
    action, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.all(action >= 0)
    assert jnp.all(action < 6)


def test_log_prob_is_non_positive(agent, obs):
    _, logprob, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.all(logprob <= 0)


def test_entropy_is_non_negative(agent, obs):
    _, _, entropy, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.all(entropy >= 0)


def test_reconstruction_is_finite(agent, obs):
    *_, recon_x, _, _ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.isfinite(recon_x).all()


def test_sampling_is_deterministic_for_a_fixed_key(agent, obs):
    first = agent.get_action_and_value(obs, jax.random.PRNGKey(7))[0]
    second = agent.get_action_and_value(obs, jax.random.PRNGKey(7))[0]

    assert jnp.array_equal(first, second)


def test_externally_supplied_actions_are_used(agent, obs):
    actions = jnp.array([0, 5], dtype=jnp.int32)

    returned, logprob, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(0), action=actions)

    assert jnp.array_equal(returned, actions)
    assert jnp.isfinite(logprob).all()


def test_value_head_does_not_depend_on_sampling(agent, obs):
    """``get_value`` must run the encoder only (no straight-through sampling)."""
    value_a = agent.get_value(obs)
    # Sampling uses the agent RNG state; the value head must be unaffected by it.
    _ = agent.get_action_and_value(obs, jax.random.PRNGKey(3))
    value_b = agent.get_value(obs)

    assert jnp.allclose(value_a, value_b)


def test_policy_head_matches_the_action_space(agent, obs):
    """Regression: the policy used to be built from ``proj`` (256 units), not ``self.actor``."""
    action, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert int(agent.actor.out_features) == 6
    assert jnp.all(action < agent.actor.out_features)
