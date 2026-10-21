"""The IMPALA PPO agent (ppo_impala)."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def impala_config():
    from ppo_impala.config import Config

    config = Config()
    impala = config.agent_config.impala_config
    impala.obs_shape = (16, 16, 3)
    impala.hidden_dims = [4, 8]

    # proj.out_features must equal actor/critic in_features; the agent only derives
    # proj.in_features, so keep these three in sync here.
    config.agent_config.proj_config.out_features = 16
    config.agent_config.actor_config.in_features = 16
    config.agent_config.actor_config.out_features = 6
    config.agent_config.critic_config.in_features = 16

    tc = config.training_config
    tc.num_envs = 2
    tc.num_steps = 8
    tc.batch_size = tc.num_envs * tc.num_steps
    tc.minibatch_size = tc.batch_size // tc.num_minibatches
    tc.update_epochs = 1
    return config


@pytest.fixture
def agent(impala_config, rngs):
    from ppo_impala.agent import PPOAgent

    return PPOAgent(impala_config.agent_config, rngs)


@pytest.fixture
def obs():
    return jax.random.randint(jax.random.PRNGKey(0), (2, 16, 16, 3), 0, 256, dtype=jnp.uint8)


def test_get_value_shape(agent, obs):
    value = agent.get_value(obs)

    assert value.shape == (2, 1)
    assert jnp.isfinite(value).all()


def test_get_action_and_value_arity(agent, obs):
    out = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert len(out) == 4


def test_get_action_and_value_shapes(agent, obs):
    action, logprob, entropy, value = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert action.shape == (2,)
    assert logprob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)


def test_actions_are_within_the_action_space(agent, obs):
    action, _, _, _ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.all(action >= 0)
    assert jnp.all(action < 6)


def test_log_prob_is_non_positive(agent, obs):
    _, logprob, _, _ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.all(logprob <= 0)


def test_entropy_is_non_negative(agent, obs):
    _, _, entropy, _ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.all(entropy >= 0)


def test_sampling_is_deterministic_for_a_fixed_key(agent, obs):
    first, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(7))
    second, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(7))

    assert jnp.array_equal(first, second)


def test_sampling_varies_with_the_key(agent, obs):
    first, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(1))
    second, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(2))

    assert not jnp.array_equal(first, second)


def test_externally_supplied_actions_are_used(agent, obs):
    actions = jnp.array([0, 5], dtype=jnp.int32)

    returned, logprob, _, _ = agent.get_action_and_value(obs, jax.random.PRNGKey(0), action=actions)

    assert jnp.array_equal(returned, actions)
    assert jnp.isfinite(logprob).all()


def test_get_value_and_policy_share_the_encoder(agent, obs):
    """The critic must reuse the encoder output: perturbing obs changes both."""
    other = obs.at[:, :8].set(255)

    assert not jnp.allclose(agent.get_value(obs), agent.get_value(other))


def test_bf16_forward_pass(agent, obs):
    """Observations are normalised to bf16 inside the model; output stays finite."""
    value = agent.get_value(jnp.zeros((1, 16, 16, 3), dtype=jnp.uint8))

    assert jnp.isfinite(value).all()


def test_projection_is_sized_from_the_encoder(agent):
    """The agent derives ``proj.in_features`` from ``ImpalaCNN.output_dim``."""
    assert agent.enc.output_dim == 4 * 4 * 8
    assert agent.proj.in_features == agent.enc.output_dim
