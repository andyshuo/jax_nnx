"""The ViT-encoder PPO agent (ppo_vit)."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def vit_config():
    from ppo_vit.config import Config

    config = Config()
    config.agent_config.vit_config.obs_shape = (16, 16, 3)
    config.agent_config.vit_config.hidden_dims = [16, 2, 2]
    config.agent_config.vit_config.patch_size = 8
    config.agent_config.proj_config.out_features = 16
    config.agent_config.actor_config.in_features = 16
    config.agent_config.actor_config.out_features = 6
    config.agent_config.critic_config.in_features = 16
    return config


@pytest.fixture
def agent(vit_config, rngs):
    from ppo_vit.agent import PPOAgent

    return PPOAgent(vit_config.agent_config, rngs)


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
    action, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

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


def test_act_greedy_is_deterministic_and_in_range(agent, obs):
    first = agent.act_greedy(obs)
    second = agent.act_greedy(obs)

    assert jnp.array_equal(first, second)
    assert jnp.all(first < 6)
