"""The JEPA PPO agent (ppo_jepa): PPO outputs plus the latent-prediction branch."""

import jax
import jax.numpy as jnp
import pytest

ACTION_DIM = 6


@pytest.fixture
def jepa_config():
    """Tiny 16x16 config, mirroring ``conftest.cleanrl_config`` for this variant."""
    from ppo_jepa.config import Config

    config = Config()
    resnet = config.agent_config.resnet_config
    resnet.obs_shape = (16, 16, 3)
    resnet.hidden_dims = [4, 8]

    # proj.out_features must equal actor/critic in_features; the agent only derives
    # proj.in_features (from the encoder) and the predictor's dimensions.
    config.agent_config.proj_config.out_features = 16
    config.agent_config.actor_config.in_features = 16
    config.agent_config.actor_config.out_features = ACTION_DIM
    config.agent_config.critic_config.in_features = 16

    tc = config.training_config
    tc.num_envs = 2
    tc.num_steps = 8
    tc.num_minibatches = 1
    tc.batch_size = tc.num_envs * tc.num_steps
    tc.minibatch_size = tc.batch_size
    tc.update_epochs = 1
    return config


@pytest.fixture
def agent(jepa_config, rngs):
    from ppo_jepa.agent import PPOAgent

    return PPOAgent(jepa_config.agent_config, rngs, shift_px=jepa_config.shift_px)


@pytest.fixture
def obs():
    return jax.random.randint(jax.random.PRNGKey(0), (2, 16, 16, 3), 0, 256, dtype=jnp.uint8)


@pytest.fixture
def float_obs():
    """A differentiable (float) observation batch for the gradient checks."""
    return jax.random.uniform(jax.random.PRNGKey(0), (2, 16, 16, 3), minval=0.0, maxval=255.0)


def test_get_value_shape(agent, obs):
    value = agent.get_value(obs)

    assert value.shape == (2, 1)
    assert jnp.isfinite(value).all()


def test_act_greedy_is_a_deterministic_int_action(agent, obs):
    first = agent.act_greedy(obs)
    second = agent.act_greedy(obs)

    assert first.shape == (2,)
    assert jnp.issubdtype(first.dtype, jnp.integer)
    assert jnp.array_equal(first, second)
    assert jnp.all(first >= 0)
    assert jnp.all(first < ACTION_DIM)


def test_get_action_and_value_arity(agent, obs):
    out = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert len(out) == 6


def test_get_action_and_value_shapes(agent, obs):
    action, logprob, entropy, value, pred, target = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert action.shape == (2,)
    assert logprob.shape == (2,)
    assert entropy.shape == (2,)
    assert value.shape == (2, 1)
    assert pred.shape == (2, agent.enc.output_dim)
    assert target.shape == (2, agent.enc.output_dim)


def test_latent_branch_is_finite(agent, obs):
    _, _, _, _, pred, target = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.isfinite(pred).all()
    assert jnp.isfinite(target).all()
    assert pred.dtype == target.dtype == jnp.bfloat16


def test_actions_are_within_the_action_space(agent, obs):
    action, _, _, _, _, _ = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.all(action >= 0)
    assert jnp.all(action < ACTION_DIM)


def test_sampling_is_deterministic_for_a_fixed_key(agent, obs):
    first = agent.get_action_and_value(obs, jax.random.PRNGKey(7))
    second = agent.get_action_and_value(obs, jax.random.PRNGKey(7))

    for a, b in zip(first, second, strict=True):
        assert jnp.array_equal(a, b)


def test_sampling_varies_with_the_key(agent, obs):
    first, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(1))
    second, *_ = agent.get_action_and_value(obs, jax.random.PRNGKey(2))

    assert not jnp.array_equal(first, second)


def test_externally_supplied_actions_are_used(agent, obs):
    actions = jnp.array([0, ACTION_DIM - 1], dtype=jnp.int32)

    returned, logprob, _, _, _, _ = agent.get_action_and_value(obs, jax.random.PRNGKey(0), action=actions)

    assert jnp.array_equal(returned, actions)
    assert jnp.isfinite(logprob).all()


def test_target_is_the_latent_of_the_shifted_view(agent, obs):
    from ppo_jepa.agent import shift

    _, _, _, _, _, target = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    expected = agent.encode(shift(obs, agent.shift_px))
    unshifted = agent.encode(obs)

    assert jnp.allclose(target, expected, atol=1e-2, rtol=1e-2)
    # The shift has to change the target, otherwise the task is a copy.
    assert not jnp.allclose(expected, unshifted, atol=1e-2, rtol=1e-2)


def test_shift_rolls_along_the_width_axis(obs):
    from ppo_jepa.agent import shift

    shifted = shift(obs, 2)

    assert shifted.shape == obs.shape
    assert jnp.array_equal(shifted, jnp.roll(obs, 2, axis=2))
    assert jnp.array_equal(shifted[:, :, 2:], obs[:, :, :-2])


def test_shift_px_is_configurable(jepa_config, rngs, obs):
    from ppo_jepa.agent import PPOAgent

    agent = PPOAgent(jepa_config.agent_config, rngs, shift_px=5)
    _, _, _, _, _, target = agent.get_action_and_value(obs, jax.random.PRNGKey(0))

    assert jnp.allclose(target, agent.encode(jnp.roll(obs, 5, axis=2)), atol=1e-2, rtol=1e-2)


def test_target_is_a_stop_gradient_but_the_prediction_is_not(agent, float_obs):
    def target_sum(x):
        *_, target = agent.get_action_and_value(x, jax.random.PRNGKey(0))
        return target.sum()

    def pred_sum(x):
        *_, pred, _ = agent.get_action_and_value(x, jax.random.PRNGKey(0))
        return pred.sum()

    target_grad = jax.grad(target_sum)(float_obs)
    pred_grad = jax.grad(pred_sum)(float_obs)

    assert target_grad.shape == float_obs.shape
    assert jnp.all(target_grad == 0)
    assert jnp.any(pred_grad != 0)


def test_get_value_reuses_the_encoder(agent, obs):
    other = obs.at[:, :8].set(255)

    assert not jnp.allclose(agent.get_value(obs), agent.get_value(other))
