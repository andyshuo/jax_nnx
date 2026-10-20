"""Generalised advantage estimation, checked against a reference implementation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

MODULES = ['ppo_cleanrl.utils', 'ppo_vae.utils']


def module(name):
    return __import__(name, fromlist=['*'])


def reference_gae(rewards, values, dones, next_done, next_value, gamma, gae_lambda):
    """Straight transcription of the original PyTorch/cleanrl GAE loop."""
    num_steps, num_envs = rewards.shape
    advantages = np.zeros_like(rewards)
    lastgaelam = np.zeros(num_envs)

    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        advantages[t] = lastgaelam

    return advantages, advantages + values


def random_rollout(num_steps, num_envs=2, seed=0):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=(num_steps, num_envs)).astype(np.float32),  # rewards
        rng.normal(size=(num_steps, num_envs)).astype(np.float32),  # values
        np.zeros((num_steps, num_envs), dtype=np.float32),  # dones
        np.zeros(num_envs, dtype=np.float32),  # next_done
        rng.normal(size=num_envs).astype(np.float32),  # next_value
    )


@pytest.mark.parametrize('name', MODULES)
@pytest.mark.parametrize('num_steps', [1, 3, 8, 17])
def test_gae_matches_reference(name, num_steps):
    rewards, values, dones, next_done, next_value = random_rollout(num_steps)

    advantages, returns = module(name).compute_advantages_and_target_returns(
        jnp.asarray(next_done),
        jnp.asarray(next_value),
        jnp.asarray(dones),
        jnp.asarray(values),
        jnp.asarray(rewards),
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected_advantages, expected_returns = reference_gae(rewards, values, dones, next_done, next_value, 0.99, 0.95)
    assert np.allclose(advantages, expected_advantages, atol=1e-5)
    assert np.allclose(returns, expected_returns, atol=1e-5)


@pytest.mark.parametrize('name', MODULES)
def test_gae_honours_gamma_and_lambda(name):
    rewards, values, dones, next_done, next_value = random_rollout(6)

    adv_a, _ = module(name).compute_advantages_and_target_returns(
        jnp.asarray(next_done),
        jnp.asarray(next_value),
        jnp.asarray(dones),
        jnp.asarray(values),
        jnp.asarray(rewards),
        gamma=0.9,
        gae_lambda=0.9,
    )
    adv_b, _ = module(name).compute_advantages_and_target_returns(
        jnp.asarray(next_done),
        jnp.asarray(next_value),
        jnp.asarray(dones),
        jnp.asarray(values),
        jnp.asarray(rewards),
        gamma=0.99,
        gae_lambda=0.5,
    )

    assert not np.allclose(adv_a, adv_b)


@pytest.mark.parametrize('name', MODULES)
def test_gae_returns_are_advantages_plus_values(name):
    rewards, values, dones, next_done, next_value = random_rollout(5)

    advantages, returns = module(name).compute_advantages_and_target_returns(
        jnp.asarray(next_done),
        jnp.asarray(next_value),
        jnp.asarray(dones),
        jnp.asarray(values),
        jnp.asarray(rewards),
    )

    assert np.allclose(returns, advantages + values, atol=1e-5)


@pytest.mark.parametrize('name', MODULES)
def test_gae_does_not_bootstrap_after_terminal(name):
    """With ``next_done=1`` the last advantage must be ``reward - value``."""
    rewards = jnp.zeros((3, 1)).at[-1, 0].set(2.0)
    values = jnp.full((3, 1), 0.5)
    dones = jnp.zeros((3, 1))
    next_done = jnp.ones((1,))
    next_value = jnp.zeros((1,))

    advantages, _ = module(name).compute_advantages_and_target_returns(
        next_done, next_value, dones, values, rewards, gamma=0.99, gae_lambda=0.95
    )

    # t=2: delta = 2.0 + 0.99*0*0 - 0.5 = 1.5, and no carry from a terminal state.
    assert np.allclose(advantages[-1], 1.5, atol=1e-6)


@pytest.mark.parametrize('name', MODULES)
def test_gae_bootstraps_when_not_terminal(name):
    rewards = jnp.zeros((1, 1))
    values = jnp.zeros((1, 1))
    dones = jnp.zeros((1, 1))
    next_done = jnp.zeros((1,))
    next_value = jnp.ones((1,))

    advantages, _ = module(name).compute_advantages_and_target_returns(
        next_done, next_value, dones, values, rewards, gamma=0.5, gae_lambda=0.9
    )

    assert np.allclose(advantages[0], 0.5, atol=1e-6)


@pytest.mark.parametrize('name', MODULES)
def test_gae_returns_are_stop_gradient(name):
    """Returns are regression targets: no gradient may flow through them."""
    _, values, dones, next_done, next_value = random_rollout(4)
    rewards = np.zeros_like(values)
    values = jnp.asarray(values)

    def objective(values):
        _, returns = module(name).compute_advantages_and_target_returns(
            jnp.asarray(next_done),
            jnp.asarray(next_value),
            jnp.asarray(dones),
            values,
            jnp.asarray(rewards),
        )
        return returns.sum()

    grads = jax.grad(objective)(values)

    assert np.allclose(grads, 0.0)
