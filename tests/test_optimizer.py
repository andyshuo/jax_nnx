"""Optimizer construction (regression: the old signature referenced a global ``agent``)."""

import jax
import optax
import pytest
from flax import nnx


def build(kind, config, rngs):
    agent_module = __import__(f'ppo_{kind}.agent', fromlist=['PPOAgent'])
    utils = __import__(f'ppo_{kind}.utils', fromlist=['*'])
    agent = agent_module.PPOAgent(config.agent_config, rngs)
    return agent, utils


@pytest.mark.parametrize(
    'kind, config_fixture',
    [
        ('cleanrl', 'cleanrl_config'),
        ('vae', 'vae_config'),
    ],
)
def test_get_optimizer_returns_nnx_optimizer(kind, config_fixture, request, rngs):
    config = request.getfixturevalue(config_fixture)
    agent, utils = build(kind, config, rngs)

    optimizer = utils.get_optimizer(config, agent)

    assert isinstance(optimizer, nnx.Optimizer)


@pytest.mark.parametrize(
    'kind, config_fixture',
    [
        ('cleanrl', 'cleanrl_config'),
        ('vae', 'vae_config'),
    ],
)
def test_get_optimizer_accepts_keyword_arguments(kind, config_fixture, request, rngs):
    config = request.getfixturevalue(config_fixture)
    agent, utils = build(kind, config, rngs)

    optimizer = utils.get_optimizer(config=config, agent=agent)

    assert isinstance(optimizer, nnx.Optimizer)


@pytest.mark.parametrize(
    'kind, config_fixture',
    [
        ('cleanrl', 'cleanrl_config'),
        ('vae', 'vae_config'),
    ],
)
def test_get_optimizer_wraps_a_gradient_transformation(kind, config_fixture, request, rngs):
    config = request.getfixturevalue(config_fixture)
    agent, utils = build(kind, config, rngs)

    optimizer = utils.get_optimizer(config, agent)

    assert isinstance(optimizer.tx, optax.GradientTransformation)


@pytest.mark.parametrize(
    'kind, config_fixture',
    [
        ('cleanrl', 'cleanrl_config'),
        ('vae', 'vae_config'),
    ],
)
def test_optimizer_state_is_created_for_all_params(kind, config_fixture, request, rngs):
    config = request.getfixturevalue(config_fixture)
    agent, utils = build(kind, config, rngs)

    optimizer = utils.get_optimizer(config, agent)

    opt_state = nnx.state(optimizer)
    assert len(jax.tree.leaves(opt_state)) > 0


@pytest.mark.parametrize(
    'kind, config_fixture',
    [
        ('cleanrl', 'cleanrl_config'),
        ('vae', 'vae_config'),
    ],
)
def test_anneal_lr_builds_an_optimizer(kind, config_fixture, request, rngs):
    config = request.getfixturevalue(config_fixture)
    config.training_config.anneal_lr = True
    agent, utils = build(kind, config, rngs)

    optimizer = utils.get_optimizer(config, agent)

    assert isinstance(optimizer, nnx.Optimizer)
    # A schedule makes optax track the step counter, so the state is non-trivial.
    assert len(jax.tree.leaves(nnx.state(optimizer))) > 0
