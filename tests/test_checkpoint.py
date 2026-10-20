"""Orbax checkpoint persistence (the training loop's save path)."""

import jax
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import nnx

pytestmark = pytest.mark.slow


def build_checkpoint(config, rngs):
    """A checkpoint payload shaped exactly like the one ``main.run`` writes."""
    agent_module = __import__('ppo_cleanrl.agent', fromlist=['PPOAgent'])
    utils = __import__('ppo_cleanrl.utils', fromlist=['*'])

    agent = agent_module.PPOAgent(config.agent_config, rngs)
    optimizer = utils.get_optimizer(config, agent)
    _, state = nnx.split((agent, optimizer))

    return {'state': state, 'global_step': 42, 'rng': jax.random.PRNGKey(0)}


def make_manager(path, max_to_keep=2):
    return ocp.CheckpointManager(
        path,
        options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep, create=True),
    )


def test_checkpoint_roundtrip(tmp_path, cleanrl_config, rngs_factory):
    ckpt = build_checkpoint(cleanrl_config, rngs_factory(0))
    manager = make_manager(tmp_path / 'checkpoints')

    manager.save(42, args=ocp.args.StandardSave(ckpt))
    manager.wait_until_finished()
    assert list(manager.all_steps()) == [42]

    restored = manager.restore(42, args=ocp.args.StandardRestore(ckpt))
    manager.close()

    assert restored['global_step'] == 42


def test_checkpoint_restores_parameters_exactly(tmp_path, cleanrl_config, rngs_factory):
    ckpt = build_checkpoint(cleanrl_config, rngs_factory(0))
    manager = make_manager(tmp_path / 'checkpoints')

    manager.save(42, args=ocp.args.StandardSave(ckpt))
    manager.wait_until_finished()
    restored = manager.restore(42, args=ocp.args.StandardRestore(ckpt))
    manager.close()

    original = jax.tree.leaves(ckpt['state'])
    loaded = jax.tree.leaves(restored['state'])
    assert len(original) == len(loaded)
    for before, after in zip(original, loaded, strict=True):
        assert np.allclose(np.asarray(before), np.asarray(after))


def test_checkpoint_manager_honours_max_to_keep(tmp_path, cleanrl_config, rngs_factory):
    ckpt = build_checkpoint(cleanrl_config, rngs_factory(0))
    manager = make_manager(tmp_path / 'checkpoints', max_to_keep=1)

    for step in (10, 20):
        manager.save(step, args=ocp.args.StandardSave(ckpt))
        manager.wait_until_finished()
    steps = list(manager.all_steps())
    manager.close()

    assert steps == [20]
