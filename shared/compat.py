"""Compatibility helpers for the Flax NNX API across the supported versions.

The repository pins Flax 0.10.x (see ``pixi.toml``) because ProcGen requires
``numpy < 2``, and JAX releases that need newer NumPy are therefore out of reach.
Newer Flax releases changed ``nnx.Optimizer.update`` from ``update(grads)`` to
``update(model, grads)``, so the call is dispatched on the installed signature
instead of guessing.
"""

import inspect

from flax import nnx

_UPDATE_PARAMETERS = inspect.signature(nnx.Optimizer.update).parameters
_UPDATE_TAKES_MODEL = 'model' in _UPDATE_PARAMETERS


def make_optimizer(agent, tx) -> nnx.Optimizer:
    """Build an ``nnx.Optimizer`` over the module's ``Param`` variables."""
    return nnx.Optimizer(agent, tx, wrt=nnx.Param)


def optimizer_update(optimizer, model, grads) -> None:
    """Apply ``grads`` to ``optimizer`` for both the old and new NNX signatures."""
    if _UPDATE_TAKES_MODEL:
        optimizer.update(model, grads)
    else:
        optimizer.update(grads)
