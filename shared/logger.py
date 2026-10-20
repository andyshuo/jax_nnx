"""Small logging abstraction so training can run with or without Weights & Biases.

``wandb`` is imported lazily, inside :class:`WandbLogger`, so that a CPU-only test
run or a ``--no-wandb`` training run never needs the package to be importable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np


class NullLogger:
    """No-op logger used by tests and ``--no-wandb`` runs."""

    enabled = False

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        del data, step

    def image(self, path: Path | str) -> str:
        return str(path)

    def finish(self) -> None:
        return None


class WandbLogger:
    """Thin wrapper around ``wandb`` that keeps jaX arrays/logging in one place."""

    enabled = True

    def __init__(self, **kwargs: Any) -> None:
        import wandb

        self._wandb = wandb
        # ``name=None`` is valid and lets wandb pick a random run name.
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        self._wandb.init(**kwargs)

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self._wandb.log({k: _to_numpy(v) for k, v in data.items()}, step=step)

    def image(self, path: Path | str) -> Any:
        return self._wandb.Image(str(path))

    def finish(self) -> None:
        self._wandb.finish()


def _to_numpy(value: Any) -> Any:
    """Convert JAX scalars/arrays (incl. bfloat16) to something wandb can log."""
    if isinstance(value, jax.Array):
        return np.asarray(jax.device_get(value))
    return value


def make_logger(enabled: bool = True, **kwargs: Any) -> NullLogger | WandbLogger:
    if not enabled:
        return NullLogger()
    return WandbLogger(**kwargs)
