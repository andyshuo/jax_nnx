# jax_nnx

PPO on [ProcGen](https://github.com/openai/procgen) written in JAX with [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).

The repo contains two reference implementations that share the same training loop,
environment wrappers and PPO objective:

| Variant   | Encoder                           | Extra losses                    | Select with        |
| --------- | --------------------------------- | ------------------------------- | ------------------ |
| `cleanrl` | IMPALA-style ResNet               | none (plain PPO)                | `--config cleanrl` |
| `vae`     | discrete VAE over the observation | reconstruction + categorical KL | `--config vae`     |

The `vae` variant encodes each frame into `num_categories × categorical_dim` discrete
latents with a straight-through estimator, then feeds the flattened latent codes to a
small projection head, the policy head and the value head. The decoder is used only for
the auxiliary reconstruction loss.

## Installation

Environments are managed with [pixi](https://pixi.sh):

```bash
pixi install          # CPU JAX: test-suite and CI
pixi install -e gpu   # CUDA 12 JAX: real training
```

The environment pins Python 3.12, JAX 0.6.x, Flax 0.10.x and NumPy 1.x. That is not
gratuitous: ProcGen (through the maintained `procgen-mirror` PyPI mirror) requires
`numpy < 2`, while JAX >= 0.7 requires `numpy >= 2.2`. `shared/compat.py` bridges the
Flax `Optimizer.update` signature change, so the code also runs on newer Flax.

## Training

```bash
# VAE encoder, logged to Weights & Biases
pixi run python main.py --config vae

# ResNet baseline, fully offline, checkpoints under runs/
pixi run python main.py --config cleanrl --no-wandb --logdir runs

# Build the model and exit (no environment, no GPU required)
pixi run python main.py --config vae --dry-run
```

Useful flags:

| Flag              | Default | Meaning                                              |
| ----------------- | ------- | ---------------------------------------------------- |
| `--config`        | `vae`   | `cleanrl` or `vae`                                   |
| `--seed`          | `1`     | Seeds Python, NumPy and JAX                          |
| `--logdir`        | `logs`  | Root for timestamped run directories and checkpoints |
| `--no-wandb`      | off     | Disable W&B entirely (no account needed)             |
| `--capture-video` | off     | Record videos with `gym.wrappers.RecordVideo`        |
| `--dry-run`       | off     | Build agent + optimizer, print a summary, exit       |

Each run writes to `<logdir>/<run-name>_<timestamp>/`, containing Orbax checkpoints
(`checkpoints/`) and VAE reconstructions (`images/`). Online logging uses the
`WANDB_PROJECT`/`WANDB_ENTITY` environment variables when the config values are unset.

## How it works

* All parameters live in a single `nnx.Module`; the agent and optimizer state are
  replicated across devices and the device mesh is a one-axis `batch` mesh sized from
  `jax.device_count()` (1 GPU, 4 GPUs or CPU all work).
* The forward pass runs in `bfloat16` with `float32` parameters.
* Advantages use GAE implemented as a `lax.scan`, so any rollout length works.
* PPO's clipped surrogate, clipped value loss, entropy bonus and global-norm clipping
  follow the cleanrl reference implementation.

### Multi-device note

The rollout forward pass shards observations across the `batch` axis. The PPO update
step currently runs **replicated** on every device: JAX >= 0.7 requires an explicit
`out_sharding` for convolution gradients whose batch axis is sharded, which Flax's
layers do not expose, so a true data-parallel update is not expressible in this
codebase. Training is still correct on any device count — it simply does not use
extra accelerators to speed up the update epochs.

## Repository layout

```
main.py                 entry point: CLI, mesh, training loop, checkpointing
shared/
  make_envs.py          ProcGen vectorised env + gym wrappers
  logger.py             Weights & Biases / no-op logging backend
ppo_cleanrl/            ResNet encoder variant
  agent.py  config.py  episode.py  modules.py  utils.py
ppo_vae/                discrete-VAE encoder variant
  agent.py  config.py  episode.py  modules.py  utils.py
tests/                  pytest suite (model, losses, GAE, CLI, env wrappers)
```

## Tests

```bash
pixi run test        # full suite
pixi run test-fast   # skip the end-to-end update steps
pixi run lint        # ruff check + format check
```

CI (`.github/workflows/ci.yml`) provisions the pixi environment, runs lint plus the
full suite on CPU, then dry-runs both variants to make sure the entry point builds.

## License

No license has been chosen for this repository yet, so no rights to use, modify or
redistribute the code are granted. If you intend to reuse it, ask the author first.
