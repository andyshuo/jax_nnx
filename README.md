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

[`uv`](https://docs.astral.sh/uv/) is the fastest way to get set up:

```bash
uv venv --python 3.12
source .venv/bin/activate

# GPU build of JAX first, otherwise pip may resolve the CPU-only wheel.
uv pip install -r requirements-gpu.txt   # CUDA 12; plain "jax" for CPU-only
uv pip install -r requirements.txt

# ProcGen is only needed for actual training (it builds from source).
uv pip install -r requirements-env.txt
```

`requirements-env.txt` is intentionally separate: building ProcGen needs a C++
toolchain and is not needed by the test-suite, which stubs the environment out.

## Training

```bash
# VAE encoder, logged to Weights & Biases
python main.py --config vae

# ResNet baseline, fully offline, checkpoints under runs/
python main.py --config cleanrl --no-wandb --logdir runs

# Build the model and exit (no environment, no GPU required)
python main.py --config vae --dry-run
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
uv pip install -r requirements-dev.txt
pytest -q                 # model, losses, GAE, CLI, env wrappers, smoke tests
pytest -q -m "not slow"   # skip the end-to-end update steps
ruff check . && ruff format --check .
```

CI (`.github/workflows/ci.yml`) runs lint plus the full suite on Python 3.11-3.13 on
CPU, then dry-runs both variants to make sure the entry point still builds.

## License

No license has been chosen for this repository yet, so no rights to use, modify or
redistribute the code are granted. If you intend to reuse it, ask the author first.
