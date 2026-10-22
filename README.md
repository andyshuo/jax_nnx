# jax_nnx

PPO on [ProcGen](https://github.com/openai/procgen) written in JAX with [Flax NNX](https://flax.readthedocs.io/en/latest/nnx_basics.html).

The repo contains two reference implementations that share the same training loop,
environment wrappers and PPO objective:

| Variant   | Encoder                           | Extra losses                    | Select with        |
| --------- | --------------------------------- | ------------------------------- | ------------------ |
| `cleanrl` | IMPALA-style ResNet               | none (plain PPO)                | `--config cleanrl` |
| `vae`     | discrete VAE over the observation | reconstruction + categorical KL | `--config vae`     |
| `nature`  | DQN "Nature" CNN                  | none (plain PPO)                | `--config nature`  |
| `impala`  | deep IMPALA CNN (2 res blocks/stage) | none (plain PPO)             | `--config impala`  |
| `jepa`    | ResNet + latent predictor         | JEPA-style latent prediction    | `--config jepa`    |
| `vit`     | small Vision Transformer          | none (plain PPO)                | `--config vit`     |
| `tiny`    | 3-stage CNN, 3,712 parameters     | none (plain PPO)                | `--config tiny`    |

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
  agent.py  config.py  episode.py  modules.py  utils.py   (same five files per variant)
shared/                 ppo.py (losses/GAE), episode.py, make_envs.py, logger, compat
tests/                  pytest suite (models, losses, GAE, CLI, env wrappers, variants)
```

## Benchmarking the variants

```bash
pixi run benchmark                              # every variant, 200k steps
pixi run python scripts/benchmark.py --variants cleanrl jepa --timesteps 500000
```

Each variant is trained in its own process with `--no-wandb` and greedy evaluation
enabled; the script parses the evaluation lines and writes `results.json` and a
markdown `results.md` table next to the run directories.

### Smoke sweep

A 16,384-step sweep (`--timesteps 16384 --eval-steps 256`, i.e. **one** PPO update and
one 256-step greedy evaluation) run on CPU. It exercises every variant end to end —
that is all it is for; the returns are noise at this budget and say nothing about
which encoder is better.

| Variant | Params | Final eval return |
| --- | --- | --- |
| `cleanrl` | 626,256 | 1.43 |
| `nature` | 342,448 | 3.33 |
| `impala` | 626,256 | 2.00 |
| `vae` | 2,393,667 | 3.00 |
| `jepa` | 2,725,968 | n/a¹ |

`vit` and `tiny` were added after this sweep and have not been benchmarked yet.

¹ The `jepa` sweep ran before its predictor was shrunk, so its return is not
comparable. The predictor was two `Linear(2048 -> 2048)` layers (8.4M parameters, i.e.
93% of the model); it now has a 512-wide bottleneck and the variant is 2.7M parameters
in total. The full sweep takes tens of minutes on CPU.

## Tests

```bash
pixi run test        # full suite
pixi run test-fast   # skip the end-to-end update steps
pixi run lint        # ruff check + format check
```

There is **no GitHub Actions workflow**: the gate runs locally, on every commit,
through a `pre-commit` hook.

```bash
git config core.hooksPath .githooks   # one-time
pixi run ci                           # lint + full suite + entrypoint smoke
```

The hook runs exactly those three checks and aborts the commit when any of them
fails, so a green history is guaranteed by the commit itself rather than by a CI
service.

## License

No license has been chosen for this repository yet, so no rights to use, modify or
redistribute the code are granted. If you intend to reuse it, ask the author first.
