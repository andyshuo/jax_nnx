"""Train PPO on ProcGen with Flax NNX.

Two implementations are available:

* ``cleanrl`` - ResNet encoder, plain PPO (cleanrl-style).
* ``vae``     - discrete-VAE encoder, PPO + reconstruction/KL auxiliary losses.

Examples
--------
Train the VAE variant, logging to Weights & Biases::

    python main.py --config vae

Train the ResNet variant offline, writing checkpoints under ``runs/``::

    python main.py --config cleanrl --no-wandb --logdir runs

Build the model, print a summary and exit (used by CI)::

    python main.py --config vae --dry-run
"""

import os

# XLA / NCCL knobs must be set before JAX initialises its backend, so this block
# has to stay above the `import jax` line. They used to sit *after* it, which made
# the preallocation and Triton flags easy to miss.
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault(
    'XLA_FLAGS',
    '--xla_gpu_triton_gemm_any=True --xla_gpu_enable_latency_hiding_scheduler=true',
)
os.environ.setdefault('NCCL_LL128_BUFFSIZE', '-2')
os.environ.setdefault('NCCL_LL_BUFFSIZE', '-2')
os.environ.setdefault('NCCL_PROTO', 'SIMPLE,LL,LL128')

import argparse
import importlib
import json
import random
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from jax import NamedSharding
from jax.sharding import PartitionSpec as P
from PIL import Image

from shared.evaluate import evaluate
from shared.logger import make_logger
from shared.make_envs import make_procgen

jax.config.update('jax_threefry_partitionable', True)

CONFIG_CHOICES = ('cleanrl', 'vae', 'nature', 'impala', 'jepa')
DEFAULT_CONFIG = 'vae'
DEFAULT_LOG_DIR = 'logs'


# --------------------------------------------------------------------------- #
# Configuration / device helpers (kept as functions so tests can call them)
# --------------------------------------------------------------------------- #
def build_config(kind: str):
    """Return ``(Config, StrConfig)`` for ``kind`` with derived sizes filled in."""
    module = importlib.import_module(f'ppo_{kind}.config')
    config = module.Config()
    strconfig = module.StrConfig()

    tc = config.training_config
    tc.batch_size = int(tc.num_envs * tc.num_steps)
    tc.minibatch_size = int(tc.batch_size // tc.num_minibatches)
    tc.num_iterations = int(tc.total_timesteps // tc.batch_size)
    return config, strconfig


def make_mesh():
    """Build a one-axis ``batch`` mesh over every visible device.

    The mesh size used to be hard-coded to ``(2,)``, so the script crashed on a
    single GPU (or CPU) and ignored the other two GPUs on a 4-GPU box.
    """
    device_count = jax.device_count()
    mesh = jax.make_mesh((device_count,), ('batch',))
    return mesh, NamedSharding(mesh, P())


def make_logdir(root: str | Path, run_name: str) -> Path:
    """Create and return a timestamped run directory under ``root``."""
    path = Path(root).expanduser() / f'{run_name}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int):
    """Seed Python/NumPy/JAX and return ``(rng, nnx.Rngs)``."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    rng = jax.random.PRNGKey(seed)
    rng, rng_params, rng_dropout, rng_sampling, rng_categorical = jax.random.split(rng, 5)
    rngs = nnx.Rngs(
        params=rng_params,
        dropout=rng_dropout,
        sampling=rng_sampling,
        categorical=rng_categorical,
    )
    return rng, rngs


def count_params(agent) -> int:
    return int(sum(leaf.size for leaf in jax.tree.leaves(nnx.state(agent, nnx.Param))))


def build_model(kind: str, config, rngs, replicated_sharding):
    """Instantiate the agent + optimizer, replicate them and split into (graphdef, state)."""
    agent_module = importlib.import_module(f'ppo_{kind}.agent')
    utils_module = importlib.import_module(f'ppo_{kind}.utils')

    agent = agent_module.PPOAgent(config.agent_config, rngs)
    optimizer = utils_module.get_optimizer(config, agent)

    state = nnx.state((agent, optimizer))
    state = jax.device_put(state, replicated_sharding)
    nnx.update((agent, optimizer), state)
    graphdef, state = nnx.split((agent, optimizer))

    return agent, optimizer, graphdef, state


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train PPO on ProcGen with JAX/Flax NNX.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--config',
        type=str,
        choices=CONFIG_CHOICES,
        default=DEFAULT_CONFIG,
        help='Which implementation to run.',
    )
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument(
        '--logdir',
        type=str,
        default=DEFAULT_LOG_DIR,
        help='Root directory for run logs and checkpoints (relative paths are recommended).',
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging (fully offline run).',
    )
    parser.add_argument(
        '--capture-video',
        action='store_true',
        help='Record environment videos with gym.wrappers.RecordVideo.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build the model and exit without touching the environment.',
    )
    parser.add_argument('--total-timesteps', type=int, default=None, help='Override the training budget.')
    parser.add_argument('--num-envs', type=int, default=None, help='Override the env count.')
    parser.add_argument('--num-steps', type=int, default=None, help='Override the rollout length.')
    parser.add_argument('--learning-rate', type=float, default=None, help='Override the learning rate.')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=0,
        metavar='N',
        help='Evaluate the greedy policy every N iterations (0 disables evaluation).',
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=4096,
        metavar='S',
        help='Number of environment steps per evaluation.',
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='RUN_DIR',
        help='Resume from the latest checkpoint inside RUN_DIR (a previous run directory).',
    )
    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #


def apply_overrides(config, args: argparse.Namespace) -> None:
    """Apply CLI hyper-parameter overrides and recompute the derived batch sizes."""
    tc = config.training_config
    if args.total_timesteps is not None:
        tc.total_timesteps = args.total_timesteps
    if args.num_envs is not None:
        tc.num_envs = args.num_envs
    if args.num_steps is not None:
        tc.num_steps = args.num_steps
    if args.learning_rate is not None:
        tc.learning_rate = args.learning_rate

    tc.batch_size = int(tc.num_envs * tc.num_steps)
    tc.minibatch_size = max(1, tc.batch_size // tc.num_minibatches)
    tc.num_iterations = int(tc.total_timesteps // tc.batch_size)


def write_config(logdir: Path, config, strconfig) -> Path:
    """Dump the fully resolved config next to the run, for reproducibility."""
    path = logdir / 'config.json'
    payload = {'config': asdict(config), 'strconfig': asdict(strconfig)}
    path.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
    return path


def restore_checkpoint(checkpoint_manager, template):
    """Restore the latest checkpoint into ``template``; ``None`` when there is none."""
    step = checkpoint_manager.latest_step()
    if step is None:
        return None
    return checkpoint_manager.restore(step, args=ocp.args.StandardRestore(template))


def run(args: argparse.Namespace) -> dict:
    """Run training (or just the model build when ``args.dry_run``) and return a summary."""
    config, strconfig = build_config(args.config)
    config.seed = args.seed
    config.capture_video = args.capture_video
    strconfig.capture_video = args.capture_video
    apply_overrides(config, args)

    rng, rngs = seed_everything(args.seed)
    if args.resume:
        logdir = Path(args.resume).expanduser()
        if not logdir.is_dir():
            raise SystemExit(f'--resume: {logdir} is not a directory')
    else:
        logdir = make_logdir(args.logdir, strconfig.wandb_config.name)
    write_config(logdir, config, strconfig)
    mesh, replicated_sharding = make_mesh()

    agent, _, graphdef, state = build_model(args.config, config, rngs, replicated_sharding)

    summary = {
        'config': args.config,
        'logdir': str(logdir),
        'device_count': int(jax.device_count()),
        'params': count_params(agent),
        'batch_size': config.training_config.batch_size,
        'num_iterations': config.training_config.num_iterations,
    }
    print(
        f'[{args.config}] devices={summary["device_count"]} params={summary["params"]:,} '
        f'batch={summary["batch_size"]:,} iterations={summary["num_iterations"]:,} '
        f'logdir={logdir}'
    )

    if args.dry_run:
        print('--dry-run: model built successfully, exiting before env creation.')
        return summary

    utils_module = importlib.import_module(f'ppo_{args.config}.utils')
    make_episode = importlib.import_module(f'ppo_{args.config}.episode').make_episode
    env_step = utils_module.env_step
    train = utils_module.train

    logger = make_logger(
        enabled=not args.no_wandb,
        **asdict(strconfig.wandb_config),
        config=asdict(config.training_config),
    )

    checkpoint_manager = ocp.CheckpointManager(
        logdir / 'checkpoints',
        options=ocp.CheckpointManagerOptions(max_to_keep=2, create=True),
    )

    envs = make_procgen(strconfig)
    eval_envs = make_procgen(strconfig) if args.eval_interval > 0 else None
    episode = make_episode(
        obs_shape=envs.single_observation_space.shape,
        act_dim=envs.single_action_space.shape,
        num_steps=config.training_config.num_steps,
        num_envs=config.training_config.num_envs,
    )

    global_step = 0
    start_iteration = 1
    if args.resume:
        restored = restore_checkpoint(
            checkpoint_manager,
            {'state': state, 'global_step': global_step, 'rng': rng},
        )
        if restored is None:
            raise SystemExit(f'--resume: no checkpoint in {logdir / "checkpoints"}')
        state, rng = restored['state'], restored['rng']
        global_step = int(restored['global_step'])
        start_iteration = global_step // max(1, config.training_config.batch_size) + 1
        print(f'resumed from step {global_step:,} (iteration {start_iteration:,})')

    start_time = time.time()
    next_obs = np.array(envs.reset())
    next_done = np.zeros(config.training_config.num_envs)

    for iteration in range(start_iteration, config.training_config.num_iterations + 1):
        rollout_start_time = time.time()
        global_step, episode = env_step(
            global_step,
            envs,
            episode,
            next_obs,
            next_done,
            mesh,
            graphdef,
            state,
            rng,
            config,
            logger=logger,
        )
        rollout_spent_time = time.time() - rollout_start_time

        data = {
            'obs': jnp.array(episode.obs, dtype=jnp.uint8),
            'actions': jnp.array(episode.actions, dtype=jnp.float32),
            'logprobs': jnp.array(episode.logprobs, dtype=jnp.float32),
            'values': jnp.array(episode.values, dtype=jnp.float32),
            'rewards': jnp.array(episode.rewards, dtype=jnp.float32),
            'dones': jnp.array(episode.dones, dtype=jnp.float32),
            'next_obs': jnp.array(next_obs, dtype=jnp.uint8),
            'next_done': jnp.array(next_done, dtype=jnp.float32),
        }

        training_start_time = time.time()
        loss, state, metrics = train(graphdef, state, data, mesh, config, rng, logger=logger)
        training_spent_time = time.time() - training_start_time

        image = metrics.pop('Images/recon_x', None)
        if image is not None:
            image_dir = logdir / 'images'
            image_dir.mkdir(parents=True, exist_ok=True)
            image_path = image_dir / f'recon_x_{global_step}.png'
            # Recon targets live on the 0-255 observation scale, so clip instead of
            # the previous `* 255.0`, which overflowed uint8.
            pixels = np.clip(np.asarray(image, dtype=np.float32), 0, 255).astype(np.uint8)
            Image.fromarray(pixels).save(image_path)
            metrics['Images/recon_x'] = logger.image(image_path)

        sps = int(global_step / (time.time() - start_time))
        logger.log(
            {
                **metrics,
                'Time/SPS': sps,
                'Time/Training_Spent_Time': training_spent_time,
                'Time/Rollout_Spent_Time': rollout_spent_time,
            },
            step=global_step,
        )
        print(
            f'iter {iteration}/{config.training_config.num_iterations} step={global_step:,} '
            f'loss={float(loss):.4f} sps={sps:,}'
        )

        if eval_envs is not None and iteration % args.eval_interval == 0:
            eval_metrics = evaluate(
                eval_envs,
                mesh,
                graphdef,
                state,
                utils_module.get_greedy_action,
                args.eval_steps,
            )
            logger.log(eval_metrics, step=global_step)
            print(
                f'eval @ step {global_step:,}: return={eval_metrics["Eval/return"]:.2f} '
                f'over {int(eval_metrics["Eval/episodes"])} episodes'
            )

        if global_step % (config.training_config.num_steps * config.training_config.num_envs * 100) == 0:
            ckpt = {
                'state': state,
                'global_step': global_step,
                'rng': rng,
                'episode': episode,
            }
            checkpoint_manager.save(global_step, args=ocp.args.StandardSave(ckpt))

    envs.close()
    if eval_envs is not None:
        eval_envs.close()
    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    logger.finish()
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
