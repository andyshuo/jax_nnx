"""Training utilities for the VAE-encoder PPO variant."""

from functools import partial
from typing import Any

import distrax
import einops
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from jax import NamedSharding
from jax.sharding import PartitionSpec as P
from tqdm import tqdm

from shared.compat import make_optimizer, optimizer_update
from shared.logger import NullLogger

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8
sg = jax.lax.stop_gradient


# --------------------------------------------------------------------------- #
# Model calls
# --------------------------------------------------------------------------- #
@jax.jit
def get_action_and_value(graphdef, state, obs, rng):
    agent, _ = nnx.merge(graphdef, state)
    return agent.get_action_and_value(obs, rng)


@jax.jit
def get_value(graphdef, state, obs):
    agent, _ = nnx.merge(graphdef, state)
    return agent.get_value(obs)


@jax.jit
def get_greedy_action(graphdef, state, obs):
    agent, _ = nnx.merge(graphdef, state)
    return agent.act_greedy(obs)


# --------------------------------------------------------------------------- #
# Loss pieces (module level so they can be unit-tested individually)
# --------------------------------------------------------------------------- #
def compute_ratio(newlogprob, logprob):
    logratio = newlogprob - logprob
    return jnp.exp(logratio), logratio


def compute_policy_loss(advantage, advantages_mean, advantages_std, ratio, clip_coef, norm_adv=True):
    if norm_adv:
        advantage = (advantage - advantages_mean) / (advantages_std + 1e-8)
    pg_loss1 = -advantage * ratio
    pg_loss2 = -advantage * jnp.clip(ratio, 1 - clip_coef, 1 + clip_coef)
    return jnp.maximum(pg_loss1, pg_loss2)


def compute_value_loss(newvalue, return_, value, clip_coef, clip_vloss=True):
    v_loss_unclipped = (newvalue - return_) ** 2
    if not clip_vloss:
        return 0.5 * v_loss_unclipped
    v_clipped = value + jnp.clip(newvalue - value, -clip_coef, clip_coef)
    v_loss_clipped = (v_clipped - return_) ** 2
    return 0.5 * jnp.maximum(v_loss_unclipped, v_loss_clipped)


def compute_vae_loss(recon_x, obs):
    return jnp.mean(jnp.square(recon_x - obs))


def compute_kl_loss(logits):
    q_dist = distrax.Categorical(logits=logits)
    p_dist = distrax.Categorical(logits=jnp.zeros_like(logits))
    return q_dist.kl_divergence(p_dist).mean()


@jax.jit
def compute_clipfracs(ratio, logratio, clip_coef):
    old_approx_kl = (-logratio).mean()
    approx_kl = ((ratio - 1) - logratio).mean()
    cf = jnp.mean(jnp.abs(ratio - 1.0) > clip_coef)
    return sg(old_approx_kl), sg(approx_kl), sg(cf)


def compute_explained_variance(y_pred, y_true):
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


# --------------------------------------------------------------------------- #
# GAE
# --------------------------------------------------------------------------- #
def compute_advantages_and_target_returns(
    next_done,
    next_value,
    dones,
    values,
    rewards,
    gamma=0.999,
    gae_lambda=0.95,
):
    """Generalised advantage estimation over a ``(T, B)`` rollout.

    Replaces the original reverse Python loop, which hard-coded a 256-step
    rollout (any other length was silently mis-indexed) and ignored
    ``config.gamma`` / ``config.gae_lambda``. A
    ``lax.scan`` over the reversed rollout supports any length and stays jittable.
    """
    not_terminal = jnp.concatenate([1.0 - dones[1:], (1.0 - next_done)[None]], axis=0)
    next_values = jnp.concatenate([values[1:], next_value[None]], axis=0)

    def step(lastgaelam, xs):
        reward_t, value_t, not_terminal_t, next_value_t = xs
        delta = reward_t + gamma * next_value_t * not_terminal_t - value_t
        lastgaelam = delta + gamma * gae_lambda * not_terminal_t * lastgaelam
        return lastgaelam, lastgaelam

    xs = (rewards[::-1], values[::-1], not_terminal[::-1], next_values[::-1])
    _, advantages = jax.lax.scan(step, jnp.zeros_like(values[0]), xs)
    advantages = advantages[::-1]

    return advantages, sg(advantages + values)


# --------------------------------------------------------------------------- #
# Objective / update step
# --------------------------------------------------------------------------- #
def loss_fn(
    agent,
    batch,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    recon_coef=1.0,
    kl_coef=1.0,
    norm_adv=True,
    clip_vloss=True,
):
    """PPO + VAE objective, as a pure function of ``agent`` so it can be diffed."""
    mb_obs = batch['mb_obs']
    mb_actions = batch['mb_actions']
    mb_logprobs = batch['mb_logprobs']
    mb_advantages = batch['mb_advantages']
    mb_returns = batch['mb_returns']
    mb_values = batch['mb_values']

    _, rng_new = jax.random.split(batch['rng'])
    _, newlogprob, entropy, newvalue, recon_x, logits, _ = agent.get_action_and_value(
        mb_obs, action=mb_actions, rng=rng_new
    )

    ratio, logratio = compute_ratio(newlogprob.astype(f32), mb_logprobs)
    pg_loss = jnp.mean(
        compute_policy_loss(
            mb_advantages,
            mb_advantages.mean(),
            mb_advantages.std(),
            ratio,
            clip_coef,
            norm_adv,
        )
    )

    newvalue = einops.rearrange(newvalue, 'b 1 -> b')
    v_loss = jnp.mean(compute_value_loss(newvalue, mb_returns, mb_values, clip_coef, clip_vloss))
    entropy_loss = entropy.mean()

    kl_loss = compute_kl_loss(logits)
    recon_loss = compute_vae_loss(recon_x, mb_obs)

    loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss + recon_coef * recon_loss + kl_coef * kl_loss
    aux = (pg_loss, v_loss, entropy_loss, recon_loss, kl_loss, logratio, ratio, recon_x, mb_obs)
    return loss, aux


@partial(jax.jit, static_argnames=('norm_adv', 'clip_vloss'))
def train_step(graphdef, state, batch, clip_coef, ent_coef, vf_coef, recon_coef, kl_coef, norm_adv, clip_vloss):
    agent, optimizer = nnx.merge(graphdef, state)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, has_aux=True)(
        agent,
        batch,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        recon_coef=recon_coef,
        kl_coef=kl_coef,
        norm_adv=norm_adv,
        clip_vloss=clip_vloss,
    )
    optimizer_update(optimizer, agent, grads)
    return loss, nnx.state((agent, optimizer)), aux


# --------------------------------------------------------------------------- #
# Rollout collection
# --------------------------------------------------------------------------- #
def env_step(
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
    logger: Any | None = None,
):
    """Collect ``config.training_config.num_steps`` steps from the vectorised env."""
    logger = logger or NullLogger()

    for step in range(config.training_config.num_steps):
        global_step += config.training_config.num_envs
        episode.obs[step] = next_obs
        episode.dones[step] = next_done

        rng, step_rng = jax.random.split(rng)
        next_obs = jax.device_put(
            jnp.array(next_obs, dtype=u8),
            NamedSharding(mesh, P('batch', None, None, None)),
        )
        action, logprob, _, value, _, _, _ = get_action_and_value(graphdef, state, next_obs, step_rng)
        action, logprob, value = sg(action), sg(logprob), sg(value)

        episode.actions[step] = action
        episode.logprobs[step] = logprob
        episode.values[step] = einops.rearrange(value, 'b 1 -> b')

        next_obs, reward, next_done, info = envs.step(np.asarray(action))
        episode.rewards[step] = reward

        for item in info:
            if 'episode' in item:
                logger.log(
                    {
                        'charts/episodic_return': item['episode']['r'],
                        'charts/episodic_length': item['episode']['l'],
                    },
                    step=global_step,
                )
                break

    return global_step, episode


# --------------------------------------------------------------------------- #
# Update loop
# --------------------------------------------------------------------------- #
def train(graphdef, state, data, mesh, config, rng, logger: Any | None = None):
    """Run PPO update epochs over one rollout and return finite metrics."""
    del logger  # logged by the caller; kept for a uniform signature across variants
    tc = config.training_config

    # Only the observations are big enough to be worth sharding. The (num_steps,
    # num_envs) scalars stay replicated: GAE shifts, concatenates and reshapes them
    # into (num_steps * num_envs,), and merging a sharded axis into a flat axis is
    # not a valid reshape. The original code sharded the *step* axis here, which
    # both mis-named the batch and broke the bootstrap row.
    env_sharding = NamedSharding(mesh, P('batch'))
    replicated = NamedSharding(mesh, P())

    next_value = get_value(graphdef, state, jax.device_put(data['next_obs'], env_sharding))
    next_value = einops.rearrange(next_value, 'b 1 -> b')
    next_value = jax.device_put(next_value, replicated)

    b_advantages, b_returns = compute_advantages_and_target_returns(
        jax.device_put(data['next_done'], replicated),
        next_value,
        jax.device_put(data['dones'], replicated),
        jax.device_put(data['values'], replicated),
        jax.device_put(data['rewards'], replicated),
        gamma=tc.gamma,
        gae_lambda=tc.gae_lambda,
    )
    b_advantages, b_returns = sg(b_advantages), sg(b_returns)

    b_obs = einops.rearrange(data['obs'], 'b l ... -> (b l) ...')
    b_actions = einops.rearrange(data['actions'], 'b l ... -> (b l) ...')
    b_logprobs = einops.rearrange(data['logprobs'], 'b l -> (b l)')
    b_advantages = einops.rearrange(b_advantages, 'b l -> (b l)')
    b_returns = einops.rearrange(b_returns, 'b l -> (b l)')
    b_values = einops.rearrange(data['values'], 'b l -> (b l)')

    b_inds = np.arange(tc.batch_size)
    clipfracs = []
    aux = None

    for _ in tqdm(range(tc.update_epochs), desc='update epochs', leave=False):
        np.random.shuffle(b_inds)
        for start in range(0, tc.batch_size, tc.minibatch_size):
            mb_inds = b_inds[start : start + tc.minibatch_size]
            mbatch = {
                'mb_obs': b_obs[mb_inds],
                'mb_actions': b_actions[mb_inds],
                'mb_logprobs': b_logprobs[mb_inds],
                'mb_advantages': b_advantages[mb_inds],
                'mb_returns': b_returns[mb_inds],
                'mb_values': b_values[mb_inds],
                'rng': rng,
            }
            # Replicated on purpose: JAX >= 0.7 requires explicit ``out_sharding``
            # for conv gradients over a sharded batch, which Flax's layers do not
            # expose, so a data-parallel update is not expressible here. Only the
            # rollout forward pass is sharded.
            mbatch = jax.device_put(mbatch, NamedSharding(mesh, P()))
            loss, state, aux = train_step(
                graphdef,
                state,
                mbatch,
                tc.clip_coef,
                tc.ent_coef,
                tc.vf_coef,
                config.recon_coef,
                config.kl_coef,
                tc.norm_adv,
                tc.clip_vloss,
            )
            pg_loss, v_loss, entropy_loss, recon_loss, kl_loss, logratio, ratio, recon_x, mb_obs = aux
            for name, value in {
                'loss': loss,
                'v_loss': v_loss,
                'pg_loss': pg_loss,
                'entropy_loss': entropy_loss,
                'recon_loss': recon_loss,
                'kl_loss': kl_loss,
            }.items():
                assert jnp.isfinite(value), f'Losses/{name} is not finite'
            old_approx_kl, approx_kl, cf = compute_clipfracs(ratio, logratio, tc.clip_coef)
            clipfracs.append(cf)
            # cleanrl-style early stop: stop minibatching once the policy moved
            # further than 1.5x the configured KL budget.
            if tc.target_kl is not None and float(approx_kl) > 1.5 * tc.target_kl:
                break

    if aux is None:
        raise ValueError(
            f'no minibatch was processed (batch_size={tc.batch_size}, '
            f'minibatch_size={tc.minibatch_size}); check the training config'
        )

    image = np.concatenate([np.asarray(mb_obs[0]), np.asarray(recon_x[0], dtype=np.float32)], axis=1)

    metrics = {
        'Losses/loss': loss,
        'Losses/v_loss': v_loss,
        'Losses/pg_loss': pg_loss,
        'Losses/entropy_loss': entropy_loss,
        'Losses/recon_loss': recon_loss,
        'Losses/kl_loss': kl_loss,
        'Losses/old_approx_kl': old_approx_kl,
        'Losses/approx_kl': approx_kl,
        'Losses/clipfracs': np.mean(clipfracs),
        'Losses/explained_var': compute_explained_variance(b_values, b_returns),
        'Images/recon_x': image,
    }
    for key, value in metrics.items():
        if 'Images' not in key:
            assert jnp.isfinite(value), f'{key} is not finite'
    return loss, state, metrics


def get_optimizer(config, agent):
    """Adam with global-norm clipping over ``agent``'s parameters."""
    tc = config.training_config
    learning_rate = tc.learning_rate
    if tc.anneal_lr:
        minibatches = max(1, tc.batch_size // max(1, tc.minibatch_size))
        learning_rate = optax.linear_schedule(
            init_value=tc.learning_rate,
            end_value=0.0,
            transition_steps=max(1, tc.num_iterations * tc.update_epochs * minibatches),
        )

    return make_optimizer(
        agent,
        optax.chain(
            optax.clip_by_global_norm(tc.max_grad_norm),
            optax.adam(learning_rate=learning_rate, eps=1e-5),
        ),
    )
