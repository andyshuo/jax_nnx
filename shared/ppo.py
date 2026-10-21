"""PPO building blocks shared by every variant.

These were duplicated in ``ppo_cleanrl/utils.py`` and ``ppo_vae/utils.py``; keeping
one copy means a fix (or a new variant) only has to happen once.
"""

import distrax
import jax
import jax.numpy as jnp
import numpy as np

sg = jax.lax.stop_gradient


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


def compute_vae_loss(recon_x, obs):
    return jnp.mean(jnp.square(recon_x - obs))


def compute_kl_loss(logits):
    q_dist = distrax.Categorical(logits=logits)
    p_dist = distrax.Categorical(logits=jnp.zeros_like(logits))
    return q_dist.kl_divergence(p_dist).mean()
