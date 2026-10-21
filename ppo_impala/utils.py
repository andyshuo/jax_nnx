"""Re-export of the plain-PPO trainer (``ppo_cleanrl.utils``) for this variant.

The IMPALA encoder only changes the model, not the objective, so the trainer is
imported rather than copied: a fix to the PPO update applies to every variant that
shares it.
"""

from ppo_cleanrl.utils import (
    env_step,
    get_action_and_value,
    get_greedy_action,
    get_optimizer,
    get_value,
    loss_fn,
    train,
    train_step,
)

__all__ = [
    'env_step',
    'get_action_and_value',
    'get_greedy_action',
    'get_optimizer',
    'get_value',
    'loss_fn',
    'train',
    'train_step',
]
