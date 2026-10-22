"""Re-export of the plain-PPO trainer for the tiny-CNN variant.

The tiny-CNN agent has exactly the ``ppo_cleanrl`` interface, so the rollout
collection, update loop and loss helpers are the same objects rather than copies.
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
