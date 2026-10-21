"""Re-export of the shared rollout buffer (kept for the per-variant API)."""

from shared.episode import Episode, make_episode

__all__ = ['Episode', 'make_episode']
