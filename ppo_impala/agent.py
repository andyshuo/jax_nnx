import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from ppo_impala.modules import ImpalaCNN

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8


class PPOAgent(nnx.Module):
    """Plain PPO agent (the ``ppo_cleanrl`` interface) with an ``ImpalaCNN`` encoder."""

    def __init__(
        self,
        agent_config,
        rngs,
    ):
        agent_config.impala_config.rngs = rngs
        agent_config.proj_config.rngs = rngs
        agent_config.actor_config.rngs = rngs
        agent_config.critic_config.rngs = rngs
        self.enc = ImpalaCNN(**agent_config.impala_config)
        agent_config.proj_config.in_features = self.enc.output_dim
        self.proj = nnx.Linear(**agent_config.proj_config)
        self.actor = nnx.Linear(**agent_config.actor_config)
        self.critic = nnx.Linear(**agent_config.critic_config)

    def _encode_hidden(self, x):
        """Shared encoder path: observations -> flattened features -> projection."""
        x = x.astype(bf16) / bf16(255.0)
        x = self.enc(x)
        return nnx.relu(self.proj(x))

    def get_value(self, x):
        return self.critic(self._encode_hidden(x))

    def act_greedy(self, x):
        """Deterministic action used for evaluation (argmax of the policy logits)."""
        return jnp.argmax(self.actor(self._encode_hidden(x)), axis=-1).astype(jnp.int32)

    def get_action_and_value(self, x, rng, action=None):
        hidden = self._encode_hidden(x)
        probs = distrax.Categorical(logits=self.actor(hidden))
        key, _ = jax.random.split(rng)
        if action is None:
            action = probs.sample(seed=key)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
