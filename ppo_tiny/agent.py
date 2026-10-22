import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from ppo_tiny.modules import TinyCNN

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8


class PPOAgent(nnx.Module):
    def __init__(
        self,
        agent_config,
        rngs,
    ):
        agent_config.tiny_config.rngs = rngs
        agent_config.proj_config.rngs = rngs
        agent_config.actor_config.rngs = rngs
        agent_config.critic_config.rngs = rngs
        self.enc = TinyCNN(**agent_config.tiny_config)
        agent_config.proj_config.in_features = self.enc.output_dim
        self.proj = nnx.Linear(**agent_config.proj_config)
        self.actor = nnx.Linear(**agent_config.actor_config)
        self.critic = nnx.Linear(**agent_config.critic_config)

    def get_value(self, x):
        x = x.astype(bf16) / bf16(255.0)
        x = self.enc(x)
        x = nnx.relu(self.proj(x))
        value = self.critic(x)
        return value

    def act_greedy(self, x):
        """Deterministic action used for evaluation (argmax of the policy logits)."""
        x = x.astype(bf16) / bf16(255.0)
        x = self.enc(x)
        hidden = nnx.relu(self.proj(x))
        return jnp.argmax(self.actor(hidden), axis=-1).astype(jnp.int32)

    def get_action_and_value(self, x, rng, action=None):
        x = x.astype(bf16) / bf16(255.0)
        x = self.enc(x)
        hidden = nnx.relu(self.proj(x))
        logits = self.actor(hidden)
        probs = distrax.Categorical(logits=logits)
        key, _ = jax.random.split(rng)
        if action is None:
            action = probs.sample(seed=key)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
