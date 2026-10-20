import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from ppo_cleanrl.modules import ResNet

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8


class PPOAgent(nnx.Module):
    def __init__(
        self,
        agent_config,
        rngs,
    ):
        agent_config.resnet_config.rngs = rngs
        agent_config.proj_config.rngs = rngs
        agent_config.actor_config.rngs = rngs
        agent_config.critic_config.rngs = rngs
        self.enc = ResNet(**agent_config.resnet_config)
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

    def get_action_and_value(self, x, rng, action=None):
        x = x.astype(bf16) / bf16(255.0)
        x = self.enc(x)
        hidden = nnx.relu(self.proj(x))
        logits = self.actor(hidden)
        probs = distrax.Categorical(logits=logits)
        #        return probs, self.critic(hidden)
        key, _ = jax.random.split(rng)
        if action is None:
            action = probs.sample(seed=key)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)
