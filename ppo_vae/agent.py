import distrax
import einops
import jax
import jax.numpy as jnp
from flax import nnx

from ppo_vae.modules import VAE

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8


class PPOAgent(nnx.Module):
    """PPO agent whose encoder is a discrete VAE.

    Notes on the two fixes relative to the original implementation:

    * the policy logits now come from ``self.actor``. Previously the projection
      output was fed to ``distrax.Categorical`` directly, so the agent sampled over
      ``proj_config.out_features`` (256) categories while ProcGen only accepts 15
      actions, and the ``actor`` head was never used at all;
    * ``get_action_and_value`` returns the *latent* categorical logits ``(b, c, d)``
      for the KL term. It used to return the projection output, so the "VAE KL" was
      measured over action logits against a uniform prior instead of over the latent
      variables.
    """

    def __init__(
        self,
        agent_config,
        rngs,
    ):
        agent_config.vae_config.rngs = rngs
        agent_config.vae_config.vae_proj_config.rngs = rngs
        agent_config.proj_config.rngs = rngs
        agent_config.actor_config.rngs = rngs
        agent_config.critic_config.rngs = rngs
        self.vae = VAE(**agent_config.vae_config)
        agent_config.proj_config.in_features = self.vae.output_dim
        self.proj = nnx.Linear(**agent_config.proj_config)
        self.actor = nnx.Linear(**agent_config.actor_config)
        self.critic = nnx.Linear(**agent_config.critic_config)

    def _encode_hidden(self, x):
        """Shared encoder path: observations -> flattened latent -> projection."""
        x = x.astype(bf16) / bf16(255.0)
        latent_logits = self.vae.encode(x)
        flat = einops.rearrange(latent_logits, 'b c d -> b (c d)')
        hidden = nnx.relu(self.proj(flat))
        return latent_logits, hidden

    def get_value(self, x):
        _, hidden = self._encode_hidden(x)
        return self.critic(hidden)

    def get_action_and_value(self, x, rng, action=None):
        latent_logits, hidden = self._encode_hidden(x)
        z = self.vae.straight_through_sample(latent_logits, rng)
        recon_x = self.vae.decode(einops.rearrange(z, 'b c d -> b (c d)'))

        logits = self.actor(hidden)
        probs = distrax.Categorical(logits=logits)
        key, _ = jax.random.split(rng)
        if action is None:
            action = probs.sample(seed=key)
        value = self.critic(hidden)
        return action, probs.log_prob(action), probs.entropy(), value, recon_x, latent_logits, z
