"""PPO agent with a JEPA-style latent-prediction auxiliary branch.

The policy, value head and encoder are exactly the cleanrl ones. On top of them
the agent predicts the encoder latent of a *shifted* view of the same observation:
because the PPO inputs are single frames, the shift is what makes the prediction
task non-trivial (the predictor has to model how the latent moves when the world
moves, not simply copy its input).
"""

import distrax
import jax
import jax.numpy as jnp
from flax import nnx

from ppo_jepa.modules import Predictor, PredictorConfig, ResNet

f32 = jnp.float32
bf16 = jnp.bfloat16
u8 = jnp.uint8
sg = jax.lax.stop_gradient


def shift(x, shift_px):
    """Build the JEPA target view by rolling ``x`` along its width axis.

    ``x`` is a ``(b, h, w, c)`` observation batch, so the width axis is ``2``.
    """
    return jnp.roll(x, shift_px, axis=2)


class PPOAgent(nnx.Module):
    """ResNet PPO agent plus a predictor of the shifted-view latent.

    ``shift_px`` is normally supplied from ``Config.shift_px``; it is fixed at
    construction time because it is part of the model's forward pass.
    """

    def __init__(
        self,
        agent_config,
        rngs,
        shift_px=2,
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
        self.predictor = Predictor(
            PredictorConfig(
                in_features=self.enc.output_dim,
                out_features=self.enc.output_dim,
                rngs=rngs,
                dtype=bf16,
                param_dtype=f32,
            )
        )
        self.shift_px = shift_px

    def encode(self, x):
        """Shared encoder path: observations -> flattened latent (bf16)."""
        x = x.astype(bf16) / bf16(255.0)
        return self.enc(x)

    def get_value(self, x):
        hidden = nnx.relu(self.proj(self.encode(x)))
        return self.critic(hidden)

    def act_greedy(self, x):
        """Deterministic action used for evaluation (argmax of the policy logits)."""
        hidden = nnx.relu(self.proj(self.encode(x)))
        return jnp.argmax(self.actor(hidden), axis=-1).astype(jnp.int32)

    def get_action_and_value(self, x, rng, action=None):
        latent = self.encode(x)
        hidden = nnx.relu(self.proj(latent))
        logits = self.actor(hidden)
        probs = distrax.Categorical(logits=logits)
        key, _ = jax.random.split(rng)
        if action is None:
            action = probs.sample(seed=key)
        pred = self.predictor(latent)
        # The target is a fixed regression target: gradients must reach the
        # encoder through ``pred`` only, never through the shifted view.
        target = sg(self.encode(shift(x, self.shift_px)))
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), pred, target
