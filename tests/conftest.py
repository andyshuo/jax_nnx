"""Shared fixtures for the test-suite.

Configs are deliberately tiny here (16x16 observations, two conv stages) so the
whole suite runs in seconds on CPU; the full-size shapes are exercised by the
``slow`` dry-run tests.
"""

import jax
import pytest
from flax import nnx


@pytest.fixture
def rngs_factory():
    """Return a factory for fresh ``nnx.Rngs`` (each model build needs new keys)."""

    def _make(seed: int = 0) -> nnx.Rngs:
        return nnx.Rngs(
            params=jax.random.PRNGKey(seed),
            dropout=jax.random.PRNGKey(seed + 1),
            sampling=jax.random.PRNGKey(seed + 2),
            categorical=jax.random.PRNGKey(seed + 3),
        )

    return _make


@pytest.fixture
def rngs(rngs_factory):
    return rngs_factory(0)


def prepare_training(config, num_envs: int = 2, num_steps: int = 8, update_epochs: int = 1):
    """Fill in the derived training sizes that ``main.build_config`` normally sets."""
    tc = config.training_config
    tc.num_envs = num_envs
    tc.num_steps = num_steps
    tc.batch_size = num_envs * num_steps
    tc.minibatch_size = tc.batch_size // tc.num_minibatches
    tc.update_epochs = update_epochs
    return config


@pytest.fixture
def cleanrl_config():
    from ppo_cleanrl.config import Config

    config = Config()
    resnet = config.agent_config.resnet_config
    resnet.obs_shape = (16, 16, 3)
    resnet.hidden_dims = [4, 8]

    # proj.out_features must equal actor/critic in_features; the agent only derives
    # proj.in_features, so keep these three in sync here.
    config.agent_config.proj_config.out_features = 16
    config.agent_config.actor_config.in_features = 16
    config.agent_config.actor_config.out_features = 6
    config.agent_config.critic_config.in_features = 16
    return prepare_training(config)


@pytest.fixture
def vae_config():
    from ppo_vae.config import Config

    config = Config()
    vae = config.agent_config.vae_config
    vae.shape = (16, 16, 3)
    vae.enc_out_features = [4, 8]
    vae.dec_out_features = [8, 3]
    vae.num_categories = 4
    vae.categorical_dim = 8

    config.agent_config.proj_config.out_features = 16
    config.agent_config.actor_config.in_features = 16
    config.agent_config.actor_config.out_features = 6
    config.agent_config.critic_config.in_features = 16
    config.recon_coef = 1.0
    config.kl_coef = 1.0
    return prepare_training(config)


@pytest.fixture(scope='session')
def mesh_and_sharding():
    from main import make_mesh

    return make_mesh()


@pytest.fixture(scope='session')
def single_device_mesh():
    """A one-device ``batch`` mesh so rollout tests work on any device count."""
    from jax import NamedSharding
    from jax.sharding import PartitionSpec as P

    mesh = jax.make_mesh((1,), ('batch',))
    return mesh, NamedSharding(mesh, P())


@pytest.fixture
def vae_module(vae_config, rngs):
    """A ``VAE`` built the same way ``PPOAgent`` builds it (rngs wired into sub-configs)."""
    from ppo_vae.modules import VAE

    cfg = vae_config.agent_config.vae_config
    cfg.rngs = rngs
    cfg.vae_proj_config.rngs = rngs
    return VAE(**cfg)
