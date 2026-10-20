"""Configuration defaults, derived sizes and the absence of shared mutable state."""

import jax.numpy as jnp
import pytest

from main import build_config

VARIANTS = ['cleanrl', 'vae']


@pytest.mark.parametrize('kind', VARIANTS)
def test_build_config_returns_matching_classes(kind):
    config, strconfig = build_config(kind)

    assert type(config).__module__ == f'ppo_{kind}.config'
    assert type(strconfig).__module__ == f'ppo_{kind}.config'


@pytest.mark.parametrize('kind', VARIANTS)
def test_build_config_derives_batch_sizes(kind):
    config, _ = build_config(kind)
    tc = config.training_config

    assert tc.batch_size == tc.num_envs * tc.num_steps
    assert tc.minibatch_size == tc.batch_size // tc.num_minibatches
    assert tc.minibatch_size > 0
    assert tc.batch_size % tc.num_minibatches == 0


@pytest.mark.parametrize('kind', VARIANTS)
def test_build_config_derives_num_iterations(kind):
    config, _ = build_config(kind)
    tc = config.training_config

    assert tc.num_iterations == tc.total_timesteps // tc.batch_size
    assert tc.num_iterations > 0


def test_cleanrl_uses_eight_minibatches():
    config, _ = build_config('cleanrl')

    assert config.training_config.num_minibatches == 8
    assert config.training_config.minibatch_size == config.training_config.batch_size // 8


def test_vae_uses_full_batch():
    config, _ = build_config('vae')

    assert config.training_config.num_minibatches == 1
    assert config.training_config.minibatch_size == config.training_config.batch_size


@pytest.mark.parametrize('kind', VARIANTS)
def test_wandb_defaults_are_impersonal(kind):
    _, strconfig = build_config(kind)

    # A cloner must not log into somebody else's account or project.
    assert strconfig.wandb_config.entity is None
    assert strconfig.wandb_config.project == 'jax_nnx'
    assert strconfig.wandb_config.name


@pytest.mark.parametrize('kind', VARIANTS)
def test_env_config_distribution_mode_is_a_string(kind):
    _, strconfig = build_config(kind)

    assert isinstance(strconfig.env_config.distribution_mode, str)
    assert strconfig.env_config.distribution_mode in {'easy', 'hard', 'memory', 'exploration'}


@pytest.mark.parametrize('kind', VARIANTS)
def test_capture_video_flag_lives_on_both_configs(kind):
    config, strconfig = build_config(kind)

    assert config.capture_video is False
    assert strconfig.capture_video is False


@pytest.mark.parametrize('kind', VARIANTS)
def test_configs_do_not_share_mutable_defaults(kind):
    """Two ``Config()`` instances must not alias their nested dataclasses."""
    config_a, _ = build_config(kind)
    config_b, _ = build_config(kind)

    config_a.agent_config.act_dim = 999

    assert config_b.agent_config.act_dim != 999


@pytest.mark.parametrize('kind', VARIANTS)
def test_rngs_default_to_none(kind):
    config, _ = build_config(kind)

    assert config.agent_config.proj_config.rngs is None


def test_vae_loss_coefficients_are_configurable():
    config, _ = build_config('vae')

    assert config.recon_coef == 1.0
    assert config.kl_coef == 1.0

    config.recon_coef = 0.5
    assert config.recon_coef == 0.5


def test_vae_config_hyperparameters():
    config, _ = build_config('vae')
    vae = config.agent_config.vae_config

    assert vae.shape == (64, 64, 3)
    assert len(vae.enc_out_features) == len(vae.dec_out_features)
    assert vae.num_categories > 1
    assert vae.categorical_dim > 1


def test_cleanrl_config_hyperparameters():
    config, _ = build_config('cleanrl')
    resnet = config.agent_config.resnet_config

    assert resnet.obs_shape == (64, 64, 3)
    assert len(resnet.hidden_dims) == 3


@pytest.mark.parametrize('kind', VARIANTS)
def test_dtypes_are_jax_dtypes(kind):
    config, _ = build_config(kind)

    assert config.agent_config.proj_config.dtype == jnp.bfloat16
    assert config.agent_config.proj_config.param_dtype == jnp.float32


@pytest.mark.parametrize('kind', VARIANTS)
def test_config_is_mappable(kind):
    """``wandb``/``asdict`` rely on the chex dataclasses being mapping-like."""
    config, _ = build_config(kind)

    assert 'seed' in config.keys()
    assert dict(**config.training_config)['batch_size'] == config.training_config.batch_size
