"""CLI parsing, config building and the device-mesh helpers in ``main``."""

import jax
import numpy as np
import pytest

import main


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #
def test_default_config_is_vae():
    args = main.parse_args([])

    assert args.config == main.DEFAULT_CONFIG == 'vae'


@pytest.mark.parametrize('kind', main.CONFIG_CHOICES)
def test_config_choice_is_accepted(kind):
    assert main.parse_args(['--config', kind]).config == kind


def test_unknown_choice_is_rejected():
    with pytest.raises(SystemExit):
        main.parse_args(['--config', 'not-a-variant'])


def test_unknown_config_is_rejected():
    with pytest.raises(SystemExit):
        main.parse_args(['--config', 'nope'])


def test_logdir_defaults_to_a_relative_path():
    args = main.parse_args([])

    assert args.logdir == main.DEFAULT_LOG_DIR == 'logs'
    assert 'andyshuo' not in args.logdir
    assert not args.logdir.startswith('/home')


def test_no_wandb_and_dry_run_flags():
    args = main.parse_args(['--no-wandb', '--dry-run'])

    assert args.no_wandb is True
    assert args.dry_run is True


def test_capture_video_flag():
    assert main.parse_args(['--capture-video']).capture_video is True


def test_seed_is_configurable():
    assert main.parse_args(['--seed', '123']).seed == 123


# --------------------------------------------------------------------------- #
# config building
# --------------------------------------------------------------------------- #
def test_build_config_unknown_variant_raises():
    with pytest.raises(ModuleNotFoundError):
        main.build_config('not-a-variant')


@pytest.mark.parametrize('kind', main.CONFIG_CHOICES)
def test_build_config_fills_training_sizes(kind):
    config, _ = main.build_config(kind)

    assert config.training_config.batch_size > 0
    assert config.training_config.minibatch_size > 0
    assert config.training_config.num_iterations > 0


# --------------------------------------------------------------------------- #
# filesystem / seeding / devices
# --------------------------------------------------------------------------- #
def test_make_logdir_creates_a_timestamped_directory(tmp_path):
    path = main.make_logdir(tmp_path, 'run')

    assert path.exists()
    assert path.is_dir()
    assert path.name.startswith('run_')
    assert path.parent == tmp_path


def test_seed_everything_is_reproducible():
    first_rng, _ = main.seed_everything(123)
    second_rng, _ = main.seed_everything(123)

    assert np.array_equal(np.asarray(first_rng), np.asarray(second_rng))


def test_seed_everything_differs_across_seeds():
    first_rng, _ = main.seed_everything(1)
    second_rng, _ = main.seed_everything(2)

    assert not np.array_equal(np.asarray(first_rng), np.asarray(second_rng))


def test_make_mesh_covers_every_device():
    mesh, sharding = main.make_mesh()

    assert mesh.size == jax.device_count()
    assert 'batch' in mesh.axis_names
    assert sharding.spec is not None


@pytest.mark.parametrize('kind', main.CONFIG_CHOICES)
def test_count_params_is_positive(kind):
    from flax import nnx

    config, _ = main.build_config(kind)
    config = _shrink(config)
    _, rngs = main.seed_everything(0)
    agent_module = __import__(f'ppo_{kind}.agent', fromlist=['PPOAgent'])
    agent = agent_module.PPOAgent(config.agent_config, rngs)

    assert main.count_params(agent) > 0
    assert isinstance(nnx.state(agent), object)


# --------------------------------------------------------------------------- #
# end-to-end dry runs (no env, no wandb, no GPU requirement)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize('kind', main.CONFIG_CHOICES)
def test_dry_run_builds_every_variant(kind, tmp_path, capsys):
    exit_code = main.main(['--config', kind, '--dry-run', '--logdir', str(tmp_path), '--no-wandb'])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert 'dry-run' in captured.out
    assert list(tmp_path.glob('*/config.json'))


def _shrink(config):
    """Shrink a default config so the dry-run tests stay fast on CPU."""
    if hasattr(config.agent_config, 'vae_config'):
        vae = config.agent_config.vae_config
        vae.shape = (16, 16, 3)
        vae.enc_out_features = [4, 8]
        vae.dec_out_features = [8, 3]
        vae.num_categories = 4
        vae.categorical_dim = 8
    elif hasattr(config.agent_config, 'vit_config'):
        vit = config.agent_config.vit_config
        vit.obs_shape = (16, 16, 3)
        vit.hidden_dims = [16, 2, 2]
        vit.patch_size = 8
    elif hasattr(config.agent_config, 'tiny_config'):
        tiny = config.agent_config.tiny_config
        tiny.obs_shape = (16, 16, 3)
        tiny.hidden_dims = [4, 8]
    elif hasattr(config.agent_config, 'nature_config'):
        nature = config.agent_config.nature_config
        nature.obs_shape = (16, 16, 3)
        nature.hidden_dims = [8, 16, 16]
    elif hasattr(config.agent_config, 'impala_config'):
        impala = config.agent_config.impala_config
        impala.obs_shape = (16, 16, 3)
        impala.hidden_dims = [8, 16]
    else:
        resnet = config.agent_config.resnet_config
        resnet.obs_shape = (16, 16, 3)
        resnet.hidden_dims = [4, 8]

    config.agent_config.proj_config.out_features = 16
    config.agent_config.actor_config.in_features = 16
    config.agent_config.actor_config.out_features = 6
    config.agent_config.critic_config.in_features = 16
    return config


def test_overrides_recompute_derived_sizes():
    args = main.parse_args(
        [
            '--num-envs',
            '4',
            '--num-steps',
            '16',
            '--total-timesteps',
            '1024',
            '--learning-rate',
            '0.001',
        ]
    )
    config, _ = main.build_config('cleanrl')

    main.apply_overrides(config, args)

    tc = config.training_config
    assert (tc.num_envs, tc.num_steps) == (4, 16)
    assert tc.batch_size == 64
    assert tc.minibatch_size == 8
    assert tc.num_iterations == 16
    assert tc.learning_rate == 0.001


def test_write_config_dumps_json(tmp_path):
    import json

    config, strconfig = main.build_config('vae')

    path = main.write_config(tmp_path, config, strconfig)
    payload = json.loads(path.read_text())

    assert payload['config']['training_config']['num_envs'] == 64
    assert 'env_config' in payload['strconfig']


def test_resume_flag_is_parsed():
    args = main.parse_args(['--resume', '/tmp/some-run'])

    assert args.resume == '/tmp/some-run'
