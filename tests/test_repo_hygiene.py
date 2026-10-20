"""Guards against re-introducing private paths, identities or dead code."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ['ppo_cleanrl', 'ppo_vae', 'shared']
SOURCE_FILES = [ROOT / 'main.py'] + sorted(path for name in SOURCE_DIRS for path in (ROOT / name).rglob('*.py'))


def read(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore')


def test_duplicate_variant_package_is_gone():
    assert not (ROOT / 'ppo_jepa').exists()


def test_variants_match_the_cli_choices():
    import main

    for kind in main.CONFIG_CHOICES:
        assert (ROOT / f'ppo_{kind}').is_dir(), kind


def test_no_absolute_home_paths_in_source():
    offenders = [
        f'{path.relative_to(ROOT)}:{line_no}'
        for path in SOURCE_FILES
        for line_no, line in enumerate(read(path).splitlines(), 1)
        if '/home/' in line or '/Users/' in line
    ]

    assert not offenders, f'absolute personal paths found: {offenders}'


def test_no_personal_identifiers_in_source():
    offenders = [
        f'{path.relative_to(ROOT)}:{line_no}'
        for path in SOURCE_FILES
        for line_no, line in enumerate(read(path).splitlines(), 1)
        if 'andyshuo' in line.lower() or 'Journal-Lets-Go' in line
    ]

    assert not offenders, f'personal identifiers found: {offenders}'


def test_no_import_from_the_removed_shared_config_module():
    offenders = [str(path.relative_to(ROOT)) for path in SOURCE_FILES if 'shared.config' in read(path)]

    assert not offenders, f'imports of the never-existing shared.config: {offenders}'


def test_no_broken_optimizer_call_signature():
    """``get_optimizer`` must take the agent explicitly."""
    for name in ('ppo_cleanrl', 'ppo_vae'):
        text = read(ROOT / name / 'utils.py')
        assert 'def get_optimizer(config, agent):' in text, name


def test_gaes_are_not_hard_coded_to_256_steps():
    for name in ('ppo_cleanrl', 'ppo_vae'):
        text = read(ROOT / name / 'utils.py')
        assert 'num_steps = 256' not in text, name
        assert 'range(num_steps)' not in text, name


def test_wandb_is_not_imported_eagerly_by_training_utils():
    for name in ('ppo_cleanrl', 'ppo_vae'):
        text = read(ROOT / name / 'utils.py')
        assert 'import wandb' not in text, name


def test_readme_is_present_and_substantive():
    readme = ROOT / 'README.md'

    assert readme.is_file()
    content = read(readme)
    assert len(content) > 500
    assert 'python main.py' in content


def test_packaging_files_exist():
    for name in ('pyproject.toml', 'requirements.txt', 'requirements-dev.txt', 'requirements-gpu.txt'):
        assert (ROOT / name).is_file(), name


def test_ci_workflow_exists_and_runs_tests():
    workflow = ROOT / '.github' / 'workflows' / 'ci.yml'

    assert workflow.is_file()
    content = read(workflow)
    assert 'pytest' in content
    assert 'ruff' in content


@pytest.mark.parametrize('pattern', ['.venv', 'wandb', 'logs', '__pycache__', '.pytest_cache'])
def test_gitignore_covers_artifacts(pattern):
    content = read(ROOT / '.gitignore')

    assert pattern in content
