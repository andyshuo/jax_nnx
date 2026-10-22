"""The benchmark harness (scripts/benchmark.py)."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark.py'


@pytest.fixture(scope='module')
def benchmark():
    spec = importlib.util.spec_from_file_location('benchmark_script', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_eval_pattern_parses_the_training_output(benchmark):
    stdout = 'iter 1/10 step=16,384 loss=0.123 sps=100\neval @ step 16,384: return=2.50 over 8 episodes\n'

    matches = benchmark.EVAL_PATTERN.findall(stdout)

    assert matches == [('2.50', '8')]


def test_to_markdown_renders_a_row_per_variant(benchmark):
    results = [
        {'variant': 'cleanrl', 'status': 'ok', 'params': 626256, 'evaluations': [], 'final_return': 2.0},
        {'variant': 'jepa', 'status': 'failed', 'returncode': 1},
    ]

    table = benchmark.to_markdown(results)

    assert '`cleanrl`' in table
    assert '626,256' in table
    assert '2.00' in table
    assert '`jepa` (failed)' in table


def test_run_variant_reports_a_failure(monkeypatch, benchmark):
    class Failed:
        returncode = 1
        stdout = ''
        stderr = 'boom'

    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: Failed())

    result = benchmark.run_variant('cleanrl', 100, Path('/tmp/x'), 1, 8, [])

    assert result['status'] == 'failed'
    assert result['returncode'] == 1


def test_run_variant_parses_evaluations(monkeypatch, benchmark):
    class Ok:
        returncode = 0
        stderr = ''
        stdout = (
            '[vae] devices=1 params=2,393,667 batch=16,384\n'
            'eval @ step 16,384: return=1.25 over 4 episodes\n'
            'eval @ step 32,768: return=3.75 over 6 episodes\n'
        )

    monkeypatch.setattr(benchmark.subprocess, 'run', lambda *a, **k: Ok())

    result = benchmark.run_variant('vae', 100, Path('/tmp/x'), 1, 8, [])

    assert result['status'] == 'ok'
    assert result['params'] == 2393667
    assert result['final_return'] == 3.75
    assert len(result['evaluations']) == 2
