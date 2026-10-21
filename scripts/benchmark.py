"""Run every variant for a fixed budget and summarise its evaluation returns.

Each variant is launched as a separate process (they share a single JAX process
poorly), with Weights & Biases disabled and greedy evaluation enabled. The
evaluation lines are parsed from stdout, and a markdown table plus a JSON file are
written to the log root.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

VARIANTS = ('cleanrl', 'nature', 'impala', 'vae', 'jepa')
EVAL_PATTERN = re.compile(r'return=([-\d.]+) over (\d+) episodes')


def run_variant(name, timesteps, logdir, eval_interval, eval_steps, extra_args):
    cmd = [
        sys.executable,
        'main.py',
        '--config',
        name,
        '--total-timesteps',
        str(timesteps),
        '--no-wandb',
        '--logdir',
        str(logdir),
        '--eval-interval',
        str(eval_interval),
        '--eval-steps',
        str(eval_steps),
        *extra_args,
    ]
    print(f'=== {name}: {" ".join(cmd)}', flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        return {'variant': name, 'status': 'failed', 'returncode': proc.returncode}

    evaluations = [
        {'return': float(ret), 'episodes': int(episodes)} for ret, episodes in EVAL_PATTERN.findall(proc.stdout)
    ]
    parameters = re.search(r'params=([\d,]+)', proc.stdout)
    return {
        'variant': name,
        'status': 'ok',
        'params': int(parameters.group(1).replace(',', '')) if parameters else None,
        'evaluations': evaluations,
        'final_return': evaluations[-1]['return'] if evaluations else None,
    }


def to_markdown(results):
    lines = ['| Variant | Params | Final eval return | Evaluations |', '| --- | --- | --- | --- |']
    for result in results:
        params = f'{result["params"]:,}' if result.get('params') else '-'
        final = f'{result["final_return"]:.2f}' if result.get('final_return') is not None else '-'
        count = len(result.get('evaluations', []))
        status = '' if result['status'] == 'ok' else ' (failed)'
        lines.append(f'| `{result["variant"]}`{status} | {params} | {final} | {count} |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--eval-interval', type=int, default=1)
    parser.add_argument('--eval-steps', type=int, default=2048)
    parser.add_argument('--logdir', type=str, default='runs/benchmark')
    parser.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=VARIANTS)
    parser.add_argument('rest', nargs=argparse.REMAINDER, help='Extra args forwarded to main.py')
    args = parser.parse_args(argv)

    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    results = [
        run_variant(name, args.timesteps, logdir, args.eval_interval, args.eval_steps, args.rest)
        for name in args.variants
    ]

    table = to_markdown(results)
    (logdir / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (logdir / 'results.md').write_text(table + '\n', encoding='utf-8')
    print()
    print(table)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
