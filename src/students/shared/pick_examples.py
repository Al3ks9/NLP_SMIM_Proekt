"""
Examples for the thesis: one (source poem, target author) pair from every
system's generation CSV, or one output of each failure mode.

Usage:
    uv run python -m students.shared.pick_examples --poem-id 43 --target "Блаже Конески"
    uv run python -m students.shared.pick_examples --failures
"""

import argparse
import csv
import random
from pathlib import Path

from common.corpus.make_splits import load_stripped_songs
from common.paths import ROOT

GENERATIONS = ROOT / 'data' / 'validation_generations'
TEACHER_TEST = ROOT / 'data' / 'synthetic' / 'teacher_test.csv'
REFERENCE_RUN = 'REAL POEMS'
# comparison labels that aren't their file's stem -> the stem
LEGACY_LABELS = {'base': 'base-Qwen3-4B', 'sft': 'qwen3-sft-merged',
                 'grpo-v1': 'qwen3-grpo', 'grpo-v2': 'qwen3-grpo-v2', 'gemma-teacher': 'teacher'}


def pick(poem_id: str, target: str, csvs: list) -> list:
    """[{system, generated_poem}] for every CSV that has the pair (system = file stem)."""
    found = []
    for path in csvs:
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['source_poem_id'] == poem_id and r['target_author'] == target:
                    found.append({'system': Path(path).stem, 'generated_poem': r['generated_poem']})
                    break
    return found


def csv_for_run(run: str, csvs: list):
    """The generation CSV a comparison run label came from, or None."""
    stem = LEGACY_LABELS.get(run, run)
    by_stem = {Path(c).stem: c for c in csvs}
    return by_stem.get(f'{stem}_test') or by_stem.get(stem)


def failure_examples(rows_csv: Path, seed: int = 42, skip=lambda run: False) -> dict:
    """One row (run, source_poem_id, target_author) per failure mode, from a
    compare_generations rows CSV: repetition, a near-copied source line, and
    an output the classifier reads as Конески when another author was asked."""
    with open(rows_csv, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['run'] != REFERENCE_RUN and not skip(r['run'])]
    kinds = {
        'degenerate': lambda r: float(r['line_uniqueness']) < 0.8,
        'copied_line': lambda r: r['copy_novelty'] != '' and float(r['copy_novelty']) < 1,
        'wrong_voice': lambda r: (r['clf_predicted'] == 'Блаже Конески'
                                  and r['target_author'] != 'Блаже Конески'),
    }
    rng = random.Random(seed)
    out = {}
    for kind, test in kinds.items():
        hits = [r for r in rows if test(r)]
        if hits:
            out[kind] = rng.choice(hits)
    return out


def _generation_csvs() -> list:
    csvs = sorted(GENERATIONS.glob('*_test.csv'))
    return csvs + ([TEACHER_TEST] if TEACHER_TEST.exists() else [])


def _print_poem(title: str, text: str) -> None:
    print(f'\n── {title} ' + '─' * max(0, 60 - len(title)))
    print(text.strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--poem-id')
    parser.add_argument('--target')
    parser.add_argument('--failures', action='store_true')
    parser.add_argument('--rows', type=Path, default=GENERATIONS / 'test_comparison_rows.csv')
    args = parser.parse_args()
    poems = {p['poem_id']: p for p in load_stripped_songs()}
    if args.failures:
        # the base model's failures (copying the source) say nothing about the method
        for kind, r in failure_examples(args.rows, skip=lambda run: run.startswith('base')).items():
            path = csv_for_run(r['run'], _generation_csvs())
            print(f'\n#### {kind}: run={r["run"]} poem={r["source_poem_id"]} → {r["target_author"]}')
            for hit in (pick(r['source_poem_id'], r['target_author'], [path]) if path else []):
                _print_poem(hit['system'], hit['generated_poem'])
    else:
        src = poems[args.poem_id]
        _print_poem(f'SOURCE · {src["author"]}, {src["song_title"]}', src['song_text'])
        for hit in pick(args.poem_id, args.target, _generation_csvs()):
            _print_poem(f'{hit["system"]} → {args.target}', hit['generated_poem'])
