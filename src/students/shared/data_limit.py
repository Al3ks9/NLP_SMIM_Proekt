"""
How much data each target author has, next to how often each system is
identified as that author. Backs the thesis's argument that the students only
learn the voices the corpus is rich in.

Usage:
    uv run --with matplotlib python -m students.shared.data_limit \\
        --rows data/validation_generations/test_comparison_rows.csv
    -> data/data_limit_<split>.csv, data/figures/identification_vs_poems_<split>.pdf
"""

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from common.corpus.make_splits import load_splits, load_stripped_songs
from common.paths import ROOT
from students.shared.sft_data import load_synthetic_rows

REFERENCE_RUN = 'REAL POEMS'
COLUMNS = ['target_author', 'corpus_poems', 'train_poems', 'teacher_examples', 'run',
           'clf_top1', 'judge_top1', 'share_predicted_as_target']


def author_table(rows_csv: Path) -> list:
    """One row per (run, target author); the real-poem reference is skipped."""
    songs = load_stripped_songs()
    splits = load_splits()
    corpus = Counter(p['author'] for p in songs)
    train = Counter(p['author'] for p in songs if splits.get(p['poem_id']) == 'train')
    teacher = Counter(r['target_author'] for r in load_synthetic_rows())

    with open(rows_csv, encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['run'] != REFERENCE_RUN]
    by_run_target = defaultdict(list)
    predicted = defaultdict(Counter)
    for r in rows:
        by_run_target[(r['run'], r['target_author'])].append(r)
        predicted[r['run']][r['clf_predicted']] += 1
    run_sizes = Counter(r['run'] for r in rows)

    table = []
    for (run, target), rs in sorted(by_run_target.items()):
        table.append({
            'target_author': target, 'corpus_poems': corpus[target],
            'train_poems': train[target], 'teacher_examples': teacher[target], 'run': run,
            'clf_top1': sum(float(r['clf_top1']) for r in rs) / len(rs),
            'judge_top1': sum(float(r['judge_top1']) for r in rs) / len(rs),
            'share_predicted_as_target': predicted[run][target] / run_sizes[run],
        })
    return table


def plot(table: list, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    runs = sorted({r['run'] for r in table})
    for run, marker in zip(runs, 'osD^vP*X'):
        pts = [r for r in table if r['run'] == run]
        ax.scatter([r['train_poems'] for r in pts], [r['judge_top1'] for r in pts],
                   marker=marker, label=run, alpha=.8)
    for r in {r['target_author']: r for r in table}.values():
        ax.annotate(r['target_author'].split()[-1], (r['train_poems'], 0), fontsize=7,
                    rotation=60, textcoords='offset points', xytext=(0, -28), ha='center')
    ax.set_xlabel('train-split poems by the target author')
    ax.set_ylabel('judge top-1 = target')
    ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--rows', type=Path, required=True, help='a compare_generations rows CSV')
    parser.add_argument('--name', default=None, help='output suffix (default: test/val from --rows)')
    args = parser.parse_args()
    name = args.name or ('test' if 'test' in args.rows.name else 'val')
    table = author_table(args.rows)
    out_csv = ROOT / 'data' / f'data_limit_{name}.csv'
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(table)
    plot(table, ROOT / 'data' / 'figures' / f'identification_vs_poems_{name}.pdf')
    print(out_csv)
