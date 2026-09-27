"""
Base-vs-tuned comparison for the Qwen SFT pipeline (stage 7): score one or
more generation runs with src/style_metrics.py and print/write the aggregate
table.

A run is any CSV with generated_poem + source_poem_id + target_author columns
-- which covers generate_validation.py's output (untuned base and LoRA-tuned
alike, see its --no-adapter flag) and generate_synthetic.py's Gemma output,
so the student can be put next to its own teacher. --corpus-reference adds the
real poems of the target authors as an upper anchor: it shows what these
metrics score on human text, without which "distinct_2 = 0.71" means nothing.

Two views are printed:

- all rows of each run, averaged. Honest per-run quality, but runs may cover
  different pairs (the teacher's rows are train-split, the validation runs'
  are val).
- paired: only (source_poem_id, target_author, sample_index) keys present in
  every run being compared, so base vs tuned is the same poems into the same
  authors and a difference can't come from an easier sample.

Usage:
    uv run python src/compare_generations.py                      # auto-discover val runs
    uv run python src/compare_generations.py --include-teacher --corpus-reference
    uv run python src/compare_generations.py --runs a.csv b.csv --labels base tuned
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import style_metrics as sm
from make_splits import load_stripped_songs
from sft_data import select_target_authors

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
GENERATIONS_DIR = DATA / 'validation_generations'
TEACHER_PATH = DATA / 'synthetic' / 'synthetic_dataset.csv'
DEFAULT_REPORT_PATH = GENERATIONS_DIR / 'comparison_summary.csv'
DEFAULT_ROWS_PATH = GENERATIONS_DIR / 'comparison_rows.csv'

# Printed in this order; (column, label, higher_is_better=True for all of
# these -- style_metrics orients every score that way on purpose).
SUMMARY_COLUMNS = [
    ('n', 'rows'),
    ('num_tokens', 'tokens'),
    ('distinct_1', 'distinct-1'),
    ('distinct_2', 'distinct-2'),
    ('distinct_3', 'distinct-3'),
    ('line_uniqueness', 'line-uniq'),
    ('max_line_repeats', 'max-rep'),
    ('is_cyrillic', 'cyrillic'),
    ('no_markup', 'no-markup'),
    ('is_nonempty', 'nonempty'),
    ('fit_structure', 'struct-fit'),
    ('clf_target_prob', 'clf-prob'),
    ('clf_top1', 'clf-top1'),
    ('tfidf_hit_rate', 'tfidf-hit'),
    ('content_recall', 'content-rec'),
    ('content_jaccard', 'content-jac'),
    ('line_novelty', 'line-novelty'),
]

METRIC_COLUMNS = [c for c, _ in SUMMARY_COLUMNS if c != 'n']

log = logging.getLogger('compare_generations')


# ── Loading runs ────────────────────────────────────────────────────────────────

def _poems_by_id() -> dict:
    return {r['poem_id']: r for r in load_stripped_songs()}


def load_run(path: Path, label: str = None) -> dict:
    """{'label', 'path', 'rows'} for one generation CSV. rows keep only what
    scoring needs, plus the pairing key."""
    with open(path, encoding='utf-8') as f:
        raw = list(csv.DictReader(f))
    rows = [{
        'key': (r['source_poem_id'], r['target_author'], r.get('sample_index', '0')),
        'source_poem_id': r['source_poem_id'],
        'target_author': r['target_author'],
        'split': r.get('split', ''),
        'generated_poem': r['generated_poem'],
    } for r in raw]
    return {'label': label or path.stem, 'path': str(path), 'rows': rows}


def corpus_reference_run(num_target_authors: int = 10, label: str = 'REAL POEMS') -> dict:
    """
    The target authors' own real poems, scored as if they were generations.

    The anchor every other number is read against: it is what a *perfect*
    style transfer would score on the style and degeneracy metrics. Content
    metrics are left blank -- there is no source poem a real poem was
    transferred from, and scoring it against itself would print a meaningless
    1.0.

    Two of its cells are not ceilings and should not be read as targets:

    - clf-top1 / clf-prob come out near 1.0 because classifier.py fits on the
      whole corpus, so the shipped model has memorized these exact poems. The
      honest ceiling is its held-out accuracy on the same 10 authors, 0.883
      (5-fold CV, same recipe) -- that, not 1.0, is what a real poem scores
      when the classifier has not seen it.
    - tfidf-hit is low (~0.05) for real poems simply because one poem contains
      few of its author's 20 top-TF-IDF words. A run that was *handed* that
      palette in its prompt (the Gemma teacher) scores far above human text
      here; it measures palette uptake, not quality.
    """
    authors = set(select_target_authors(num_target_authors))
    rows = [{
        'key': (r['poem_id'], r['author'], '0'),
        'source_poem_id': r['poem_id'],
        'target_author': r['author'],
        'split': 'corpus',
        'generated_poem': r['song_text'],
        'source_text': None,
    } for r in load_stripped_songs() if r['author'] in authors]
    return {'label': label, 'path': 'data/stripped_songs.csv', 'rows': rows}


# ── Scoring ─────────────────────────────────────────────────────────────────────

def score_run(run: dict) -> list:
    """Every row of one run scored. source_text comes from stripped_songs.csv
    by poem_id (one source of truth for poem text, as elsewhere in this
    pipeline) unless the row already carries one explicitly."""
    poems = _poems_by_id()
    scored = []
    for row in run['rows']:
        source_text = row.get('source_text', ...)
        if source_text is ...:
            source = poems.get(row['source_poem_id'])
            source_text = source['song_text'] if source else None

        metrics = {
            **sm.repetition_metrics(row['generated_poem']),
            **sm.language_metrics(row['generated_poem']),
            **sm.structure_fit(row['generated_poem'], row['target_author']),
            **sm.style_match(row['generated_poem'], row['target_author']),
        }
        if source_text:
            metrics.update(sm.content_preservation(source_text, row['generated_poem']))
        else:
            metrics.update({'content_recall': None, 'content_jaccard': None,
                            'line_novelty': None})

        scored.append({'run': run['label'], 'key': row['key'],
                       'source_poem_id': row['source_poem_id'],
                       'target_author': row['target_author'],
                       'split': row['split'], **metrics})
    return scored


def aggregate(scored_rows: list) -> dict:
    """Mean of each metric over the rows, skipping None (content metrics on
    the corpus reference) rather than treating a missing value as zero."""
    summary = {'n': len(scored_rows)}
    for col in METRIC_COLUMNS:
        values = [r[col] for r in scored_rows if r.get(col) is not None]
        summary[col] = sum(values) / len(values) if values else None
    return summary


def paired_keys(runs_scored: dict) -> set:
    """Keys present in every run -- the fair-comparison subset."""
    key_sets = [{r['key'] for r in rows} for rows in runs_scored.values()]
    return set.intersection(*key_sets) if key_sets else set()


# ── Reporting ───────────────────────────────────────────────────────────────────

def _fmt(value) -> str:
    if value is None:
        return '--'
    if isinstance(value, int) or float(value).is_integer() and abs(value) >= 100:
        return f'{int(value)}'
    return f'{value:.3f}'


def print_table(title: str, summaries: dict) -> None:
    labels = list(summaries)
    width = max((len(l) for l in labels), default=10)
    width = max(width, 11)
    print(f'\n{title}')
    print('-' * (14 + (width + 2) * len(labels)))
    print(f'{"metric":<14}' + ''.join(f'{l:>{width + 2}}' for l in labels))
    for col, label in SUMMARY_COLUMNS:
        cells = ''.join(f'{_fmt(summaries[l].get(col)):>{width + 2}}' for l in labels)
        print(f'{label:<14}{cells}')


def write_reports(all_scored: list, summaries: dict, rows_path: Path,
                  report_path: Path) -> None:
    rows_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['run', 'source_poem_id', 'target_author', 'split'] + METRIC_COLUMNS + [
        'clf_predicted', 'actual_lines', 'actual_stanzas', 'actual_tokens_per_line']
    with open(rows_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_scored)

    with open(report_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['run', 'view'] + [c for c, _ in SUMMARY_COLUMNS])
        writer.writeheader()
        for (label, view), summary in summaries.items():
            writer.writerow({'run': label, 'view': view, **summary})
    log.info('wrote %s and %s', rows_path, report_path)


# ── Orchestration ───────────────────────────────────────────────────────────────

def discover_runs(split: str = 'val') -> list:
    """Generation CSVs in data/validation_generations/ for one split, base
    model first so the table reads baseline -> tuned left to right."""
    found = sorted(GENERATIONS_DIR.glob(f'*_{split}.csv'))
    return sorted(found, key=lambda p: (not p.name.startswith('base'), p.name))


def main(run_paths: list, labels: list = None, include_teacher: bool = False,
         corpus_reference: bool = False, num_target_authors: int = 10,
         rows_path: Path = DEFAULT_ROWS_PATH, report_path: Path = DEFAULT_REPORT_PATH) -> None:
    runs = [load_run(p, labels[i] if labels and i < len(labels) else None)
            for i, p in enumerate(run_paths)]
    comparable = [r['label'] for r in runs]  # the runs the paired view covers

    if include_teacher and TEACHER_PATH.exists():
        runs.append(load_run(TEACHER_PATH, 'gemma-teacher'))
    if corpus_reference:
        runs.append(corpus_reference_run(num_target_authors))

    scored = {run['label']: score_run(run) for run in runs}
    for label, rows in scored.items():
        log.info('scored %d rows for %s', len(rows), label)

    summaries = {(label, 'all'): aggregate(rows) for label, rows in scored.items()}
    print_table('All rows (runs may cover different pairs)',
                {label: summaries[(label, 'all')] for label in scored})

    paired = {label: scored[label] for label in comparable if label in scored}
    if len(paired) > 1:
        keys = paired_keys(paired)
        log.info('%d pairs present in all of %s', len(keys), ', '.join(paired))
        paired_summaries = {label: aggregate([r for r in rows if r['key'] in keys])
                            for label, rows in paired.items()}
        for label, summary in paired_summaries.items():
            summaries[(label, 'paired')] = summary
        print_table(f'Paired ({len(keys)} identical source/target pairs)', paired_summaries)

    all_scored = [r for rows in scored.values() for r in rows]
    write_reports(all_scored, summaries, rows_path, report_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', type=Path, default=None,
                       help='generation CSVs (default: auto-discover in '
                            'data/validation_generations/)')
    parser.add_argument('--labels', nargs='+', default=None,
                       help='display names for --runs, in the same order')
    parser.add_argument('--split', default='val', help='split to auto-discover runs for')
    parser.add_argument('--include-teacher', action='store_true',
                       help="also score generate_synthetic.py's Gemma output")
    parser.add_argument('--corpus-reference', action='store_true',
                       help="also score the target authors' real poems as an upper anchor")
    parser.add_argument('--num-target-authors', type=int, default=10)
    parser.add_argument('--rows-out', type=Path, default=DEFAULT_ROWS_PATH)
    parser.add_argument('--report-out', type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    paths = args.runs or discover_runs(args.split)
    if not paths:
        parser.error(f'no generation CSVs found in {GENERATIONS_DIR} for split '
                     f'{args.split!r} -- pass --runs explicitly')
    main(paths, labels=args.labels, include_teacher=args.include_teacher,
         corpus_reference=args.corpus_reference, num_target_authors=args.num_target_authors,
         rows_path=args.rows_out, report_path=args.report_out)
