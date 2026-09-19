"""
Synthetic style-transfer dataset generation (Qwen SFT pipeline, stage 2).

Samples source poems from the train split (src/make_splits.py) and, for each,
runs llm_style_transfer.run_transfer() against a fixed pool of target authors --
the existing Gemma pipeline, unmodified. Every generated example is appended to
data/synthetic/synthetic_dataset.csv with enough metadata to trace it back to
its source (poem_id, source author, target author, the poem itself, and which
split it belongs to).

Output rows keep the split of the SOURCE poem, so downstream consumers
(sft_data.load_synthetic_rows, train_sft.py) can select the train slice without
re-deriving anything: a poem's split, once assigned by make_splits.py, is fixed,
and this script never trains on it -- it only calls an LLM API with it.

Resumable: the output CSV is the source of truth for "already generated" --
pairs already present (by source_poem_id, target_author, sample_index) are
skipped, so a run killed partway through (rate limit, network blip) picks back
up without re-spending API quota or duplicating rows. A per-pair failure is
logged to data/synthetic/synthetic_errors.csv and skipped rather than aborting
the whole batch.

Usage:
    uv run python src/generate_synthetic.py \\
        --num-target-authors 10 --poems-per-author 100 --samples-per-pair 1 --seed 42

--poems-per-author is the number of source poems sampled ONCE from train and
reused across every target author (not resampled per author) -- with the
defaults above that's 100 source poems x 10 target authors x 1 sample = up to
1000 examples (fewer whenever a source poem's own author is one of the 10
target authors -- see SKIP_SELF_TRANSFER).
"""

import argparse
import csv
import logging
import random
import sys
from pathlib import Path

from make_splits import poems_in_split
from sft_data import select_target_authors
import llm_style_transfer as lst

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
OUTPUT_PATH = DATA / 'synthetic' / 'synthetic_dataset.csv'
ERRORS_PATH = DATA / 'synthetic' / 'synthetic_errors.csv'

DEFAULT_NUM_TARGET_AUTHORS = 10
DEFAULT_POEMS_PER_AUTHOR = 100
DEFAULT_SAMPLES_PER_PAIR = 1
DEFAULT_SEED = 42

# A source poem transferred to its own author isn't a style transfer example --
# skipped rather than counted toward the requested total.
SKIP_SELF_TRANSFER = True

FIELDNAMES = [
    'example_id', 'source_poem_id', 'source_author', 'source_title',
    'target_author', 'split', 'sample_index', 'generated_poem',
    'model', 'backend', 'delta_lines', 'delta_stanzas',
    'delta_tokens_per_line', 'delta_lines_per_stanza', 'log_path', 'timestamp',
]

log = logging.getLogger('generate_synthetic')


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_source_poems(n: int, seed: int = DEFAULT_SEED, split: str = 'train') -> list:
    """n poems sampled without replacement from `split`, in a fixed order
    determined by `seed` -- independent of --num-target-authors, so widening
    the target-author list doesn't reshuffle which source poems were already
    used. generate_validation.py reuses this with split='val'/'test' rather
    than re-deriving its own sampling."""
    pool = poems_in_split(split)
    if n > len(pool):
        raise ValueError(f'requested {n} source poems but {split} split only has '
                         f'{len(pool)}')
    rng = random.Random(seed)
    return rng.sample(pool, n)


def build_pairs(source_poems: list, target_authors: list,
                samples_per_pair: int = DEFAULT_SAMPLES_PER_PAIR) -> list:
    """[(source_poem_row, target_author, sample_index), ...], skipping
    self-transfer pairs. sample_index distinguishes multiple stochastic
    samples of the same (source, target) pair when samples_per_pair > 1."""
    pairs = []
    for poem in source_poems:
        for target_author in target_authors:
            if SKIP_SELF_TRANSFER and poem['author'] == target_author:
                continue
            for sample_index in range(samples_per_pair):
                pairs.append((poem, target_author, sample_index))
    return pairs


# ── Resumability ──────────────────────────────────────────────────────────────

def load_done_keys(path: Path = OUTPUT_PATH) -> set:
    """{(source_poem_id, target_author, sample_index), ...} already present in
    the output CSV -- skip these on a rerun."""
    if not path.exists():
        return set()
    with open(path, encoding='utf-8') as f:
        return {(r['source_poem_id'], r['target_author'], int(r['sample_index']))
               for r in csv.DictReader(f)}


def _open_append(path: Path, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    f = open(path, 'a', newline='', encoding='utf-8')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
        f.flush()
    return f, writer


# ── Orchestration ─────────────────────────────────────────────────────────────

def generate(num_target_authors: int = DEFAULT_NUM_TARGET_AUTHORS,
            poems_per_author: int = DEFAULT_POEMS_PER_AUTHOR,
            samples_per_pair: int = DEFAULT_SAMPLES_PER_PAIR,
            seed: int = DEFAULT_SEED, target_authors: list = None,
            model: str = lst.DEFAULT_MODEL, backend: str = lst.DEFAULT_BACKEND,
            output_path: Path = OUTPUT_PATH, errors_path: Path = ERRORS_PATH) -> dict:
    authors = select_target_authors(num_target_authors, explicit=target_authors)
    source_poems = sample_source_poems(poems_per_author, seed=seed)
    pairs = build_pairs(source_poems, authors, samples_per_pair)

    done = load_done_keys(output_path)
    todo = [p for p in pairs if (p[0]['poem_id'], p[1], p[2]) not in done]

    log.info('target authors: %s', ', '.join(authors))
    log.info('%d source poems, %d total pairs, %d already done, %d to generate',
            len(source_poems), len(pairs), len(pairs) - len(todo), len(todo))

    out_f, out_writer = _open_append(output_path, FIELDNAMES)
    err_f, err_writer = _open_append(errors_path,
                                     ['source_poem_id', 'source_author', 'target_author',
                                      'sample_index', 'error'])

    generated, failed = 0, 0
    try:
        for i, (poem, target_author, sample_index) in enumerate(todo, 1):
            try:
                result = lst.run_transfer(
                    poem['author'], poem['song_title'], target_author,
                    model=model, backend=backend, source_poem_id=poem['poem_id'])
            except Exception as e:
                log.warning('pair %d/%d FAILED poem_id=%s target=%s: %s',
                           i, len(todo), poem['poem_id'], target_author, e)
                err_writer.writerow({
                    'source_poem_id': poem['poem_id'], 'source_author': poem['author'],
                    'target_author': target_author, 'sample_index': sample_index,
                    'error': str(e),
                })
                err_f.flush()
                failed += 1
                continue

            fit = result['structural_fit']
            out_writer.writerow({
                'example_id': f'{poem["poem_id"]}_{target_author}_{sample_index}',
                'source_poem_id': result['source_poem_id'],
                'source_author': result['source_author'],
                'source_title': result['source_title'],
                'target_author': result['target_author'],
                'split': 'train',
                'sample_index': sample_index,
                'generated_poem': result['generated_poem'],
                'model': result['model'],
                'backend': result['backend'],
                'delta_lines': fit['delta_lines'],
                'delta_stanzas': fit['delta_stanzas'],
                'delta_tokens_per_line': fit['delta_tokens_per_line'],
                'delta_lines_per_stanza': fit['delta_lines_per_stanza'],
                'log_path': result['log_path'],
                'timestamp': result['timestamp'],
            })
            out_f.flush()
            generated += 1
            log.info('pair %d/%d OK poem_id=%s (%s) -> %s',
                    i, len(todo), poem['poem_id'], poem['author'], target_author)
    finally:
        out_f.close()
        err_f.close()

    log.info('done: %d generated, %d failed, output=%s', generated, failed, output_path)
    return {'generated': generated, 'failed': failed, 'output_path': output_path}


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--num-target-authors', type=int, default=DEFAULT_NUM_TARGET_AUTHORS)
    parser.add_argument('--poems-per-author', type=int, default=DEFAULT_POEMS_PER_AUTHOR,
                       help='number of source poems sampled once from the train split '
                            'and reused across every target author (NOT per-author-each; '
                            'see module docstring)')
    parser.add_argument('--samples-per-pair', type=int, default=DEFAULT_SAMPLES_PER_PAIR)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--target-authors', nargs='+', default=None,
                       help='explicit target-author list, overriding auto-selection')
    parser.add_argument('--model', default=lst.DEFAULT_MODEL)
    parser.add_argument('--backend', default=lst.DEFAULT_BACKEND,
                       choices=sorted(lst.llm_client.BACKENDS))
    parser.add_argument('--output-path', type=Path, default=OUTPUT_PATH,
                       help='dataset CSV to append to -- override so parallel '
                            'per-author invocations (see scripts/generate_synthetic_parallel.sh) '
                            'write to separate files instead of racing on one')
    parser.add_argument('--errors-path', type=Path, default=ERRORS_PATH,
                       help='errors CSV to append to -- same override as --output-path')
    args = parser.parse_args()

    try:
        generate(num_target_authors=args.num_target_authors,
                poems_per_author=args.poems_per_author,
                samples_per_pair=args.samples_per_pair, seed=args.seed,
                target_authors=args.target_authors, model=args.model,
                backend=args.backend, output_path=args.output_path,
                errors_path=args.errors_path)
    except (KeyError, ValueError) as e:
        sys.exit(str(e))
