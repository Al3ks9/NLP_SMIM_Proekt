"""
Word substitution over the same (source poem, target author) pairs as another
system, written in the generation-CSV format compare_generations.py scores.

The pairs are read from an existing generation CSV rather than resampled, so
they are exactly the students' pairs, and this pipeline needs nothing from a
later one. source_author is always passed: it turns on transfer_style's
criterion (a), source-distinctive words, as for any real transfer.

Resumable: pairs already in the output file are skipped.

Usage:
    uv run python -m word_substitution.generate_pairs \\
        --pairs-from data/validation_generations/base-Qwen3-4B_test.csv \\
        --output data/validation_generations/word-substitution_test.csv
"""

import argparse
import csv
import logging
from pathlib import Path

from common.corpus.make_splits import load_stripped_songs

log = logging.getLogger('generate_pairs')

COLUMNS = ['example_id', 'source_poem_id', 'source_author', 'source_title',
           'target_author', 'split', 'sample_index', 'generated_poem']


def read_pairs(path: Path) -> list:
    """Ordered, de-duplicated (source_poem_id, target_author) pairs of a generation CSV."""
    with open(path, encoding='utf-8') as f:
        keys = [(r['source_poem_id'], r['target_author']) for r in csv.DictReader(f)]
    return list(dict.fromkeys(keys))


def _done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    with open(out_path, encoding='utf-8') as f:
        return {(r['source_poem_id'], r['target_author']) for r in csv.DictReader(f)}


def generate(pairs: list, out_path: Path, transfer=None, split: str = 'test') -> int:
    """Run transfer_style on every pair not yet in out_path; returns rows written."""
    if transfer is None:
        # the vocabularies and embeddings load on demand, so read_pairs stays cheap
        from word_substitution import style_transfer
        style_transfer.load_resources()
        transfer = style_transfer.transfer_style
    poems = {p['poem_id']: p for p in load_stripped_songs()}
    done = _done(out_path)
    todo = [p for p in pairs if p not in done]
    new_file = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for i, (pid, target) in enumerate(todo, 1):
            poem = poems[pid]
            generated = transfer(poem['song_text'], target, source_author=poem['author'])
            writer.writerow({
                'example_id': f'{pid}_{target}_0', 'source_poem_id': pid,
                'source_author': poem['author'], 'source_title': poem['song_title'],
                'target_author': target, 'split': split, 'sample_index': 0,
                'generated_poem': generated,
            })
            f.flush()
            log.info('%d/%d poem_id=%s -> %s', i, len(todo), pid, target)
    return len(todo)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--pairs-from', type=Path, required=True,
                        help='generation CSV whose (source_poem_id, target_author) pairs to reuse')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split', default='test')
    args = parser.parse_args()
    n = generate(read_pairs(args.pairs_from), args.output, split=args.split)
    log.info('wrote %d rows to %s', n, args.output)
