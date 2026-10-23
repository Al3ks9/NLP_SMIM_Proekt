"""
Train / validation / test split over the real poem corpus (Qwen SFT pipeline,
stage 1). Every downstream stage (synthetic-data generation, SFT training,
validation generation) reads the split from here rather than re-deriving it,
so which poems are "train" can't drift between scripts.

Split unit is the poem (poem_id), not the (author, song_title) pair — see
CLAUDE.md's "Data quality context" on why song_title alone isn't a safe key.

Per-author 80/10/10, with a floor of 1 poem in each split once an author has
enough poems to support it (MIN_POEMS_FOR_3WAY), matching the MIN_POEMS
convention already used by classifier.py and style_profiler.py for
"does this author have enough data" decisions:

- n >= 3: every split gets >= 1 poem (val/test floor of 1, remainder to train).
  This is standard 80/10/10 for anyone with enough poems to make that
  meaningful, and degrades gracefully down to a 1/1/1 split at n=3.
- n in {1, 2}: cannot appear in three disjoint splits at all -- one or two
  poems can't be cut three ways. These go entirely to train. This is a real
  corpus-shape fact (5 authors have exactly 1 poem, 3 have exactly 2 -- see
  the printed summary), not a bug: such authors were never going to work as
  well-profiled style-transfer *targets* either, since style_profiler /
  exemplar_selection need volume regardless of this split.

Reproducible: a single seeded Random, authors processed in sorted order, each
author's poems shuffled in poem_id order before slicing -- rerunning with the
same seed reproduces the exact same assignment.

Run once (or whenever stripped_songs.csv changes):
    uv run python src/make_splits.py
"""

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
SPLITS_PATH = DATA / 'splits' / 'poem_splits.csv'

SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1
MIN_POEMS_FOR_3WAY = 3
SPLIT_NAMES = ('train', 'val', 'test')

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


# ── Core computation (pure -- no file I/O) ──────────────────────────────────────

def _val_test_counts(n: int) -> tuple:
    """(n_val, n_test) for an author with n poems: >=1 each once n allows it,
    shrunk as needed so n_train stays >= 1. n < MIN_POEMS_FOR_3WAY is the
    caller's job to special-case -- this assumes n >= MIN_POEMS_FOR_3WAY."""
    n_val = max(1, round(n * VAL_FRAC))
    n_test = max(1, round(n * TEST_FRAC))
    while n_val + n_test >= n:
        if n_val >= n_test and n_val > 1:
            n_val -= 1
        elif n_test > 1:
            n_test -= 1
        else:
            break  # n == MIN_POEMS_FOR_3WAY: 1/1/(n-2), nothing left to shrink
    return n_val, n_test


def compute_splits(poems: list, seed: int = SEED) -> dict:
    """
    poems: [{'poem_id': ..., 'author': ...}, ...] (extra keys ignored).
    Returns {poem_id (str): split_name}.

    Authors below MIN_POEMS_FOR_3WAY poems are assigned to 'train' in full --
    see module docstring. Every other author is guaranteed >= 1 poem in each
    of train/val/test.
    """
    rng = random.Random(seed)
    by_author = defaultdict(list)
    for p in poems:
        by_author[p['author']].append(str(p['poem_id']))

    assignment = {}
    for author in sorted(by_author):
        ids = sorted(by_author[author], key=int)
        rng.shuffle(ids)
        n = len(ids)

        if n < MIN_POEMS_FOR_3WAY:
            for pid in ids:
                assignment[pid] = 'train'
            continue

        n_val, n_test = _val_test_counts(n)
        n_train = n - n_val - n_test
        for pid in ids[:n_train]:
            assignment[pid] = 'train'
        for pid in ids[n_train:n_train + n_val]:
            assignment[pid] = 'val'
        for pid in ids[n_train + n_val:]:
            assignment[pid] = 'test'

    return assignment


# ── File I/O ────────────────────────────────────────────────────────────────────

def load_stripped_songs() -> list:
    def build():
        with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    return _cached('stripped_songs', build)


def build_and_save(seed: int = SEED) -> dict:
    """Compute the split over the full corpus and write poem_splits.csv.
    Returns the {poem_id: split} assignment."""
    rows = load_stripped_songs()
    assignment = compute_splits(rows, seed=seed)

    author_counts = Counter(r['author'] for r in rows)
    by_author_below_min = sorted(a for a, n in author_counts.items() if n < MIN_POEMS_FOR_3WAY)

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['poem_id', 'author', 'song_title', 'split'])
        for row in sorted(rows, key=lambda r: int(r['poem_id'])):
            writer.writerow([row['poem_id'], row['author'], row['song_title'],
                            assignment[row['poem_id']]])

    counts = defaultdict(int)
    for split in assignment.values():
        counts[split] += 1
    print(f'Saved {SPLITS_PATH.relative_to(ROOT)} ({len(assignment)} poems)')
    print(f"  train={counts['train']} val={counts['val']} test={counts['test']}")
    if by_author_below_min:
        print(f'  {len(by_author_below_min)} authors below MIN_POEMS_FOR_3WAY='
             f'{MIN_POEMS_FOR_3WAY} poems went entirely to train: '
             f'{", ".join(by_author_below_min)}')
    return assignment


def load_splits() -> dict:
    """poem_id (str) -> split name, read back from poem_splits.csv. Downstream
    scripts (generate_synthetic.py, train_sft.py, generate_validation.py) use
    this rather than recomputing -- compute_splits() is only ever invoked here."""
    def build():
        if not SPLITS_PATH.exists():
            raise FileNotFoundError(
                f'{SPLITS_PATH} does not exist -- run `uv run python src/make_splits.py` first.')
        with open(SPLITS_PATH, encoding='utf-8') as f:
            return {r['poem_id']: r['split'] for r in csv.DictReader(f)}
    return _cached('splits', build)


def poems_in_split(split: str) -> list:
    """[{'poem_id', 'author', 'song_title', 'song_text'}, ...] for one split
    ('train' / 'val' / 'test'), in stripped_songs.csv row order."""
    if split not in SPLIT_NAMES:
        raise ValueError(f'split must be one of {SPLIT_NAMES}, got {split!r}')
    assignment = load_splits()
    return [r for r in load_stripped_songs() if assignment.get(r['poem_id']) == split]


if __name__ == '__main__':
    build_and_save()
