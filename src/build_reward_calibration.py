"""
Reward calibration for the GRPO stage: per-author distributions of the
per-poem style features (style_features.poem_features) over TRAIN-split real
poems, plus the content-similarity baseline. grpo_rewards.py scores a
generated poem against these numbers.

Train split only, unlike author_style_profiles.csv (all poems): GRPO must not
see val/test poems in any form, and the reward is part of training.

Tags come from pos_tagged.csv (the corpus build), not a fresh classla run --
grpo_rewards tags generated text with tagging.tag_poem, which reproduces that
build, so both sides of the comparison are tagged the same way.

Deterministic given --seed. Writes data/reward_calibration.json (tracked).

Usage:
    uv run python src/build_reward_calibration.py
"""

import argparse
import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import style_features as sf
import style_metrics as sm
from content_similarity import DEFAULT_CONTENT_MODEL, LineEmbedder, line_f1
from make_splits import SPLITS_PATH, poems_in_split
from sft_data import load_profile_rows

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_PATH = DATA / 'reward_calibration.json'
SEED = 42
N_CONTENT_PAIRS = 2000

log = logging.getLogger('build_reward_calibration')


def load_train_tokens(train_ids: set) -> dict:
    """poem_id -> [(lemma, pos), ...] from pos_tagged.csv, train poems only."""
    tokens = defaultdict(list)
    with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['poem_id'] in train_ids:
                tokens[r['poem_id']].append((r['lemma'], r['pos']))
    return tokens


def author_calibration(poem_feats: list, hit_rates: list, palette_size: int) -> dict:
    scalar = {}
    for f in sf.SCALAR_FEATURES:
        values = np.array([p[f] for p in poem_feats], dtype=float)
        scalar[f] = {'mean': float(values.mean()), 'std': float(values.std())}

    dist = {}
    for f in sf.DIST_FEATURES:
        total = Counter()
        for p in poem_feats:
            total.update(p[f])
        cosines = []
        for p in poem_feats:
            # leave-one-out: a poem's similarity to the rest of its author,
            # not to an aggregate that already contains it
            rest = total - p[f] if len(poem_feats) > 1 else total
            cosines.append(sf.dist_cosine(p[f], sf.normalise(rest)))
        dist[f] = {'dist': sf.normalise(total),
                   'cos_mean': float(np.mean(cosines)), 'cos_std': float(np.std(cosines))}

    floor = 1.0 / palette_size if palette_size else 1.0
    p90 = float(np.percentile(hit_rates, 90)) if hit_rates else 0.0
    return {'n_poems': len(poem_feats), 'scalar': scalar, 'dist': dist,
            'tfidf_hit_p90': max(p90, floor)}


def content_baseline(poems: list, embedder, n_pairs: int, seed: int) -> float:
    """Mean line-F1 between random pairs of poems by different authors: what
    'about nothing in particular' scores, subtracted out by the reward."""
    rng = random.Random(seed)
    f1s = []
    attempts = 0
    while len(f1s) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        a, b = rng.sample(poems, 2)
        if a['author'] == b['author']:
            continue
        ea = embedder.embed_lines(sm.lines(a['song_text']), cache=True)
        eb = embedder.embed_lines(sm.lines(b['song_text']), cache=True)
        f1s.append(line_f1(ea, eb)[2])
    return float(np.mean(f1s)) if f1s else 0.0


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj


def build_calibration(seed: int = SEED, n_content_pairs: int = N_CONTENT_PAIRS,
                      embedder=None, content_model: str = DEFAULT_CONTENT_MODEL) -> dict:
    train = poems_in_split('train')
    tokens = load_train_tokens({p['poem_id'] for p in train})
    feats = {p['poem_id']: sf.poem_features(tokens.get(p['poem_id'], []), p['song_text'])
             for p in train}

    corpus_std = {f: float(np.std([feats[pid][f] for pid in feats])) for f in sf.SCALAR_FEATURES}

    by_author = defaultdict(list)
    for p in train:
        by_author[p['author']].append(p)

    palettes = sm.load_author_tfidf()
    authors = {}
    for author in sorted(r['author'] for r in load_profile_rows()):
        poems = by_author.get(author, [])
        if not poems:
            log.warning('no train poems for profiled author %s -- skipped', author)
            continue
        hits = [sm.style_match(p['song_text'], author)['tfidf_hit_rate'] for p in poems]
        authors[author] = author_calibration(
            [feats[p['poem_id']] for p in poems], hits, len(palettes.get(author, ())))

    embedder = embedder or LineEmbedder(content_model)
    baseline = content_baseline(train, embedder, n_content_pairs, seed)

    return _round({
        'meta': {'seed': seed, 'split': 'train',
                 'splits_sha256': hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest(),
                 'content_model': content_model, 'n_train_poems': len(train),
                 'n_content_pairs': n_content_pairs},
        'corpus_std': corpus_std,
        'content_baseline': baseline,
        'authors': authors,
    })


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n-content-pairs', type=int, default=N_CONTENT_PAIRS)
    parser.add_argument('--content-model', default=DEFAULT_CONTENT_MODEL)
    parser.add_argument('--output', type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()
    cal = build_calibration(args.seed, args.n_content_pairs, content_model=args.content_model)
    args.output.write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding='utf-8')
    log.info('wrote %s: %d authors, content baseline %.3f', args.output,
             len(cal['authors']), cal['content_baseline'])
