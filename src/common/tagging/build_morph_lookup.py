"""
Build a feats-keyed morphological lookup from pos_tagged.csv.

Groups each author's surface forms by the MULTEXT-East morphological features
classla assigned them, so a lookup asks for the grammatical form it actually
wants rather than guessing from the last three characters.

Emits two indices:
  by_author  author -> lemma -> pos -> feats -> most common surface form
  pooled     lemma -> pos -> feats -> most common surface form (all authors)

The pooled index exists because per-author data is sparse: 81.5% of
(author, lemma, pos) cells hold exactly one feats value, and pooling makes 118%
more forms retrievable.

Run: uv run python src/build_morph_lookup.py
"""

import csv
import json
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'

MIN_POEMS = 5

with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
    songs = list(csv.DictReader(f))

author_counts = Counter(r['author'] for r in songs)
eligible = {a for a, c in author_counts.items() if c >= MIN_POEMS}
print(f"Building morph lookup for {len(eligible)} authors from pos_tagged.csv...")

# author -> lemma -> pos -> feats -> Counter(surface)
raw = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(Counter))))
# lemma -> pos -> feats -> Counter(surface)
pooled_raw = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['author'] not in eligible:
            continue
        raw[r['author']][r['lemma']][r['pos']][r['feats']][r['word']] += 1
        pooled_raw[r['lemma']][r['pos']][r['feats']][r['word']] += 1


def _collapse_author(tree):
    return {
        author: {
            lemma: {
                pos: {feats: counter.most_common(1)[0][0]
                      for feats, counter in feats_map.items()}
                for pos, feats_map in pos_map.items()
            }
            for lemma, pos_map in lemma_map.items()
        }
        for author, lemma_map in tree.items()
    }


def _collapse_pooled(tree):
    return {
        lemma: {
            pos: {feats: counter.most_common(1)[0][0]
                  for feats, counter in feats_map.items()}
            for pos, feats_map in pos_map.items()
        }
        for lemma, pos_map in tree.items()
    }


result = {'by_author': _collapse_author(raw), 'pooled': _collapse_pooled(pooled_raw)}

with open(MODELS / 'morph_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

by_author_entries = sum(
    len(feats_map)
    for lemmas in result['by_author'].values()
    for pos_map in lemmas.values()
    for feats_map in pos_map.values()
)
pooled_entries = sum(
    len(feats_map)
    for pos_map in result['pooled'].values()
    for feats_map in pos_map.values()
)
print(f"Saved morph_lookup.json  ({by_author_entries:,} per-author entries, "
      f"{pooled_entries:,} pooled entries, {len(result['by_author'])} authors)")
