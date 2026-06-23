"""
Build suffix-based morphological lookup from pos_tagged.csv.

Groups each author's surface forms by their last 3 characters (suffix), which
in Macedonian is strongly tied to grammatical form (person, number, tense,
definiteness, gender). No spaCy morphological features needed.

Lookup structure: author → lemma → pos → suffix → most common surface form

Run once: uv run python build_morph_lookup.py
"""

import csv
import json
from collections import defaultdict, Counter

MIN_POEMS = 5
SUFFIX_LEN = 3

# Load eligible authors
songs = []
with open('stripped_songs.csv', encoding='utf-8') as f:
    songs = list(csv.DictReader(f))

author_counts = Counter(r['author'] for r in songs)
eligible = {a for a, c in author_counts.items() if c >= MIN_POEMS}
print(f"Building morph lookup for {len(eligible)} authors from pos_tagged.csv...")

# raw[author][lemma][pos][suffix] → Counter(surface_form)
raw: dict = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
)

with open('pos_tagged.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        if r['author'] not in eligible:
            continue
        word = r['word']
        suffix = word[-SUFFIX_LEN:].lower() if len(word) >= SUFFIX_LEN else word.lower()
        raw[r['author']][r['lemma']][r['pos']][suffix][word] += 1

# Collapse to most common surface form per (author, lemma, pos, suffix)
result = {
    author: {
        lemma: {
            pos: {
                suffix: counter.most_common(1)[0][0]
                for suffix, counter in suffix_map.items()
            }
            for pos, suffix_map in pos_map.items()
        }
        for lemma, pos_map in lemma_map.items()
    }
    for author, lemma_map in raw.items()
}

with open('morph_lookup.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False)

total = sum(
    len(suffix_map)
    for a in result.values()
    for l in a.values()
    for suffix_map in l.values()
)
print(f"Saved morph_lookup.json  ({total:,} (lemma, pos, suffix) entries, {len(result)} authors)")
