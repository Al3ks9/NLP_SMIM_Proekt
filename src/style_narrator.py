"""
Style profile -> descriptive text (LLM style-transfer pipeline, step 3).

Loads author_style_profiles.csv and turns its numeric columns into prose
relative to the corpus — never absolute thresholds, since "24% adjectives" means
nothing on its own but "more adjective-heavy than most poets here" does.

Percentile rank (0-100, rank / (n-1) * 100 across all 24 authors, ties broken by
author name) is binned into tertiles — bottom/middle/top third — which splits
cleanly at 24 authors (8/8/8) and keeps the descriptor table small (10 columns
x 3 bands). top_pos_bigrams and top_rhyme_endings get their own, separate
sentence-level translation.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

NUMERIC_COLUMNS = [
    'pct_NOUN', 'pct_VERB', 'pct_ADJ', 'pct_ADV', 'pct_PROPN',
    'mattr', 'avg_tokens_per_line', 'avg_lines_per_poem',
    'avg_stanzas_per_poem', 'avg_lines_per_stanza',
]

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


# ── Percentile ranking + banding ────────────────────────────────────────────────

def author_percentiles(rows: list, columns: list = NUMERIC_COLUMNS) -> dict:
    """author -> {column: percentile (0-100)}, ranked across the given rows."""
    percentiles = {r['author']: {} for r in rows}
    n = len(rows)
    for col in columns:
        ordered = sorted(rows, key=lambda r: (float(r[col]), r['author']))
        for rank, r in enumerate(ordered):
            pct = 0.0 if n <= 1 else rank / (n - 1) * 100
            percentiles[r['author']][col] = round(pct, 2)
    return percentiles


def band(percentile: float) -> str:
    """Tertile band: bottom third / middle third / top third."""
    if percentile < 100 / 3:
        return 'low'
    if percentile < 200 / 3:
        return 'mid'
    return 'high'


# ── Descriptor tables ────────────────────────────────────────────────────────────

DESCRIPTORS = {
    'pct_NOUN': {
        'low': 'leans away from nouns, toward action and description over naming',
        'mid': 'balances nouns with the other content classes',
        'high': 'is noun-heavy, favouring concrete naming over action or description',
    },
    'pct_VERB': {
        'low': 'verbs are comparatively scarce — static rather than active scenes',
        'mid': 'uses verbs at a middling rate for the corpus',
        'high': 'is verb-driven, favouring action and event over static imagery',
    },
    'pct_ADJ': {
        'low': 'uses few adjectives — spare, unadorned diction',
        'mid': 'uses adjectives at a middling rate for the corpus',
        'high': 'leans on adjective-heavy, descriptive imagery',
    },
    'pct_ADV': {
        'low': 'rarely modifies with adverbs',
        'mid': 'uses adverbs at a middling rate for the corpus',
        'high': 'modifies frequently with adverbs — manner and degree are foregrounded',
    },
    'pct_PROPN': {
        'low': 'names few proper nouns — places and people stay unnamed',
        'mid': 'names proper nouns at a middling rate for the corpus',
        'high': 'names proper nouns often — specific people and places anchor the verse',
    },
    'mattr': {
        'low': 'has comparatively low lexical diversity, returning to the same words',
        'mid': 'has middling lexical diversity for the corpus',
        'high': 'has high lexical diversity, rarely repeating the same words',
    },
    'avg_tokens_per_line': {
        'low': 'writes short lines compared to most poets in the corpus',
        'mid': 'writes lines of middling length for the corpus',
        'high': 'writes long lines compared to most poets in the corpus',
    },
    'avg_lines_per_poem': {
        'low': 'writes shorter poems than most poets in the corpus',
        'mid': 'writes poems of middling length for the corpus',
        'high': 'writes longer poems than most poets in the corpus',
    },
    'avg_stanzas_per_poem': {
        'low': 'organizes poems into fewer stanzas than most in the corpus',
        'mid': 'organizes poems into a middling number of stanzas for the corpus',
        'high': 'organizes poems into more stanzas than most in the corpus',
    },
    'avg_lines_per_stanza': {
        'low': 'keeps stanzas short compared to most poets in the corpus',
        'mid': 'keeps stanzas of middling length for the corpus',
        'high': 'writes long stanzas compared to most poets in the corpus',
    },
}

# POS-bigram -> plain-English phrase. Fallback below covers any pair not listed
# here — 5 content POS gives 20 possible ordered pairs, not all worth a bespoke
# phrase.
BIGRAM_PHRASES = {
    'NOUN>NOUN': 'stacks nouns directly one after another',
    'ADJ>NOUN': 'frequently pairs adjectives directly before nouns',
    'NOUN>ADJ': 'frequently places adjectives directly after the noun they modify',
    'NOUN>VERB': 'often moves straight from a noun into a verb',
    'VERB>NOUN': 'often moves straight from a verb into its noun',
    'ADV>VERB': 'often modifies a verb with a preceding adverb',
    'VERB>ADV': 'often follows a verb immediately with an adverb',
    'ADJ>ADJ': 'stacks adjectives directly one after another',
}


def _bigram_phrase(pair: str) -> str:
    if pair in BIGRAM_PHRASES:
        return BIGRAM_PHRASES[pair]
    a, b = pair.split('>')
    return f'often places {a} directly before {b}'


def describe_bigrams(top_pos_bigrams: str, n: int = 2) -> str:
    """'NOUN>NOUN|ADJ>NOUN|...' -> a sentence about the top n bigrams."""
    pairs = [p for p in top_pos_bigrams.split('|') if p][:n]
    if not pairs:
        return ''
    phrases = [_bigram_phrase(p) for p in pairs]
    return 'In word order, this poet ' + ' and '.join(phrases) + '.'


def describe_rhymes(top_rhyme_endings: str, n: int = 5) -> str:
    """'ни|ле|и|ња|ил' -> a sentence about characteristic line-ending sounds."""
    endings = [e for e in top_rhyme_endings.split('|') if e][:n]
    if not endings:
        return ''
    quoted = ', '.join(f'"-{e}"' for e in endings)
    return f'Lines often end in sounds like {quoted}.'


# ── Composition ──────────────────────────────────────────────────────────────────

def load_profile_rows() -> list:
    def build():
        with open(DATA / 'author_style_profiles.csv', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    return _cached('profile_rows', build)


def load_percentiles() -> dict:
    return _cached('percentiles', lambda: author_percentiles(load_profile_rows()))


def describe_style(author: str) -> str:
    """A paragraph of descriptive style text for `author`, relative to the corpus."""
    rows = {r['author']: r for r in load_profile_rows()}
    if author not in rows:
        raise KeyError(f'no style profile for author={author!r}')
    row = rows[author]
    pct = load_percentiles()[author]

    sentences = [
        f'{author} {DESCRIPTORS[col][band(pct[col])]}.'
        for col in NUMERIC_COLUMNS
    ]
    sentences.append(describe_bigrams(row['top_pos_bigrams']))
    sentences.append(describe_rhymes(row['top_rhyme_endings']))

    return ' '.join(s for s in sentences if s)
