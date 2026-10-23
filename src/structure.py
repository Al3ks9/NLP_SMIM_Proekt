"""
Structural targets and fit: how many lines, stanzas and words per line the
target author's poems have, and how far a generated poem is from that.

Used by the Gemma teacher's prompt (llm_style_transfer.py, which re-exports
these names) and by the evaluation metrics / GRPO reward (style_metrics.py),
so it lives in common code rather than in the teacher's pipeline.
"""

import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def load_profile_rows() -> dict:
    def build():
        with open(DATA / 'author_style_profiles.csv', encoding='utf-8') as f:
            return {r['author']: r for r in csv.DictReader(f)}
    return _cached('profile_rows', build)


STRUCTURAL_COLUMNS = [
    'avg_lines_per_poem', 'avg_tokens_per_line',
    'avg_stanzas_per_poem', 'avg_lines_per_stanza',
]


def structural_targets(target_author: str) -> dict:
    """avg_lines_per_poem, avg_tokens_per_line, avg_stanzas_per_poem,
    avg_lines_per_stanza straight off the target author's profile row."""
    rows = load_profile_rows()
    if target_author not in rows:
        raise KeyError(f'no style profile for author={target_author!r}')
    row = rows[target_author]
    return {col: float(row[col]) for col in STRUCTURAL_COLUMNS}


def _tokens(line: str) -> list:
    return re.findall(r'\S+', line)


def validate_structure(generated_text: str, targets: dict) -> dict:
    """
    Actual lines/stanzas/tokens-per-line in generated_text vs. targets.
    Measures fit only — never regenerates.
    """
    stanzas = [s for s in generated_text.strip().split('\n\n') if s.strip()]
    lines = [l for l in generated_text.strip().splitlines() if l.strip()]
    token_counts = [len(_tokens(l)) for l in lines]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0.0

    actual = {
        'actual_lines': len(lines),
        'actual_stanzas': len(stanzas),
        'actual_tokens_per_line': round(avg_tokens, 2),
        'actual_lines_per_stanza': round(len(lines) / len(stanzas), 2) if stanzas else 0.0,
    }
    return {
        **actual,
        'delta_lines': round(actual['actual_lines'] - targets['avg_lines_per_poem'], 2),
        'delta_stanzas': round(actual['actual_stanzas'] - targets['avg_stanzas_per_poem'], 2),
        'delta_tokens_per_line': round(
            actual['actual_tokens_per_line'] - targets['avg_tokens_per_line'], 2),
        'delta_lines_per_stanza': round(
            actual['actual_lines_per_stanza'] - targets['avg_lines_per_stanza'], 2),
    }
