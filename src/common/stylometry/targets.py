"""
Target-author selection: which authors a style-transfer run writes into.

Shared by the Gemma teacher (generate_synthetic.py), the students' training
and evaluation, and the evaluation-only judge, so it lives in common code
rather than in the students' sft_data.py (which re-exports both names).
"""

import csv
from pathlib import Path

from common.paths import ROOT
DATA = ROOT / 'data'

_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def load_profile_rows() -> list:
    def build():
        with open(DATA / 'author_style_profiles.csv', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    return _cached('profile_rows', build)


def select_target_authors(n: int, explicit: list = None) -> list:
    """
    n distinct target authors for style transfer.

    explicit, when given, is used as-is (after checking every name has a
    style profile) -- this is generate_synthetic.py's/generate_validation.py's
    --target-authors override. Otherwise: the n authors with the most poems
    among those with a style profile at all (author_style_profiles.csv already
    applies style_profiler.py's own MIN_POEMS=5 floor), ties broken by author
    name for determinism. Richer authors give exemplar_selection and
    vocabulary_palette more to work with, and being real distinct poets makes
    "distinct target styles" automatic without a bespoke diversity metric.
    """
    rows = load_profile_rows()
    by_author = {r['author']: r for r in rows}

    if explicit:
        missing = [a for a in explicit if a not in by_author]
        if missing:
            raise KeyError(f'no style profile for author(s): {missing}')
        return list(explicit)

    if n > len(rows):
        raise ValueError(
            f'requested {n} target authors but only {len(rows)} authors have a '
            f'style profile (author_style_profiles.csv)')

    ranked = sorted(rows, key=lambda r: (-int(r['num_poems']), r['author']))
    return [r['author'] for r in ranked[:n]]
