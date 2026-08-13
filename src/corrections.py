"""Hand corrections layered on top of classla's tagging.

`data/pos_corrections.csv` is the durable record of every manual fix. It is the
one file here that is never regenerated — `pos_tagged.csv` and `pos_flags.csv`
are both rebuilt from scratch on each tagging run, so edits that live only in
those files are lost the next time the corpus is re-tagged.

The workflow:

  1. review `data/pos_flags.csv`, filling in `correction` / `lemma_fix` / `scope`
  2. `uv run python src/apply_corrections.py` — harvests those edits into
     `pos_corrections.csv` and patches `pos_tagged.csv`
  3. `pos_tag_corpus.py` re-applies the whole file on every later run

To undo a correction, delete its row from `pos_corrections.csv`.

A correction keys on the word *and the tag that was wrong*, so it only ever
touches tokens tagged the way you judged incorrect — a token already carrying
the right tag is never rewritten. Scope narrows that further:

  (blank)  every matching token in the corpus
  poem     matching tokens within one poem
  row      the single occurrence in one sentence
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CORRECTIONS_CSV = DATA / 'pos_corrections.csv'

FIELDS = ['word', 'pos', 'correction', 'lemma_fix', 'scope',
          'author', 'song_title', 'context']

DROP = 'DROP'
VALID_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', DROP}
VALID_SCOPES = {'', 'poem', 'row'}


def key_for(row, scope):
    """Key a correction at the requested precision.

    Keys are tuples of different lengths, so one dict holds all three scopes and
    a longer (more specific) key never collides with a shorter one.
    """
    base = (row['word'].lower(), row['pos'])
    if scope == 'poem':
        return base + (row['author'], row['song_title'])
    if scope == 'row':
        return base + (row['author'], row['song_title'], row['context'])
    return base


def validate(row):
    """Return an error string, or None if the correction is usable."""
    fix, lemma_fix = row['correction'].strip(), row['lemma_fix'].strip()
    scope = row['scope'].strip().lower()
    if not fix and not lemma_fix:
        return 'neither correction nor lemma_fix is set'
    if fix and fix not in VALID_POS:
        return f"correction {fix!r} is not one of {', '.join(sorted(VALID_POS))}"
    if scope not in VALID_SCOPES:
        return f"scope {scope!r} is not one of (blank), poem, row"
    if fix and fix == row['pos']:
        return f"correction {fix!r} is already the current tag"
    return None


def load(path=CORRECTIONS_CSV):
    """Read the durable corrections file into a lookup keyed by scope."""
    if not Path(path).exists():
        return {}
    table = {}
    with open(path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            scope = row.get('scope', '').strip().lower()
            table[key_for(row, scope)] = (
                row['correction'].strip(), row['lemma_fix'].strip())
    return table


def lookup(table, word, pos, author, title, context):
    """Most specific scope wins: row, then poem, then global."""
    base = (word.lower(), pos)
    for k in (base + (author, title, context), base + (author, title), base):
        if k in table:
            return table[k]
    return None


def apply_to(table, word, pos, lemma, xpos, feats, author, title, context):
    """Return the corrected (pos, lemma, xpos, feats), or None to drop the token.

    A hand-corrected POS invalidates the morphology classla derived alongside
    the old tag, so xpos and feats are cleared rather than left contradicting
    the new reading. An empty xpos in pos_tagged.csv means "corrected by hand".
    """
    fix = lookup(table, word, pos, author, title, context)
    if fix is None:
        return pos, lemma, xpos, feats
    new_pos, new_lemma = fix
    if new_pos == DROP:
        return None
    if new_lemma:
        lemma = new_lemma
    if new_pos and new_pos != pos:
        return new_pos, lemma, '', ''
    return pos, lemma, xpos, feats
