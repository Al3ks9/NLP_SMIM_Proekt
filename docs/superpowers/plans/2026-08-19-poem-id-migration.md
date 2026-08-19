# Poem-Level ID Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every poem a stable, unique `poem_id` and switch every script that currently
groups tokens by `(author, song_title)` — which is ambiguous for 15 pairs in this corpus —
to key on `poem_id` instead.

**Architecture:** `poem_id` is a plain sequential integer assigned once, by row order, in
`data/stripped_songs.csv`. It rides as a passenger column through `pos_tagged.csv`,
`pos_flags.csv`, and `pos_corrections.csv`, and every script that currently groups by
`(author, song_title)` switches its grouping key to it. One script gets a real interface
change (`candidate_selection.py`'s `corpus_token_stream`/`find_source_poem`); the rest are
mechanical key swaps. The corpus gets tagged once early (Task 3) so every later task can be
verified against real, poem_id-bearing data rather than waiting for one final rebuild.

**Tech Stack:** Python 3.10, `csv` stdlib module, `pytest`, `uv`.

**Spec:** `docs/superpowers/specs/2026-08-19-poem-id-migration-design.md`

## Global Constraints

- `poem_id` is a plain sequential integer (row position in `stripped_songs.csv`), not a UUID.
- Every poem gets an id — no partial/ambiguous-only coverage.
- `author`/`song_title` stay as human-readable columns everywhere; only the *keys code
  computes on* change to `poem_id`.
- No row reordering/deletion assumed anywhere — only appends preserve id stability, and
  nothing in this plan reorders existing rows.
- Follow existing project conventions: `pathlib.Path(__file__)`-relative paths, `csv.DictReader`/`DictWriter`,
  TDD for pure/testable logic, no dedicated test file for scripts that don't already have one
  (matches existing repo pattern — `build_skg.py`/`enrich_skg.py`/`style_profiler.py`/
  `pos_tag_corpus.py` are verified by running them and inspecting output, not unit tests).

---

## Task 1: `poem_id` source of truth

**Files:**
- Create: `src/add_poem_ids.py`
- Create: `tests/test_add_poem_ids.py`
- Modify: `src/strip_lines.py`

**Interfaces:**
- Produces: `resolve_correction_poem_id(row, ids_by_title) -> int` (raises `SystemExit` on
  an ambiguous, unresolvable poem/row-scoped correction) — pure, used by
  `migrate_pos_corrections()`.
- Produces on disk: `data/stripped_songs.csv` gains a first column `poem_id` (int, row
  order); `data/pos_corrections.csv` gains a `poem_id` column (blank for `scope=''` rows).

- [ ] **Step 1: Write the failing test for the pure resolution helper**

```python
# tests/test_add_poem_ids.py
"""poem_id migration: the pure helper that resolves a poem/row-scoped correction's
(author, song_title) to exactly one poem_id, refusing rather than guessing."""

import pytest

import add_poem_ids as api


IDS_BY_TITLE = {
    ('А', 'Уникатна'): [5],
    ('А', 'Дупликат'): [12, 40],
}


def test_resolves_an_unambiguous_title_to_its_poem_id():
    row = {'author': 'А', 'song_title': 'Уникатна', 'scope': 'poem', 'word': 'х', 'pos': 'NOUN'}
    assert api.resolve_correction_poem_id(row, IDS_BY_TITLE) == 5


def test_global_scope_rows_are_not_resolved():
    row = {'author': 'А', 'song_title': 'Дупликат', 'scope': '', 'word': 'х', 'pos': 'NOUN'}
    assert api.resolve_correction_poem_id(row, IDS_BY_TITLE) is None


def test_raises_on_an_ambiguous_title_rather_than_guessing():
    row = {'author': 'А', 'song_title': 'Дупликат', 'scope': 'poem', 'word': 'х', 'pos': 'NOUN'}
    with pytest.raises(SystemExit):
        api.resolve_correction_poem_id(row, IDS_BY_TITLE)


def test_raises_when_the_title_matches_no_poem_at_all():
    row = {'author': 'А', 'song_title': 'Не Постои', 'scope': 'row', 'word': 'х', 'pos': 'NOUN'}
    with pytest.raises(SystemExit):
        api.resolve_correction_poem_id(row, IDS_BY_TITLE)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_add_poem_ids.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'add_poem_ids'`

- [ ] **Step 3: Write `src/add_poem_ids.py`**

```python
"""
One-time migration: add a stable poem_id to stripped_songs.csv and (retroactively)
to pos_corrections.csv.

poem_id is a plain sequential integer, assigned by row order in stripped_songs.csv --
see docs/superpowers/specs/2026-08-19-poem-id-migration-design.md for why (not a UUID,
not partial coverage). Idempotent: running it again on an already-migrated file prints
a message and does nothing.

Run once:
    uv run python src/add_poem_ids.py
"""

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'


def resolve_correction_poem_id(row: dict, ids_by_title: dict):
    """
    The single poem_id a poem/row-scoped correction's (author, song_title) resolves
    to, or None for a global-scope row (which never used author/song_title to begin
    with). Raises SystemExit -- refuses rather than guessing -- if the title matches
    zero or more than one poem; there's no way to recover which physical poem a
    hand-written correction meant in that case.
    """
    scope = row['scope'].strip().lower()
    if scope not in ('poem', 'row'):
        return None
    matches = ids_by_title.get((row['author'], row['song_title']), [])
    if len(matches) != 1:
        raise SystemExit(
            f"Cannot migrate: {row['word']!r} ({row['pos']}) is scope={scope!r} for "
            f"author={row['author']!r} song_title={row['song_title']!r}, which matches "
            f"{len(matches)} poems. Disambiguate this correction by hand before migrating."
        )
    return matches[0]


def migrate_stripped_songs() -> dict:
    """Add poem_id to stripped_songs.csv. Returns {(author, song_title): [poem_id, ...]}
    for migrate_pos_corrections() to resolve corrections against."""
    path = DATA / 'stripped_songs.csv'
    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if fieldnames[0] == 'poem_id':
        print(f'{path.name} already has poem_id -- skipping')
        ids_by_title = {}
        for row in rows:
            ids_by_title.setdefault((row['author'], row['song_title']), []).append(
                int(row['poem_id']))
        return ids_by_title

    ids_by_title = {}
    for poem_id, row in enumerate(rows):
        row['poem_id'] = poem_id
        ids_by_title.setdefault((row['author'], row['song_title']), []).append(poem_id)

    shutil.copy(path, path.with_suffix('.csv.bak'))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['poem_id'] + fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{path.name}: added poem_id to {len(rows)} rows (backup: {path.with_suffix(".csv.bak").name})')
    return ids_by_title


def migrate_pos_corrections(ids_by_title: dict) -> None:
    """Add poem_id to pos_corrections.csv, resolving each poem/row-scoped row's
    (author, song_title) via resolve_correction_poem_id()."""
    path = DATA / 'pos_corrections.csv'
    if not path.exists():
        print(f'{path.name} does not exist -- nothing to migrate')
        return

    with open(path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if 'poem_id' in fieldnames:
        print(f'{path.name} already has poem_id -- skipping')
        return

    for row in rows:
        poem_id = resolve_correction_poem_id(row, ids_by_title)
        row['poem_id'] = poem_id if poem_id is not None else ''

    out_fields = ['word', 'pos', 'correction', 'lemma_fix', 'scope', 'poem_id',
                 'author', 'song_title', 'context']
    shutil.copy(path, path.with_suffix('.csv.bak'))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f'{path.name}: added poem_id to {len(rows)} rows (backup: {path.with_suffix(".csv.bak").name})')


if __name__ == '__main__':
    ids_by_title = migrate_stripped_songs()
    migrate_pos_corrections(ids_by_title)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/test_add_poem_ids.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the migration against the real corpus**

Run: `uv run python src/add_poem_ids.py`
Expected output: `stripped_songs.csv: added poem_id to 2711 rows (backup: stripped_songs.csv.bak)`
and `pos_corrections.csv: added poem_id to 87 rows (backup: pos_corrections.csv.bak)`.

Verify: `head -3 data/stripped_songs.csv` shows `poem_id,author,song_title,song_text` as the
header and `0`, `1`, ... in the first column. `head -3 data/pos_corrections.csv` shows
`word,pos,correction,lemma_fix,scope,poem_id,author,song_title,context` with `poem_id` blank
on every row (all 87 existing corrections are global scope, confirmed during design).

- [ ] **Step 6: Rewrite `src/strip_lines.py` to match, for future regenerations**

Read the current file first — it does a naive per-line `.strip()` copy from `songs.csv`,
without using the `csv` module at all, so it can't insert a column as written.

```python
"""
Regenerate data/stripped_songs.csv from the raw songs.csv, assigning poem_id by row
order -- see docs/superpowers/specs/2026-08-19-poem-id-migration-design.md.

Not part of CLAUDE.md's regular pipeline order (data/stripped_songs.csv is the
checked-in base input that order starts from) -- only needed if songs.csv itself
changes and stripped_songs.csv must be regenerated from scratch. Uses the csv module
(the previous version read songs.csv line-by-line without parsing it as CSV, which
can't survive multi-line quoted fields safely and can't insert a column).
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
in_file = ROOT / 'songs.csv'
out_file = ROOT / 'data' / 'stripped_songs.csv'

with open(in_file, encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

for poem_id, row in enumerate(rows):
    row['author'] = row['author'].strip()
    row['song_title'] = row['song_title'].strip()
    # Strip each physical line of song_text individually, matching the original
    # script's per-line behavior, rather than only the field's outer whitespace.
    row['song_text'] = '\n'.join(line.strip() for line in row['song_text'].split('\n'))
    row['poem_id'] = poem_id

with open(out_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['poem_id', 'author', 'song_title', 'song_text'])
    writer.writeheader()
    writer.writerows(rows)

print(f'Wrote {len(rows)} poems to {out_file}')
```

- [ ] **Step 7: Verify the rewritten strip_lines.py reproduces the migrated file's content**

Run it to a scratch copy and diff against the real (already-migrated) file, ignoring
nothing — they should now match exactly, since Step 5 already added the same `poem_id`
scheme by row order:

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
import strip_lines  # runs at import time, writing data/stripped_songs.csv
"
git diff --stat data/stripped_songs.csv
```
Expected: `git diff --stat` shows no changes (or only trivial whitespace-equivalent ones you
inspect by hand) — confirming the rewritten script reproduces what Step 5 already produced.
If it doesn't match, do not proceed — investigate the discrepancy before continuing (most
likely cause: a `song_text` field whose per-line stripping differs from the raw-line-copy
original, worth eyeballing directly).

- [ ] **Step 8: Commit**

```bash
git add src/add_poem_ids.py tests/test_add_poem_ids.py src/strip_lines.py \
        data/stripped_songs.csv data/pos_corrections.csv
git commit -m "Add poem_id to stripped_songs.csv and pos_corrections.csv"
```

---

## Task 2: `corrections.py` keys on `poem_id`

**Files:**
- Modify: `src/corrections.py`
- Create: `tests/test_corrections.py`

**Interfaces:**
- Consumes: `data/pos_corrections.csv`'s `poem_id` column (Task 1).
- Produces: `key_for(row, scope) -> tuple`, `lookup(table, word, pos, poem_id, context) -> tuple | None`,
  `apply_to(table, word, pos, lemma, xpos, feats, poem_id, context) -> tuple | None` — all now
  take `poem_id` instead of `(author, title)`. Consumed by `pos_tag_corpus.py` (Task 3) and
  `apply_corrections.py` (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corrections.py
"""poem_id-based correction keying — replaces (author, song_title), which is not a
unique poem key in this corpus (~15 shared titles; see CLAUDE.md's Data quality
context)."""

import corrections as C


def test_key_for_global_scope_ignores_poem_id():
    row = {'word': 'гора', 'pos': 'NOUN', 'poem_id': ''}
    assert C.key_for(row, '') == ('гора', 'NOUN')


def test_key_for_poem_scope_uses_poem_id_not_title():
    row = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '42'}
    assert C.key_for(row, 'poem') == ('гора', 'NOUN', 42)


def test_key_for_row_scope_uses_poem_id_and_context():
    row = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '42', 'context': 'некоја реченица'}
    assert C.key_for(row, 'row') == ('гора', 'NOUN', 42, 'некоја реченица')


def test_two_same_titled_poems_do_not_collide_at_poem_scope():
    # The whole point: (author, song_title) alone used to be the key, so two poems
    # sharing a title would collide. poem_id keeps them apart.
    row_a = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '1'}
    row_b = {'word': 'гора', 'pos': 'NOUN', 'poem_id': '2'}
    assert C.key_for(row_a, 'poem') != C.key_for(row_b, 'poem')


def test_lookup_finds_a_poem_scoped_correction_by_poem_id():
    table = {('гора', 'NOUN', 42): ('PROPN', '')}
    assert C.lookup(table, 'гора', 'NOUN', 42, 'иргелава реченица') == ('PROPN', '')


def test_lookup_does_not_match_a_different_poem_id():
    table = {('гора', 'NOUN', 42): ('PROPN', '')}
    assert C.lookup(table, 'гора', 'NOUN', 43, 'иргелава реченица') is None


def test_lookup_row_scope_beats_poem_scope():
    table = {
        ('гора', 'NOUN', 42): ('PROPN', ''),
        ('гора', 'NOUN', 42, 'точна реченица'): ('VERB', ''),
    }
    assert C.lookup(table, 'гора', 'NOUN', 42, 'точна реченица') == ('VERB', '')


def test_apply_to_drops_a_token_on_DROP():
    table = {('гора', 'NOUN', 42): (C.DROP, '')}
    assert C.apply_to(table, 'гора', 'NOUN', 'гора', 'Ncfsn', 'Case=Nom', 42, '') is None


def test_apply_to_clears_xpos_and_feats_on_a_pos_change():
    table = {('гора', 'NOUN', 42): ('PROPN', '')}
    result = C.apply_to(table, 'гора', 'NOUN', 'гора', 'Ncfsn', 'Case=Nom', 42, '')
    assert result == ('PROPN', 'гора', '', '')
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_corrections.py -v`
Expected: FAIL — `key_for`/`lookup`/`apply_to` raise `KeyError: 'poem_id'` or produce the old
`(word, pos, author, title, ...)`-shaped keys, so the assertions on tuple shape/value fail.

- [ ] **Step 3: Update `src/corrections.py`**

Read the file first (you'll need exact current content to edit precisely). Make these changes:

`FIELDS`:
```python
FIELDS = ['word', 'pos', 'correction', 'lemma_fix', 'scope', 'poem_id',
          'author', 'song_title', 'context']
```

`key_for()`:
```python
def key_for(row, scope):
    """Key a correction at the requested precision.

    Keys are tuples of different lengths, so one dict holds all three scopes and
    a longer (more specific) key never collides with a shorter one. poem_id
    disambiguates scope='poem'/'row' -- (author, song_title) is not a unique poem
    key in this corpus (~15 pairs share a title); see CLAUDE.md's Data quality
    context.
    """
    base = (row['word'].lower(), row['pos'])
    if scope == 'poem':
        return base + (int(row['poem_id']),)
    if scope == 'row':
        return base + (int(row['poem_id']), row['context'])
    return base
```

`lookup()`:
```python
def lookup(table, word, pos, poem_id, context):
    """Most specific scope wins: row, then poem, then global."""
    base = (word.lower(), pos)
    pid = int(poem_id)
    for k in (base + (pid, context), base + (pid,), base):
        if k in table:
            return table[k]
    return None
```

`apply_to()`:
```python
def apply_to(table, word, pos, lemma, xpos, feats, poem_id, context):
    """Return the corrected (pos, lemma, xpos, feats), or None to drop the token.

    A hand-corrected POS invalidates the morphology classla derived alongside
    the old tag, so xpos and feats are cleared rather than left contradicting
    the new reading. An empty xpos in pos_tagged.csv means "corrected by hand".
    """
    fix = lookup(table, word, pos, poem_id, context)
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_corrections.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Run the full test suite to check for regressions in callers**

Run: `uv run pytest -q`
Expected: `pos_tag_corpus.py` and `apply_corrections.py` are not yet updated (Tasks 3-4), so
this is expected to still pass at this point — nothing else in the test suite calls
`corrections.py` directly yet. If anything fails, investigate before proceeding.

- [ ] **Step 6: Commit**

```bash
git add src/corrections.py tests/test_corrections.py
git commit -m "corrections.py: key on poem_id instead of (author, song_title)"
```

---

## Task 3: `pos_tag_corpus.py` propagates `poem_id`, then re-tag the corpus

**Files:**
- Modify: `src/pos_tag_corpus.py`

**Interfaces:**
- Consumes: `data/stripped_songs.csv`'s `poem_id` column (Task 1), `corrections.apply_to(..., poem_id, context)` (Task 2).
- Produces on disk: `data/pos_tagged.csv` and `data/pos_flags.csv` both gain a first column
  `poem_id`. Every later task (5-9) depends on this.

- [ ] **Step 1: Read `src/pos_tag_corpus.py` in full**

You need the exact current content of the tagging loop, the correction-application loop,
the tag-distribution loop, the results/flags-building loop, and both `csv.DictWriter` calls
to edit precisely — read the whole file before making changes.

- [ ] **Step 2: Thread `poem_id` through the `tokens` tuples**

In the tagging loop (`for i, row in enumerate(poems): ... tokens.append((...))`), add
`row['poem_id']` as the first element of the tuple:

```python
tokens.append((
    row['poem_id'], row['author'], row['song_title'], w.text, w.lemma or '',
    w.upos or '', w.xpos or '', w.feats or '', sid,
))
```

Update the comment above `tokens = []` to match:
```python
tokens = []       # (poem_id, author, title, word, lemma, pos, xpos, feats, sentence_id)
```

- [ ] **Step 3: Update the correction-application loop's unpacking and call**

```python
table = C.load()
if table:
    corrected, changed, removed = [], 0, 0
    for poem_id, author, title, word, lemma, pos, xpos, feats, sid in tokens:
        result = C.apply_to(table, word, pos, lemma, xpos, feats,
                            poem_id, sentences[sid])
        if result is None:
            removed += 1
            continue
        if (result[0], result[1]) != (pos, lemma):
            changed += 1
        pos, lemma, xpos, feats = result
        corrected.append((poem_id, author, title, word, lemma, pos, xpos, feats, sid))
    print(f"Applied {len(table)} hand corrections from {C.CORRECTIONS_CSV.name}: "
          f"{changed} tokens retagged, {removed} dropped")
    tokens = corrected
```

- [ ] **Step 4: Update the tag-distribution/verb-lemma loop's unpacking**

```python
tag_dists = defaultdict(Counter)
verb_lemmas = Counter()
for _, _, _, word, lemma, pos, xpos, _, _ in tokens:
    if is_content(pos, xpos):
        tag_dists[word.lower()][pos] += 1
    if pos == 'VERB' and lemma:
        verb_lemmas[lemma.lower()] += 1
```

- [ ] **Step 5: Update the results/flags-building loop**

```python
results = []
flags = []
dropped_relativizers = 0
gerunds_relemmatised = 0
gerunds_guessed = 0
for poem_id, author, title, word, lemma, pos, xpos, feats, sid in tokens:
    if pos in KEEP_POS and xpos == RELATIVIZER_XPOS:
        dropped_relativizers += 1
    if is_content(pos, xpos):
        out_lemma = lemma.lower()
        if xpos == 'Rv' and word.lower().endswith('јќи'):
            out_lemma, known = gerund_lemma(word.lower(), verb_lemmas)
            gerunds_relemmatised += known
            gerunds_guessed += not known
        results.append({
            'poem_id': poem_id,
            'author': author,
            'song_title': title,
            'word': word.lower(),
            'lemma': out_lemma,
            'pos': pos,
            'xpos': xpos,
            'feats': feats,
        })
    if pos == 'PUNCT' or not word.strip() or word.isnumeric():
        continue
    for reason, detail in flag_reasons(word, lemma, pos, xpos, tag_dists[word.lower()]):
        flags.append({
            'reason': reason,
            'word': word,
            'lemma': lemma,
            'pos': pos,
            'correction': '',
            'lemma_fix': '',
            'scope': '',
            'xpos': xpos,
            'detail': detail,
            'poem_id': poem_id,
            'author': author,
            'song_title': title,
            'context': sentences[sid],
        })
```

- [ ] **Step 6: Update both `csv.DictWriter` fieldnames lists**

```python
with open(DATA / 'pos_tagged.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f, fieldnames=['poem_id', 'author', 'song_title', 'word', 'lemma', 'pos', 'xpos', 'feats'])
    writer.writeheader()
    writer.writerows(results)
```

```python
with open(DATA / 'pos_flags.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(
        f, fieldnames=['reason', 'word', 'lemma', 'pos',
                       'correction', 'lemma_fix', 'scope',
                       'xpos', 'detail', 'poem_id', 'author', 'song_title', 'context'])
    writer.writeheader()
    writer.writerows(flags)
```

- [ ] **Step 7: Run the full re-tag**

Run: `uv run python src/pos_tag_corpus.py`
This re-runs classla tagging over the whole corpus (~70k tokens) — expect several minutes.

Expected: completes with the same summary shape as before (content-token count, POS
distribution, dropped-relativizer count, etc. — these numbers should be unchanged from before
this migration, since no tagging logic changed, only which columns get written).

- [ ] **Step 8: Verify the new schema on disk**

```bash
head -3 data/pos_tagged.csv
head -3 data/pos_flags.csv
```
Expected: both start with `poem_id,...` and show integer values in that column.

```bash
uv run python -c "
import csv
with open('data/pos_tagged.csv', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
konteski_pesna = {r['poem_id'] for r in rows if r['author'] == 'Блаже Конески' and r['song_title'] == 'ПЕСНА'}
print('distinct poem_ids for Конески/ПЕСНА:', len(konteski_pesna))
"
```
Expected: `distinct poem_ids for Конески/ПЕСНА: 4` — confirming the 4 previously-colliding
poems now carry 4 distinct ids in the tagged output.

- [ ] **Step 9: Commit**

```bash
git add src/pos_tag_corpus.py data/pos_tagged.csv data/pos_flags.csv
git commit -m "pos_tag_corpus.py: propagate poem_id into pos_tagged.csv and pos_flags.csv"
```

---

## Task 4: `apply_corrections.py` passes `poem_id`

**Files:**
- Modify: `src/apply_corrections.py`

**Interfaces:**
- Consumes: `corrections.apply_to(..., poem_id, context)` (Task 2), `pos_tagged.csv`'s
  `poem_id` column (Task 3).

- [ ] **Step 1: Update the patch loop's call to `C.apply_to()`**

Read the file first. In the loop that patches `pos_tagged.csv` in place, change:

```python
result = C.apply_to(
    table, row['word'], row['pos'], row['lemma'], row['xpos'], row['feats'],
    row['author'], row['song_title'], '')
```
to:
```python
result = C.apply_to(
    table, row['word'], row['pos'], row['lemma'], row['xpos'], row['feats'],
    row['poem_id'], '')
```

- [ ] **Step 2: Run it against the real, freshly-migrated files to verify a no-op**

Run: `uv run python src/apply_corrections.py`

Expected: since Task 3 already applied every correction in `pos_corrections.csv` during the
full re-tag, and no new edits exist in `pos_flags.csv` yet (nothing has been hand-edited this
session), this should report `No new edits in pos_flags.csv -- re-applying 87 rule(s) from
pos_corrections.csv.` and `0 tokens retagged, 0 dropped` in the patch summary — a true no-op,
proving the poem_id-based lookup reproduces the same result the tagging-time correction pass
already applied.

If it reports any tokens changed or dropped, stop and investigate — that would mean the
poem_id-keyed lookup disagrees with the corpus-wide corrections already baked into
`pos_tagged.csv` by Task 3, which should not happen since all 87 corrections are
`scope=''` (global, unaffected by the poem_id change).

- [ ] **Step 3: Commit**

```bash
git add src/apply_corrections.py
git commit -m "apply_corrections.py: pass poem_id instead of (author, song_title)"
```

(No `pos_tagged.csv`/`.csv.bak` changes expected from Step 2's no-op run, but check
`git status` — if `apply_corrections.py` wrote a `.csv.bak`, it's gitignored already via
the existing `*.csv.bak` rule.)

---

## Task 5: `build_skg.py`, `enrich_skg.py`, `style_profiler.py` key on `poem_id`

**Files:**
- Modify: `src/build_skg.py`
- Modify: `src/enrich_skg.py`
- Modify: `src/style_profiler.py`

**Interfaces:**
- Consumes: `pos_tagged.csv`'s `poem_id` column (Task 3).

- [ ] **Step 1: `src/build_skg.py` — swap the co-occurrence grouping key**

Change:
```python
poem_tokens = defaultdict(set)  # (author, song_title) -> set of (lemma, pos)
for r in pos_rows:
    key = (r['author'], r['song_title'])
    poem_tokens[key].add((r['lemma'], r['pos']))
```
to:
```python
poem_tokens = defaultdict(set)  # poem_id -> set of (lemma, pos)
for r in pos_rows:
    poem_tokens[r['poem_id']].add((r['lemma'], r['pos']))
```

- [ ] **Step 2: `src/enrich_skg.py` — swap the eligibility count and the POS-bigram grouping key**

Change:
```python
author_poem_counts = Counter((r['author'], r['song_title']) for r in pos_rows)
author_counts = Counter(r['author'] for r in pos_rows)
eligible_authors = {
    author for author in author_counts
    if len({(r['author'], r['song_title']) for r in pos_rows if r['author'] == author}) >= MIN_POEMS
}
```
to:
```python
author_poem_counts = Counter((r['author'], r['poem_id']) for r in pos_rows)
author_counts = Counter(r['author'] for r in pos_rows)
eligible_authors = {
    author for author in author_counts
    if len({r['poem_id'] for r in pos_rows if r['author'] == author}) >= MIN_POEMS
}
```

And:
```python
poem_pos_seqs = defaultdict(list)
for r in pos_rows:
    if r['author'] not in eligible_authors:
        continue
    poem_pos_seqs[(r['author'], r['song_title'])].append(r['pos'])
```
to:
```python
poem_pos_seqs = defaultdict(list)
for r in pos_rows:
    if r['author'] not in eligible_authors:
        continue
    poem_pos_seqs[(r['author'], r['poem_id'])].append(r['pos'])
```
(kept as `(author, poem_id)` here, not bare `poem_id` — the very next loop groups bigrams
back up by author: `for (author, _), seq in poem_pos_seqs.items()`, so the tuple shape must
stay a 2-tuple. Only the second element's meaning changes, from title to poem_id.)

- [ ] **Step 3: `src/style_profiler.py` — swap the POS-bigram grouping key**

Change:
```python
poem_token_map = defaultdict(list)  # (author, song_title) -> list of pos tags
for row in pos_rows:
    if row['author'] not in eligible_authors:
        continue
    poem_token_map[(row['author'], row['song_title'])].append(row['pos'])
```
to:
```python
poem_token_map = defaultdict(list)  # (author, poem_id) -> list of pos tags
for row in pos_rows:
    if row['author'] not in eligible_authors:
        continue
    poem_token_map[(row['author'], row['poem_id'])].append(row['pos'])
```
(same reasoning as Step 2 — the consuming loop `for (a, title), pos_seq in poem_token_map.items()`
only uses the first element (`a`) for the author filter, so the second element's rename from
`title` to `poem_id`/whatever local name is cosmetic; leave the loop variable name as-is or
rename to `_pid` for clarity, your choice, but the logic is unchanged.)

- [ ] **Step 4: Run all three and inspect output**

```bash
uv run python src/style_profiler.py
uv run python src/build_skg.py
uv run python src/enrich_skg.py
```

Expected: `style_profiler.py` reports "Style profiles written for 24 authors" (same count as
before — this migration doesn't change which authors qualify). `build_skg.py`/`enrich_skg.py`
report node/edge counts in the same ballpark as before (exact `top_pos_bigrams` values may
shift very slightly for the previously-affected authors — e.g. Конески, Клетников — since
spurious cross-poem bigrams at the seam of merged same-titled poems no longer form; this is
the fix working as intended, not a regression).

Verify the specific fix:
```bash
uv run python -c "
import csv
with open('data/author_style_profiles.csv', encoding='utf-8') as f:
    rows = {r['author']: r for r in csv.DictReader(f)}
print(rows['Блаже Конески']['top_pos_bigrams'])
"
```
(No fixed expected value — just confirm it prints a sensible pipe-separated bigram list, same
shape as before.)

- [ ] **Step 5: Commit**

```bash
git add src/build_skg.py src/enrich_skg.py src/style_profiler.py \
        data/author_style_profiles.csv models/skg.gexf models/author_vocab.json \
        models/pos_transitions.json
git commit -m "build_skg.py, enrich_skg.py, style_profiler.py: key on poem_id"
```

(`models/skg_enriched.gexf` is gitignored per the existing `.gitignore` rule — nothing to add
for it.)

---

## Task 6: `candidate_selection.py` — `corpus_token_stream`/`find_source_poem` take `poem_id`

**Files:**
- Modify: `src/candidate_selection.py:535-539` (`corpus_token_stream`)
- Modify: `src/candidate_selection.py:573-591` (`find_source_poem`)
- Modify: `src/candidate_selection.py:594-660` (`slot_contexts`, its call site)
- Modify: `tests/test_candidate_selection.py`

**Interfaces:**
- Produces: `corpus_token_stream(poem_id, pos_rows) -> list` (was `(author, song_title, pos_rows)`).
- Produces: `find_source_poem(text, author, pos_rows) -> (poem_id, song_title, coverage)`
  (was `(song_title, coverage)`).
- Consumed by: `exemplar_selection.py` (Task 8).

- [ ] **Step 1: Update the existing tests to the new return shape (RED first)**

In `tests/test_candidate_selection.py`, change:
```python
monkeypatch.setattr(cs, 'find_source_poem', lambda *a, **k: (None, 0.0))
```
to:
```python
monkeypatch.setattr(cs, 'find_source_poem', lambda *a, **k: (None, None, 0.0))
```

And change:
```python
def test_find_source_poem_identifies_the_stanzas_poem():
    title, coverage = cs.find_source_poem(STANZA, RACIN, cs.load_pos_rows())
    assert title == 'Балада за непознатиот'
    assert coverage > 0.5
```
to:
```python
def test_find_source_poem_identifies_the_stanzas_poem():
    poem_id, title, coverage = cs.find_source_poem(STANZA, RACIN, cs.load_pos_rows())
    assert title == 'Балада за непознатиот'
    assert poem_id is not None
    assert coverage > 0.5


def test_find_source_poem_resolves_a_duplicated_title_to_the_right_poem():
    # Confirmed in stripped_songs.csv: 4 distinct poems by Конески are titled
    # 'ПЕСНА'. Aligning against one specific poem's own text must resolve to its
    # own poem_id, not silently merge with the other 3 sharing the title.
    import csv
    with open(cs.DATA / 'stripped_songs.csv', encoding='utf-8') as f:
        pesna_poems = [r for r in csv.DictReader(f)
                      if r['author'] == KONESKI and r['song_title'] == 'ПЕСНА']
    assert len(pesna_poems) > 1, 'fixture assumption: Конески has >1 poem titled ПЕСНА'

    target = pesna_poems[0]
    stanza = '\n'.join(target['song_text'].strip().splitlines()[:4])

    poem_id, title, coverage = cs.find_source_poem(stanza, KONESKI, cs.load_pos_rows())

    assert poem_id == target['poem_id']
    assert coverage > 0.5
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_candidate_selection.py -v`
Expected: FAIL — `corpus_token_stream()`/`find_source_poem()` still have the old signature/return
shape, so the new assertions don't match.

- [ ] **Step 3: Update `corpus_token_stream()`**

```python
def corpus_token_stream(poem_id, pos_rows) -> list:
    """The ordered content-token stream of one poem, as tagged in pos_tagged.csv.

    Keyed on poem_id, not (author, song_title) -- ~15 (author, song_title) pairs
    in this corpus are shared by multiple distinct poems (see CLAUDE.md's Data
    quality context), so title alone can't identify one.
    """
    poem_id = str(poem_id)
    return [r for r in pos_rows if r['poem_id'] == poem_id]
```

- [ ] **Step 4: Update `find_source_poem()`**

```python
def find_source_poem(text: str, author: str, pos_rows) -> tuple:
    """
    Locate the poem a stanza came from, by picking the poem of that author whose
    token stream the stanza aligns into most completely.

    Iterates the author's distinct poem_ids, not distinct titles -- titles alone
    would silently collapse same-titled poems into one alignment candidate.

    Returns (poem_id, song_title, coverage) or (None, None, 0.0).
    """
    wanted = [w for line in text.splitlines() for w in _words(line)]
    if not wanted:
        return None, None, 0.0

    poem_ids = sorted({r['poem_id'] for r in pos_rows if r['author'] == author}, key=int)
    best_id, best_title, best_cov = None, None, 0.0
    for pid in poem_ids:
        tokens = corpus_token_stream(pid, pos_rows)
        title = tokens[0]['song_title'] if tokens else None
        cov = len(_align(text, tokens)) / len(wanted)
        if cov > best_cov:
            best_id, best_title, best_cov = pid, title, cov
    return best_id, best_title, best_cov
```

- [ ] **Step 5: Update `slot_contexts()`'s call site**

Change:
```python
title, coverage = find_source_poem(text, source_author, pos_rows)
if title is not None and coverage >= min_coverage:
    tokens = corpus_token_stream(source_author, title, pos_rows)
    mapping = _align(text, tokens)
    tagged = {(li, wi): tokens[ti] for (li, wi), ti in mapping.items()}
    log.info('source context from pos_tagged.csv: author=%s poem=%r coverage=%.2f',
             source_author, title, coverage)
```
to:
```python
poem_id, title, coverage = find_source_poem(text, source_author, pos_rows)
if poem_id is not None and coverage >= min_coverage:
    tokens = corpus_token_stream(poem_id, pos_rows)
    mapping = _align(text, tokens)
    tagged = {(li, wi): tokens[ti] for (li, wi), ti in mapping.items()}
    log.info('source context from pos_tagged.csv: author=%s poem=%r poem_id=%s coverage=%.2f',
             source_author, title, poem_id, coverage)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/test_candidate_selection.py -v`
Expected: PASS, including the new `test_find_source_poem_resolves_a_duplicated_title_to_the_right_poem`.

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass — this is `candidate_selection.py`'s only public interface change, and
`exemplar_selection.py` isn't updated yet (Task 8), but it calls `corpus_token_stream` with
`(author, title, pos_rows)` today, which no longer matches the new 2-arg signature. Check
whether `tests/test_exemplar_selection.py` fails here; if it does, that's expected and will be
fixed in Task 8, not a mistake in this task.

- [ ] **Step 8: Commit**

```bash
git add src/candidate_selection.py tests/test_candidate_selection.py
git commit -m "candidate_selection.py: corpus_token_stream/find_source_poem key on poem_id"
```

---

## Task 7: `poem_tfidf.py` keys on `poem_id`

**Files:**
- Modify: `src/poem_tfidf.py`
- Modify: `tests/test_poem_tfidf.py`

**Interfaces:**
- Produces: `build_documents(pos_rows, keep_pos=KEEP_POS) -> {poem_id: [lemma, ...]}` (was
  `{(author, title): [...]}`). `top_content_words(poem_id, n=10) -> list` (was
  `(author, song_title, n=10)`).
- Produces on disk: `data/poem_tfidf_results.csv` gains a first column `poem_id`.
- Consumed by: `llm_style_transfer.py` (Task 9).

- [ ] **Step 1: Update the existing tests to key by poem_id (RED first)**

```python
# tests/test_poem_tfidf.py
def test_word_unique_to_one_poem_ranks_above_a_word_common_to_all():
    documents = {
        '0': ['нива', 'сокол', 'сокол', 'нива'],
        '1': ['нива', 'река', 'нива'],
        '2': ['нива', 'облак'],
    }
    scores = pt.compute_tfidf(documents)
    top_word_a = scores['0'][0][0]
    assert top_word_a == 'сокол'


def test_scores_within_a_document_are_sorted_descending():
    documents = {
        '0': ['алфа', 'алфа', 'бета', 'гама'],
        '1': ['делта', 'делта', 'делта'],
    }
    scores = pt.compute_tfidf(documents)
    doc_scores = [s for _, s in scores['0']]
    assert doc_scores == sorted(doc_scores, reverse=True)


def test_a_word_absent_from_a_document_does_not_appear_in_its_results():
    documents = {
        '0': ['алфа', 'бета'],
        '1': ['гама'],
    }
    scores = pt.compute_tfidf(documents)
    words_in_0 = {w for w, _ in scores['0']}
    assert 'гама' not in words_in_0


def test_build_documents_keeps_same_titled_poems_separate():
    # The whole point: two distinct poems sharing a title must not blend their
    # vocabulary into one TF-IDF document.
    pos_rows = [
        {'poem_id': '1', 'pos': 'NOUN', 'lemma': 'сокол'},
        {'poem_id': '2', 'pos': 'NOUN', 'lemma': 'камен'},
    ]
    documents = pt.build_documents(pos_rows)
    assert documents == {'1': ['сокол'], '2': ['камен']}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_poem_tfidf.py -v`
Expected: FAIL — `build_documents()` still groups by `(r['author'], r['song_title'])`, and
`compute_tfidf` is unaffected in isolation but the fixtures above no longer match its current
key shape assumptions in the surrounding test file before this edit.

- [ ] **Step 3: Update `src/poem_tfidf.py`**

```python
def build_documents(pos_rows, keep_pos=KEEP_POS) -> dict:
    """poem_id -> [lemma, lemma, ...] in corpus order, content POS only."""
    documents = defaultdict(list)
    for r in pos_rows:
        if r['pos'] in keep_pos:
            documents[r['poem_id']].append(r['lemma'])
    return dict(documents)
```

`build_and_save()`:
```python
def build_and_save(top_n: int = TOP_N) -> None:
    """Compute per-poem TF-IDF over the full corpus and write poem_tfidf_results.csv."""
    pos_rows = load_pos_rows()
    documents = build_documents(pos_rows)
    scores = compute_tfidf(documents)

    # poem_id -> (author, song_title), for the human-readable output columns.
    poem_info = {r['poem_id']: (r['author'], r['song_title']) for r in pos_rows}

    with open(DATA / 'poem_tfidf_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['poem_id', 'author', 'song_title', 'rank', 'lemma', 'tfidf_score'])
        for poem_id, words in sorted(scores.items(), key=lambda kv: int(kv[0])):
            author, title = poem_info[poem_id]
            for rank, (lemma, score) in enumerate(words[:top_n], 1):
                writer.writerow([poem_id, author, title, rank, lemma, score])

    print(f'Saved poem_tfidf_results.csv ({len(scores)} poems)')
```

`top_content_words()`:
```python
def top_content_words(poem_id, n: int = TOP_N) -> list:
    """A poem's top-N content lemmas, read from poem_tfidf_results.csv."""
    def build():
        rows = defaultdict(list)
        with open(DATA / 'poem_tfidf_results.csv', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                rows[row['poem_id']].append((int(row['rank']), row['lemma']))
        return {k: [lemma for _, lemma in sorted(v)] for k, v in rows.items()}

    by_poem = _cached('poem_tfidf', build)
    return by_poem.get(str(poem_id), [])[:n]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_poem_tfidf.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Rerun against the real corpus**

Run: `uv run python src/poem_tfidf.py`
Expected: `Saved poem_tfidf_results.csv (1169 poems)` (same poem count as before this migration
— this fixes content blending for the 15 affected pairs, it doesn't change the poem count).

Verify the fix directly:
```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
import poem_tfidf as pt
import csv
with open('data/stripped_songs.csv', encoding='utf-8') as f:
    pesna = [r for r in csv.DictReader(f)
            if r['author'] == 'Блаже Конески' and r['song_title'] == 'ПЕСНА']
for r in pesna:
    print(r['poem_id'], pt.top_content_words(r['poem_id'], n=5))
"
```
Expected: 4 different `poem_id`s, each with its own distinct top-5 keyword list (not
identical lists across all 4 — that would indicate the blending bug is still present).

- [ ] **Step 6: Commit**

```bash
git add src/poem_tfidf.py tests/test_poem_tfidf.py data/poem_tfidf_results.csv
git commit -m "poem_tfidf.py: key on poem_id instead of (author, song_title)"
```

---

## Task 8: `exemplar_selection.py` passes `poem_id`

**Files:**
- Modify: `src/exemplar_selection.py`

**Interfaces:**
- Consumes: `corpus_token_stream(poem_id, pos_rows)` (Task 6's new signature).
- No public signature change — `select_exemplars(target_author, ...)` still takes an author
  name; the fix is entirely internal to `qualifying_lines()`.

- [ ] **Step 1: Update `qualifying_lines()`'s call to `corpus_token_stream()`**

Change:
```python
poem_tokens = corpus_token_stream(target_author, title, pos_rows)
```
to:
```python
poem_tokens = corpus_token_stream(row['poem_id'], pos_rows)
```

(`title` stays used elsewhere in the function for the returned `'song_title'` field — only
the `corpus_token_stream()` call itself changes.)

- [ ] **Step 2: Run the existing tests to verify they still pass**

Run: `uv run pytest tests/test_exemplar_selection.py -v`
Expected: PASS (10 tests) — no test changes needed, since `select_exemplars()`'s public
output shape is unchanged and `pos_tagged.csv` already has `poem_id` from Task 3.

- [ ] **Step 3: Commit**

```bash
git add src/exemplar_selection.py
git commit -m "exemplar_selection.py: pass poem_id to corpus_token_stream"
```

---

## Task 9: `llm_style_transfer.py` — `load_poem_text_by_id` and `--source-poem-id`

**Files:**
- Modify: `src/llm_style_transfer.py`
- Modify: `tests/test_llm_style_transfer.py`

**Interfaces:**
- Produces: `load_poem_text_by_id(poem_id) -> str`, `resolve_poem_id(author, song_title) -> str`
  (raises `KeyError`/`ValueError`; `load_poem_text()` becomes a thin wrapper over both).
- Produces: `assemble_prompt(source_poem_text, source_poem_id, source_author, source_title, target_author, ...)`
  — gains `source_poem_id` as its second positional parameter.
- Produces: `run_transfer(source_author, source_title, target_author, ..., source_poem_id=None, ...)`
  — gains an optional `source_poem_id` keyword parameter that, when given, bypasses title
  resolution entirely.
- Produces: CLI gains `--source-poem-id`.
- Consumes: `poem_tfidf.top_content_words(poem_id, n=...)` (Task 7's new signature).

- [ ] **Step 1: Write the failing tests**

In `tests/test_llm_style_transfer.py`, replace the existing `load_poem_text` tests (from
the previous session's fix) with:

```python
# ── load_poem_text / load_poem_text_by_id / resolve_poem_id ────────────────────

def test_load_poem_text_returns_the_poem_for_an_unambiguous_title():
    text = lst.load_poem_text(KONESKI, 'Разделба')
    assert isinstance(text, str) and text.strip()


def test_load_poem_text_raises_on_a_real_duplicated_title():
    # Verified in stripped_songs.csv: 4 distinct poems by Конески are titled
    # 'ПЕСНА'. Silently returning one would be a plausible-looking wrong poem.
    with pytest.raises(ValueError):
        lst.load_poem_text(KONESKI, 'ПЕСНА')


def test_resolve_poem_id_error_lists_the_candidate_ids():
    with pytest.raises(ValueError, match=r'--source-poem-id'):
        lst.resolve_poem_id(KONESKI, 'ПЕСНА')


def test_load_poem_text_by_id_resolves_one_of_the_duplicated_pesna_poems():
    import csv
    with open(lst.DATA / 'stripped_songs.csv', encoding='utf-8') as f:
        pesna_ids = [r['poem_id'] for r in csv.DictReader(f)
                    if r['author'] == KONESKI and r['song_title'] == 'ПЕСНА']
    assert len(pesna_ids) > 1, 'fixture assumption: Конески has >1 poem titled ПЕСНА'

    text = lst.load_poem_text_by_id(pesna_ids[0])
    assert isinstance(text, str) and text.strip()


def test_load_poem_text_by_id_raises_for_an_unknown_id():
    with pytest.raises(KeyError):
        lst.load_poem_text_by_id('999999')
```

Update `_stub_components()`'s `poem_tfidf.top_content_words` stub to match the new signature:
```python
monkeypatch.setattr(poem_tfidf, 'top_content_words',
                    lambda poem_id, n=10: ['пролет', 'цвет'])
```

Update the three `assemble_prompt(...)` calls in the prompt-assembly tests to include a
`source_poem_id` second positional argument (any placeholder value — it's only threaded
through to the stubbed `top_content_words`):
```python
result = lst.assemble_prompt('изворен текст', '0', 'Извор Автор', 'Извор Наслов', 'Целен Автор')
```
(three call sites: `test_assemble_prompt_includes_every_component`,
`test_assemble_prompt_strips_poem_attribution_from_exemplar_lines`,
`test_assemble_prompt_returns_components_for_logging`)

Replace `test_run_transfer_orchestrates_the_full_chain`:
```python
def test_run_transfer_orchestrates_the_full_chain(monkeypatch, tmp_path):
    monkeypatch.setattr(lst, 'LOG_DIR', tmp_path)
    monkeypatch.setattr(lst, 'load_poem_text_by_id', lambda poem_id: 'изворен текст')
    monkeypatch.setattr(lst, '_load_stripped_songs_indexed', lambda: {
        '7': {'author': 'Извор Автор', 'song_title': 'Извор Наслов', 'song_text': 'изворен текст'},
    })
    monkeypatch.setattr(lst, 'assemble_prompt', lambda *a, **kw: {
        'prompt': 'составен промпт', 'summary': 'резиме', 'content_keywords': ['зб'],
        'style_text': 'стил', 'vocab_palette': ['palette'], 'exemplars': [],
        'structural_targets': {'avg_lines_per_poem': 4, 'avg_tokens_per_line': 3,
                               'avg_stanzas_per_poem': 1, 'avg_lines_per_stanza': 4},
    })
    monkeypatch.setattr(lst, 'generate_poem', lambda prompt, **kw: 'нова\nпесна')

    result = lst.run_transfer('Извор Автор', 'Извор Наслов', 'Целен Автор', source_poem_id='7')

    assert result['generated_poem'] == 'нова\nпесна'
    assert result['structural_fit']['actual_lines'] == 2
    assert result['source_author'] == 'Извор Автор'
    assert result['source_poem_id'] == '7'
    assert result['target_author'] == 'Целен Автор'
    logged = list(tmp_path.glob('*.json'))
    assert len(logged) == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_llm_style_transfer.py -v`
Expected: FAIL — `load_poem_text_by_id`/`resolve_poem_id` don't exist yet, `assemble_prompt`
doesn't accept a `source_poem_id` argument yet, `run_transfer` doesn't accept `source_poem_id=`.

- [ ] **Step 3: Update `load_poem_text` and add the new functions**

Replace the current `load_poem_text()` (added in the previous session) with:

```python
def _load_stripped_songs_indexed() -> dict:
    """poem_id (str) -> {'author', 'song_title', 'song_text'} (that row's dict, as read)."""
    def build():
        with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
            return {r['poem_id']: r for r in csv.DictReader(f)}
    return _cached('stripped_songs_by_id', build)


def load_poem_text_by_id(poem_id) -> str:
    """A poem's text by its unambiguous poem_id."""
    rows = _load_stripped_songs_indexed()
    row = rows.get(str(poem_id))
    if row is None:
        raise KeyError(f'no poem with poem_id={poem_id!r}')
    return row['song_text']


def resolve_poem_id(author: str, song_title: str) -> str:
    """
    The single poem_id matching (author, song_title).

    (author, song_title) is NOT a unique key in stripped_songs.csv -- ~15 pairs
    are shared by multiple distinct poems (e.g. Конески has 4 poems titled
    'ПЕСНА'; see CLAUDE.md's Data quality context). Raises rather than guessing
    on an ambiguous title -- the error lists the actual candidate poem_ids so
    the caller can pick one via --source-poem-id.
    """
    by_id = _load_stripped_songs_indexed()
    matches = sorted(
        (pid for pid, r in by_id.items()
         if r['author'] == author and r['song_title'] == song_title),
        key=int,
    )
    if not matches:
        raise KeyError(f'no poem for author={author!r} song_title={song_title!r}')
    if len(matches) > 1:
        raise ValueError(
            f'{len(matches)} poems share author={author!r} song_title={song_title!r} in '
            "stripped_songs.csv — (author, song_title) isn't a unique poem key here. "
            f'Use --source-poem-id to pick one of: {matches}.'
        )
    return matches[0]


def load_poem_text(author: str, song_title: str) -> str:
    """A poem's text by (author, song_title) -- raises if the title is ambiguous;
    see resolve_poem_id(). Use load_poem_text_by_id() when the poem_id is already
    known (e.g. from --source-poem-id)."""
    return load_poem_text_by_id(resolve_poem_id(author, song_title))
```

- [ ] **Step 4: Add `source_poem_id` to `assemble_prompt()` and thread it to `poem_tfidf`**

Change the signature and the `top_content_words` call:
```python
def assemble_prompt(source_poem_text: str, source_poem_id, source_author: str,
                    source_title: str, target_author: str, model: str = DEFAULT_MODEL,
                    backend: str = DEFAULT_BACKEND, instr_lang: str = DEFAULT_INSTR_LANG,
                    n_keywords: int = N_CONTENT_KEYWORDS,
                    n_vocab: int = N_VOCAB_PALETTE,
                    n_clusters: int = N_EXEMPLAR_CLUSTERS,
                    per_cluster: int = EXEMPLARS_PER_CLUSTER) -> dict:
    """..."""
    summary = summarize_poem(source_poem_text, author=source_author,
                             title=source_title, model=model, backend=backend)
    content_keywords = poem_tfidf.top_content_words(source_poem_id, n=n_keywords)
    ...
```
(only the `content_keywords` line's arguments change — everything else in the function body
is untouched.)

- [ ] **Step 5: Add `source_poem_id` to `run_transfer()`**

```python
def run_transfer(source_author: str, source_title: str, target_author: str,
                 model: str = DEFAULT_MODEL, backend: str = DEFAULT_BACKEND,
                 instr_lang: str = DEFAULT_INSTR_LANG, source_poem_id=None,
                 **assemble_kw) -> dict:
    """
    Full step 1-9 chain for one source poem -> target author transfer.

    source_poem_id, when given, identifies the source poem directly --
    source_author/source_title are then overwritten from the poem's own record
    (safe to pass placeholders for them in that case). This is what actually
    reaches every one of the corpus's ~15 same-titled poem groups, since
    (author, song_title) alone can't. Without it, source_author/source_title
    resolve through resolve_poem_id(), which raises on an ambiguous title.
    """
    poem_id = source_poem_id if source_poem_id is not None else resolve_poem_id(
        source_author, source_title)
    source_text = load_poem_text_by_id(poem_id)
    source_row = _load_stripped_songs_indexed()[str(poem_id)]
    source_author, source_title = source_row['author'], source_row['song_title']

    assembled = assemble_prompt(source_text, poem_id, source_author, source_title,
                                target_author, model=model, backend=backend,
                                instr_lang=instr_lang, **assemble_kw)
    generated = generate_poem(assembled['prompt'], model=model, backend=backend)
    fit = validate_structure(generated, assembled['structural_targets'])

    record = {
        **assembled,
        'generated_poem': generated,
        'structural_fit': fit,
        'source_poem_id': poem_id,
        'source_author': source_author,
        'source_title': source_title,
        'target_author': target_author,
        'model': model,
        'backend': backend,
        'instr_lang': instr_lang,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    log_path = log_run(record)
    log.info('transfer source=%s (%s, poem_id=%s) -> target=%s  lines=%d/%d stanzas=%d/%d  log=%s',
            source_author, source_title, poem_id, target_author,
            fit['actual_lines'], assembled['structural_targets']['avg_lines_per_poem'],
            fit['actual_stanzas'], assembled['structural_targets']['avg_stanzas_per_poem'],
            log_path)
    return {**record, 'log_path': str(log_path)}
```

- [ ] **Step 6: Add `--source-poem-id` to the CLI**

```python
parser.add_argument('--source-author', default='Кочо Рацин')
parser.add_argument('--source-title')
parser.add_argument('--source-poem-id', default=None)
parser.add_argument('--target-author', default='Блаже Конески')
parser.add_argument('--model', default=DEFAULT_MODEL)
parser.add_argument('--backend', default=DEFAULT_BACKEND, choices=['ollama', 'openrouter'])
parser.add_argument('--lang', default=DEFAULT_INSTR_LANG, choices=['en', 'mk'],
                   dest='instr_lang')
args = parser.parse_args()

if args.source_poem_id is None and not args.source_title:
    # No title or id given: pick this author's first poem in the corpus.
    with open(DATA / 'stripped_songs.csv', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row['author'] == args.source_author:
                args.source_title = row['song_title']
                break
    if not args.source_title:
        sys.exit(f'No poems found for source author {args.source_author!r}')

try:
    result = run_transfer(args.source_author, args.source_title, args.target_author,
                          model=args.model, backend=args.backend, instr_lang=args.instr_lang,
                          source_poem_id=args.source_poem_id)
except KeyError as e:
    sys.exit(e.args[0])
except ValueError as e:
    sys.exit(str(e))
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest tests/test_llm_style_transfer.py -v`
Expected: PASS (all tests, including the 5 new/updated `load_poem_text`/`resolve_poem_id` cases).

- [ ] **Step 8: Manual smoke test of the new CLI flag (no live LLM needed)**

```bash
uv run python -c "
import sys; sys.path.insert(0, 'src')
import llm_style_transfer as lst
import csv
with open('data/stripped_songs.csv', encoding='utf-8') as f:
    pesna_ids = [r['poem_id'] for r in csv.DictReader(f)
                if r['author'] == 'Блаже Конески' and r['song_title'] == 'ПЕСНА']
print('4 ПЕСНА poems, ids:', pesna_ids)
for pid in pesna_ids:
    text = lst.load_poem_text_by_id(pid)
    print(pid, '->', text.strip().splitlines()[0])
"
```
Expected: 4 distinct first lines printed, one per poem_id — confirming all 4 are now
individually reachable, not just the one the old code happened to return.

- [ ] **Step 9: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/llm_style_transfer.py tests/test_llm_style_transfer.py
git commit -m "llm_style_transfer.py: add load_poem_text_by_id and --source-poem-id"
```

---

## Task 10: Remaining pipeline rebuild, full verification, spot-check

**Files:** none (no code changes — this task reruns the remaining pipeline scripts, none of
which reference `song_title` at all, and verifies the migration end-to-end).

- [ ] **Step 1: Rerun the remaining pipeline scripts for artifact freshness**

These don't reference `song_title` anywhere (confirmed by grep during design) and therefore
need no code changes, but should be rerun so every artifact is fresh relative to the
already-changed `pos_tagged.csv`/`author_style_profiles.csv`/`skg_enriched.gexf`:

```bash
uv run python src/tfidf_authors.py       # unaffected (reads stripped_songs.csv directly,
                                          # author-level only) — rerun for hygiene
uv run python src/add_embeddings.py      # depends on skg_enriched.gexf (Task 5) and
                                          # pos_tagged.csv's lemma set (Task 3)
uv run python src/graph_analytics.py     # reads skg_final.gexf
uv run python src/classifier.py          # reads pos_tagged.csv
uv run python src/build_morph_lookup.py  # reads pos_tagged.csv
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -q`
Expected: every test passes — this is the first point every test in the suite runs against
fully migrated, fully rebuilt real data end-to-end.

- [ ] **Step 3: Spot-check all 15 previously-colliding pairs resolve distinctly**

```bash
uv run python -c "
import csv
from collections import Counter

with open('data/stripped_songs.csv', encoding='utf-8') as f:
    songs = list(csv.DictReader(f))
title_count = Counter((r['author'], r['song_title']) for r in songs)
dupes = {k: v for k, v in title_count.items() if v > 1}
print(f'{len(dupes)} duplicated (author, title) pairs, {sum(dupes.values())} poems total')

with open('data/pos_tagged.csv', encoding='utf-8') as f:
    tagged = list(csv.DictReader(f))

for (author, title), count in sorted(dupes.items(), key=lambda kv: -kv[1]):
    ids = {r['poem_id'] for r in tagged if r['author'] == author and r['song_title'] == title}
    status = 'OK' if len(ids) == count else f'MISMATCH ({len(ids)} distinct ids, expected {count})'
    print(f'  {count:2d}  {author} / {title!r}  -> {status}')
"
```
Expected: every row prints `OK` — each duplicated title's poems carry as many distinct
`poem_id`s in `pos_tagged.csv` as there are physical poems in `stripped_songs.csv`. (A poem
with zero content tokens at all would show up as a mismatch here for an unrelated reason —
investigate any `MISMATCH` before considering this task done, rather than assuming it's
expected.)

- [ ] **Step 4: Commit any remaining regenerated artifacts**

```bash
git status --short
git add data/ models/
git commit -m "Rebuild remaining pipeline artifacts after poem_id migration"
```

---

## Task 11: `data/report.tex` check

**Files:**
- Modify: `data/report.tex` (only if Step 1 finds affected numbers)

- [ ] **Step 1: Check whether report.tex cites any numbers this migration could shift**

Per `CLAUDE.md`'s instruction to check `report.tex` after a change affecting reported
numbers: search it for any reference to per-poem statistics, stanza/line counts, or
POS-bigram fingerprints for the affected authors (Конески, Клетников, Урошевиќ, and the other
~9 authors with a duplicated title).

```bash
grep -n "Конески\|Клетников\|Урошевиќ\|top_pos_bigrams\|stanza" data/report.tex
```

- [ ] **Step 2: If any cited number changed, update it and note why**

If Step 1 finds a specific figure that changed (e.g. a reported `top_pos_bigrams` value, or
an author-level average that shifted because 43 poems' stats are no longer contaminated by
title collisions), update that number in `data/report.tex` and add a short note explaining the
poem_id fix, in the same style as the document's existing validation-finding notes. If nothing
cited is affected, state that explicitly rather than silently skipping — add a one-line note
to the relevant section (or skip entirely if `report.tex` has no per-poem-level content at
all — check before deciding).

- [ ] **Step 3: Recompile if changed**

If you edited `report.tex`:
```bash
cd data && xelatex report.tex
```
Expected: compiles cleanly (matches `CLAUDE.md`'s note that `xelatex`, not `pdflatex`, is
required here for Cyrillic support).

- [ ] **Step 4: Commit**

```bash
git add data/report.tex data/report.pdf
git commit -m "report.tex: note poem_id migration's effect on per-poem-derived numbers"
```

(Skip this commit entirely if Step 2 found nothing to change — say so instead of committing
a no-op.)

---

## Self-Review Notes

**Spec coverage:** Every section of `docs/superpowers/specs/2026-08-19-poem-id-migration-design.md`
maps to a task — schema (Task 1), `corrections.py` (Task 2), `pos_tag_corpus.py` (Task 3),
`apply_corrections.py` (Task 4), the three mechanical key-swap files (Task 5),
`candidate_selection.py`'s interface change (Task 6), and the three files from the previous
session (Tasks 7-9), followed by rebuild/verification (Task 10) and the `report.tex` check
(Task 11) the spec's migration section calls for.

**Type consistency:** `poem_id` is consistently a string everywhere it crosses a CSV
boundary (`csv.DictReader` always yields strings) and cast to `int` only where numeric
comparison/sorting is needed (`corrections.key_for`/`lookup`, `find_source_poem`'s sort,
`poem_tfidf.build_and_save`'s output ordering) — `corpus_token_stream` and
`load_poem_text_by_id` both normalize their input via `str(poem_id)` so callers can pass
either an int or a string without it mattering.
