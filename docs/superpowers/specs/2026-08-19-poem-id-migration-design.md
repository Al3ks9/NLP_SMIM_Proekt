# Poem-level id: `(author, song_title)` is not a unique poem key

**Date:** 2026-08-19
**Status:** design, awaiting implementation plan
**Scope:** add a stable `poem_id` to `stripped_songs.csv` and thread it through every
pipeline stage and every existing script that currently groups tokens by
`(author, song_title)`. Then rebuild all artifacts once.

## Background

Surfaced during code review of the new LLM style-transfer pipeline
(`src/llm_style_transfer.py`, `src/poem_tfidf.py`, `src/exemplar_selection.py`):
`(author, song_title)` is not a unique key into `data/stripped_songs.csv`. 15 pairs are
shared by multiple distinct physical poems — e.g. Блаже Конески has 4 poems titled
`ПЕСНА`, Ефтим Клетников has 20 poems titled `***`. `pos_tagged.csv` carries no
finer-grained id, so this ambiguity is inherited by every downstream file keyed the same
way: `pos_tag_corpus.py` (the origin), `apply_corrections.py`/`corrections.py` (the
`scope=poem` hand-correction key), `build_skg.py`, `enrich_skg.py`, `style_profiler.py`,
`candidate_selection.py`'s `corpus_token_stream`/`find_source_poem`, and this session's
three new files.

Two distinct failure modes, both real and both verified against the corpus:

- **Silent wrong answer.** A lookup that expects one poem back (`load_poem_text`,
  `corpus_token_stream` as used by `find_source_poem`) picks one arbitrary poem among
  the duplicates — currently "whichever the dict comprehension wrote last" — with no
  signal to the caller that a choice was made.
- **Silent content blending.** An aggregation keyed by `(author, song_title)`
  (`poem_tfidf.py`'s per-poem TF-IDF, `style_profiler.py`'s per-poem stanza/line
  counts, `build_skg.py`'s co-occurrence counts) merges the *distinct* poems sharing a
  title into one artificial document/count before computing anything.

`candidate_selection.py`'s `find_source_poem()` already has an independent instance of
the first failure mode today, unrelated to this session's new files: it collapses
same-titled poems into one alignment candidate (iterating the *set* of an author's
titles, not their poems), so aligning free text against an ambiguous-titled author can
silently score against the wrong physical poem.

## Measured constraints

From `data/stripped_songs.csv`, 2711 rows, 1169 poems by the 24 qualifying (≥5 poems)
authors that `pos_tag_corpus.py` tags:

- 15 `(author, song_title)` pairs are duplicated, covering 43 of 1169 poems (3.7%).
- Worst cases: `('Ефтим Клетников', '***')` — 20 poems; `('Блаже Конески', 'ПЕСНА')` —
  4 poems; `('Влада Урошевиќ', 'Летен дожд')` — 3 poems.
- `songs.csv` (raw input to `strip_lines.py`) still exists in the repo root (1.4 MB), so
  a full from-scratch regeneration of `stripped_songs.csv` remains possible.
- `strip_lines.py` does not currently parse CSV at all — it copies `songs.csv`
  line-by-line, `.strip()`ing each physical line. It cannot insert a column as written.
- `data/pos_corrections.csv` currently has 87 rows, **all corpus-wide scope** — zero
  use `scope=poem` or `scope=row` today. Retroactively assigning `poem_id` to existing
  corrections is therefore a no-op in practice, but the migration script must still
  handle the general case correctly (see Migration step 1) since it's the mechanism any
  future poem/row-scoped correction relies on.

## Design decisions

**Plain sequential integer, not a UUID.** This is a static, single-writer,
version-controlled corpus of 1169 poems, not a system with independent concurrent
writers needing global uniqueness. A UUID adds 36 bytes/row for no benefit here and is
unreadable by eye. A UUID's one real advantage — stability under row reordering —
doesn't apply: nothing in this pipeline reorders or deletes from the middle of
`stripped_songs.csv`, only appends. `poem_id` = row position (0-indexed) at the point
`stripped_songs.csv` is written.

**Every poem gets an id, not just the 43 ambiguous ones.** A mixed schema forces every
consumer to branch ("if ambiguous, use id; else use title") — more code, a second place
to get it wrong, and no simpler than always having the column.

**`poem_id` is assigned once, at the earliest point poems exist as distinct rows** —
`stripped_songs.csv` — and carried through as a plain passenger column everywhere else.
Nothing downstream re-derives it.

## Schema changes

| File | Change |
|---|---|
| `data/stripped_songs.csv` | new first column `poem_id` (int, row order) |
| `data/pos_tagged.csv` | new first column `poem_id`, copied from the source poem's row |
| `data/pos_flags.csv` | new column `poem_id`, alongside existing `author`/`song_title` |
| `data/pos_corrections.csv` | new column `poem_id`, alongside existing `author`/`song_title` |
| `data/poem_tfidf_results.csv` | new first column `poem_id`; `author`/`song_title` kept for readability |

`author`/`song_title` are kept everywhere as human-readable columns — they're what a
person reviewing `pos_flags.csv` or grepping a CSV actually reads. Only the *keys code
computes on* change to `poem_id`.

## Component changes

**`src/strip_lines.py`** — rewritten to parse `songs.csv` with `csv.DictReader` and
write `stripped_songs.csv` with `csv.DictWriter`, adding `poem_id` by row order. (Its
current naive line-copy approach can't insert a column; this also incidentally fixes it
reading raw text instead of parsed CSV.) Not part of `CLAUDE.md`'s regular pipeline
order — this only matters if `stripped_songs.csv` is ever regenerated from scratch — but
kept consistent so a future regeneration reproduces the same scheme.

A one-time migration script adds `poem_id` to the *current* `stripped_songs.csv`
directly (read + rewrite via `csv`), since regenerating via `strip_lines.py` isn't
necessary for this migration and risks reintroducing whitespace differences.

**`src/pos_tag_corpus.py`** — the `poems` list already carries each `stripped_songs.csv`
row; `poem_id` rides along through the existing `tokens` tuples into both `results`
(→ `pos_tagged.csv`) and `flags` (→ `pos_flags.csv`).

**`src/corrections.py`** — `FIELDS` gains `poem_id`. `key_for()`: `scope='poem'` becomes
`base + (row['poem_id'],)`; `scope='row'` becomes `base + (row['poem_id'], row['context'])`
(replacing `(author, song_title)` in both). `lookup()`/`apply_to()` take `poem_id`
instead of `(author, title)`.

**`src/apply_corrections.py`** — its patch loop over `pos_tagged.csv` rows passes
`row['poem_id']` to `C.apply_to()` instead of `(row['author'], row['song_title'])`.

**`src/build_skg.py`, `src/enrich_skg.py`, `src/style_profiler.py`** — mechanical:
replace the `(author, song_title)` tuple key with `poem_id` (int) wherever poems are
grouped. No interface changes outside these files — the key change is internal to each
script's own aggregation.

**`src/candidate_selection.py`** — the one real interface change:
- `corpus_token_stream(author, song_title, pos_rows)` → `corpus_token_stream(poem_id, pos_rows)`.
- `find_source_poem(text, author, pos_rows)` iterates the author's distinct `poem_id`s
  (not distinct titles) and returns `(poem_id, song_title, coverage)` instead of
  `(song_title, coverage)`.
- `slot_contexts()`/`build_slots()` update their one call site each to match.

This is `candidate_selection.py`'s tested public API (`tests/test_candidate_selection.py`),
so those call sites and their tests are part of this change, not incidental fallout.

**`src/poem_tfidf.py`** — `build_documents()` groups by `poem_id`. `top_content_words()`
becomes `top_content_words(poem_id, n=10)`; `poem_tfidf_results.csv` gains a `poem_id`
column.

**`src/exemplar_selection.py`** — `qualifying_lines()` already iterates
`stripped_songs.csv` rows one at a time (one physical poem each); it passes that row's
`poem_id` to `corpus_token_stream()` instead of its title. No public signature change —
`select_exemplars(target_author, ...)` still takes an author name, since exemplar
selection was never about one specific poem.

**`src/llm_style_transfer.py`** — `load_poem_text(author, song_title)` keeps today's
"raise on ambiguous title" behavior (still useful — most titles remain unambiguous, and
a human referring to a poem by title should get a clear error, not silent wrong output).
New: `load_poem_text_by_id(poem_id)` for unambiguous lookup, and a `--source-poem-id`
CLI flag that bypasses title resolution entirely — the CLI can now actually reach all 4
of Конески's `ПЕСНА` poems, not merely refuse them. `assemble_prompt()`/`run_transfer()`
resolve `poem_id` once (from either path) and pass it to `poem_tfidf.top_content_words()`
instead of `(source_author, source_title)`.

## Migration / rebuild

1. Add `poem_id` to the current `stripped_songs.csv` (one-time script), and add a
   `poem_id` column to `data/pos_corrections.csv`'s existing 87 rows — trivial today
   (all corpus-wide scope, so `poem_id` is simply blank/unused for every existing row),
   but written as a real `(author, song_title) -> poem_id` lookup, not a hardcoded
   no-op, so it does the right thing if corrections are ever added before this step is
   next run. If a future run finds a `scope=poem`/`row` row whose `(author, song_title)`
   is itself ambiguous, the script refuses and reports it rather than guessing — the
   correction's original author would need to disambiguate by hand (there is no
   automatic way to recover which of several same-titled poems a hand-written
   correction was meant for).
2. Update `strip_lines.py` to match, for future regenerations.
3. Land all the code changes above.
4. Rebuild from scratch: `pos_tag_corpus.py` (applies `pos_corrections.csv` internally,
   now `poem_id`-keyed — corpus-wide corrections are unaffected since global scope never
   used `author`/`title`; `apply_corrections.py` itself is a separate incremental-patch
   path for the flag-review workflow, not a required step in a from-scratch rebuild) →
   steps 2-9 of `CLAUDE.md`'s pipeline order (`tfidf_authors.py` through
   `build_morph_lookup.py`) → `poem_tfidf.py` (this session's addition, not yet in the
   numbered list).
5. Full test suite (existing + new).
6. Spot-check: confirm each of the 15 previously-colliding pairs now resolves to
   distinct, correct poems (e.g. all 4 of Конески's `ПЕСНА` poems individually
   reachable via `--source-poem-id`).
7. Check `data/report.tex` per `CLAUDE.md`'s instruction — this can shift
   poem-level-derived numbers for the 43 affected poems (`style_profiler.py`'s
   per-poem stanza/line stats feed corpus-relative author aggregates).

## Testing

- New tests for `corrections.py`'s `poem_id`-based `key_for`/`lookup` (replacing the
  `(author, title)` cases, plus a case proving two same-titled poems no longer collide).
- `tests/test_candidate_selection.py` updates for `corpus_token_stream`'s new signature,
  plus a new case: aligning text against an author with a duplicated title resolves to
  the correct one of the two poems, not their merged token stream.
- `tests/test_poem_tfidf.py`: a case with two same-titled documents confirming they stay
  separate TF-IDF documents (this already passes today, since the existing tests use
  literal dict keys rather than the real corpus — this becomes a regression test using
  duplicate-titled synthetic input).
- `tests/test_llm_style_transfer.py`: replace `test_load_poem_text_raises_on_a_real_duplicated_title`
  with a pair of tests — title lookup still raises, `load_poem_text_by_id` succeeds and
  returns the correct one of the 4 poems.
- `tests/test_exemplar_selection.py`: existing tests continue to pass unchanged (no
  public signature change); add nothing new here since the fix is internal.

## Out of scope

- Reordering or deleting rows from `stripped_songs.csv` (would break the row-order id
  scheme — not something this pipeline does today, and not addressed defensively here).
- A `poem_id` -> UUID migration path, or any external system needing globally unique
  ids across independent writers.
- Deduplicating actual duplicate *content* (e.g. `'Бура'` vs `'БУРА'`, same author,
  differing only in title case, spotted during the LLM pipeline's manual smoke test) —
  that's a data-cleaning question distinct from the identity problem this spec solves,
  and those two rows already get distinct `poem_id`s under this scheme regardless.
