# Feats-Keyed Morph Lookup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-key `models/morph_lookup.json` on the `feats` column instead of 3-character suffixes, move inference-time tagging from spaCy to classla, and share tag normalization between the corpus build and the inference path — then rebuild all artifacts once.

**Architecture:** A new `src/tagging.py` owns the classla pipeline plus the two corrections `pos_tag_corpus.py` currently applies inline (`is_content()` relativizer filtering, `gerund_lemma()` re-lemmatization), so the corpus build and inference cannot drift. `build_morph_lookup.py` emits two indices — per-author and pooled-across-authors — both keyed on the feats string. `surface_form()` resolves through five tiers, with a new `relaxed` tier that drops features in a corpus-measured order.

**Tech Stack:** Python 3.10+, `uv`, classla (`mk` models), pytest. spaCy is removed by this work.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-13-feats-morph-lookup-design.md`. Read it before Task 1.
- All scripts resolve paths via `pathlib.Path(__file__)` and must keep working from any working directory.
- Run everything through `uv run` — e.g. `uv run python src/<script>.py`, `uv run pytest`.
- `pytest.ini_options` already sets `pythonpath = ["src"]`, so tests import modules bare (`import candidate_selection as cs`).
- classla feats strings are pipe-separated `Key=Value` pairs in **alphabetical key order** (`Case=Nom|Definite=Ind|Gender=Fem|Number=Sing`). Any feats string this code constructs must use the same ordering or keys will not match.
- `data/pos_corrections.csv` is never regenerated. Nothing in this plan writes to it.
- Existing morph tier names `exact`, `cross_author`, `nearest_attested`, `unresolved` keep their meaning. `relaxed` is added between `cross_author` and `nearest_attested`.
- Do not re-review `data/pos_flags.csv`, and do not tune `MIN_POEMS`, `SEM_THRESHOLD`, or the composite scoring weights.

## Correction to the spec, found while writing this plan

The spec's relaxation tier says "drop features cumulatively ... retry tiers 1 then 2 at each width". Taken literally that does not work: the stored keys are always **full** feats bundles from the corpus, so a query with `Polarity` dropped (`Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Tense=Pres`) will never equal a stored key, which always contains `Polarity=Pos`. Exact-match relaxation would resolve nothing.

Relaxation is therefore implemented as **subset matching**: a relaxed bundle is a *constraint set*, and a stored key matches when its parsed features are a superset of the constraint. Per-cell dicts hold 1.28 entries on average, so the scan is trivial. Tiers 1 and 2 remain direct dict lookups. This is a correction to the mechanism, not to the design's intent or its drop order.

## File Structure

| File | Responsibility |
|---|---|
| `src/tagging.py` | **new** — classla pipeline; `Token`; `is_content()`; `gerund_lemma()`; correction application; line-oriented `tag_lines()` for inference |
| `src/morph.py` | **new** — feats parsing, `relax_feats()`, `surface_form()`. Kept out of `candidate_selection.py`, which is already 700+ lines |
| `src/build_morph_lookup.py` | modified — key on feats; emit `by_author` + `pooled` |
| `src/pos_tag_corpus.py` | modified — imports its normalization from `tagging.py` |
| `src/candidate_selection.py` | modified — re-export `surface_form` from `morph.py`; `_retag()` uses classla; feats flow through `slot_contexts`/`build_slot` |
| `src/style_transfer.py` | modified — classla via `tagging.py`; feats-based `get_surface_morph()`; offset splicing |
| `tests/test_tagging.py` | **new** |
| `tests/test_morph.py` | **new** |
| `tests/test_candidate_selection.py` | modified — suffix-based `surface_form` tests rewritten |

---

### Task 1: Extract `src/tagging.py`

Pure extraction — `is_content()` and `gerund_lemma()` move verbatim, so `pos_tag_corpus.py` behaviour must not change.

**Files:**
- Create: `src/tagging.py`
- Modify: `src/pos_tag_corpus.py:26-77` (remove moved definitions, import instead)
- Test: `tests/test_tagging.py`

**Interfaces:**
- Consumes: `corrections` module (`C.load()`, `C.apply_to(table, word, pos, lemma, xpos, feats, author, title, context)` → `(pos, lemma, xpos, feats)` or `None`)
- Produces: `KEEP_POS`, `RELATIVIZER_XPOS`, `is_content(pos, xpos) -> bool`, `gerund_lemma(word, verb_lemmas) -> (str, bool)`, `Token`, `pipeline()`, `tag_lines(text, verb_lemmas) -> list[Token]`, `load_verb_lemmas(pos_rows) -> Counter`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tagging.py
"""Tag normalization shared by the corpus build and the inference path."""

from collections import Counter

import pytest

import tagging


def test_is_content_keeps_a_real_adverb():
    assert tagging.is_content('ADV', 'Rgp') is True


def test_is_content_drops_a_bare_rg_relativizer():
    # што/кога/како carry bare 'Rg' — no degree slot — where adverbs get Rgp.
    assert tagging.is_content('ADV', 'Rg') is False


def test_is_content_drops_a_function_word():
    assert tagging.is_content('ADP', 'Sps') is False


def test_gerund_lemma_rebuilds_an_a_verb():
    assert tagging.gerund_lemma('барајќи', Counter({'бара': 40})) == ('бара', True)


def test_gerund_lemma_prefers_the_frequent_candidate():
    # бране occurs once off a mistagged token; брани is the real lemma.
    verb_lemmas = Counter({'брани': 20, 'бране': 1})
    assert tagging.gerund_lemma('бранејќи', verb_lemmas) == ('брани', True)


def test_gerund_lemma_falls_back_to_the_i_verb_when_unattested():
    lemma, known = tagging.gerund_lemma('велејќи', Counter())
    assert (lemma, known) == ('вели', False)


def test_load_verb_lemmas_counts_only_verbs():
    rows = [
        {'pos': 'VERB', 'lemma': 'бара'},
        {'pos': 'VERB', 'lemma': 'бара'},
        {'pos': 'NOUN', 'lemma': 'град'},
    ]
    assert tagging.load_verb_lemmas(rows) == Counter({'бара': 2})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tagging.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tagging'`

- [ ] **Step 3: Create `src/tagging.py`**

Move `KEEP_POS`, `RELATIVIZER_XPOS`, `is_content`, and `gerund_lemma` out of `pos_tag_corpus.py` **unchanged** — including their docstrings and comments, which explain why the relativizer and gerund rules exist.

```python
"""Tagging and tag normalization, shared by the corpus build and inference.

pos_tag_corpus.py does not use raw classla output: it drops bare-Rg
relativizers and re-lemmatizes -јќи verbal adverbs, and pos_tagged.csv
reflects both. Inference that called classla directly would produce a source
'што' tagged ADV/Degree=Pos and a 'барајќи' lemmatized to itself — neither of
which exists in morph_lookup.json, sending every such slot to 'unresolved'.
Defining the normalization once makes that divergence impossible.
"""

from collections import Counter
from dataclasses import dataclass

KEEP_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN'}

# classla tags што/кога/како/колку/каде as ADV, but they are relativizers, not
# manner adverbs ("Зборовите што ти ги дадов"). They carry the bare xpos 'Rg' —
# no degree slot — where real adverbs get Rgp/Rgc/Rgs, so xpos separates them
# cleanly. Left in, these five forms are the most frequent "content" words in
# the corpus and swamp any POS ratio built on it.
RELATIVIZER_XPOS = 'Rg'


def is_content(pos, xpos):
    return pos in KEEP_POS and xpos != RELATIVIZER_XPOS


def gerund_lemma(word, verb_lemmas):
    """Lemmatise a -јќи verbal adverb (xpos Rv) back to its base verb.

    classla tags these ADV and leaves them unlemmatised, so барајќи lemmatises
    to барајќи rather than бара. Stripping -јќи gives the stem, but the ending
    that goes back on is ambiguous: -ајќи always rebuilds an -а verb (барајќи ->
    бара), while -ејќи comes from both и-verbs (велејќи -> вели) and e-verbs
    (знаејќи -> знае). Rather than guess, weigh each candidate by how often
    classla lemmatised a finite verb to it elsewhere in the corpus.

    Frequency rather than first-match matters: бране and гале each occur once as
    a verb lemma, off a single mistagged token, and would beat the correct брани
    (20) and гали (15) on a bare membership test. и-verbs are the larger class,
    so they win when the corpus has nothing to say.
    """
    stem = word[:-len('јќи')]
    candidates = [stem]
    if stem.endswith('е'):
        candidates += [stem[:-1] + 'и', stem + 'е']
    attested = [c for c in candidates if verb_lemmas[c]]
    if attested:
        return max(attested, key=lambda c: verb_lemmas[c]), True
    return (stem[:-1] + 'и' if stem.endswith('е') else stem), False


def load_verb_lemmas(pos_rows):
    """VERB lemma frequencies, the evidence gerund_lemma() weighs candidates on.

    Built from pos_tagged.csv at inference time so a -јќи form resolves to the
    same base verb it did during the corpus build.
    """
    counts = Counter()
    for r in pos_rows:
        if r['pos'] == 'VERB' and r['lemma']:
            counts[r['lemma'].lower()] += 1
    return counts


@dataclass(frozen=True)
class Token:
    """One tagged token. Offsets are relative to the line it came from."""
    text: str
    lemma: str
    pos: str
    xpos: str
    feats: str
    line: int
    start_char: int
    end_char: int


_PIPELINE = None


def pipeline():
    """The classla mk pipeline, loaded once per process."""
    global _PIPELINE
    if _PIPELINE is None:
        import classla
        try:
            _PIPELINE = classla.Pipeline(
                'mk', processors='tokenize,pos,lemma', download_method=None)
        except Exception as e:
            raise SystemExit(
                f"Could not load the classla 'mk' models ({e}).\n"
                "Run this once to fetch them:  "
                "uv run python -c \"import classla; classla.download('mk')\"")
    return _PIPELINE


def tag_lines(text, verb_lemmas):
    """Tag text line by line, returning normalized content tokens.

    Offsets are line-relative because every consumer rebuilds output one line
    at a time. Normalization matches pos_tag_corpus.py exactly: relativizers
    dropped, -јќи forms re-lemmatized, words and lemmas lowercased.
    """
    nlp = pipeline()
    out = []
    for li, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        doc = nlp(line)
        for sent in doc.sentences:
            for w in sent.words:
                pos, xpos = w.upos or '', w.xpos or ''
                if not is_content(pos, xpos):
                    continue
                lemma = (w.lemma or '').lower()
                if xpos == 'Rv' and w.text.lower().endswith('јќи'):
                    lemma, _ = gerund_lemma(w.text.lower(), verb_lemmas)
                start, end = _span(w, line)
                out.append(Token(
                    text=w.text.lower(), lemma=lemma, pos=pos, xpos=xpos,
                    feats=w.feats or '', line=li,
                    start_char=start, end_char=end,
                ))
    return out


def _span(word, line):
    """Character span of a classla word within its line.

    classla exposes offsets as a 'start_char|end_char' string in word.misc on
    most builds; fall back to locating the surface form when it is absent.
    """
    misc = getattr(word, 'misc', None) or ''
    start = end = None
    for part in misc.split('|'):
        if part.startswith('start_char='):
            start = int(part.split('=', 1)[1])
        elif part.startswith('end_char='):
            end = int(part.split('=', 1)[1])
    if start is None or end is None:
        start = line.find(word.text)
        end = start + len(word.text) if start >= 0 else 0
        start = max(start, 0)
    return start, end
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tagging.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Point `pos_tag_corpus.py` at the shared module**

Delete lines 26–77's definitions of `KEEP_POS`, `RELATIVIZER_XPOS`, `is_content`, and `gerund_lemma` (leave `MIN_POEMS`, `XPOS_CATEGORY`, `CYRILLIC`, `LATIN_GREEK`, `MIN_OCC`, `MAX_SHARE` in place — they belong to the audit, not the tagging). Add the import beside `import corrections as C`:

```python
import corrections as C
from tagging import KEEP_POS, RELATIVIZER_XPOS, is_content, gerund_lemma
```

Replace the inline pipeline construction at lines 136–141 with:

```python
from tagging import pipeline
nlp = pipeline()
```

- [ ] **Step 6: Verify the corpus build is unchanged**

This is an extraction, so tagging a fixed sample must give byte-identical output. Run:

```bash
uv run python -c "
import tagging, pos_tag_corpus
" 2>&1 | head -5
```

Expected: no `ImportError`, no `NameError`. (This runs the whole tagging script; let it finish or interrupt after it prints its token count — the full rebuild happens in Task 8.)

- [ ] **Step 7: Commit**

```bash
git add src/tagging.py src/pos_tag_corpus.py tests/test_tagging.py
git commit -m "Extract tagging normalization into src/tagging.py

is_content() and gerund_lemma() move verbatim so the corpus build and the
inference path cannot drift apart."
```

---

### Task 2: `src/morph.py` — feats parsing and relaxation

**Files:**
- Create: `src/morph.py`
- Test: `tests/test_morph.py`

**Interfaces:**
- Produces: `parse_feats(str) -> dict`, `render_feats(dict) -> str`, `relax_feats(feats, pos) -> Iterator[str]`, `DROP_ORDER`, `NEVER_DROP`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_morph.py
"""Feats parsing and the relaxation order (spec §Relaxation order)."""

import morph

NOUN_FEM_SG_DEF = 'Case=Nom|Definite=Def|Gender=Fem|Number=Sing'
VERB_PRES_3SG = 'Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres'
ADJ_FEM_SG = 'Definite=Ind|Degree=Pos|Gender=Fem|Number=Sing'


def test_parse_feats_round_trips():
    assert morph.render_feats(morph.parse_feats(NOUN_FEM_SG_DEF)) == NOUN_FEM_SG_DEF


def test_render_feats_is_alphabetical():
    # classla writes keys alphabetically; anything we build must match or keys miss.
    assert morph.render_feats({'Number': 'Sing', 'Case': 'Nom'}) == 'Case=Nom|Number=Sing'


def test_parse_feats_handles_the_empty_string():
    assert morph.parse_feats('') == {}


def test_relaxation_drops_polarity_first_on_a_verb():
    first = next(morph.relax_feats(VERB_PRES_3SG, 'VERB'))
    assert 'Polarity' not in first
    assert 'Person=3' in first and 'Tense=Pres' in first


def test_relaxation_is_cumulative():
    steps = list(morph.relax_feats(VERB_PRES_3SG, 'VERB'))
    widths = [len(morph.parse_feats(s)) for s in steps]
    assert widths == sorted(widths, reverse=True)
    assert len(set(widths)) == len(widths), 'each step must drop exactly one feature'


def test_relaxation_drops_gender_for_a_noun():
    steps = list(morph.relax_feats(NOUN_FEM_SG_DEF, 'NOUN'))
    assert any('Gender' not in s for s in steps)


def test_relaxation_never_drops_gender_for_an_adjective():
    # Gender is agreement on ADJ — dropping it produces 'тивка гроб'.
    for step in morph.relax_feats(ADJ_FEM_SG, 'ADJ'):
        assert 'Gender=Fem' in step


def test_relaxation_never_drops_number_or_definite():
    for step in morph.relax_feats(NOUN_FEM_SG_DEF, 'NOUN'):
        assert 'Number=Sing' in step and 'Definite=Def' in step


def test_relaxation_never_empties_the_bundle():
    # '' is reserved for genuinely featless tokens; matching it from a relaxed
    # lookup would return an arbitrary form while claiming a morphological match.
    # Degree is an ADV's only feature 76.9% of the time and is droppable.
    assert list(morph.relax_feats('Degree=Pos', 'ADV')) == []


def test_relaxation_of_an_empty_bundle_yields_nothing():
    assert list(morph.relax_feats('', 'NOUN')) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_morph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'morph'`

- [ ] **Step 3: Create `src/morph.py`**

```python
"""Morphological feats: parsing, relaxation, and surface realisation.

Kept separate from candidate_selection.py, which is already large and owns
candidate *ranking* rather than candidate *inflection*.
"""

from collections import Counter

# Drop order, most-droppable first, measured over pos_tagged.csv. Polarity is
# 100% present on verbs with a single value and so carries zero information;
# Case is 99.8% Nom; Mood 96.8% Ind; Degree 96.6%/95.1% Pos on ADJ/ADV.
#
# Gender is droppable for NOUN only. Macedonian nouns carry inherent lexical
# gender — град is masculine in every form — so once the lemma is fixed Gender
# is redundant. On ADJ and VERB it is agreement, and dropping it yields the
# 'тивка гроб' class of error.
DROP_ORDER = ['Polarity', 'Case', 'Mood', 'Degree', 'Gender',
              'Aspect', 'Person', 'Tense']

GENDER_DROPPABLE_POS = {'NOUN'}

# Number/Definite are the surface distinction itself. VerbForm, Poss, NumType,
# PronType and AdvType each have a single value but appear on only a fraction
# of their POS — their *presence* is the signal, so single-valuedness must not
# be mistaken for uninformativeness.
NEVER_DROP = {'Number', 'Definite', 'VerbForm', 'Poss',
              'NumType', 'PronType', 'AdvType'}


def parse_feats(feats):
    """'Case=Nom|Number=Sing' -> {'Case': 'Nom', 'Number': 'Sing'}."""
    if not feats:
        return {}
    return dict(kv.split('=', 1) for kv in feats.split('|') if '=' in kv)


def render_feats(d):
    """Inverse of parse_feats. Alphabetical, matching classla's own ordering."""
    return '|'.join(f'{k}={v}' for k, v in sorted(d.items()))


def relax_feats(feats, pos):
    """Yield progressively weaker feats *constraints*, widest match last.

    Each step drops one more feature. Yields constraint sets rather than keys:
    stored keys are always full bundles, so a relaxed bundle is matched by
    superset containment (see surface_form), never by equality.

    Never empties the bundle — '' is reserved for genuinely featless tokens.
    """
    d = parse_feats(feats)
    for feature in DROP_ORDER:
        if feature in NEVER_DROP or feature not in d:
            continue
        if feature == 'Gender' and pos not in GENDER_DROPPABLE_POS:
            continue
        if len(d) <= 1:
            return
        del d[feature]
        yield render_feats(d)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_morph.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add src/morph.py tests/test_morph.py
git commit -m "Add feats parsing and corpus-measured relaxation order"
```

---

### Task 3: `surface_form()` on feats, with subset matching

**Files:**
- Modify: `src/morph.py` (append)
- Test: `tests/test_morph.py` (append)

**Interfaces:**
- Consumes: `parse_feats`, `relax_feats` from Task 2
- Produces: `surface_form(target_author, candidate_lemma, pos, source_feats, morph_lookup) -> (str, str)`. `morph_lookup` is `{'by_author': {author: {lemma: {pos: {feats: surface}}}}, 'pooled': {lemma: {pos: {feats: surface}}}}`. Tier is one of `exact`, `cross_author`, `relaxed`, `nearest_attested`, `unresolved`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_morph.py
FEM_SG_DEF = 'Case=Nom|Definite=Def|Gender=Fem|Number=Sing'
NEUT_PL_DEF = 'Case=Nom|Definite=Def|Gender=Neut|Number=Plur'

LOOKUP = {
    'by_author': {
        'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}},
        'B': {'жена': {'NOUN': {NEUT_PL_DEF: 'женине'}}},
    },
    'pooled': {'жена': {'NOUN': {FEM_SG_DEF: 'жената', NEUT_PL_DEF: 'женине'}}},
}


def test_exact_tier_hits_the_target_author():
    assert morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, LOOKUP) == ('жената', 'exact')


def test_the_bug_this_fixes_ata_no_longer_conflates_two_readings():
    # Both surface forms end in 'ата'/'ине' but differ in gender+number. Under
    # the old 3-char suffix key these collided; NOUN+ата covered 2,685 tokens
    # across 8 feats bundles.
    fem, _ = morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, LOOKUP)
    neut, _ = morph.surface_form('A', 'жена', 'NOUN', NEUT_PL_DEF, LOOKUP)
    assert fem != neut


def test_cross_author_tier_borrows_from_the_pooled_index():
    form, tier = morph.surface_form('A', 'жена', 'NOUN', NEUT_PL_DEF, LOOKUP)
    assert (form, tier) == ('женине', 'cross_author')


def test_relaxed_tier_matches_a_stored_superset_key():
    # Query carries Polarity; the stored key does not differ on anything else.
    lookup = {
        'by_author': {'A': {'бара': {'VERB': {
            'Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres': 'бара'}}}},
        'pooled': {},
    }
    # Same bundle but Mood=Sub, which no stored key has — relaxation drops Mood.
    query = 'Aspect=Prog|Mood=Sub|Number=Sing|Person=3|Polarity=Pos|Tense=Pres'
    form, tier = morph.surface_form('A', 'бара', 'VERB', query, lookup)
    assert (form, tier) == ('бара', 'relaxed')


def test_relaxed_never_reports_exact():
    lookup = {
        'by_author': {'A': {'бара': {'VERB': {
            'Aspect=Prog|Mood=Ind|Number=Sing|Person=3|Polarity=Pos|Tense=Pres': 'бара'}}}},
        'pooled': {},
    }
    query = 'Aspect=Prog|Mood=Sub|Number=Sing|Person=3|Polarity=Pos|Tense=Pres'
    assert morph.surface_form('A', 'бара', 'VERB', query, lookup)[1] != 'exact'


def test_nearest_attested_when_nothing_matches():
    lookup = {'by_author': {'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}}}, 'pooled': {}}
    other = 'Case=Nom|Definite=Ind|Gender=Masc|Number=Plur'
    form, tier = morph.surface_form('A', 'жена', 'NOUN', other, lookup)
    assert (form, tier) == ('жената', 'nearest_attested')


def test_unresolved_returns_the_bare_lemma():
    empty = {'by_author': {}, 'pooled': {}}
    assert morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, empty) == ('жена', 'unresolved')


def test_featless_token_degrades_to_nearest_attested_not_an_exception():
    lookup = {'by_author': {'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}}}, 'pooled': {}}
    assert morph.surface_form('A', 'жена', 'NOUN', '', lookup) == ('жената', 'nearest_attested')


def test_target_author_is_preferred_over_the_pooled_index():
    lookup = {
        'by_author': {'A': {'жена': {'NOUN': {FEM_SG_DEF: 'жената'}}}},
        'pooled': {'жена': {'NOUN': {FEM_SG_DEF: 'женава'}}},
    }
    assert morph.surface_form('A', 'жена', 'NOUN', FEM_SG_DEF, lookup)[0] == 'жената'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_morph.py -v -k surface or tier or bug or featless`
Expected: FAIL — `AttributeError: module 'morph' has no attribute 'surface_form'`

- [ ] **Step 3: Append `surface_form()` to `src/morph.py`**

```python
def _match(forms, constraint):
    """Best stored form whose feats are a superset of `constraint`.

    Stored keys are always full bundles from the corpus, so a relaxed
    constraint is matched by containment rather than equality. Cells hold 1.28
    entries on average, so the scan is cheap. Ties break on frequency then
    alphabetically, so the result is deterministic across runs.
    """
    wanted = parse_feats(constraint).items()
    hits = [surface for key, surface in forms.items()
            if parse_feats(key).items() >= wanted]
    if not hits:
        return None
    counts = Counter(hits)
    return max(sorted(set(hits)), key=lambda f: counts[f])


def surface_form(target_author, candidate_lemma, pos, source_feats, morph_lookup):
    """Realise a candidate lemma in the source token's grammatical form.

    Returns (surface, tier). The tier is required output, not optional logging:
    it says how much of the inflection the model still has to do, and it is what
    a batch run is audited on.

      1. by_author[target][lemma][pos][feats]   -> 'exact'
      2. pooled[lemma][pos][feats]              -> 'cross_author'
      3. relaxed constraint, target then pooled -> 'relaxed'
      4. any attested form for [target][lemma][pos] -> 'nearest_attested'
      5. the lemma unchanged                    -> 'unresolved'

    Tier 2 borrows another author's spelling of an inflection. That is the
    right trade: a wrong-gender word is a visible grammatical error, a
    correctly-inflected word from a neighbouring idiolect is not. Pooling makes
    118% more forms retrievable than the per-author index alone.
    """
    target_forms = (morph_lookup.get('by_author', {})
                    .get(target_author, {}).get(candidate_lemma, {}).get(pos, {}))
    pooled_forms = (morph_lookup.get('pooled', {})
                    .get(candidate_lemma, {}).get(pos, {}))

    if source_feats and source_feats in target_forms:
        return target_forms[source_feats], 'exact'
    if source_feats and source_feats in pooled_forms:
        return pooled_forms[source_feats], 'cross_author'

    for constraint in relax_feats(source_feats, pos):
        hit = _match(target_forms, constraint) or _match(pooled_forms, constraint)
        if hit:
            return hit, 'relaxed'

    if target_forms:
        counts = Counter(target_forms.values())
        return max(sorted(set(target_forms.values())), key=lambda f: counts[f]), 'nearest_attested'

    return candidate_lemma, 'unresolved'
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_morph.py -v`
Expected: PASS (19 tests)

- [ ] **Step 5: Commit**

```bash
git add src/morph.py tests/test_morph.py
git commit -m "Add feats-keyed surface_form with subset-matched relaxation

Relaxation matches stored keys by superset containment: stored keys are
always full bundles, so a relaxed query never equals one."
```

---

### Task 4: Re-key `build_morph_lookup.py` on feats

**Files:**
- Modify: `src/build_morph_lookup.py` (whole file)
- Test: `tests/test_morph.py` (append)

**Interfaces:**
- Produces: `models/morph_lookup.json` as `{'by_author': ..., 'pooled': ...}`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_morph.py
import json
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / 'models'


def test_morph_lookup_has_both_indices():
    with open(MODELS / 'morph_lookup.json', encoding='utf-8') as f:
        lookup = json.load(f)
    assert set(lookup) == {'by_author', 'pooled'}


def test_morph_lookup_keys_are_feats_not_suffixes():
    with open(MODELS / 'morph_lookup.json', encoding='utf-8') as f:
        lookup = json.load(f)
    keys = [k
            for lemmas in lookup['by_author'].values()
            for pos_map in lemmas.values()
            for forms in pos_map.values()
            for k in forms]
    assert keys, 'lookup is empty'
    # A feats key contains '='; a 3-char suffix never does.
    assert sum('=' in k for k in keys) / len(keys) > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_morph.py -v -k morph_lookup`
Expected: FAIL — `KeyError` / assertion failure, since the committed `morph_lookup.json` is still suffix-keyed

- [ ] **Step 3: Rewrite `src/build_morph_lookup.py`**

Replace the file body from the docstring through the final print:

```python
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
```

- [ ] **Step 4: Build the lookup and verify**

Run: `uv run python src/build_morph_lookup.py`
Expected: prints roughly `45,333 per-author entries` and a larger pooled count over 24 authors. This reads the *current* `pos_tagged.csv`; Task 8 rebuilds it against the final corpus.

Run: `uv run pytest tests/test_morph.py -v`
Expected: PASS (21 tests)

- [ ] **Step 5: Commit**

```bash
git add src/build_morph_lookup.py models/morph_lookup.json tests/test_morph.py
git commit -m "Key morph_lookup.json on feats and add a pooled index

The 3-char suffix key was ambiguous for 76.5% of tokens — NOUN+ите cannot
distinguish feminine from masculine plural at all."
```

---

### Task 5: Move `candidate_selection.py` onto feats and classla

**Files:**
- Modify: `src/candidate_selection.py:52` (drop `SUFFIX_LEN`), `:475-521` (delete `_shared_suffix_len` and `surface_form`), `:526-559` (`build_slot`), `:609-619` (`_retag`), `:643-680` (`slot_contexts`)
- Modify: `tests/test_candidate_selection.py:26` (`MORPH_TIERS`), `:216-245` (suffix-based tests)

**Interfaces:**
- Consumes: `morph.surface_form` (Task 3), `tagging.tag_lines` / `tagging.load_verb_lemmas` (Task 1)
- Produces: `build_slot(source_word, source_lemma, source_pos, source_feats, prev_pos, next_pos, target_author, resources)` — note the added `source_feats` parameter, fourth positional

- [ ] **Step 1: Update the tests first**

In `tests/test_candidate_selection.py`, extend the tier set at line 26:

```python
MORPH_TIERS = {'exact', 'cross_author', 'relaxed', 'nearest_attested', 'unresolved'}
```

Delete the five suffix-based tests at lines 218–245 (`test_surface_form_exact_tier`,
`test_surface_form_falls_through_to_another_author`,
`test_surface_form_nearest_attested_prefers_the_closest_suffix`,
`test_surface_form_unresolved_returns_the_lemma`,
`test_surface_form_short_source_word_uses_the_whole_word`) — `tests/test_morph.py` now
covers every tier against the feats key. Replace them with one test that the re-export
is wired up:

```python
def test_surface_form_is_re_exported_from_morph():
    lookup = {
        'by_author': {'A': {'жена': {'NOUN': {'Case=Nom|Number=Sing': 'жена'}}}},
        'pooled': {},
    }
    assert cs.surface_form('A', 'жена', 'NOUN', 'Case=Nom|Number=Sing', lookup) == ('жена', 'exact')
```

Add a test that feats reach the slot:

```python
def test_slot_contexts_carries_feats(resources):
    contexts = cs.slot_contexts(STANZA, MASK_WORDS, RACIN)
    assert all('feats' in c for c in contexts)
    assert any(c['feats'] for c in contexts), 'no slot carried any morphology'
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_candidate_selection.py -v -k "re_exported or carries_feats"`
Expected: FAIL — `TypeError` on the old 5-arg `surface_form`, and `KeyError: 'feats'`

- [ ] **Step 3: Rewire `candidate_selection.py`**

Delete `SUFFIX_LEN = 3` (line 52), and delete `_shared_suffix_len` and `surface_form` (lines 475–521). Replace with a re-export next to the other imports:

```python
from morph import surface_form  # noqa: F401 — re-exported for callers and tests
```

Replace `_retag` (lines 609–619) with the classla path:

```python
def _retag(text: str) -> list:
    """classla fallback for a source poem that is not in the corpus (§5).

    Uses the same normalization as the corpus build, so a source 'што' is
    dropped and a '-јќи' form resolves to the base verb the lookup knows.
    """
    import tagging
    verb_lemmas = _cached('verb_lemmas', lambda: tagging.load_verb_lemmas(load_pos_rows()))
    return [{'line': t.line, 'word': t.text, 'lemma': t.lemma,
             'pos': t.pos, 'feats': t.feats}
            for t in tagging.tag_lines(text, verb_lemmas)]
```

In `slot_contexts` (lines 643 onward), the corpus branch reads rows straight from
`pos_tagged.csv`, which already carries a `feats` column — so `tagged[(li, wi)]` already
has it. Ensure each emitted context dict includes `'feats': token.get('feats', '')`
alongside its existing `lemma` and `pos` keys.

In `build_slot`, take `source_feats` as the fourth positional parameter and pass it
through:

```python
def build_slot(
    source_word: str, source_lemma: str, source_pos: str, source_feats: str,
    prev_pos, next_pos,
    target_author: str, resources: dict,
) -> dict:
```

and inside the candidate loop:

```python
        surface, morph_tier = surface_form(
            target_author, c['lemma'], source_pos, source_feats,
            resources['morph_lookup'],
        )
```

Add `'feats': source_feats` to the returned slot dict. Update `build_slots` (the caller
used by the tests) to read `context['feats']` and pass it.

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: PASS. If `test_cooccurrence_index_builds_in_acceptable_time` is slow, that is pre-existing.

- [ ] **Step 5: Commit**

```bash
git add src/candidate_selection.py tests/test_candidate_selection.py
git commit -m "Move candidate selection onto feats keys and classla retagging"
```

---

### Task 6: `style_transfer.py` — classla, feats, offset splicing

**Files:**
- Modify: `src/style_transfer.py:18` (drop spacy import), `:31` (pipeline), `:55-73` (`get_surface_morph`), `:94-95` (lookup load), `:180-236` (line rebuild)

**Interfaces:**
- Consumes: `morph.surface_form`, `tagging.tag_lines`, `tagging.load_verb_lemmas`
- Produces: `splice(line, edits) -> str` where `edits` is `[(start_char, end_char, surface), ...]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_style_transfer.py
"""Offset splicing — the mechanism that makes out-of-region edits impossible."""

import style_transfer as st


def test_splice_replaces_one_span():
    assert st.splice('чемрее врба проклета', [(7, 11, 'бука')]) == 'чемрее бука проклета'


def test_splice_preserves_untouched_text_exactly():
    # 'в поле' must survive: normalising it to 'во полето' destroys period voice.
    line = 'Натаму – в поле битолско'
    assert st.splice(line, []) == line


def test_splice_handles_multiple_edits_in_order():
    out = st.splice('чемрее врба проклета', [(0, 6, 'вене'), (12, 20, 'самотна')])
    assert out == 'вене врба самотна'


def test_splice_is_order_independent():
    edits = [(12, 20, 'самотна'), (0, 6, 'вене')]
    assert st.splice('чемрее врба проклета', edits) == 'вене врба самотна'
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_style_transfer.py -v`
Expected: FAIL — `AttributeError: module 'style_transfer' has no attribute 'splice'`

Note: `style_transfer.py` loads `word_embeddings.json` (299 MB) and `classifier.pkl` at
import. If import time makes this painful, that is expected — the test still runs.

- [ ] **Step 3: Rewire `style_transfer.py`**

Remove `import spacy` (line 18) and replace line 31:

```python
import tagging
from morph import surface_form
```

Replace `get_surface_morph` (lines 55–73):

```python
def get_surface_morph(author: str, lemma: str, pos: str, source_feats: str) -> str:
    """Surface form of (lemma, pos) for author, in the source token's morphology.

    Delegates to morph.surface_form and keeps only the form — callers here do
    not audit tiers; candidate_selection.py does.
    """
    surface, _tier = surface_form(author, lemma, pos, source_feats, morph_lookup)
    return surface
```

Delete `get_surface` (lines 47–52) and the `surface_counter` build (lines 43–44) — tier 4
of `surface_form` subsumes both. Add `splice`:

```python
def splice(line: str, edits: list) -> str:
    """Write surfaces into the original line at recorded character offsets.

    Rebuilding from the source text rather than from tagger tokens makes
    out-of-region edits structurally impossible and preserves spacing and
    punctuation exactly.
    """
    out, prev = [], 0
    for start, end, surface in sorted(edits):
        out.append(line[prev:start])
        out.append(surface)
        prev = end
    out.append(line[prev:])
    return ''.join(out)
```

Rewrite the rebuild loop (lines 180–236). It currently appends `token.text_with_ws` for
every skipped token and `surface + token.whitespace_` for replacements. Replace that with
collecting edits and splicing once per line. Tag the whole poem once:

```python
verb_lemmas = tagging.load_verb_lemmas(pos_rows)
tokens = tagging.tag_lines(text, verb_lemmas)
by_line = defaultdict(list)
for t in tokens:
    by_line[t.line].append(t)

rebuilt_lines = []
for li, line in enumerate(text.splitlines()):
    edits = []
    for token in by_line[li]:
        # ... existing should_replace / scoring logic, using token.text,
        # token.lemma, token.pos in place of token.text_ / token.lemma_ / token.pos_
        if not replace:
            continue
        surface = get_surface_morph(target_author, best_lemma, token.pos, token.feats)
        edits.append((token.start_char, token.end_char, surface))
    rebuilt_lines.append(splice(line, edits))
```

`tag_lines` already filters to content POS, so the `token.pos_ not in CONTENT_POS or
token.is_punct or token.is_space` guard at line 186 becomes unnecessary — skipped tokens
simply never appear. Keep the `-ски/-цки` adjective guard at line 191, reading
`token.text` instead of `token.text.lower()` (`tag_lines` already lowercases).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_style_transfer.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Smoke-test the demo end to end**

Run: `uv run python src/style_transfer.py`
Expected: loads without `ImportError`, prints `Ready.`, and transfers the built-in Рацин stanza without a traceback. Confirm `в поле` and `в гроб` survive verbatim in the output — that is the region-discipline property the splice is there to guarantee.

- [ ] **Step 6: Commit**

```bash
git add src/style_transfer.py tests/test_style_transfer.py
git commit -m "Move style transfer to classla, feats lookup, and offset splicing"
```

---

### Task 7: Drop spaCy and update the docs

**Files:**
- Modify: `pyproject.toml:14`
- Modify: `CLAUDE.md` (pipeline order, POS tagging section)

- [ ] **Step 1: Confirm nothing imports spaCy**

Run: `grep -rn "spacy" src/ tests/`
Expected: no output. If anything remains, fix it before continuing.

- [ ] **Step 2: Remove the dependency**

Delete `"spacy>=3.8.14",` from `pyproject.toml` line 14, then:

```bash
uv sync
```

Expected: resolves without spaCy. `mk_core_news_lg` was installed out-of-band and is not in
`pyproject.toml`; leave it on disk, nothing imports it now.

- [ ] **Step 3: Update `CLAUDE.md`**

Two edits:

In the "POS tagging quality" section, replace the sentence noting that
`build_morph_lookup.py` "still guesses gender/number/definiteness from suffixes and could
read these directly instead" with:

```markdown
`build_morph_lookup.py` keys on the `feats` column directly — see
`docs/superpowers/specs/2026-08-13-feats-morph-lookup-design.md`.
```

Add to the "Folder structure" / module notes a line for the two new modules:

```markdown
`src/tagging.py` owns the classla pipeline and the two corrections applied on top of it
(relativizer filtering, `-јќи` re-lemmatization). Both `pos_tag_corpus.py` and the
inference path import it, so corpus tags and inference tags cannot drift.

`src/morph.py` owns feats parsing, the relaxation order, and `surface_form()`.
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock CLAUDE.md
git commit -m "Drop spaCy; document the shared tagging and morph modules"
```

---

### Task 8: Rebuild all artifacts

Nothing here changes code. Run the nine steps in order, once. Step 6 is the long one — it
regenerates `models/word_embeddings.json` (299 MB, excluded from git).

**Files:**
- Regenerates: everything in `data/` except `stripped_songs.csv` and `pos_corrections.csv`; everything in `models/`

- [ ] **Step 1: Confirm the classla models are present**

Run: `uv run python -c "import classla; classla.Pipeline('mk', processors='tokenize,pos,lemma', download_method=None)"`
Expected: loads without error. If it fails, run `uv run python -c "import classla; classla.download('mk')"` once.

- [ ] **Step 2: Re-tag the corpus**

Run: `uv run python src/pos_tag_corpus.py`
Expected: prints the corrections summary (`Applied 86 hand corrections ... N tokens retagged, M dropped`) and a final content-token count. Both `data/pos_tagged.csv` and `data/pos_flags.csv` are rewritten.

- [ ] **Step 3: Run steps 2–8 of the pipeline**

```bash
uv run python src/tfidf_authors.py
uv run python src/style_profiler.py
uv run python src/build_skg.py
uv run python src/enrich_skg.py
uv run python src/add_embeddings.py
uv run python src/graph_analytics.py
uv run python src/classifier.py
```

Expected: each completes without a traceback. `add_embeddings.py` is the slow one.

- [ ] **Step 4: Rebuild the morph lookup against the final corpus**

Run: `uv run python src/build_morph_lookup.py`
Expected: per-author and pooled entry counts printed, 24 authors.

- [ ] **Step 5: Verify against the rebuilt artifacts**

Run: `uv run pytest tests/ -v`
Expected: PASS. These tests read the real `pos_tagged.csv` and `morph_lookup.json`, so this is the check that the rebuild is coherent, not just that the code runs.

- [ ] **Step 6: Spot-check the demo**

Run: `uv run python src/style_transfer.py`
Expected: transfers the Рацин stanza. As a Macedonian speaker, read the output for agreement errors — that is the property this whole change exists to protect, and no automated test substitutes for it.

- [ ] **Step 7: Commit the rebuilt artifacts**

```bash
git add data/ models/
git commit -m "Rebuild all artifacts against corrected tags and feats-keyed lookup"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: feats keying and the pooled index → Task 4; the five-tier chain → Task 3; the relaxation order including NOUN-only Gender → Task 2; shared normalization via `src/tagging.py` → Task 1; offset rebuilding → Task 6; dropping spaCy → Task 7; the empty-key edge case → Task 2 (`test_relaxation_never_empties_the_bundle`); featless tokens → Task 3; the rebuild order → Task 8. The spec's testing section maps item-for-item onto tests in Tasks 2, 3, 5, and 6.

**Deviation from the spec, deliberate and flagged above.** Exact-match relaxation cannot work against full-bundle stored keys; relaxation uses subset matching instead. Documented under "Correction to the spec" and implemented in `_match()`.

**Type consistency.** `surface_form(target_author, candidate_lemma, pos, source_feats, morph_lookup)` has the same five-parameter signature everywhere it appears (Tasks 3, 5, 6). `morph_lookup` is `{'by_author', 'pooled'}` in Tasks 3, 4, 5, and 6. `Token` fields (`text, lemma, pos, xpos, feats, line, start_char, end_char`) are consistent between Task 1's definition and its uses in Tasks 5 and 6. `build_slot` gains `source_feats` as its fourth positional parameter, and Task 5 updates its only caller.

**Known risk.** Task 1 Step 6 verifies the extraction by import rather than by diffing tagger output, because a full tagging run takes minutes and Task 8 re-runs it anyway. If Task 8 Step 2 reports a materially different content-token count than the ~70,531 rows currently in `pos_tagged.csv`, suspect the extraction and diff against `data/pos_tagged.csv.bak`.
