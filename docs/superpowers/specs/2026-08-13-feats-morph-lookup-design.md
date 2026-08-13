# Feats-keyed morphological lookup and a single tagger across the pipeline

**Date:** 2026-08-13
**Status:** design, awaiting implementation plan
**Scope:** re-key `models/morph_lookup.json` on the `feats` column, move inference-time
tagging from spaCy to classla, and share tag normalization between the corpus build and
the inference path. Then rebuild all artifacts once.

## Background

Hand-correction of the POS tags is largely complete: 86 corrections in
`data/pos_corrections.csv` (21 `DROP`, 4 lemma-only, all corpus-wide scope), and
`data/stripped_songs.csv` itself has changed. A full rebuild is required regardless of
anything in this document.

The corrected corpus also makes a deferred improvement viable. `build_morph_lookup.py`
keys surface forms on the last 3 characters of the word, a decision made when tagging was
spaCy-based and no morphological features were available. Tagging is now classla-based and
`pos_tagged.csv` carries a populated `feats` column, so the lookup can key on the actual
morphology instead of a proxy for it.

### Why this matters more than it looks

`docs/superpowers/specs/2026-08-09-llm-filling-stage-design.md` established, from probes
against four models, that every model tested breaks Macedonian agreement when generating
verse — `тивка гроб`, `мрачен врба`, `тивок ноќ` — including one whose leaked reasoning
recites the correct rule immediately before violating it. That finding drove its central
architectural decision: **the LLM proposes lemmas only and never emits a surface form**
(line 62), with all inflection routed through `get_surface_morph()` and
`morph_lookup.json`.

That makes this lookup the single component carrying morphological correctness for the
entire LLM filling path. Its key is currently the weakest part of it.

## Measured constraints

All figures from `data/pos_tagged.csv` at 70,531 rows, restricted to the 24 authors with
≥5 poems where noted.

**`feats` coverage is 98.9%.** 790 rows have an empty `feats`, of which 111 are hand-
corrections where `xpos`/`feats` were deliberately cleared. Empty by POS: NOUN 333, VERB
168, PROPN 154, ADV 102, ADJ 33.

**The suffix key is ambiguous for 76.5% of tokens.** Of 3,713 distinct `(pos, suffix)`
keys, 1,303 map to more than one feats bundle, covering 53,966 of 70,531 tokens. The
worst cases are the most common endings in the language:

| key | tokens | distinct feats | top two readings |
|---|---|---|---|
| `NOUN` + `ата` | 2,685 | 8 | Fem-Sing-Def (2,303) vs **Neut-Plur-Def** (304) |
| `NOUN` + `ите` | 2,029 | 7 | Fem-Plur-Def (1,059) vs **Masc-Plur-Def** (954) |
| `ADJ` + `ата` | 1,165 | 8 | Fem-Sing-Poss (433) vs Fem-Sing-Degree=Pos (428) |
| `ADJ` + `ите` | 1,092 | 10 | Plur-Degree=Pos (468) vs Plur-Poss (338) |

`NOUN`+`ите` is the clearest failure: the suffix cannot distinguish feminine from
masculine plural at all, and the split is near 50/50, so the lookup returns the wrong
gender roughly half the time it fires — while reporting tier `exact`.

**Switching keys costs nothing in size.** Across eligible authors: 45,092 suffix entries
vs 45,333 feats entries over 35,471 `(author, lemma, pos)` cells. Average 1.27 suffixes
vs 1.28 feats per cell.

**Per-author data is sparse; pooling more than doubles it.** 81.5% of
`(author, lemma, pos)` cells hold exactly one feats value. Authors collectively cover
45,333 feats slots, while 53,481 additional slots exist corpus-wide for the same lemmas —
**cross-author pooling makes 118% more forms retrievable.**

**Feature informativeness, per POS, over tokens with non-empty feats:**

```
NOUN   Number   100.0% present   Sing 74.9%       Case  100.0% present   Nom 99.8%
       Definite 100.0% present   Ind  66.1%       Gender 99.9% present   Fem 42.5%

VERB   Mood     100.0% present   Ind  96.8%       Polarity 100% present  Pos 100.0%
       Number   100.0% present   Sing 79.2%       Aspect  99.6% present  Prog 62.0%
       Tense     92.9% present   Pres 75.4%       Person  75.6% present  3    58.7%

ADJ    Number   100.0% present   Sing 72.9%       Definite 100% present  Ind 72.5%
       Gender    72.6% present   Fem  39.8%       Degree  54.6% present  Pos 96.6%
       VerbForm  16.0% present   Part 100.0%      Poss    13.5% present  Yes 100.0%

ADV    Degree    96.2% present   Pos  95.1%       PronType 17.6% present Dem 57.3%
```

Two readings of this table drive the relaxation order below. `Polarity` is 100% present
with a single value and therefore carries literally zero information. Conversely
`VerbForm=Part` and `Poss=Yes` have a single value each but appear on only 16% and 13.5%
of adjectives — their *presence* is the signal, so single-valuedness must not be mistaken
for uninformativeness.

**spaCy emits no morphological features.** `mk_core_news_lg` lists a `morphologizer` in
its pipeline, but `token.morph` is empty for every token tested. It also mistags against
classla on basic cases — `жената` as ADJ with an uninflected lemma, `што` as SCONJ. There
is no path to a feats key on the spaCy side.

## Architecture

```
CORPUS BUILD                          INFERENCE
pos_tag_corpus.py                     style_transfer.py / candidate_selection.py
  │                                     │
  └──────────► src/tagging.py ◄─────────┘
               classla pipeline
               is_content()      drops bare-Rg relativizers
               gerund_lemma()    re-lemmatizes -јќи forms
               corrections       applies pos_corrections.csv
               │
               └─► tokens: text, lemma, pos, xpos, feats, start_char, end_char
                        │
                        ├─► pos_tagged.csv ──► build_morph_lookup.py
                        │                        │
                        │                        ├─► author → lemma → pos → feats → surface
                        │                        └─► pooled → lemma → pos → feats → surface
                        │
                        └─► surface_form(target, lemma, pos, source_feats) ──► (surface, tier)
```

### Design decisions

**Key on `feats`, not suffix.** The suffix was a proxy for morphology chosen when
morphology was unavailable. It is now available, is no larger as an index, and is
unambiguous where the suffix conflates 76.5% of tokens.

**Emit a pooled index alongside the per-author one.** Sparsity, not ambiguity, is the
binding constraint at inference: 81.5% of cells hold one form. Pooling adds 118% more
retrievable forms at the cost of borrowing another author's spelling of a common
inflection — a trade worth taking, because a wrong-gender word is a visible grammatical
error while a correctly-inflected word from a neighbouring idiolect is not.

**Share normalization between corpus and inference.** This is the correctness-critical
decision. `pos_tag_corpus.py` does not use raw classla output — it drops bare-`Rg`
relativizers via `is_content()` and re-lemmatizes `-јќи` gerunds via `gerund_lemma()`, and
`pos_tagged.csv` reflects those corrections. Inference calling classla directly would
produce a source `што` tagged ADV/`Degree=Pos` and a source `барајќи` lemmatized to
itself — neither of which exists in the lookup, sending every such slot silently to
`unresolved`. One shared module makes divergence structurally impossible rather than a
thing to remember.

**Rebuild by character offset.** classla has no equivalent of spaCy's `token.text_with_ws`
/ `token.whitespace_`, which `style_transfer.py` uses to reassemble lines (lines 186–235).
It provides `start_char`/`end_char` instead. This is the same mechanism the phase-1 LLM
spec independently chose for splicing (line 112), so the two converge.

**Drop spaCy entirely.** With inference on classla, `mk_core_news_lg` has no remaining
consumer.

## Components

| File | Role |
|---|---|
| `src/tagging.py` | **new** — classla pipeline wrapper; `is_content()`, `gerund_lemma()`, correction application moved here; yields tokens with `text, lemma, pos, xpos, feats, start_char, end_char` |
| `src/build_morph_lookup.py` | modified — key on `feats`; emit the pooled index |
| `src/candidate_selection.py` | modified — `surface_form()` takes `source_feats`; new `relaxed` tier; `SUFFIX_LEN` removed |
| `src/style_transfer.py` | modified — classla via `tagging.py`; `get_surface_morph()` takes feats; offset-based line rebuild |
| `src/pos_tag_corpus.py` | modified — imports its normalization from `tagging.py` rather than defining it |

## Resolution chain

`surface_form(target_author, lemma, pos, source_feats) -> (surface, tier)`

| # | Tier | Source |
|---|---|---|
| 1 | `exact` | `author[lemma][pos][feats]` |
| 2 | `cross_author` | `pooled[lemma][pos][feats]` |
| 3 | `relaxed` | drop features cumulatively per the order below; retry 1 then 2 at each width |
| 4 | `nearest_attested` | author's most frequent form for `[lemma][pos]`, any feats |
| 5 | `unresolved` | bare lemma, flagged for model-side inflection |

Tiers 1, 2, 4 and 5 keep the names they have today in `candidate_selection.py:492–521`,
so existing `morph_tier` audit logging keeps working. Tier 3 is new. A relaxed match
reports `relaxed` and never `exact` — the current implementation reports `exact` for a
suffix hit even when it matched the wrong gender.

### Relaxation order

Drops are **cumulative** — each step removes one more feature from the bundle and retries
tiers 1 then 2 at that width before the next step. Features not present on the token are
skipped, so the single ordering below serves all four POS without per-POS branching.

```
1. Polarity   (VERB)      100% present, 1 value — zero information
2. Case       (NOUN)      99.8% Nom
3. Mood       (VERB)      96.8% Ind
4. Degree     (ADJ, ADV)  96.6% / 95.1% Pos
5. Gender     (NOUN ONLY) lexically fixed by the lemma — see below
6. Aspect     (VERB)
7. Person     (VERB)
8. Tense      (VERB)

Never dropped:
   Gender, Number, Definite  on ADJ and VERB   — agreement
   Number, Definite          on NOUN           — surface distinction
   VerbForm, Poss, NumType, PronType, AdvType  — presence is the signal
```

**`Gender` is droppable for NOUN only.** Macedonian nouns carry inherent lexical gender —
`град` is masculine in every form it takes — so once the lemma is fixed, `Gender` is
redundant and dropping it only widens the match. On adjectives and verbs gender is
agreement with the subject, and dropping it yields exactly the `тивка гроб` class of error
the phase-1 probes caught the LLMs producing.

**`Definite` is never dropped, including on NOUN.** Definiteness is a syntactic choice
rather than agreement, so a mismatch stays grammatical — but swapping `град` for `градот`
changes what the line says, and this pipeline exists to preserve content while changing
voice.

## Edge cases

- **Empty `feats` (790 rows, 1.1%).** Keyed under `''`. Matches source tokens that are
  themselves featless; otherwise reachable via `nearest_attested`. No special-casing.
- **Relaxation must never reduce a bundle to the empty key.** The `''` key is reserved
  for genuinely featless tokens, and matching it from a relaxed lookup would return an
  arbitrary form while claiming a morphological match. This is reachable in practice:
  `Degree` is an ADV's only feature 76.9% of the time and is droppable at step 4. When the
  next drop would empty the bundle, relaxation stops and falls through to
  `nearest_attested` instead.
- **Hand-corrected tokens (111).** A corrected POS clears `xpos`/`feats` by existing
  convention, so these are a subset of the above and need no separate handling.
- **Author absent from the lookup.** Authors under the `MIN_POEMS = 5` threshold are not
  indexed; such a request falls to tier 2 and then tier 5.
- **Lemma absent entirely.** Tier 5, `unresolved` — the existing contract.

## Relationship to the 2026-08-09 phase-1 spec

This document supersedes two items in it, deliberately:

- **Line 181** lists `build_morph_lookup.py` as out of scope for phase 1. It was deferred
  pending exactly the tag corrections that have now landed.
- **Line 100** has the MASK stage POS-tagging with spaCy. That becomes classla via
  `src/tagging.py`.

Everything else in that spec stands unchanged. In particular its decision that the LLM
proposes lemmas and never emits surface forms (line 62) is *reinforced* here, since this
work strengthens the component that decision depends on.

## Testing

Extends `tests/test_candidate_selection.py`, following its existing style.

- **The bug this fixes:** `NOUN` + `ата` under Fem-Sing-Def and under Neut-Plur-Def must
  resolve to *different* surface forms. Same for `NOUN` + `ите` across Fem-Plur-Def and
  Masc-Plur-Def, where the suffix key is near 50/50 and currently coin-flips.
- **Each tier fires and is labelled correctly** — construct a case per tier, including a
  `relaxed` case, and assert the returned tier string.
- **Relaxation respects the never-drop list** — assert an ADJ request never resolves to a
  form disagreeing in gender, and that a NOUN request may drop `Gender`.
- **Featless token** degrades to `nearest_attested` rather than raising.
- **Shared normalization** — `што` and a `-јќи` form normalize identically whether
  reached through the corpus path or the inference path.
- **Existing Рацин→Конески case** (`чемрее`, `проклета`, `незнаен`) still returns ≥1
  candidate per slot with a resolved surface form.

## Rebuild

The rework lands and is tested first; then steps 1–9 run exactly once, so no artifact is
built twice:

```
1. src/pos_tag_corpus.py      → data/pos_tagged.csv, data/pos_flags.csv
2. src/tfidf_authors.py       → data/tfidf_results.csv
3. src/style_profiler.py      → data/author_style_profiles.csv
4. src/build_skg.py           → models/skg.gexf
5. src/enrich_skg.py          → models/skg_enriched.gexf, author_vocab.json, pos_transitions.json
6. src/add_embeddings.py      → models/skg_final.gexf, models/word_embeddings.json
7. src/graph_analytics.py     → data/graph_analytics.csv
8. src/classifier.py          → models/classifier.pkl
9. src/build_morph_lookup.py  → models/morph_lookup.json
```

Step 1 is required independently of this design: `stripped_songs.csv` has changed, and the
`DROP` and `unknown_tag` corrections need a full tagging run rather than
`apply_corrections.py`.

## Out of scope

- Any change to the phase-1 LLM filling stage beyond the two superseded lines above.
- Reading `xpos` positionally in `build_morph_lookup.py`. `CLAUDE.md` notes the suffix-
  guessing of gender/number/definiteness could read `xpos` slots directly; keying on
  `feats` resolves this more directly and makes the note obsolete.
- Tuning `MIN_POEMS`, `SEM_THRESHOLD`, or the composite scoring weights.
- Re-review of the remaining 1,827 rows in `pos_flags.csv`.
