# Style-Transfer Candidate Selection Pipeline — Implementation Spec

## 1. Goal

Replace the current global "top-20 TF-IDF words" style spec with a **per-slot, grammatically valid, thematically grounded, corpus-attested candidate list** for each masked word in the infill probes. This is inference-time logic — it runs at prompt-build time, not a one-time offline build like the SKG scripts.

Bypass `skg_final.gexf` for this path entirely. The graph's 371-word cap and frequency floors exist for visualization legibility, not because the underlying data is unavailable elsewhere — every signal the graph encodes also exists directly in the source JSON/CSV files, ungated. Query those directly.

## 2. Inputs (existing files, no changes needed to any of them)

FileShapeUsed for`data/pos_tagged.csv`author, song_title, word, lemma, poslive co-occurrence, POS lookups`data/tfidf_results.csv`author, rank, word, tfidf_scoredistinctiveness boost`models/author_vocab.json`author → pos → \[(lemma, freq), ...\]candidate pool (uncapped)`models/pos_transitions.json`author → pos → {next_pos: prob}syntactic plausibility`models/morph_lookup.json`author → lemma → pos → suffix → surface formsurface realization`models/word_embeddings.json`lemma → 768-dim normalized vector (ALL corpus lemmas)semantic narrowing

## 3. New module: `candidate_selection.py`

```python
# --- loaders (module-level, load once) ---
def load_author_vocab() -> dict: ...          # author_vocab.json
def load_tfidf() -> dict: ...                 # author -> {word: score}
def load_pos_transitions() -> dict: ...       # pos_transitions.json
def load_morph_lookup() -> dict: ...          # morph_lookup.json
def load_word_embeddings() -> dict: ...       # lemma -> np.ndarray
def build_cooccurrence_index(pos_rows) -> dict:
    """
    (lemma, pos) -> Counter{(other_lemma, other_pos): shared_poem_count}
    Same logic as build_skg.py's `cooccurrence`, but keyed for lookup by
    ANY lemma, not filtered to graph-node membership. Build once, reuse
    across all slot queries in a run.
    """
```

```python
def get_embedding(lemma: str, embeddings: dict) -> np.ndarray | None:
    """
    Direct dict lookup. word_embeddings.json covers every corpus lemma,
    so this should always hit for in-corpus source words. Return None
    (not an exception) if truly absent — e.g. a lemma introduced only
    via hardcoded FALLBACK_STANZA — and let the caller skip the semantic
    filter for that slot rather than crash.
    """

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Plain dot product — embeddings are pre-normalized at generation time."""
    return float(a @ b)
```

```python
def candidate_pool(target_author: str, pos: str, author_vocab: dict) -> list[tuple[str, int]]:
    """
    author_vocab[target_author][pos], already frequency-sorted.
    Returns [] if author/pos combo absent — caller must handle via
    fallback tier (see §5).
    """

def rank_candidates(
    source_lemma: str, source_pos: str,
    target_author: str,
    prev_pos: str | None, next_pos: str | None,
    author_vocab: dict, tfidf: dict, embeddings: dict,
    cooc_index: dict, transitions: dict,
    top_k: int = 3,
) -> list[dict]:
    """
    1. pool = candidate_pool(target_author, source_pos, author_vocab)
       Drop the source_lemma itself if present (no-op replacement).
    2. For each candidate lemma in pool, compute:
         - tfidf_boost: candidate's tfidf score for target_author if present
           in tfidf[target_author], else 0
         - sem_sim: cosine(get_embedding(source_lemma), get_embedding(candidate))
           if both vectors exist, else None
         - cooc_score: cooc_index.get((source_lemma, source_pos), {}).get(
           (candidate, source_pos), 0) — thematic proximity to the SOURCE word,
           not the target author's usage
         - transition_score: average of
           transitions[target_author].get(prev_pos, {}).get(source_pos, 0.0) and
           transitions[target_author].get(source_pos, {}).get(next_pos, 0.0)
           (skip whichever side is None, e.g. line-initial/final)
    3. Filter: drop candidates with sem_sim is not None and sem_sim < SEM_THRESHOLD
       (default 0.65). If this empties the pool, see §5 relaxation tiers.
    4. Composite score (defaults — treat as tunable, log raw components):
         score = 0.4 * norm(tfidf_boost) + 0.35 * (sem_sim or 0)
                 + 0.15 * norm(cooc_score) + 0.10 * transition_score
    5. Return top_k as [{'lemma': ..., 'score': ..., 'tfidf': ..., 'sem_sim': ...,
       'cooc': ..., 'transition': ..., 'tier': 'primary'}], sorted descending.
    """
```

```python
def surface_form(
    target_author: str, candidate_lemma: str, pos: str, source_word: str,
    morph_lookup: dict,
) -> tuple[str, str]:
    """
    Returns (surface_form, tier_used).
    suffix = source_word[-3:].lower() if len(source_word) >= 3 else source_word.lower()

    Tier chain:
      1. morph_lookup[target_author][candidate_lemma][pos][suffix]        -> 'exact'
      2. any other author: first author whose
         morph_lookup[a][candidate_lemma][pos] has this suffix            -> 'cross_author'
      3. most frequent surface form for
         morph_lookup[target_author][candidate_lemma][pos] (any suffix)   -> 'nearest_attested'
      4. candidate_lemma unchanged, flagged for model-side inflection     -> 'unresolved'

    Always return which tier fired — this is required output, not optional
    logging (see §8).
    """

def build_slot(
    source_word: str, source_lemma: str, source_pos: str,
    prev_pos: str | None, next_pos: str | None,
    target_author: str, resources: dict,
) -> dict:
    """
    Orchestrates rank_candidates + surface_form per candidate.
    Returns:
    {
      'source_word': 'незнаен',
      'pos': 'ADJ',
      'candidates': [
        {'surface': 'самотен', 'lemma': 'самотен', 'score': 0.81,
         'morph_tier': 'exact'},
        ...
      ]
    }
    """
```

## 4. Fallback / relaxation tiers for sparse candidate pools

Author-specific POS pools can be thin (e.g. a handful of adjectives for a minor author). If step 3 in `rank_candidates` leaves fewer than `MIN_CANDIDATES` (default 3):

1. Relax `SEM_THRESHOLD` in steps of 0.05 down to a floor (e.g. 0.45).
2. If still short, pull candidates from Jaccard-similar authors (`author_similarity`-equivalent — recompute directly from `pos_tagged.csv` lemma sets per §3, don't depend on the graph's precomputed edges) with a `tier: 'cross_author_similar'` tag, ranked by that author's own tfidf/embedding scores against the same source word.
3. If still empty, fall back to the current global TF-IDF top-N as a last resort, tagged `tier: 'legacy_fallback'` — this should be rare enough to be worth alerting on, not silent.

Every candidate emitted carries its tier. This is required for evaluation — you want to know what fraction of replacements in a run were fully grounded vs. relaxed vs. legacy-fallback.

## 5. Integration point

In the probe-building code (`_mask` / `build_probes` equivalent):

- Masking already knows each masked word's surface form and position in the line. Add: look up its lemma+POS (from `pos_tagged.csv` for that author/poem, or re-tag if the source poem/author isn't in the corpus), and the POS of the immediately preceding/following token in the line.
- Call `build_slot(...)` once per masked word.
- Replace the current single `vocab` string (`load_target_words` output) with a per-slot structure passed into prompt construction.
- Keep `load_target_words()` around, unused by the new template, as the `legacy_fallback` tier's data source.

## 6. Logging requirement

For each probe run, emit (stdout or a structured log line is fine) per slot: `source_word, pos, chosen_candidate, composite_score, morph_tier, semantic_tier`. This is what lets you audit, after a batch run, how often the pipeline is actually grounded vs. falling back — directly answers "is this worth the added complexity" empirically rather than by assumption.

## 7. Testing / acceptance criteria

- Unit test the exact three-word Рацин→Конески case from the existing probe (`чемрее`, `проклета`, `незнаен`) — assert each slot returns ≥1 candidate, correct POS, and a resolved surface form (any tier).
- Unit test a deliberately sparse case (pick a low-frequency POS for a minor author) — assert the relaxation chain fires and is logged, not silently empty.
- Unit test `get_embedding` on a lemma known absent from `word_embeddings.json` — assert graceful `None`, not an exception.
- Confirm `cooccurrence` build time is acceptable at full corpus scale (pure Python double loop over poem token sets — profile once, it's the most expensive step here).