# GRPO Stage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune the SFT Qwen3-4B style-transfer model with TRL GRPO against a calibrated style reward and an embedding content reward, with a standalone reward inspector, a smoke test, Slurm jobs, and a three-way (Gemma / SFT / SFT+GRPO) evaluation on identical val pairs.

**Architecture:** Per-poem style features are extracted into `style_features.py` (shared with `style_profiler.py`). `build_reward_calibration.py` turns train-split real poems into per-author feature distributions (`data/reward_calibration.json`). `grpo_rewards.py` scores a generation against that calibration (style), against the source poem via line-level embedding F1 (content, in `content_similarity.py`), and applies multiplicative gates from `style_metrics.py`; it exposes two TRL reward functions. `train_grpo.py` loads the SFT adapter merged into the base weights (`merge_sft_adapter.py`) and trains a fresh LoRA with `GRPOTrainer`, so the KL reference is the SFT model.

**Tech Stack:** Python 3.10, TRL 1.13 (`GRPOTrainer`/`GRPOConfig`), PEFT 0.21, transformers 5.9, sentence-transformers 5.6 (`paraphrase-multilingual-mpnet-base-v2`), classla (`mk`), numpy, pytest, Singularity + Slurm.

**Spec:** `docs/superpowers/specs/2026-09-27-grpo-stage-design.md`

## Global Constraints

- Train split only for GRPO prompts and reward calibration; val/test never enter training, calibration or in-training eval.
- Gemma output is never a reward target; content reference = source poem; style reference = target author's real train poems.
- Self-transfer pairs are skipped (`generate_synthetic.build_pairs`).
- `author_style_profiles.csv` must be byte-identical after the `style_profiler.py` refactor.
- Every reward component and gate is logged separately.
- Reward = `α·R_style_g + β·R_content_g`, defaults α = β = 0.5; flags `--style-weight` / `--content-weight` (KL stays `--beta`, default 0.04).
- GRPO defaults: G=4, per-device batch 8, grad accum 4, lr 1e-5 constant with 10 warmup steps, LoRA r16/α32/dropout 0.0, max completion 512, temperature 0.9, max steps 500, seed 42, bf16.
- Prompts longer than `--max-prompt-length` (1024) are dropped, never truncated.
- Nothing starts a large training run automatically; GPU steps are commands for the user.
- Run everything via `uv run`; tests via `uv run pytest`. Tests that load classla, sentence-transformers or download a model are marked `@pytest.mark.slow`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Deviations from the spec (decided while planning — flagged to the user)

1. **Copy gate metric:** the spec's "token Jaccard ≥ 0.7" misses one-word edits on this corpus's ~5-token lines (4/6 = 0.67). Replaced by **containment**: a generated line (≥ 3 tokens) is a copy if ≥ 75 % of its token set occurs in one source line; lines under 3 tokens count as copies only on exact match. Parameter `copy_overlap_threshold = 0.75`.
2. **Component logging:** TRL 1.13 passes `log_metric` / `log_extra` into reward functions, so no `TrainerCallback` is needed.
3. **Smoke check (d):** "greedy generations differ" can legitimately fail after 3 small steps. Replaced by "the log-probability of a fixed completion changes by > 1e-3", which is deterministic; before/after generations are still printed.
4. **Tagging:** `tagging.tag_lines` tags line by line; the corpus was tagged whole-poem. A new `tagging.tag_poem` mirrors the corpus build (whole text, same corrections/filtering) so reward features are computed like the profile features.
5. **Content embedding code** lives in its own `content_similarity.py` (calibration and reward both need it; avoids a circular import).
6. **SFT candidates in fixtures:** SFT outputs exist only for val pairs, and fixtures use train sources, so the SFT candidate is dropped from the fixture set; SFT rows are inspected via `evaluate_reward.py --generations-csv`.

## Review Focus

1. A target author with no calibration entry (e.g. an arbitrary `--target-authors` name) must raise a clear `KeyError` naming the author, not score 0 silently — test in Task 5.
2. A completion that is whitespace-only, a single line, or pure markdown must score 0 without crashing classla or the embedder — tests in Task 5 (fakes) and Task 6 (real classla).
3. A source poem whose rendered prompt exceeds `--max-prompt-length` must be dropped (and counted), never truncated — test in Task 8.
4. A GRPO adapter whose `adapter_config.json` records a container path (`/workspace/models/qwen3-sft-merged`) must load on the host — test in Task 12.
5. Gemma val rows must never land in `synthetic_dataset.csv` (which `train_sft.py` reads) — test in Task 12.

---

### Task 1: Whole-poem tagging for inference (`tagging.tag_poem`) + test marker

**Files:**
- Modify: `src/tagging.py` (add `tag_poem` after `tag_lines`)
- Modify: `pyproject.toml` (`[tool.pytest.ini_options]` markers)
- Test: `tests/test_tagging.py` (append)

**Interfaces:**
- Produces: `tagging.tag_poem(text: str, verb_lemmas: Counter) -> list[tuple[str, str]]` — `(lemma, pos)` for every content token, identical selection/normalisation to `pos_tag_corpus.py`.

- [ ] **Step 1: Register the `slow` marker**

In `pyproject.toml`, extend `[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
markers = [
    "slow: loads classla / sentence-transformers / downloads a model (deselect with -m 'not slow')",
]
```

- [ ] **Step 2: Write the failing test**

Append to `tests/test_tagging.py`:

```python
import csv
from difflib import SequenceMatcher
from pathlib import Path

import pytest


@pytest.mark.slow
def test_tag_poem_reproduces_the_corpus_tags_for_a_real_poem():
    import tagging
    from make_splits import load_stripped_songs

    root = Path(__file__).resolve().parent.parent
    with open(root / 'data' / 'pos_tagged.csv', encoding='utf-8') as f:
        pos_rows = list(csv.DictReader(f))
    verb_lemmas = tagging.load_verb_lemmas(pos_rows)
    poem = next(r for r in load_stripped_songs() if r['poem_id'] == '301')
    corpus = [(r['lemma'], r['pos']) for r in pos_rows if r['poem_id'] == '301']

    got = tagging.tag_poem(poem['song_text'], verb_lemmas)

    # poem/row-scoped hand corrections can't fire at inference, so allow a
    # sliver of disagreement -- but the token selection must be the corpus's.
    assert SequenceMatcher(None, got, corpus).ratio() >= 0.97


def test_tag_poem_returns_nothing_for_blank_text():
    import tagging
    assert tagging.tag_poem('  \n\n ', verb_lemmas={}) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_tagging.py -k tag_poem -v`
Expected: FAIL with `AttributeError: module 'tagging' has no attribute 'tag_poem'`

- [ ] **Step 4: Implement `tag_poem`**

Add to `src/tagging.py` directly after `tag_lines`:

```python
def tag_poem(text, verb_lemmas):
    """Tag a whole poem the way pos_tag_corpus.py tags the corpus, returning
    [(lemma, pos), ...] for its content tokens.

    tag_lines() tags line by line (its consumers splice per line); the corpus
    build hands classla the whole poem, so sentence context -- and therefore
    some tags -- differ between the two. Anything compared against
    pos_tagged.csv-derived statistics (the GRPO style reward) must be tagged
    this way instead. Corrections use global scope only (poem_id '-1'), as in
    tag_lines, since generated text has no corpus position.
    """
    if not text.strip():
        return []
    nlp = pipeline()
    table = corrections.load()
    out = []
    for sent in nlp(text).sentences:
        context = ' '.join(t.text for t in sent.tokens)
        for w in sent.words:
            result = corrections.apply_to(
                table, w.text, w.upos or '', w.lemma or '', w.xpos or '',
                w.feats or '', '-1', context)
            if result is None:
                continue
            pos, lemma, xpos, _ = result
            if not is_content(pos, xpos):
                continue
            lemma = lemma.lower()
            if xpos == 'Rv' and w.text.lower().endswith('јќи'):
                lemma, _ = gerund_lemma(w.text.lower(), verb_lemmas)
            out.append((lemma, pos))
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_tagging.py -v`
Expected: all PASS. If the ratio test fails, print `got` vs `corpus` side by side and stop to report: a large mismatch means the reward cannot reproduce profile features.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/tagging.py tests/test_tagging.py
git commit -m "Add tagging.tag_poem: whole-poem tagging matching the corpus build

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: `style_features.py` and the byte-identical `style_profiler.py` refactor

**Files:**
- Create: `src/style_features.py`
- Modify: `src/style_profiler.py` (import helpers, wrap script body in `main()`)
- Test: `tests/test_style_features.py`

**Interfaces:**
- Produces (in `style_features`):
  - `CONTENT_POS`, `MATTR_WINDOW`, `line_final_syllable(line) -> str | None`, `mattr(lemmas, window=MATTR_WINDOW) -> float` (moved verbatim)
  - `SCALAR_FEATURES: tuple[str, ...]` = `('pct_NOUN','pct_VERB','pct_ADJ','pct_ADV','pct_PROPN','lexical_diversity','avg_tokens_per_line','num_lines','num_stanzas','avg_lines_per_stanza')`
  - `DIST_FEATURES: tuple[str, ...]` = `('pos_bigrams','rhyme_endings')`
  - `line_stats(text) -> dict` with int keys `tokens, lines, stanzas, stanza_lines`
  - `pos_bigrams(pos_seq: list[str]) -> Counter[str]` (keys `'NOUN>VERB'`)
  - `rhyme_endings(text) -> Counter[str]`
  - `poem_features(tokens: list[tuple[str,str]], text: str) -> dict` (scalar floats + the two Counters)
  - `normalise(counts) -> dict[str, float]`, `dist_cosine(counts, dist) -> float`

- [ ] **Step 1: Snapshot the current profile CSV**

```bash
cp data/author_style_profiles.csv /private/tmp/claude-502/-Users-Aleks-Documents-Programming-Python-NLP/60b53d13-c3a0-44c3-9970-6ac90a540361/scratchpad/author_style_profiles.before.csv
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_style_features.py`:

```python
from collections import Counter

import pytest

import style_features as sf

POEM = 'сонце грее над гора\nптица пее\n\nреката тече\nкон морето сина'


def test_line_stats_counts_lines_stanzas_and_whitespace_tokens():
    assert sf.line_stats(POEM) == {'tokens': 11, 'lines': 4, 'stanzas': 2, 'stanza_lines': 4}


def test_poem_features_scalars():
    tokens = [('сонце', 'NOUN'), ('грее', 'VERB'), ('гора', 'NOUN'), ('птица', 'NOUN'),
              ('пее', 'VERB'), ('река', 'NOUN'), ('тече', 'VERB'), ('море', 'NOUN'),
              ('син', 'ADJ')]
    f = sf.poem_features(tokens, POEM)
    assert f['pct_NOUN'] == pytest.approx(5 / 9)
    assert f['pct_VERB'] == pytest.approx(3 / 9)
    assert f['pct_ADJ'] == pytest.approx(1 / 9)
    assert f['pct_ADV'] == 0.0 and f['pct_PROPN'] == 0.0
    assert f['lexical_diversity'] == 1.0          # 9 distinct lemmas, below MATTR window -> TTR
    assert f['avg_tokens_per_line'] == 2.75
    assert f['num_lines'] == 4.0 and f['num_stanzas'] == 2.0
    assert f['avg_lines_per_stanza'] == 2.0
    assert f['pos_bigrams']['NOUN>VERB'] == 3
    assert sum(f['rhyme_endings'].values()) == 4


def test_poem_features_on_empty_input_is_all_zero_not_an_error():
    f = sf.poem_features([], '')
    assert all(f[k] == 0.0 for k in sf.SCALAR_FEATURES)
    assert f['pos_bigrams'] == Counter() and f['rhyme_endings'] == Counter()


def test_dist_cosine():
    assert sf.dist_cosine(Counter({'a': 2}), {'a': 1.0}) == pytest.approx(1.0)
    assert sf.dist_cosine(Counter({'a': 1}), {'b': 1.0}) == 0.0
    assert sf.dist_cosine(Counter(), {'a': 1.0}) == 0.0


def test_normalise_sums_to_one():
    assert sf.normalise(Counter({'a': 1, 'b': 3})) == {'a': 0.25, 'b': 0.75}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_style_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'style_features'`

- [ ] **Step 4: Create `src/style_features.py`**

Move `CONTENT_POS`, `_VOWELS`, `_EDGE_PUNCT_RE`, `line_final_syllable`, the `MATTR_WINDOW` comment + constant, and `mattr` **verbatim** from `style_profiler.py`, then add:

```python
"""
Per-poem stylometric features, shared by style_profiler.py (author profiles),
build_reward_calibration.py (per-author feature distributions) and
grpo_rewards.py (scoring a generated poem) -- one implementation, so the GRPO
style reward measures exactly what the profiles describe.

style_profiler.py aggregates author-level *totals* (e.g. total NOUN tokens /
total tokens); the reward needs per-poem values, because a single poem is
what gets scored. poem_features() returns those, unrounded.
"""

import math
import re
from collections import Counter

# ... CONTENT_POS, _VOWELS, _EDGE_PUNCT_RE, line_final_syllable,
#     MATTR_WINDOW (+ its comment), mattr -- moved verbatim ...

POS_PCT = ('NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN')
SCALAR_FEATURES = tuple(f'pct_{p}' for p in POS_PCT) + (
    'lexical_diversity', 'avg_tokens_per_line', 'num_lines', 'num_stanzas',
    'avg_lines_per_stanza')
DIST_FEATURES = ('pos_bigrams', 'rhyme_endings')


def line_stats(text: str) -> dict:
    """Whitespace-token, line and blank-line-stanza counts -- the exact
    conventions style_profiler.py's line-level features use."""
    lines = [l for l in text.splitlines() if l.strip()]
    stanzas = [s for s in text.strip().split('\n\n') if s.strip()]
    return {
        'tokens': sum(len(l.split()) for l in lines),
        'lines': len(lines),
        'stanzas': len(stanzas),
        'stanza_lines': sum(len([l for l in s.splitlines() if l.strip()]) for s in stanzas),
    }


def pos_bigrams(pos_seq: list) -> Counter:
    return Counter(f'{a}>{b}' for a, b in zip(pos_seq, pos_seq[1:]))


def rhyme_endings(text: str) -> Counter:
    counts = Counter()
    for line in text.splitlines():
        ending = line_final_syllable(line)
        if ending:
            counts[ending] += 1
    return counts


def poem_features(tokens: list, text: str) -> dict:
    """
    tokens: [(lemma, pos), ...] content tokens of the poem, selected and
    normalised exactly as pos_tagged.csv holds them (tagging.tag_poem for
    generated text). text: the raw poem.
    """
    total = len(tokens)
    pos_counts = Counter(p for _, p in tokens)
    stats = line_stats(text)
    feats = {f'pct_{p}': (pos_counts[p] / total if total else 0.0) for p in POS_PCT}
    feats['lexical_diversity'] = float(mattr([l for l, _ in tokens]))
    feats['avg_tokens_per_line'] = stats['tokens'] / stats['lines'] if stats['lines'] else 0.0
    feats['num_lines'] = float(stats['lines'])
    feats['num_stanzas'] = float(stats['stanzas'])
    feats['avg_lines_per_stanza'] = (stats['stanza_lines'] / stats['stanzas']
                                     if stats['stanzas'] else 0.0)
    feats['pos_bigrams'] = pos_bigrams([p for _, p in tokens])
    feats['rhyme_endings'] = rhyme_endings(text)
    return feats


def normalise(counts) -> dict:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()} if total else {}


def dist_cosine(counts, dist: dict) -> float:
    """Cosine between a count vector and a (normalised) distribution; 0.0
    when either side is empty."""
    dot = sum(v * dist.get(k, 0.0) for k, v in counts.items())
    na = math.sqrt(sum(v * v for v in counts.values()))
    nb = math.sqrt(sum(v * v for v in dist.values()))
    return dot / (na * nb) if na and nb else 0.0
```

- [ ] **Step 5: Refactor `src/style_profiler.py`**

1. Delete the moved definitions; replace with
   `from style_features import CONTENT_POS, MATTR_WINDOW, line_final_syllable, line_stats, mattr  # noqa: F401 (re-exported)`.
2. Move everything from `# Load raw songs` to the final `print` loop into `def main():` and add `if __name__ == '__main__': main()`.
3. Replace the per-poem body of the line-stats loop with:

```python
    for poem in poems:
        s = line_stats(poem)
        author_line_stats[author]['total_tokens'] += s['tokens']
        author_line_stats[author]['total_lines'] += s['lines']
        author_line_stats[author]['total_poems'] += 1
        author_line_stats[author]['total_stanzas'] += s['stanzas']
        author_line_stats[author]['total_stanza_lines'] += s['stanza_lines']
```

Leave all aggregation, rounding and CSV writing untouched.

- [ ] **Step 6: Run tests and the byte-identity check**

```bash
uv run pytest tests/test_style_features.py -v
uv run python src/style_profiler.py
cmp data/author_style_profiles.csv /private/tmp/claude-502/-Users-Aleks-Documents-Programming-Python-NLP/60b53d13-c3a0-44c3-9970-6ac90a540361/scratchpad/author_style_profiles.before.csv && echo IDENTICAL
```
Expected: tests PASS, then `IDENTICAL`. Any diff is a refactor bug — fix it; do not commit a changed profile CSV.

- [ ] **Step 7: Commit**

```bash
git add src/style_features.py src/style_profiler.py tests/test_style_features.py
git commit -m "Extract per-poem style features into style_features.py

style_profiler.py output verified byte-identical.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `content_similarity.py` and `style_metrics.copy_novelty`

**Files:**
- Create: `src/content_similarity.py`
- Modify: `src/style_metrics.py` (add `COPY_OVERLAP_THRESHOLD`, `copy_novelty`)
- Test: `tests/test_content_similarity.py`, `tests/test_style_metrics.py` (create if absent; append otherwise)

**Interfaces:**
- Produces:
  - `content_similarity.DEFAULT_CONTENT_MODEL = 'paraphrase-multilingual-mpnet-base-v2'`
  - `class LineEmbedder(model_name=DEFAULT_CONTENT_MODEL, device=None)` with `embed_lines(lines: list[str], cache: bool) -> np.ndarray` (shape `(n, d)`, L2-normalised, `(0, d)` for no lines)
  - `line_f1(src_emb: np.ndarray, gen_emb: np.ndarray) -> tuple[float, float, float]` → `(precision, recall, f1)`
  - `style_metrics.copy_novelty(source_text, text, threshold=COPY_OVERLAP_THRESHOLD) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_content_similarity.py`:

```python
import numpy as np
import pytest

import content_similarity as cs


def _unit(*rows):
    a = np.array(rows, dtype=float)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def test_line_f1_identical_is_one():
    e = _unit([1, 0], [0, 1])
    assert cs.line_f1(e, e) == pytest.approx((1.0, 1.0, 1.0))


def test_line_f1_partial_coverage_lowers_recall_not_precision():
    src = _unit([1, 0], [0, 1])
    gen = _unit([1, 0])
    p, r, f1 = cs.line_f1(src, gen)
    assert p == pytest.approx(1.0)
    assert r == pytest.approx(0.5)
    assert f1 == pytest.approx(2 / 3)


def test_line_f1_empty_side_is_zero():
    assert cs.line_f1(np.zeros((0, 2)), _unit([1, 0])) == (0.0, 0.0, 0.0)


@pytest.mark.slow
def test_line_embedder_caches_only_when_asked():
    emb = cs.LineEmbedder()
    out = emb.embed_lines(['сонце грее', 'птица пее'], cache=True)
    assert out.shape[0] == 2
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-4)
    assert set(emb._cache) == {'сонце грее', 'птица пее'}
    emb.embed_lines(['нов ред'], cache=False)
    assert 'нов ред' not in emb._cache
    assert emb.embed_lines([], cache=False).shape[0] == 0
```

Create `tests/test_style_metrics.py` (it does not exist yet):

```python
import style_metrics as sm

SRC = 'сонце грее над темната гора\nптица пее во зелената шума'


def test_copy_novelty_verbatim_copy_is_zero():
    assert sm.copy_novelty(SRC, SRC) == 0.0


def test_copy_novelty_one_word_changed_per_line_still_counts_as_copy():
    near = 'сонце грее над темната ноќ\nптица пее во зелената ноќ'
    assert sm.copy_novelty(SRC, near) == 0.0


def test_copy_novelty_new_lines_are_novel():
    assert sm.copy_novelty(SRC, 'ветер носи лисје низ полето\nмајка чека пред портата') == 1.0


def test_copy_novelty_short_lines_need_an_exact_match():
    assert sm.copy_novelty('сонце грее', 'сонце гори') == 1.0
    assert sm.copy_novelty('сонце грее', 'Сонце, грее!') == 0.0


def test_copy_novelty_empty_generation_is_zero():
    assert sm.copy_novelty(SRC, '') == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_content_similarity.py tests/test_style_metrics.py -v`
Expected: FAIL (`No module named 'content_similarity'`, `no attribute 'copy_novelty'`)

- [ ] **Step 3: Implement `src/content_similarity.py`**

```python
"""
Line-level semantic similarity between a source poem and a generated poem
(GRPO content reward, src/grpo_rewards.py; baseline in
src/build_reward_calibration.py).

BERTScore-style over lines rather than one whole-poem embedding: the
multilingual mpnet model truncates at 128 tokens (many poems are longer), and
a poem that only covers its first stanza's content should not score full
recall. Embedding model is frozen; source-line embeddings are cached by text
(each source recurs G times per GRPO step and across steps), generated lines
are not (they rarely repeat and would grow the cache without bound).
"""

import numpy as np

DEFAULT_CONTENT_MODEL = 'paraphrase-multilingual-mpnet-base-v2'


class LineEmbedder:
    def __init__(self, model_name: str = DEFAULT_CONTENT_MODEL, device: str = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._cache: dict = {}

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self._model.eval()
        return self._model

    def _encode(self, lines: list) -> np.ndarray:
        return np.asarray(self._load().encode(
            lines, normalize_embeddings=True, batch_size=64, show_progress_bar=False),
            dtype=np.float32)

    def embed_lines(self, lines: list, cache: bool) -> np.ndarray:
        if not lines:
            dim = self._load().get_sentence_embedding_dimension()
            return np.zeros((0, dim), dtype=np.float32)
        if not cache:
            return self._encode(lines)
        missing = [l for l in dict.fromkeys(lines) if l not in self._cache]
        if missing:
            for line, vec in zip(missing, self._encode(missing)):
                self._cache[line] = vec
        return np.stack([self._cache[l] for l in lines])


def line_f1(src_emb: np.ndarray, gen_emb: np.ndarray) -> tuple:
    """(precision, recall, f1): recall = mean over source lines of their best
    match among generated lines; precision the reverse."""
    if len(src_emb) == 0 or len(gen_emb) == 0:
        return 0.0, 0.0, 0.0
    sim = src_emb @ gen_emb.T
    recall = float(sim.max(axis=1).mean())
    precision = float(sim.max(axis=0).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1
```

- [ ] **Step 4: Add `copy_novelty` to `src/style_metrics.py`**

Directly after `_normalize_line`:

```python
# A generated line is a copy when at least this share of its word types
# occurs in a single source line. Containment, not Jaccard: with this
# corpus's ~5-token lines one swapped word gives Jaccard 4/6 = 0.67, which a
# Jaccard threshold of 0.7 would wave through as "novel".
COPY_OVERLAP_THRESHOLD = 0.75
_COPY_MIN_TOKENS = 3


def copy_novelty(source_text: str, text: str,
                 threshold: float = COPY_OVERLAP_THRESHOLD) -> float:
    """
    Share of generated lines that are NOT (near-)copies of a source line --
    the GRPO content reward's copy gate. line_novelty (exact match after
    normalisation) is kept unchanged for the reported SFT numbers; this is
    its stricter sibling, catching the one-or-two-words-changed copy that an
    embedding content reward would otherwise pay out on.

    Lines under _COPY_MIN_TOKENS tokens count as copies only on an exact
    match: containment on a two-word line is too coarse to mean anything.
    Empty generations score 0.0 (nothing novel was written).
    """
    gen_lines = [tokens(l) for l in lines(text)]
    gen_lines = [g for g in gen_lines if g]
    if not gen_lines:
        return 0.0
    src_sets = [set(tokens(l)) for l in lines(source_text)]
    src_exact = {' '.join(tokens(l)) for l in lines(source_text)}
    copied = 0
    for g in gen_lines:
        if len(g) < _COPY_MIN_TOKENS:
            copied += ' '.join(g) in src_exact
            continue
        gset = set(g)
        if any(len(gset & s) / len(gset) >= threshold for s in src_sets):
            copied += 1
    return round(1.0 - copied / len(gen_lines), 4)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_content_similarity.py tests/test_style_metrics.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/content_similarity.py src/style_metrics.py tests/test_content_similarity.py tests/test_style_metrics.py
git commit -m "Add line-level embedding F1 and the copy_novelty gate metric

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `build_reward_calibration.py` → `data/reward_calibration.json`

**Files:**
- Create: `src/build_reward_calibration.py`
- Create (generated, tracked): `data/reward_calibration.json`
- Test: `tests/test_build_reward_calibration.py`

**Interfaces:**
- Consumes: `style_features.poem_features/normalise/dist_cosine/SCALAR_FEATURES/DIST_FEATURES`, `content_similarity.LineEmbedder/line_f1`, `make_splits.poems_in_split`, `sft_data.load_profile_rows`, `style_metrics.style_match/load_author_tfidf/lines`.
- Produces: `CALIBRATION_PATH`; `author_calibration(poem_feats: list[dict], hit_rates: list[float], palette_size: int) -> dict`; `content_baseline(poems, embedder, n_pairs, seed) -> float`; `build_calibration(seed=42, n_content_pairs=2000, embedder=None) -> dict`. JSON schema:

```json
{
  "meta": {"seed": 42, "split": "train", "splits_sha256": "...", "content_model": "...",
           "n_train_poems": 990, "n_content_pairs": 2000},
  "corpus_std": {"pct_NOUN": 0.1, "...": 0.0},
  "content_baseline": 0.31,
  "authors": {
    "<author>": {
      "n_poems": 140,
      "scalar": {"pct_NOUN": {"mean": 0.5, "std": 0.08}, "...": {}},
      "dist": {"pos_bigrams": {"dist": {"NOUN>VERB": 0.1}, "cos_mean": 0.9, "cos_std": 0.05},
               "rhyme_endings": {"dist": {}, "cos_mean": 0.3, "cos_std": 0.1}},
      "tfidf_hit_p90": 0.1
    }
  }
}
```

- [ ] **Step 1: Write the failing tests**

`tests/test_build_reward_calibration.py`:

```python
from collections import Counter

import numpy as np
import pytest

import build_reward_calibration as brc
import style_features as sf


def _feats(noun, bigrams):
    f = {k: 0.0 for k in sf.SCALAR_FEATURES}
    f['pct_NOUN'] = noun
    f['pos_bigrams'] = Counter(bigrams)
    f['rhyme_endings'] = Counter({'ла': 1})
    return f


def test_author_calibration_scalar_mean_std_and_leave_one_out_cosine():
    poems = [_feats(0.4, {'NOUN>VERB': 2}), _feats(0.6, {'NOUN>VERB': 1, 'ADJ>NOUN': 1})]
    cal = brc.author_calibration(poems, hit_rates=[0.0, 0.1], palette_size=20)
    assert cal['n_poems'] == 2
    assert cal['scalar']['pct_NOUN']['mean'] == pytest.approx(0.5)
    assert cal['scalar']['pct_NOUN']['std'] == pytest.approx(0.1)
    assert cal['dist']['pos_bigrams']['dist'] == pytest.approx({'NOUN>VERB': 0.75, 'ADJ>NOUN': 0.25})
    # leave-one-out: poem 1 vs poem 2's dist and vice versa, never itself
    c1 = sf.dist_cosine(Counter({'NOUN>VERB': 2}), {'NOUN>VERB': 0.5, 'ADJ>NOUN': 0.5})
    c2 = sf.dist_cosine(Counter({'NOUN>VERB': 1, 'ADJ>NOUN': 1}), {'NOUN>VERB': 1.0})
    assert cal['dist']['pos_bigrams']['cos_mean'] == pytest.approx((c1 + c2) / 2)


def test_tfidf_p90_is_floored_at_one_palette_word():
    cal = brc.author_calibration([_feats(0.5, {}), _feats(0.5, {})],
                                 hit_rates=[0.0, 0.0], palette_size=20)
    assert cal['tfidf_hit_p90'] == pytest.approx(1 / 20)


class _OneHotEmbedder:
    """Identical lines -> cosine 1, different lines -> 0."""
    def __init__(self):
        self.vocab = {}

    def embed_lines(self, lines, cache):
        idx = [self.vocab.setdefault(l, len(self.vocab)) for l in lines]
        out = np.zeros((len(lines), 64))
        out[np.arange(len(lines)), idx] = 1.0
        return out


def test_content_baseline_only_pairs_different_authors():
    poems = [{'author': 'А', 'song_text': 'ист ред'},
             {'author': 'А', 'song_text': 'ист ред'},
             {'author': 'Б', 'song_text': 'друг ред'}]
    # same-author pairs would score 1.0; cross-author pairs score 0.0
    assert brc.content_baseline(poems, _OneHotEmbedder(), n_pairs=20, seed=1) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_build_reward_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/build_reward_calibration.py`**

```python
"""
Reward calibration for the GRPO stage: per-author distributions of the
per-poem style features (style_features.poem_features) over TRAIN-split real
poems, plus the content-similarity baseline. grpo_rewards.py scores a
generated poem against these numbers.

Train split only, unlike author_style_profiles.csv (all poems): GRPO must not
see val/test poems in any form, and the reward is part of training.

Tags come from pos_tagged.csv (the corpus build), not a fresh classla run --
grpo_rewards tags generated text with tagging.tag_poem, which reproduces that
build, so both sides of the comparison are tagged the same way.

Deterministic given --seed. Writes data/reward_calibration.json (tracked).

Usage:
    uv run python src/build_reward_calibration.py
"""

import argparse
import csv
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import style_features as sf
import style_metrics as sm
from content_similarity import DEFAULT_CONTENT_MODEL, LineEmbedder, line_f1
from make_splits import SPLITS_PATH, poems_in_split
from sft_data import load_profile_rows

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_PATH = DATA / 'reward_calibration.json'
SEED = 42
N_CONTENT_PAIRS = 2000

log = logging.getLogger('build_reward_calibration')


def load_train_tokens(train_ids: set) -> dict:
    """poem_id -> [(lemma, pos), ...] from pos_tagged.csv, train poems only."""
    tokens = defaultdict(list)
    with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['poem_id'] in train_ids:
                tokens[r['poem_id']].append((r['lemma'], r['pos']))
    return tokens


def author_calibration(poem_feats: list, hit_rates: list, palette_size: int) -> dict:
    scalar = {}
    for f in sf.SCALAR_FEATURES:
        values = np.array([p[f] for p in poem_feats], dtype=float)
        scalar[f] = {'mean': float(values.mean()), 'std': float(values.std())}

    dist = {}
    for f in sf.DIST_FEATURES:
        total = Counter()
        for p in poem_feats:
            total.update(p[f])
        cosines = []
        for p in poem_feats:
            # leave-one-out: a poem's similarity to the rest of its author,
            # not to an aggregate that already contains it
            rest = total - p[f] if len(poem_feats) > 1 else total
            cosines.append(sf.dist_cosine(p[f], sf.normalise(rest)))
        dist[f] = {'dist': sf.normalise(total),
                   'cos_mean': float(np.mean(cosines)), 'cos_std': float(np.std(cosines))}

    floor = 1.0 / palette_size if palette_size else 1.0
    p90 = float(np.percentile(hit_rates, 90)) if hit_rates else 0.0
    return {'n_poems': len(poem_feats), 'scalar': scalar, 'dist': dist,
            'tfidf_hit_p90': max(p90, floor)}


def content_baseline(poems: list, embedder, n_pairs: int, seed: int) -> float:
    """Mean line-F1 between random pairs of poems by different authors: what
    'about nothing in particular' scores, subtracted out by the reward."""
    rng = random.Random(seed)
    f1s = []
    attempts = 0
    while len(f1s) < n_pairs and attempts < n_pairs * 50:
        attempts += 1
        a, b = rng.sample(poems, 2)
        if a['author'] == b['author']:
            continue
        ea = embedder.embed_lines(sm.lines(a['song_text']), cache=True)
        eb = embedder.embed_lines(sm.lines(b['song_text']), cache=True)
        f1s.append(line_f1(ea, eb)[2])
    return float(np.mean(f1s)) if f1s else 0.0


def _round(obj):
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    return obj


def build_calibration(seed: int = SEED, n_content_pairs: int = N_CONTENT_PAIRS,
                      embedder=None, content_model: str = DEFAULT_CONTENT_MODEL) -> dict:
    train = poems_in_split('train')
    tokens = load_train_tokens({p['poem_id'] for p in train})
    feats = {p['poem_id']: sf.poem_features(tokens.get(p['poem_id'], []), p['song_text'])
             for p in train}

    corpus_std = {f: float(np.std([feats[pid][f] for pid in feats])) for f in sf.SCALAR_FEATURES}

    by_author = defaultdict(list)
    for p in train:
        by_author[p['author']].append(p)

    palettes = sm.load_author_tfidf()
    authors = {}
    for author in sorted(r['author'] for r in load_profile_rows()):
        poems = by_author.get(author, [])
        if not poems:
            log.warning('no train poems for profiled author %s -- skipped', author)
            continue
        hits = [sm.style_match(p['song_text'], author)['tfidf_hit_rate'] for p in poems]
        authors[author] = author_calibration(
            [feats[p['poem_id']] for p in poems], hits, len(palettes.get(author, ())))

    embedder = embedder or LineEmbedder(content_model)
    baseline = content_baseline(train, embedder, n_content_pairs, seed)

    return _round({
        'meta': {'seed': seed, 'split': 'train',
                 'splits_sha256': hashlib.sha256(SPLITS_PATH.read_bytes()).hexdigest(),
                 'content_model': content_model, 'n_train_poems': len(train),
                 'n_content_pairs': n_content_pairs},
        'corpus_std': corpus_std,
        'content_baseline': baseline,
        'authors': authors,
    })


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--n-content-pairs', type=int, default=N_CONTENT_PAIRS)
    parser.add_argument('--content-model', default=DEFAULT_CONTENT_MODEL)
    parser.add_argument('--output', type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()
    cal = build_calibration(args.seed, args.n_content_pairs, content_model=args.content_model)
    args.output.write_text(json.dumps(cal, ensure_ascii=False, indent=1), encoding='utf-8')
    log.info('wrote %s: %d authors, content baseline %.3f', args.output,
             len(cal['authors']), cal['content_baseline'])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_build_reward_calibration.py -v`
Expected: all PASS.

- [ ] **Step 5: Build the calibration and sanity-check it**

```bash
uv run python src/build_reward_calibration.py
uv run python -c "
import json; c=json.load(open('data/reward_calibration.json'))
print('baseline', c['content_baseline'], 'authors', len(c['authors']))
a=c['authors']['Блаже Конески']; print(a['n_poems'], a['scalar']['pct_NOUN'], a['dist']['pos_bigrams']['cos_mean'], a['tfidf_hit_p90'])"
```
Expected: 30-ish authors; content baseline roughly 0.2–0.45; Конески `n_poems` ≈ 145, `pct_NOUN.mean` ≈ 0.5, `cos_mean` ≈ 0.8–0.95. Run the build twice and `cmp` the outputs to confirm determinism. Values far outside these ranges → stop and report.

- [ ] **Step 6: Commit**

```bash
git add src/build_reward_calibration.py tests/test_build_reward_calibration.py data/reward_calibration.json
git commit -m "Add train-split reward calibration for the GRPO style/content rewards

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `grpo_rewards.py` — style, content, gates, combination, TRL reward functions

**Files:**
- Create: `src/grpo_rewards.py`
- Test: `tests/test_grpo_rewards.py`

**Interfaces:**
- Consumes: Task 1 `tagging.tag_poem`, `tagging.load_verb_lemmas`; Task 2 `style_features`; Task 3 `LineEmbedder`, `line_f1`, `sm.copy_novelty`; Task 4 `CALIBRATION_PATH` + schema.
- Produces:
  - `STYLE_GROUPS: dict[str, tuple[str, ...]]`, `DEFAULT_STYLE_WEIGHTS: dict[str, float]`
  - `@dataclass RewardConfig(style_weights, style_weight=0.5, content_weight=0.5, copy_overlap_threshold=0.75, sigma_floor_frac=0.5, content_model=DEFAULT_CONTENT_MODEL, calibration_path=CALIBRATION_PATH, components_log_path=None)`
  - `score_batch(sources: list[str], generations: list[str], targets: list[str], cfg: RewardConfig = None, tagger=None, embedder=None) -> list[dict]`
  - `score(source: str, generated: str, target: str, cfg=None, tagger=None, embedder=None) -> dict`
  - `make_reward_funcs(cfg=None, tagger=None, embedder=None) -> tuple[Callable, Callable]` — functions named `style_reward`, `content_reward`
  - Row keys: `R`, `R_style`, `R_content`, `R_style_g`, `R_content_g`, `group/<g>`, `score/<f>`, `z/<f>`, `raw/<f>`, `content/precision|recall|f1`, `gate/validity|is_cyrillic|no_markup|is_nonempty|line_uniqueness|copy_novelty`

- [ ] **Step 1: Write the failing tests**

`tests/test_grpo_rewards.py`:

```python
import json
import math
from collections import Counter

import numpy as np
import pytest

import grpo_rewards as gr
import style_features as sf
import style_metrics as sm

AUTHOR = 'Автор'


def _calibration(tmp_path):
    scalar = {f: {'mean': 0.0, 'std': 1.0} for f in sf.SCALAR_FEATURES}
    scalar['pct_NOUN'] = {'mean': 0.5, 'std': 0.1}
    scalar['num_lines'] = {'mean': 2.0, 'std': 1.0}
    cal = {
        'meta': {}, 'content_baseline': 0.2,
        'corpus_std': {f: 1.0 for f in sf.SCALAR_FEATURES},
        'authors': {AUTHOR: {
            'n_poems': 10, 'scalar': scalar, 'tfidf_hit_p90': 0.1,
            'dist': {'pos_bigrams': {'dist': {'NOUN>VERB': 1.0}, 'cos_mean': 0.8, 'cos_std': 0.1},
                     'rhyme_endings': {'dist': {'ла': 1.0}, 'cos_mean': 0.5, 'cos_std': 0.2}}}},
    }
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps(cal, ensure_ascii=False), encoding='utf-8')
    return path


class OneHotEmbedder:
    def __init__(self):
        self.vocab = {}

    def embed_lines(self, lines, cache):
        idx = [self.vocab.setdefault(l, len(self.vocab)) for l in lines]
        out = np.zeros((len(lines), 64))
        out[np.arange(len(lines)), idx] = 1.0
        return out


def fake_tagger(text):
    return [(w, 'NOUN' if i % 2 == 0 else 'VERB') for i, w in enumerate(sm.tokens(text))]


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, 'style_match', lambda text, author: {
        'clf_target_prob': 0.5, 'clf_top1': 0.0, 'clf_predicted': '', 'tfidf_hit_rate': 0.05})
    gr._CACHE.clear()
    return gr.RewardConfig(calibration_path=_calibration(tmp_path))


SRC = 'сонце грее над гора\nптица пее во шума'
NEW = 'ветер носи лисје низ поле\nмајка чека пред порта'


def _score(cfg, gen, src=SRC):
    return gr.score(src, gen, AUTHOR, cfg, tagger=fake_tagger, embedder=OneHotEmbedder())


def test_gaussian_score_floors_sigma():
    z, s = gr.gaussian_score(1.0, mean=0.0, std=0.0, corpus_std=1.0, floor_frac=0.5)
    assert z == pytest.approx(2.0)
    assert s == pytest.approx(math.exp(-2.0))


def test_one_sided_score_is_one_at_or_above_mean():
    assert gr.one_sided_score(0.9, mean=0.8, std=0.1) == 1.0
    assert gr.one_sided_score(0.7, mean=0.8, std=0.1) == pytest.approx(math.exp(-0.5))


def test_content_score_rescales_and_clips():
    assert gr.content_score(0.2, baseline=0.2) == 0.0
    assert gr.content_score(0.6, baseline=0.2) == pytest.approx(0.5)
    assert gr.content_score(0.1, baseline=0.2) == 0.0


def test_all_components_in_unit_interval(cfg):
    row = _score(cfg, NEW)
    for key, value in row.items():
        if key.startswith(('R', 'group/', 'score/', 'gate/', 'content/')):
            assert 0.0 <= value <= 1.0, key


def test_verbatim_copy_has_zero_gated_content_but_high_raw_content(cfg):
    row = _score(cfg, SRC)
    assert row['R_content'] == 1.0
    assert row['gate/copy_novelty'] == 0.0
    assert row['R_content_g'] == 0.0


def test_combination_uses_weights(cfg):
    row = _score(cfg, NEW)
    assert row['R'] == pytest.approx(0.5 * row['R_style_g'] + 0.5 * row['R_content_g'])


def test_style_weight_zero_removes_a_group(cfg):
    cfg.style_weights = {g: 0.0 for g in gr.STYLE_GROUPS} | {'classifier': 1.0}
    assert _score(cfg, NEW)['R_style'] == pytest.approx(0.5)


def test_unknown_style_group_rejected():
    with pytest.raises(ValueError):
        gr.RewardConfig(style_weights={'no_such_group': 1.0})


# Review Focus 1
def test_unknown_target_author_raises_with_its_name(cfg):
    with pytest.raises(KeyError, match='Непознат'):
        gr.score_batch([SRC], [NEW], ['Непознат'], cfg, tagger=fake_tagger,
                       embedder=OneHotEmbedder())


# Review Focus 2
@pytest.mark.parametrize('gen', ['', '   \n\n  ', '**Poem**\n- a line', 'Here is the poem in English'])
def test_invalid_generations_score_zero_without_crashing(cfg, gen):
    row = _score(cfg, gen)
    assert row['gate/validity'] == 0.0
    assert row['R'] == 0.0


def test_score_batch_matches_score(cfg):
    emb = OneHotEmbedder()
    batch = gr.score_batch([SRC, SRC], [NEW, SRC], [AUTHOR, AUTHOR], cfg,
                           tagger=fake_tagger, embedder=emb)
    single = gr.score(SRC, SRC, AUTHOR, cfg, tagger=fake_tagger, embedder=emb)
    assert batch[1]['R'] == pytest.approx(single['R'])


def test_reward_funcs_score_once_per_batch_and_log(cfg, tmp_path):
    calls = []

    def counting_tagger(text):
        calls.append(text)
        return fake_tagger(text)

    cfg.components_log_path = tmp_path / 'components.jsonl'
    style_fn, content_fn = gr.make_reward_funcs(cfg, tagger=counting_tagger,
                                                embedder=OneHotEmbedder())
    assert style_fn.__name__ == 'style_reward' and content_fn.__name__ == 'content_reward'
    logged = {}
    kwargs = dict(prompts=['p', 'p'], completions=[NEW, SRC], source_text=[SRC, SRC],
                  target_author=[AUTHOR, AUTHOR], source_poem_id=['1', '1'],
                  log_metric=lambda k, v: logged.__setitem__(k, v), trainer_state=None)
    s = style_fn(**kwargs)
    c = content_fn(**kwargs)
    assert len(s) == len(c) == 2
    assert len(calls) == 2                       # classla once per completion, not per reward func
    assert c[1] == 0.0                            # the copy
    assert 'gate/copy_novelty' in logged and 'group/pos_profile' in logged
    lines = (tmp_path / 'components.jsonl').read_text(encoding='utf-8').splitlines()
    assert len(lines) == 2 and json.loads(lines[0])['target_author'] == AUTHOR
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grpo_rewards.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'grpo_rewards'`

- [ ] **Step 3: Implement `src/grpo_rewards.py`**

```python
"""
GRPO rewards for the Qwen style-transfer pipeline (stage 2, after SFT).

For one generated poem, given the source poem and the target author:

- style: the poem's per-poem features (style_features.poem_features -- the
  same definitions as author_style_profiles.csv) scored against the target
  author's per-poem distribution over TRAIN poems (reward_calibration.json),
  plus the stylometric classifier's probability and TF-IDF palette uptake;
  weighted mean over feature groups.
- content: line-level embedding F1 against the SOURCE poem (never the Gemma
  output), rescaled so unrelated poems score 0.
- gates (multiplicative, style_metrics): validity = cyrillic * no markup *
  non-empty on both; line_uniqueness on style; copy_novelty on content.

R = style_weight * R_style_g + content_weight * R_content_g. Every component
is returned (and, during training, logged) separately. See
docs/superpowers/specs/2026-09-27-grpo-stage-design.md §4.

Inputs are only (generated poem, source poem, target author) -- no reference
answer -- so there is no path for the Gemma output or any val/test poem to
leak into the reward.
"""

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import style_features as sf
import style_metrics as sm
from content_similarity import DEFAULT_CONTENT_MODEL, LineEmbedder, line_f1

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
CALIBRATION_PATH = DATA / 'reward_calibration.json'

STYLE_GROUPS = {
    'pos_profile': ('pct_NOUN', 'pct_VERB', 'pct_ADJ', 'pct_ADV', 'pct_PROPN'),
    'lexical_diversity': ('lexical_diversity',),
    'structure': ('avg_tokens_per_line', 'num_lines', 'num_stanzas', 'avg_lines_per_stanza'),
    'pos_bigrams': ('pos_bigrams',),
    'classifier': ('clf_target_prob',),
    'rhyme': ('rhyme_endings',),
    'tfidf': ('tfidf',),
}
DEFAULT_STYLE_WEIGHTS = {'pos_profile': 1.0, 'lexical_diversity': 1.0, 'structure': 1.0,
                         'pos_bigrams': 1.0, 'classifier': 1.0, 'rhyme': 0.5, 'tfidf': 0.5}

# Prefixes pushed to the trainer's logged metrics each step (raw/z/score go
# to the JSONL only -- dozens of per-feature curves would bury the plot).
_LOGGED_PREFIXES = ('R', 'group/', 'gate/', 'content/')


@dataclass
class RewardConfig:
    style_weights: dict = field(default_factory=lambda: dict(DEFAULT_STYLE_WEIGHTS))
    style_weight: float = 0.5
    content_weight: float = 0.5
    copy_overlap_threshold: float = sm.COPY_OVERLAP_THRESHOLD
    sigma_floor_frac: float = 0.5
    content_model: str = DEFAULT_CONTENT_MODEL
    calibration_path: Path = CALIBRATION_PATH
    components_log_path: Path = None

    def __post_init__(self):
        unknown = set(self.style_weights) - set(STYLE_GROUPS)
        if unknown:
            raise ValueError(f'unknown style group(s) {sorted(unknown)}; '
                             f'known: {sorted(STYLE_GROUPS)}')
        if sum(self.style_weights.values()) <= 0:
            raise ValueError('style_weights must have a positive total')


_CACHE: dict = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def load_calibration(path: Path) -> dict:
    return _cached(('calibration', str(path)),
                   lambda: json.loads(Path(path).read_text(encoding='utf-8')))


def _default_tagger():
    import tagging

    def build():
        with open(DATA / 'pos_tagged.csv', encoding='utf-8') as f:
            verb_lemmas = tagging.load_verb_lemmas(csv.DictReader(f))
        return lambda text: tagging.tag_poem(text, verb_lemmas)
    return _cached('tagger', build)


def _default_embedder(model_name: str) -> LineEmbedder:
    return _cached(('embedder', model_name), lambda: LineEmbedder(model_name))


# ── Scoring primitives ─────────────────────────────────────────────────────────

def gaussian_score(x, mean, std, corpus_std, floor_frac) -> tuple:
    """(z, exp(-z^2/2)); sigma floored at floor_frac * corpus-wide per-poem std
    so a few-poem or unusually uniform author doesn't zero every small miss."""
    sigma = max(std, floor_frac * corpus_std, 1e-6)
    z = (x - mean) / sigma
    return z, math.exp(-0.5 * z * z)


def one_sided_score(c, mean, std) -> float:
    """Distribution similarity: 1 at or above the author's own typical
    self-similarity, Gaussian decay below it."""
    if c >= mean:
        return 1.0
    z = (c - mean) / max(std, 1e-6)
    return math.exp(-0.5 * z * z)


def content_score(f1: float, baseline: float) -> float:
    return min(1.0, max(0.0, (f1 - baseline) / (1.0 - baseline)))


def style_components(features: dict, clf_prob: float, tfidf_hit: float,
                     author_cal: dict, corpus_std: dict, cfg: RewardConfig) -> dict:
    out, scores = {}, {}
    for f in sf.SCALAR_FEATURES:
        st = author_cal['scalar'][f]
        z, s = gaussian_score(features[f], st['mean'], st['std'], corpus_std[f],
                              cfg.sigma_floor_frac)
        out[f'raw/{f}'], out[f'z/{f}'], scores[f] = features[f], z, s
    for f in sf.DIST_FEATURES:
        d = author_cal['dist'][f]
        c = sf.dist_cosine(features[f], d['dist'])
        out[f'raw/{f}'] = c
        scores[f] = one_sided_score(c, d['cos_mean'], d['cos_std'])
    out['raw/clf_target_prob'] = scores['clf_target_prob'] = clf_prob
    out['raw/tfidf'] = tfidf_hit
    scores['tfidf'] = min(1.0, tfidf_hit / author_cal['tfidf_hit_p90'])
    for f, s in scores.items():
        out[f'score/{f}'] = s

    num = den = 0.0
    for group, members in STYLE_GROUPS.items():
        g = sum(scores[m] for m in members) / len(members)
        out[f'group/{group}'] = g
        w = cfg.style_weights.get(group, 0.0)
        num, den = num + w * g, den + w
    out['R_style'] = num / den
    return out


def gates(source: str, text: str, cfg: RewardConfig) -> dict:
    lang = sm.language_metrics(text)
    return {
        'gate/is_cyrillic': lang['is_cyrillic'],
        'gate/no_markup': lang['no_markup'],
        'gate/is_nonempty': lang['is_nonempty'],
        'gate/validity': lang['is_cyrillic'] * lang['no_markup'] * lang['is_nonempty'],
        'gate/line_uniqueness': sm.repetition_metrics(text)['line_uniqueness'],
        'gate/copy_novelty': sm.copy_novelty(source, text, cfg.copy_overlap_threshold),
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def score_batch(sources: list, generations: list, targets: list, cfg: RewardConfig = None,
                tagger=None, embedder=None) -> list:
    cfg = cfg or RewardConfig()
    cal = load_calibration(cfg.calibration_path)
    missing = sorted({t for t in targets if t not in cal['authors']})
    if missing:
        raise KeyError(f'no reward calibration for target author(s) {missing} -- '
                       f'rebuild {cfg.calibration_path.name} or pick profiled authors')
    tagger = tagger or _default_tagger()
    embedder = embedder or _default_embedder(cfg.content_model)

    rows = []
    for src, gen, tgt in zip(sources, generations, targets, strict=True):
        feats = sf.poem_features(tagger(gen) if sm.lines(gen) else [], gen)
        match = sm.style_match(gen, tgt)
        style = style_components(feats, match['clf_target_prob'], match['tfidf_hit_rate'],
                                 cal['authors'][tgt], cal['corpus_std'], cfg)
        p, r, f1 = line_f1(embedder.embed_lines(sm.lines(src), cache=True),
                           embedder.embed_lines(sm.lines(gen), cache=False))
        g = gates(src, gen, cfg)
        r_content = content_score(f1, cal['content_baseline'])
        r_style_g = g['gate/validity'] * g['gate/line_uniqueness'] * style['R_style']
        r_content_g = g['gate/validity'] * g['gate/copy_novelty'] * r_content
        rows.append({
            'target_author': tgt, **style, **g,
            'content/precision': p, 'content/recall': r, 'content/f1': f1,
            'R_content': r_content, 'R_style_g': r_style_g, 'R_content_g': r_content_g,
            'R': cfg.style_weight * r_style_g + cfg.content_weight * r_content_g,
        })
    return rows


def score(source: str, generated: str, target: str, cfg: RewardConfig = None,
          tagger=None, embedder=None) -> dict:
    return score_batch([source], [generated], [target], cfg, tagger, embedder)[0]


def _completion_text(completion) -> str:
    """TRL hands plain strings for string prompts, message lists for
    conversational ones; grpo_data uses string prompts."""
    if isinstance(completion, str):
        return completion
    return completion[-1]['content']


def make_reward_funcs(cfg: RewardConfig = None, tagger=None, embedder=None) -> tuple:
    """
    (style_reward, content_reward) with TRL's reward-function signature.
    Both read one memoised score_batch per generation batch, so classla and
    the embedder run once per completion per step, not once per function.
    Dataset columns source_text / target_author / source_poem_id arrive as
    keyword lists; log_metric / trainer_state are TRL's.
    """
    cfg = cfg or RewardConfig()
    memo = {}

    def _rows(completions, source_text, target_author, kwargs):
        texts = [_completion_text(c) for c in completions]
        key = (tuple(texts), tuple(source_text), tuple(target_author))
        if memo.get('key') != key:
            memo.update(key=key, logged=False,
                        rows=score_batch(list(source_text), texts, list(target_author),
                                         cfg, tagger, embedder))
        if not memo['logged']:
            memo['logged'] = True
            _log(memo['rows'], texts, kwargs)
        return memo['rows']

    def _log(rows, texts, kwargs):
        log_metric = kwargs.get('log_metric')
        numeric = [k for k, v in rows[0].items() if isinstance(v, (int, float))]
        if log_metric:
            for k in numeric:
                if k.startswith(_LOGGED_PREFIXES):
                    log_metric(k, sum(r[k] for r in rows) / len(rows))
        if cfg.components_log_path:
            state = kwargs.get('trainer_state')
            step = state.global_step if state is not None else None
            ids = kwargs.get('source_poem_id') or [None] * len(rows)
            path = Path(cfg.components_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'a', encoding='utf-8') as f:
                for row, text, pid in zip(rows, texts, ids):
                    record = {'step': step, 'source_poem_id': pid,
                              'target_author': row['target_author'], 'completion': text,
                              **{k: float(row[k]) for k in numeric}}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

    def style_reward(prompts, completions, source_text, target_author, **kwargs):
        return [r['R_style_g'] for r in _rows(completions, source_text, target_author, kwargs)]

    def content_reward(prompts, completions, source_text, target_author, **kwargs):
        return [r['R_content_g'] for r in _rows(completions, source_text, target_author, kwargs)]

    return style_reward, content_reward
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grpo_rewards.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/grpo_rewards.py tests/test_grpo_rewards.py
git commit -m "Add GRPO style/content rewards with multiplicative gates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Reward fixtures and `evaluate_reward.py` (single + `--fixtures`)

**Files:**
- Create: `tests/fixtures/reward_cases.json`
- Create: `src/evaluate_reward.py`
- Test: `tests/test_evaluate_reward.py`

**Interfaces:**
- Consumes: `grpo_rewards.score_batch/RewardConfig/STYLE_GROUPS`, `make_splits.load_stripped_songs`, `style_features.SCALAR_FEATURES/DIST_FEATURES`.
- Produces: `evaluate_reward.FIXTURES_PATH`, `build_candidates(case: dict) -> dict[str, str]` (keys `real_target`, `verbatim_source`, `near_copy`, `gemma`, `repetition`, `markdown_english`, `empty`), `check_expectations(scores: dict[str, dict]) -> list[str]` (failure messages), `format_report(row: dict, cfg) -> str`, `main(argv=None) -> int`.

- [ ] **Step 1: Create the fixture file**

`tests/fixtures/reward_cases.json` (train-split sources, target ≠ source author, Gemma row present in `synthetic_dataset.csv`; picked by a seeded scan during planning):

```json
[
  {"source_poem_id": "301", "source_author": "Веле Смилевски", "target_author": "Адем Гајтани", "target_poem_id": "212"},
  {"source_poem_id": "935", "source_author": "Ефтим Клетников", "target_author": "Влада Урошевиќ", "target_poem_id": "3"},
  {"source_poem_id": "689", "source_author": "Блаже Конески", "target_author": "Братислав Ташковски", "target_poem_id": "1036"},
  {"source_poem_id": "42", "source_author": "Богомил Ѓузел", "target_author": "Пијан Славеј", "target_poem_id": "718"}
]
```

- [ ] **Step 2: Write the failing tests**

`tests/test_evaluate_reward.py`:

```python
import json

import pytest

import evaluate_reward as er


def test_candidates_cover_every_kind():
    case = json.loads(er.FIXTURES_PATH.read_text(encoding='utf-8'))[0]
    cands = er.build_candidates(case)
    assert set(cands) == {'real_target', 'verbatim_source', 'near_copy', 'gemma',
                          'repetition', 'markdown_english', 'empty'}
    assert cands['empty'] == ''
    assert cands['near_copy'] != cands['verbatim_source']


def test_check_expectations_reports_violations():
    ok = {'R_content_g': 0.0, 'R_style': 0.4, 'R_content': 0.9, 'R': 0.2}
    scores = {
        'verbatim_source': dict(ok),
        'near_copy': dict(ok),
        'real_target': {'R_content_g': 0.1, 'R_style': 0.3, 'R_content': 0.1, 'R': 0.2},  # style too low
        'gemma': {'R_content_g': 0.3, 'R_style': 0.5, 'R_content': 0.3, 'R': 0.4},
        'repetition': {'R_content_g': 0.0, 'R_style': 0.1, 'R_content': 0.0, 'R': 0.05},
        'markdown_english': {'R_content_g': 0.0, 'R_style': 0.0, 'R_content': 0.0, 'R': 0.0},
        'empty': {'R_content_g': 0.0, 'R_style': 0.0, 'R_content': 0.0, 'R': 0.0},
    }
    failures = er.check_expectations(scores)
    assert len(failures) == 1 and 'real_target' in failures[0]


@pytest.mark.slow
def test_fixtures_pass_with_the_real_reward():
    """The point of the whole reward-validation step: real classla, real
    embedder, real calibration. A failure here is a finding to report, not a
    threshold to loosen."""
    assert er.main(['--fixtures']) == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluate_reward.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `src/evaluate_reward.py`**

```python
"""
Standalone GRPO reward inspector -- run it before any long GRPO job to check
that the reward behaves sensibly.

Single generation:
    uv run python src/evaluate_reward.py --source-poem-id 301 \\
        --target-author "Адем Гајтани" --generated-file poem.txt
    uv run python src/evaluate_reward.py \\
        --generations-csv data/validation_generations/qwen3-lora-sft_val.csv --row 0

Hand-built fixture cases (tests/fixtures/reward_cases.json), with ordering
checks; exits 1 if an expectation fails:
    uv run python src/evaluate_reward.py --fixtures

Within-group reward spread from the SFT policy (GPU; see --groups help).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import grpo_rewards as gr
import style_features as sf
from make_splits import load_stripped_songs

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
FIXTURES_PATH = ROOT / 'tests' / 'fixtures' / 'reward_cases.json'
SYNTHETIC_PATH = DATA / 'synthetic' / 'synthetic_dataset.csv'

MARKDOWN_ENGLISH = ('**Here is the poem rewritten in the requested style:**\n\n'
                    '- The sun rises over the mountain\n- A bird sings in the forest')


def _poem(poem_id: str) -> dict:
    return next(r for r in load_stripped_songs() if r['poem_id'] == poem_id)


def _near_copy(text: str) -> str:
    """Source with the last word of every line of >= 4 words replaced -- the
    'changed one word per line' copy the copy gate must still catch."""
    out = []
    for line in text.splitlines():
        words = line.split()
        out.append(' '.join(words[:-1] + ['ноќта']) if len(words) >= 4 else line)
    return '\n'.join(out)


def _gemma(source_poem_id: str, target_author: str) -> str:
    with open(SYNTHETIC_PATH, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['source_poem_id'], r['target_author'], r['sample_index']) == (
                    source_poem_id, target_author, '0'):
                return r['generated_poem']
    raise KeyError(f'no Gemma row for ({source_poem_id}, {target_author})')


def build_candidates(case: dict) -> dict:
    source = _poem(case['source_poem_id'])['song_text']
    first_line = next(l for l in source.splitlines() if l.strip())
    return {
        'real_target': _poem(case['target_poem_id'])['song_text'],
        'verbatim_source': source,
        'near_copy': _near_copy(source),
        'gemma': _gemma(case['source_poem_id'], case['target_author']),
        'repetition': '\n'.join([first_line] * 12),
        'markdown_english': MARKDOWN_ENGLISH,
        'empty': '',
    }


def check_expectations(scores: dict) -> list:
    """Ordering checks from the spec §3.6. Returns human-readable failures."""
    s = scores
    checks = [
        (s['verbatim_source']['R_content_g'] == 0.0, 'verbatim_source: gated content should be 0'),
        (s['near_copy']['R_content_g'] <= 0.3 * max(s['near_copy']['R_content'], 1e-9),
         'near_copy: copy gate should remove >= 70% of its content reward'),
        (s['real_target']['R_style'] > s['verbatim_source']['R_style'],
         'real_target: style should beat verbatim_source (target voice vs source voice)'),
        (s['verbatim_source']['R_content'] > s['real_target']['R_content'],
         'content scorer: raw content of the source should beat an unrelated poem'),
        (all(s[k]['R'] < s['gemma']['R'] for k in ('repetition', 'markdown_english', 'empty')),
         'gemma should out-score repetition / markdown_english / empty'),
        (s['markdown_english']['R'] == 0.0 and s['empty']['R'] == 0.0,
         'markdown_english and empty should be zeroed by the validity gate'),
    ]
    return [msg for ok, msg in checks if not ok]


def format_report(row: dict, cfg: gr.RewardConfig) -> str:
    lines = [
        f"Style reward:   {row['R_style']:.3f}   (gated: {row['R_style_g']:.3f})",
        f"Content reward: {row['R_content']:.3f}   (gated: {row['R_content_g']:.3f})",
        f"Total reward:   {row['R']:.3f}   "
        f"(= {cfg.style_weight} * style_g + {cfg.content_weight} * content_g)",
        '',
        'Gates: ' + '  '.join(f"{k[5:]}={row[k]:.2f}" for k in row if k.startswith('gate/')),
        f"Content: precision={row['content/precision']:.3f} recall={row['content/recall']:.3f} "
        f"f1={row['content/f1']:.3f}",
        '',
        f"{'group':<18}{'weight':>7}{'score':>7}   features (raw -> z -> score)",
    ]
    for group, members in gr.STYLE_GROUPS.items():
        parts = []
        for m in members:
            z = f" z={row[f'z/{m}']:+.2f}" if f'z/{m}' in row else ''
            parts.append(f"{m}={row[f'raw/{m}']:.3f}{z} -> {row[f'score/{m}']:.2f}")
        lines.append(f"{group:<18}{cfg.style_weights.get(group, 0.0):>7.2f}"
                     f"{row[f'group/{group}']:>7.3f}   " + '; '.join(parts))
    return '\n'.join(lines)


def run_fixtures(cfg: gr.RewardConfig) -> int:
    cases = json.loads(FIXTURES_PATH.read_text(encoding='utf-8'))
    failed = 0
    for case in cases:
        cands = build_candidates(case)
        source = _poem(case['source_poem_id'])['song_text']
        rows = gr.score_batch([source] * len(cands), list(cands.values()),
                              [case['target_author']] * len(cands), cfg)
        scores = dict(zip(cands, rows))
        print(f"\n{case['source_author']} (poem {case['source_poem_id']}) -> {case['target_author']}")
        print(f"  {'candidate':<18}{'style':>7}{'style_g':>9}{'content':>9}{'content_g':>11}{'R':>7}")
        for name, r in scores.items():
            print(f"  {name:<18}{r['R_style']:>7.3f}{r['R_style_g']:>9.3f}"
                  f"{r['R_content']:>9.3f}{r['R_content_g']:>11.3f}{r['R']:>7.3f}")
        for msg in check_expectations(scores):
            print(f'  FAIL: {msg}')
            failed += 1
    print(f'\n{failed} expectation(s) failed' if failed else '\nall expectations hold')
    return 1 if failed else 0


def run_single(args, cfg: gr.RewardConfig) -> int:
    if args.generations_csv:
        with open(args.generations_csv, encoding='utf-8') as f:
            row = list(csv.DictReader(f))[args.row]
        source = _poem(row['source_poem_id'])['song_text']
        target, generated = row['target_author'], row['generated_poem']
    else:
        source = (_poem(args.source_poem_id)['song_text'] if args.source_poem_id
                  else Path(args.source_file).read_text(encoding='utf-8'))
        target = args.target_author
        generated = Path(args.generated_file).read_text(encoding='utf-8')
    print(format_report(gr.score(source, generated, target, cfg), cfg))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--fixtures', action='store_true')
    p.add_argument('--source-poem-id')
    p.add_argument('--source-file')
    p.add_argument('--target-author')
    p.add_argument('--generated-file')
    p.add_argument('--generations-csv', type=Path)
    p.add_argument('--row', type=int, default=0)
    p.add_argument('--style-weight', type=float, default=0.5)
    p.add_argument('--content-weight', type=float, default=0.5)
    p.add_argument('--style-weights', type=json.loads, default=None,
                   help='JSON {group: weight}; omitted groups keep their default')
    return p


def config_from_args(args) -> gr.RewardConfig:
    weights = dict(gr.DEFAULT_STYLE_WEIGHTS, **(args.style_weights or {}))
    return gr.RewardConfig(style_weights=weights, style_weight=args.style_weight,
                           content_weight=args.content_weight)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.fixtures:
        return run_fixtures(cfg)
    if not (args.generations_csv or (args.target_author and args.generated_file
                                     and (args.source_poem_id or args.source_file))):
        build_parser().error('give --fixtures, --generations-csv, or source + '
                             '--target-author + --generated-file')
    return run_single(args, cfg)


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 5: Run the fast tests, then the real fixtures**

```bash
uv run pytest tests/test_evaluate_reward.py -v -m "not slow"
uv run python src/evaluate_reward.py --fixtures
uv run python src/evaluate_reward.py --generations-csv data/validation_generations/qwen3-lora-sft_val.csv --row 0
uv run python src/evaluate_reward.py --generations-csv data/validation_generations/base-Qwen3-4B_val.csv --row 0
```
Expected: fast tests PASS; `--fixtures` prints four tables and `all expectations hold`; the two single reports print the full breakdown (the base row, typically a markdown-wrapped copy, should show validity or copy gate at 0). **If any fixture expectation fails, stop and report the table to the user** — do not tune thresholds to make it pass. Then run `uv run pytest tests/test_evaluate_reward.py -v` (includes slow).

- [ ] **Step 6: Commit**

```bash
git add tests/fixtures/reward_cases.json src/evaluate_reward.py tests/test_evaluate_reward.py
git commit -m "Add standalone reward inspector with hand-built fixture cases

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `merge_sft_adapter.py`

**Files:**
- Create: `src/merge_sft_adapter.py`
- Modify: `.gitignore` (add `models/qwen3-sft-merged/`, `models/qwen3-grpo/`)
- Test: `tests/test_merge_sft_adapter.py`

**Interfaces:**
- Consumes: `generate_validation._infer_base_model(adapter_dir) -> str`.
- Produces: `DEFAULT_MERGED_DIR = MODELS / 'qwen3-sft-merged'`, `DEFAULT_SFT_ADAPTER = MODELS / 'qwen3-lora-sft'`, `adapter_sha256(adapter_dir) -> str`, `merge(adapter_dir, output_dir, base_model=None, force=False) -> Path`. Output dir contains `config.json`, weights, tokenizer files, `merge_info.json` = `{"base_model", "adapter_dir", "adapter_sha256"}`.

- [ ] **Step 1: Add gitignore entries**

Append to `.gitignore` after the `models/qwen3-lora-sft/` block:

```gitignore
# GRPO stage (src/merge_sft_adapter.py, src/train_grpo.py): the SFT adapter
# merged into bf16 base weights (GRPO's start policy + KL reference) and the
# GRPO LoRA + checkpoints -- regenerate via slurm/train_grpo.slurm.
models/qwen3-sft-merged/
models/qwen3-grpo/
```

- [ ] **Step 2: Write the failing tests**

`tests/test_merge_sft_adapter.py`:

```python
import json

import pytest
import torch

import merge_sft_adapter as msa

TINY = 'trl-internal-testing/tiny-Qwen3ForCausalLM'


@pytest.fixture
def tiny_adapter(tmp_path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(TINY)
    peft = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=['q_proj'],
                                            task_type='CAUSAL_LM'))
    with torch.no_grad():
        for n, p in peft.named_parameters():
            if 'lora_B' in n:
                p.fill_(0.01)   # non-zero so the merge actually changes weights
    adapter = tmp_path / 'adapter'
    peft.save_pretrained(adapter)
    AutoTokenizer.from_pretrained(TINY).save_pretrained(adapter)
    return adapter, peft


@pytest.mark.slow
def test_merge_matches_the_adapter_model_and_records_provenance(tiny_adapter, tmp_path):
    from transformers import AutoModelForCausalLM
    adapter, peft = tiny_adapter
    out = msa.merge(adapter, tmp_path / 'merged', base_model=TINY)

    info = json.loads((out / 'merge_info.json').read_text())
    assert info['base_model'] == TINY and info['adapter_sha256'] == msa.adapter_sha256(adapter)

    merged = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float32)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(merged(ids).logits, peft(ids).logits.float(), atol=5e-2)


@pytest.mark.slow
def test_merge_is_idempotent_and_refuses_a_different_adapter(tiny_adapter, tmp_path):
    adapter, _ = tiny_adapter
    out = msa.merge(adapter, tmp_path / 'merged', base_model=TINY)
    assert msa.merge(adapter, out, base_model=TINY) == out   # same adapter: no-op
    info = json.loads((out / 'merge_info.json').read_text())
    info['adapter_sha256'] = 'something-else'
    (out / 'merge_info.json').write_text(json.dumps(info))
    with pytest.raises(SystemExit):
        msa.merge(adapter, out, base_model=TINY)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_merge_sft_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `src/merge_sft_adapter.py`**

```python
"""
Fold the SFT LoRA adapter into Qwen3-4B's weights (GRPO stage, step 0).

Why: TRL's GRPOTrainer, given a PEFT model, gets its KL reference by
switching the trainable adapter off. Training the SFT adapter directly would
make that reference the *untuned* base model; merging SFT into the weights
and attaching a fresh LoRA makes both the start policy and the KL reference
exactly the SFT model. (TRL itself refuses a PeftModel + peft_config and
says to merge first.)

Idempotent: re-running with the same adapter is a no-op; a merge from a
different adapter is refused without --force.

Usage:
    uv run python src/merge_sft_adapter.py            # models/qwen3-lora-sft -> models/qwen3-sft-merged
"""

import argparse
import hashlib
import json
import logging
import shutil
from pathlib import Path

import torch

from generate_validation import _infer_base_model

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
DEFAULT_SFT_ADAPTER = MODELS / 'qwen3-lora-sft'
DEFAULT_MERGED_DIR = MODELS / 'qwen3-sft-merged'

log = logging.getLogger('merge_sft_adapter')


def adapter_sha256(adapter_dir: Path) -> str:
    return hashlib.sha256((Path(adapter_dir) / 'adapter_model.safetensors').read_bytes()).hexdigest()


def merge(adapter_dir: Path, output_dir: Path, base_model: str = None, force: bool = False) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir, output_dir = Path(adapter_dir), Path(output_dir)
    digest = adapter_sha256(adapter_dir)
    info_path = output_dir / 'merge_info.json'
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding='utf-8'))
        if info.get('adapter_sha256') == digest:
            log.info('%s already holds a merge of %s -- nothing to do', output_dir, adapter_dir)
            return output_dir
        if not force:
            raise SystemExit(f'{output_dir} holds a merge of a different adapter '
                             f'({info.get("adapter_dir")}); pass --force to replace it')
        shutil.rmtree(output_dir)

    base_model = base_model or _infer_base_model(adapter_dir)
    log.info('merging %s into %s', adapter_dir, base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(output_dir)
    info_path.write_text(json.dumps({
        'base_model': base_model, 'adapter_dir': str(adapter_dir), 'adapter_sha256': digest,
    }, indent=2), encoding='utf-8')
    log.info('merged model saved to %s', output_dir)
    return output_dir


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--adapter', type=Path, default=DEFAULT_SFT_ADAPTER)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_MERGED_DIR)
    parser.add_argument('--base-model', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    merge(args.adapter, args.output_dir, args.base_model, args.force)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_merge_sft_adapter.py -v`
Expected: PASS (downloads the tiny model once).

- [ ] **Step 6: Commit**

```bash
git add .gitignore src/merge_sft_adapter.py tests/test_merge_sft_adapter.py
git commit -m "Add SFT adapter merge: GRPO start policy and KL reference

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: `grpo_data.py` — train-split prompt dataset

**Files:**
- Create: `src/grpo_data.py`
- Test: `tests/test_grpo_data.py`

**Interfaces:**
- Consumes: `make_splits.poems_in_split`, `generate_synthetic.build_pairs`, `sft_data.select_target_authors/render_inference_prompt`.
- Produces: `build_prompt_rows(target_authors=None, num_target_authors=10, seed=42) -> list[dict]` (keys `source_poem_id, source_author, source_text, target_author`, shuffled); `build_grpo_dataset(tokenizer, num_prompts=None, max_prompt_length=1024, target_authors=None, num_target_authors=10, seed=42) -> tuple[datasets.Dataset, dict]` (columns `prompt, source_poem_id, source_author, source_text, target_author`; stats `{'kept', 'dropped_too_long', 'candidates'}`).

- [ ] **Step 1: Write the failing tests**

`tests/test_grpo_data.py`:

```python
import grpo_data as gd
from make_splits import load_splits


class FakeTokenizer:
    """Renders to the user text; 'token' count = whitespace words."""
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False and add_generation_prompt is True
        return messages[1]['content']

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': text.split()}


def test_prompt_rows_are_train_only_and_never_self_transfer():
    splits = load_splits()
    rows = gd.build_prompt_rows()
    assert rows
    assert all(splits[r['source_poem_id']] == 'train' for r in rows)
    assert all(r['source_author'] != r['target_author'] for r in rows)


def test_prompt_rows_are_seeded():
    assert gd.build_prompt_rows(seed=1)[:20] == gd.build_prompt_rows(seed=1)[:20]
    assert gd.build_prompt_rows(seed=1)[:20] != gd.build_prompt_rows(seed=2)[:20]


def test_dataset_columns_and_cap():
    ds, stats = gd.build_grpo_dataset(FakeTokenizer(), num_prompts=5)
    assert len(ds) == 5 and stats['kept'] == 5
    assert set(ds.column_names) == {'prompt', 'source_poem_id', 'source_author',
                                    'source_text', 'target_author'}
    assert ds[0]['source_text'].strip() in ds[0]['prompt']


# Review Focus 3
def test_too_long_prompts_are_dropped_not_truncated():
    ds, stats = gd.build_grpo_dataset(FakeTokenizer(), max_prompt_length=60)
    assert stats['dropped_too_long'] > 0
    for row in ds:
        assert len(row['prompt'].split()) <= 60
        assert row['source_text'].strip() in row['prompt']   # never cut
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_grpo_data.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `src/grpo_data.py`**

```python
"""
GRPO prompt dataset: (train-split source poem, target author) pairs rendered
in the exact chat format SFT trained on (sft_data.render_inference_prompt,
thinking disabled), with the raw source text and target author kept as
columns -- TRL passes extra columns to the reward functions as keyword lists,
and those two are all the reward is allowed to see.

Train split only (make_splits.poems_in_split('train')); the same target
authors as SFT (sft_data.select_target_authors); self-transfer skipped
(generate_synthetic.build_pairs). No Gemma output is involved.

Prompts over max_prompt_length tokens are dropped, never truncated: a cut
source poem would have the content reward compare against text the model
never saw.
"""

import logging
import random

from generate_synthetic import build_pairs
from make_splits import poems_in_split
from sft_data import render_inference_prompt, select_target_authors

DEFAULT_SEED = 42
DEFAULT_MAX_PROMPT_LENGTH = 1024

log = logging.getLogger('grpo_data')


def build_prompt_rows(target_authors: list = None, num_target_authors: int = 10,
                      seed: int = DEFAULT_SEED) -> list:
    authors = select_target_authors(num_target_authors, explicit=target_authors)
    pairs = build_pairs(poems_in_split('train'), authors, samples_per_pair=1)
    random.Random(seed).shuffle(pairs)
    return [{'source_poem_id': p['poem_id'], 'source_author': p['author'],
             'source_text': p['song_text'], 'target_author': t} for p, t, _ in pairs]


def build_grpo_dataset(tokenizer, num_prompts: int = None,
                       max_prompt_length: int = DEFAULT_MAX_PROMPT_LENGTH,
                       target_authors: list = None, num_target_authors: int = 10,
                       seed: int = DEFAULT_SEED) -> tuple:
    from datasets import Dataset

    candidates = build_prompt_rows(target_authors, num_target_authors, seed)
    kept, dropped = [], 0
    for row in candidates:
        prompt = render_inference_prompt(tokenizer, row['source_text'], row['target_author'])
        if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) > max_prompt_length:
            dropped += 1
            continue
        kept.append({'prompt': prompt, **row})
        if num_prompts and len(kept) >= num_prompts:
            break
    stats = {'kept': len(kept), 'dropped_too_long': dropped, 'candidates': len(candidates)}
    log.info('GRPO prompts: %(kept)d kept, %(dropped_too_long)d dropped as too long '
             '(of %(candidates)d train pairs)', stats)
    if not kept:
        raise ValueError('no GRPO prompts fit max_prompt_length')
    return Dataset.from_list(kept), stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_grpo_data.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/grpo_data.py tests/test_grpo_data.py
git commit -m "Add train-split GRPO prompt dataset

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `train_grpo.py` + `--smoke` + local plumbing test

**Files:**
- Create: `src/train_grpo.py`
- Test: `tests/test_train_grpo.py`

**Interfaces:**
- Consumes: Task 5 `RewardConfig`, `make_reward_funcs`, `DEFAULT_STYLE_WEIGHTS`; Task 7 `DEFAULT_MERGED_DIR`; Task 8 `build_grpo_dataset`; `train_sft.DEFAULT_TARGET_MODULES`, `train_sft.build_lora_config`.
- Produces: `parse_args(argv=None) -> Namespace`, `apply_smoke_overrides(args) -> Namespace`, `build_grpo_config(args) -> GRPOConfig`, `completion_logprob(model, tokenizer, prompt, completion) -> float`, `train(args) -> Path`, `DEFAULT_OUTPUT_DIR = MODELS / 'qwen3-grpo'`. Output dir: adapter, tokenizer, `run_config.json`, `reward_components.jsonl`, `checkpoint-*/`.

- [ ] **Step 1: Confirm the GRPOConfig field names in the installed TRL**

```bash
uv run python -c "
import dataclasses; from trl import GRPOConfig as C
need = {'beta','num_generations','max_completion_length','temperature','loss_type','scale_rewards','num_iterations','mask_truncated_completions','reward_weights','log_completions','num_completions_to_print','warmup_steps','use_cpu'}
print(sorted(need - {f.name for f in dataclasses.fields(C)}))"
```
Expected: `[]`. If a name is missing, adapt `build_grpo_config` to the installed field and note it in the commit message.

- [ ] **Step 2: Write the failing tests**

`tests/test_train_grpo.py`:

```python
import json

import pytest

import train_grpo as tg

TINY = 'trl-internal-testing/tiny-Qwen3ForCausalLM'


def test_defaults_match_the_spec():
    a = tg.parse_args([])
    assert (a.num_generations, a.per_device_train_batch_size, a.gradient_accumulation_steps) == (4, 8, 4)
    assert a.learning_rate == 1e-5 and a.beta == 0.04 and a.temperature == 0.9
    assert (a.lora_r, a.lora_alpha, a.lora_dropout) == (16, 32, 0.0)
    assert (a.style_weight, a.content_weight) == (0.5, 0.5)
    assert a.max_steps == 500 and a.seed == 42 and a.max_prompt_length == 1024


def test_grpo_config_carries_reward_weights_and_masks_truncation(tmp_path):
    a = tg.parse_args(['--output-dir', str(tmp_path), '--style-weight', '0.7',
                       '--content-weight', '0.3', '--no-bf16'])
    c = tg.build_grpo_config(a)
    assert c.reward_weights == [0.7, 0.3]
    assert c.mask_truncated_completions is True      # Review Focus: truncated poems get no credit
    assert c.beta == 0.04 and c.num_generations == 4


def test_smoke_overrides_are_tiny():
    a = tg.apply_smoke_overrides(tg.parse_args(['--smoke']))
    assert a.max_steps == 3 and a.num_prompts == 4 and a.max_completion_length == 128
    assert (a.per_device_train_batch_size * a.gradient_accumulation_steps) % a.num_generations == 0


def test_style_weights_json_merges_with_defaults():
    a = tg.parse_args(['--style-weights', '{"rhyme": 0.0}'])
    assert tg.reward_config(a).style_weights['rhyme'] == 0.0
    assert tg.reward_config(a).style_weights['pos_profile'] == 1.0


def test_missing_merge_info_is_refused(tmp_path):
    with pytest.raises(SystemExit, match='merge_info'):
        tg.load_policy(tmp_path, 'sdpa')


@pytest.mark.slow
def test_plumbing_end_to_end_on_a_tiny_model(tmp_path, monkeypatch):
    """Wiring only: dataset columns reach the reward functions, TRL steps,
    the adapter + run_config are saved. Rewards are stubbed; the real smoke
    test (slurm/grpo_smoke.slurm) runs the real ones on the real model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    merged = tmp_path / 'merged'
    AutoModelForCausalLM.from_pretrained(TINY).save_pretrained(merged)
    AutoTokenizer.from_pretrained(TINY).save_pretrained(merged)
    (merged / 'merge_info.json').write_text(json.dumps({'adapter_dir': 'x', 'adapter_sha256': 'y'}))

    seen = {}

    def fake_make_reward_funcs(cfg=None, **_):
        def style_reward(prompts, completions, source_text, target_author, **kw):
            seen['n'] = len(completions)
            seen['cols'] = (len(source_text), len(target_author))
            return [float(len(c) % 3) for c in completions]

        def content_reward(prompts, completions, source_text, target_author, **kw):
            return [0.5] * len(completions)
        return style_reward, content_reward

    monkeypatch.setattr(tg, 'make_reward_funcs', fake_make_reward_funcs)
    out = tmp_path / 'out'
    tg.train(tg.parse_args([
        '--sft-merged', str(merged), '--output-dir', str(out), '--max-steps', '2',
        '--num-prompts', '4', '--num-generations', '2', '--per-device-train-batch-size', '4',
        '--gradient-accumulation-steps', '1', '--max-completion-length', '8',
        '--no-bf16', '--no-gradient-checkpointing', '--save-steps', '100', '--use-cpu']))

    assert (out / 'adapter_config.json').exists()
    cfg = json.loads((out / 'run_config.json').read_text())
    assert cfg['args']['num_generations'] == 2 and 'calibration_sha256' in cfg
    assert seen['cols'] == (seen['n'], seen['n'])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_train_grpo.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement `src/train_grpo.py`**

```python
"""
GRPO fine-tuning of the SFT Qwen3-4B style-transfer model (stage 2).

Start policy = the SFT adapter merged into the base weights
(src/merge_sft_adapter.py -> models/qwen3-sft-merged), plus a FRESH LoRA.
With beta > 0, TRL's KL reference is the same model with the adapter off --
i.e. exactly the SFT model. The base weights stay frozen.

Prompts: train-split (source poem, target author) pairs (src/grpo_data.py),
in the SFT chat format. Rewards: src/grpo_rewards.py's style_reward and
content_reward, combined by TRL as reward_weights=[style_weight,
content_weight]. Every component is logged via TRL's log_metric and written
per completion to <output-dir>/reward_components.jsonl.

--smoke runs 3 tiny steps and asserts the loop works end to end (see
run_smoke_checks). No evaluation inside training: checkpoints are evaluated
afterwards with generate_validation.py (slurm/eval_grpo.slurm).

Usage:
    uv run python src/train_grpo.py --smoke
    uv run python src/train_grpo.py --max-steps 500
"""

import argparse
import hashlib
import json
import logging
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

import grpo_rewards as gr
from grpo_data import DEFAULT_MAX_PROMPT_LENGTH, build_grpo_dataset
from grpo_rewards import make_reward_funcs
from merge_sft_adapter import DEFAULT_MERGED_DIR
from sft_data import DEFAULT_MODEL_NAME
from train_sft import DEFAULT_TARGET_MODULES, build_lora_config

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / 'models'
DEFAULT_OUTPUT_DIR = MODELS / 'qwen3-grpo'

log = logging.getLogger('train_grpo')


# ── Model ────────────────────────────────────────────────────────────────────────

def load_policy(sft_merged: Path, attn_implementation: str):
    sft_merged = Path(sft_merged)
    if not (sft_merged / 'merge_info.json').exists():
        raise SystemExit(f'{sft_merged}/merge_info.json missing -- build the start policy with '
                         '`python src/merge_sft_adapter.py` (GRPO must start from SFT, not base)')
    tokenizer = AutoTokenizer.from_pretrained(str(sft_merged))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(sft_merged), dtype=torch.bfloat16, attn_implementation=attn_implementation)
    return model, tokenizer


def completion_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    """Sum of log p(completion | prompt) -- the smoke test's deterministic
    'did the update change the policy' probe."""
    n_prompt = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    ids = tokenizer(prompt + completion, add_special_tokens=False,
                    return_tensors='pt')['input_ids'].to(model.device)
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1].float()
    logp = torch.log_softmax(logits, dim=-1).gather(1, ids[0, 1:, None])[:, 0]
    return logp[n_prompt - 1:].sum().item()


# ── Config ───────────────────────────────────────────────────────────────────────

def reward_config(args) -> gr.RewardConfig:
    return gr.RewardConfig(
        style_weights=dict(gr.DEFAULT_STYLE_WEIGHTS, **(args.style_weights or {})),
        style_weight=args.style_weight, content_weight=args.content_weight,
        components_log_path=Path(args.output_dir) / 'reward_components.jsonl')


def build_grpo_config(args) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        mask_truncated_completions=True,
        temperature=args.temperature,
        beta=args.beta,
        loss_type=args.loss_type,
        scale_rewards=args.scale_rewards,
        num_iterations=args.num_iterations,
        reward_weights=[args.style_weight, args.content_weight],
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to=args.report_to,
        log_completions=True,
        num_completions_to_print=2,
        use_cpu=args.use_cpu,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if Path(path).exists() else None


def _git_sha() -> str:
    """HEAD without calling git (the container has no git binary)."""
    head = ROOT / '.git' / 'HEAD'
    if not head.exists():
        return None
    ref = head.read_text().strip()
    if ref.startswith('ref: '):
        ref_path = ROOT / '.git' / ref[5:]
        if ref_path.exists():
            return ref_path.read_text().strip()
        packed = ROOT / '.git' / 'packed-refs'
        for line in packed.read_text().splitlines() if packed.exists() else []:
            if line.endswith(ref[5:]):
                return line.split()[0]
        return None
    return ref


def write_run_config(args, grpo_config: GRPOConfig, dataset_stats: dict) -> None:
    import peft
    import transformers
    import trl
    merge_info = json.loads((Path(args.sft_merged) / 'merge_info.json').read_text())
    record = {
        'args': vars(args), 'git_sha': _git_sha(), 'merge_info': merge_info,
        'calibration_sha256': _sha256(gr.CALIBRATION_PATH), 'dataset': dataset_stats,
        'versions': {'trl': trl.__version__, 'transformers': transformers.__version__,
                     'peft': peft.__version__, 'torch': torch.__version__},
        'grpo_config': grpo_config.to_dict(),
    }
    out = Path(args.output_dir) / 'run_config.json'
    out.write_text(json.dumps(record, default=str, ensure_ascii=False, indent=2), encoding='utf-8')


# ── Smoke checks ────────────────────────────────────────────────────────────────

def run_smoke_checks(trainer, tokenizer, probe: tuple, before_logp: float) -> list:
    failures = []
    history = trainer.state.log_history
    losses = [h['loss'] for h in history if 'loss' in h]
    if not losses or not all(math.isfinite(l) for l in losses):
        failures.append(f'loss not finite: {losses}')
    stds = [h['reward_std'] for h in history if 'reward_std' in h]
    if not any(s > 0 for s in stds):
        failures.append(f'reward constant within every group (reward_std={stds})')
    rewards = [h['reward'] for h in history if 'reward' in h]
    if not all(math.isfinite(r) for r in rewards):
        failures.append(f'reward not finite: {rewards}')
    lora_b = sum(p.detach().float().norm().item()
                 for n, p in trainer.model.named_parameters() if 'lora_B' in n)
    if lora_b == 0.0:
        failures.append('LoRA B weights still zero -- no update reached the adapter')
    after_logp = completion_logprob(trainer.model, tokenizer, *probe)
    log.info('probe log-prob before=%.4f after=%.4f', before_logp, after_logp)
    if abs(after_logp - before_logp) <= 1e-3:
        failures.append('probe completion log-prob unchanged -- update did not move the policy')
    return failures


def _probe(dataset) -> tuple:
    row = dataset[0]
    first_lines = '\n'.join(l for l in row['source_text'].splitlines() if l.strip())
    return row['prompt'], '\n'.join(first_lines.splitlines()[:3])


# ── Train ────────────────────────────────────────────────────────────────────────

def train(args) -> Path:
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_policy(args.sft_merged, args.attn_implementation)
    dataset, stats = build_grpo_dataset(
        tokenizer, num_prompts=args.num_prompts, max_prompt_length=args.max_prompt_length,
        target_authors=args.target_authors, num_target_authors=args.num_target_authors,
        seed=args.seed)

    style_reward, content_reward = make_reward_funcs(reward_config(args))
    grpo_config = build_grpo_config(args)
    lora = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout, args.target_modules)

    probe = _probe(dataset)
    if args.smoke:
        model.to('cuda' if torch.cuda.is_available() and not args.use_cpu else 'cpu')
        before_logp = completion_logprob(model, tokenizer, *probe)

    trainer = GRPOTrainer(model=model, reward_funcs=[style_reward, content_reward],
                          args=grpo_config, train_dataset=dataset,
                          processing_class=tokenizer, peft_config=lora)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    write_run_config(args, grpo_config, stats)
    log.info('GRPO adapter saved to %s', args.output_dir)

    if args.smoke:
        failures = run_smoke_checks(trainer, tokenizer, probe, before_logp)
        for f in failures:
            log.error('SMOKE FAIL: %s', f)
        if failures:
            sys.exit(1)
        log.info('SMOKE OK: generate -> reward -> GRPO update -> policy changed')
    return args.output_dir


# ── CLI ─────────────────────────────────────────────────────────────────────────

def apply_smoke_overrides(args):
    args.max_steps, args.num_prompts, args.num_generations = 3, 4, 4
    args.per_device_train_batch_size, args.gradient_accumulation_steps = 8, 1
    args.max_completion_length, args.save_steps = 128, 10_000
    args.learning_rate = 1e-4   # large enough that 3 steps visibly move the probe
    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = MODELS / 'qwen3-grpo-smoke'
    return args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sft-merged', type=Path, default=DEFAULT_MERGED_DIR,
                   help='SFT-merged start policy (merge_sft_adapter.py output)')
    p.add_argument('--base-model', default=DEFAULT_MODEL_NAME,
                   help='recorded for provenance; the weights come from --sft-merged')
    p.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument('--smoke', action='store_true')

    d = p.add_argument_group('data')
    d.add_argument('--num-prompts', type=int, default=None)
    d.add_argument('--num-target-authors', type=int, default=10)
    d.add_argument('--target-authors', nargs='+', default=None)
    d.add_argument('--max-prompt-length', type=int, default=DEFAULT_MAX_PROMPT_LENGTH)

    l = p.add_argument_group('LoRA')
    l.add_argument('--lora-r', type=int, default=16)
    l.add_argument('--lora-alpha', type=int, default=32)
    l.add_argument('--lora-dropout', type=float, default=0.0)
    l.add_argument('--target-modules', nargs='+', default=DEFAULT_TARGET_MODULES)

    r = p.add_argument_group('reward')
    r.add_argument('--style-weight', type=float, default=0.5, help='alpha')
    r.add_argument('--content-weight', type=float, default=0.5, help='beta (content)')
    r.add_argument('--style-weights', type=json.loads, default=None,
                   help='JSON {group: weight}; omitted groups keep their default')

    t = p.add_argument_group('GRPO')
    t.add_argument('--learning-rate', type=float, default=1e-5)
    t.add_argument('--warmup-steps', type=int, default=10)
    t.add_argument('--num-generations', type=int, default=4)
    t.add_argument('--per-device-train-batch-size', type=int, default=8)
    t.add_argument('--gradient-accumulation-steps', type=int, default=4)
    t.add_argument('--max-completion-length', type=int, default=512)
    t.add_argument('--temperature', type=float, default=0.9)
    t.add_argument('--beta', type=float, default=0.04, help='KL coefficient (0 disables the reference)')
    t.add_argument('--loss-type', default='dapo')
    t.add_argument('--scale-rewards', default='group')
    t.add_argument('--num-iterations', type=int, default=1)
    t.add_argument('--max-steps', type=int, default=500)
    t.add_argument('--logging-steps', type=int, default=1)
    t.add_argument('--save-steps', type=int, default=50)
    t.add_argument('--save-total-limit', type=int, default=5)
    t.add_argument('--seed', type=int, default=42)
    t.add_argument('--attn-implementation', default='sdpa',
                   choices=['sdpa', 'eager', 'flash_attention_2'])
    t.add_argument('--report-to', default='none')
    t.add_argument('--resume-from-checkpoint', default=None)
    t.add_argument('--use-cpu', action='store_true', help='tests only')

    bf = t.add_mutually_exclusive_group()
    bf.add_argument('--bf16', dest='bf16', action='store_true', default=True)
    bf.add_argument('--no-bf16', dest='bf16', action='store_false')
    gc = t.add_mutually_exclusive_group()
    gc.add_argument('--gradient-checkpointing', dest='gradient_checkpointing',
                    action='store_true', default=True)
    gc.add_argument('--no-gradient-checkpointing', dest='gradient_checkpointing',
                    action='store_false')
    return p.parse_args(argv)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    arguments = parse_args()
    if arguments.smoke:
        arguments = apply_smoke_overrides(arguments)
    train(arguments)
```

- [ ] **Step 5: Run the tests**

```bash
uv run pytest tests/test_train_grpo.py -v -m "not slow"
uv run pytest tests/test_train_grpo.py -v -k plumbing
```
Expected: fast tests PASS; the plumbing test PASSES on CPU (a minute or two). If TRL rejects a CPU/`use_cpu` combination on macOS (MPS auto-selection), set `PYTORCH_ENABLE_MPS_FALLBACK=1` or skip the test on darwin with a reason — record which in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/train_grpo.py tests/test_train_grpo.py
git commit -m "Add GRPO trainer over the SFT-merged policy with a fresh LoRA

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: `evaluate_reward.py --groups` (within-group reward spread)

**Files:**
- Modify: `src/evaluate_reward.py` (add `summarise_groups`, `run_groups`, CLI flags)
- Test: `tests/test_evaluate_reward.py` (append)

**Interfaces:**
- Consumes: `grpo_data.build_prompt_rows`, `generate_validation.load_adapter_model/generate_one`, `merge_sft_adapter.DEFAULT_MERGED_DIR`.
- Produces: `summarise_groups(rows_by_prompt: list[list[dict]], keys: list[str]) -> dict[str, dict]` with `mean`, `between_prompt_std`, `within_group_std` per key; CSV at `--groups-out` (default `data/grpo/reward_groups.csv`, tracked).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_evaluate_reward.py`:

```python
def test_summarise_groups_separates_within_and_between_spread():
    groups = [[{'R': 0.1}, {'R': 0.3}], [{'R': 0.5}, {'R': 0.7}]]
    s = er.summarise_groups(groups, ['R'])['R']
    assert s['mean'] == pytest.approx(0.4)
    assert s['within_group_std'] == pytest.approx(0.1)
    assert s['between_prompt_std'] == pytest.approx(0.2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/test_evaluate_reward.py -k summarise -v`
Expected: FAIL with `AttributeError: ... 'summarise_groups'`

- [ ] **Step 3: Implement**

Add to `src/evaluate_reward.py`:

```python
import numpy as np

GROUP_KEYS = ['R', 'R_style_g', 'R_content_g', 'R_style', 'R_content'] + \
    [f'group/{g}' for g in gr.STYLE_GROUPS] + \
    ['gate/validity', 'gate/line_uniqueness', 'gate/copy_novelty']
DEFAULT_GROUPS_OUT = DATA / 'grpo' / 'reward_groups.csv'


def summarise_groups(rows_by_prompt: list, keys: list) -> dict:
    """Under GRPO's per-group advantage normalisation, what decides which
    component drives the update is its spread *within* a group of samples for
    one prompt, not its mean -- so that is what alpha/beta are set from."""
    out = {}
    for k in keys:
        per_group = [[r[k] for r in rows] for rows in rows_by_prompt]
        out[k] = {
            'mean': float(np.mean([v for g in per_group for v in g])),
            'within_group_std': float(np.mean([np.std(g) for g in per_group])),
            'between_prompt_std': float(np.std([np.mean(g) for g in per_group])),
        }
    return out


def run_groups(args, cfg: gr.RewardConfig) -> int:
    import torch
    import generate_validation as gv
    from grpo_data import build_prompt_rows

    model, tokenizer = gv.load_adapter_model(args.adapter, base_model=str(args.merged_model))
    prompts = build_prompt_rows(seed=args.seed)[:args.num_prompts]
    rows_by_prompt = []
    for i, p in enumerate(prompts):
        gens = []
        for g in range(args.num_generations):
            torch.manual_seed(args.seed + i * args.num_generations + g)
            gens.append(gv.generate_one(model, tokenizer, p['source_text'], p['target_author'],
                                        max_new_tokens=512, temperature=args.temperature))
        rows_by_prompt.append(gr.score_batch([p['source_text']] * len(gens), gens,
                                             [p['target_author']] * len(gens), cfg))
        print(f'{i + 1}/{len(prompts)} poem {p["source_poem_id"]} -> {p["target_author"]}: '
              f'R={[round(r["R"], 3) for r in rows_by_prompt[-1]]}')

    summary = summarise_groups(rows_by_prompt, GROUP_KEYS)
    print(f"\n{'component':<26}{'mean':>8}{'within-grp std':>16}{'between std':>13}")
    for k, s in summary.items():
        print(f"{k:<26}{s['mean']:>8.3f}{s['within_group_std']:>16.3f}{s['between_prompt_std']:>13.3f}")

    args.groups_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.groups_out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['component', 'mean', 'within_group_std', 'between_prompt_std'])
        for k, s in summary.items():
            w.writerow([k, s['mean'], s['within_group_std'], s['between_prompt_std']])
    print(f'\nwrote {args.groups_out}')
    return 0
```

In `build_parser()` add:

```python
    p.add_argument('--groups', action='store_true',
                   help='sample --num-generations completions per train prompt from the '
                        'SFT policy and report each component\'s within-group spread (GPU)')
    p.add_argument('--merged-model', type=Path, default=ROOT / 'models' / 'qwen3-sft-merged')
    p.add_argument('--adapter', type=Path, default=None,
                   help='optional LoRA on top of --merged-model (e.g. a GRPO checkpoint)')
    p.add_argument('--num-prompts', type=int, default=20)
    p.add_argument('--num-generations', type=int, default=4)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--groups-out', type=Path, default=DEFAULT_GROUPS_OUT)
```

In `main()` before the fixtures branch: `if args.groups: return run_groups(args, cfg)`.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_evaluate_reward.py -v -m "not slow"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/evaluate_reward.py tests/test_evaluate_reward.py
git commit -m "Add within-group reward spread report for choosing alpha/beta

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Slurm jobs — smoke, reward groups, full training

**Files:**
- Create: `slurm/grpo_common.sh`, `slurm/grpo_smoke.slurm`, `slurm/reward_groups.slurm`, `slurm/train_grpo.slurm`

**Interfaces:**
- Consumes: `src/merge_sft_adapter.py`, `src/evaluate_reward.py --fixtures/--groups`, `src/train_grpo.py [--smoke]`.
- Produces: `grpo_preflight`, `run_in_container BODY [ARGS...]` shell functions (used again in Task 12).

- [ ] **Step 1: Create `slurm/grpo_common.sh`**

```bash
# Shared by the GRPO Slurm jobs (sourced, not submitted). Same container
# pattern as train_sft.slurm, plus a classla model bind -- the GRPO style
# reward POS-tags every generated poem (tagging.tag_poem).
#
# classla's mk models must be fetched once on a node with internet:
#   uv run python -c "import classla; classla.download('mk')"

SIF="${SIF:-singularity/qwen-sft.sif}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CLASSLA_DIR="${CLASSLA_RESOURCES_DIR:-$HOME/classla_resources}"

grpo_preflight() {
    [[ -f "$SIF" ]] || { echo "Missing $SIF -- build it first: ./singularity/build.sh" >&2; exit 1; }
    [[ -d models/qwen3-lora-sft ]] || { echo "Missing models/qwen3-lora-sft -- run train_sft.slurm first." >&2; exit 1; }
    [[ -d "$CLASSLA_DIR/mk" ]] || { echo "Missing classla mk models in $CLASSLA_DIR -- see slurm/grpo_common.sh" >&2; exit 1; }
    [[ -f data/reward_calibration.json ]] || { echo "Missing data/reward_calibration.json -- run: uv run python src/build_reward_calibration.py" >&2; exit 1; }
    mkdir -p "$HF_CACHE" logs
    echo "Host: $(hostname)"
}

# run_in_container 'shell body using "$@"' [args forwarded to the body...]
run_in_container() {
    local body="$1"; shift
    singularity exec --nv \
        -B "$HF_CACHE:/root/.cache/huggingface" \
        -B "$CLASSLA_DIR:/root/classla_resources" \
        -B "$PWD:/workspace" \
        "$SIF" \
        bash -c "
            set -euo pipefail
            export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
            export HF_HOME=/root/.cache/huggingface CLASSLA_RESOURCES_DIR=/root/classla_resources
            cd /workspace
            nvidia-smi -L || true
            PY=/opt/venv/bin/python
            $body
        " _ "$@"
}
```

- [ ] **Step 2: Create the three job scripts**

`slurm/grpo_smoke.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen3-grpo-smoke
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Minutes-long end-to-end check before any real GRPO run: merge the SFT
# adapter (if not done), score the hand-built reward fixtures, then 3 tiny
# GRPO steps with the real rewards on the real model. Exits non-zero if the
# fixtures' ordering expectations or the smoke assertions fail.
#
# Usage: sbatch slurm/grpo_smoke.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/grpo_smoke.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

run_in_container '
    $PY src/merge_sft_adapter.py
    $PY src/evaluate_reward.py --fixtures
    $PY src/train_grpo.py --smoke "$@"
' "$@"
```

`slurm/reward_groups.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen3-reward-groups
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# G samples per train prompt from the SFT policy, scored by the GRPO reward;
# reports each component's within-group spread (what dominates GRPO's
# group-normalised advantage) -> data/grpo/reward_groups.csv. Run before the
# long job to choose --style-weight / --content-weight.
#
# Usage: sbatch slurm/reward_groups.slurm [--num-prompts 20 --num-generations 4]

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/reward_groups.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

run_in_container '
    $PY src/merge_sft_adapter.py
    $PY src/evaluate_reward.py --groups "$@"
' "$@"
```

`slurm/train_grpo.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen3-grpo
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Full single-A100 GRPO run: SFT-merged policy + fresh LoRA, style/content
# rewards (src/grpo_rewards.py). Output: models/qwen3-grpo/ (adapter,
# run_config.json, reward_components.jsonl, checkpoint-*/). Run
# grpo_smoke.slurm first. Extra args go straight to train_grpo.py.
#
# Usage:
#   sbatch slurm/train_grpo.slurm
#   sbatch slurm/train_grpo.slurm --max-steps 200 --style-weight 0.6 --content-weight 0.4

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/train_grpo.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

run_in_container '
    $PY src/merge_sft_adapter.py
    $PY src/train_grpo.py "$@"
' "$@"
```

- [ ] **Step 3: Syntax-check and verify argument forwarding locally**

```bash
chmod +x slurm/grpo_smoke.slurm slurm/reward_groups.slurm slurm/train_grpo.slurm
for f in slurm/grpo_common.sh slurm/grpo_smoke.slurm slurm/reward_groups.slurm slurm/train_grpo.slurm; do bash -n "$f" && echo "ok $f"; done
bash -c 'source slurm/grpo_common.sh
singularity() { while [[ $1 != bash ]]; do shift; done; "$@"; }   # drop exec flags + image
cd() { :; }; export -f cd                                         # no /workspace on the Mac
run_in_container "echo got: \"\$@\"" --max-steps 7 --style-weight 0.6' 2>/dev/null | grep "got:"
```
Expected: four `ok` lines, then `got: --max-steps 7 --style-weight 0.6` (the stub drops the singularity flags, runs the same `bash -c` body with the forwarded args, and a no-op `cd` stands in for the container's `/workspace`).

- [ ] **Step 4: Commit**

```bash
git add slurm/grpo_common.sh slurm/grpo_smoke.slurm slurm/reward_groups.slurm slurm/train_grpo.slurm
git commit -m "Add Slurm jobs for GRPO smoke test, reward spread and training

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Evaluation — loader fixes, Gemma on val, `compare_generations --rewards`, `eval_grpo.slurm`

**Files:**
- Modify: `src/generate_validation.py` (`_infer_base_model`, `_run_label`)
- Modify: `src/generate_synthetic.py` (`generate(split=...)`, `--split`)
- Modify: `src/merge_synthetic_parts.py` (`paths_for_split`, CLI `paths` subcommand)
- Modify: `scripts/generate_synthetic_parallel.sh` (5th arg `split`)
- Modify: `src/compare_generations.py` (`--rewards`)
- Modify: `.gitignore` (`data/synthetic/parts_val/`, `data/synthetic/teacher_val_errors.csv`)
- Create: `slurm/eval_grpo.slurm`
- Test: `tests/test_generate_validation.py`, `tests/test_generate_synthetic.py`, `tests/test_merge_synthetic_parts.py`, `tests/test_compare_generations.py` (append / create)

**Interfaces:**
- Produces: `merge_synthetic_parts.paths_for_split(split: str) -> dict` with keys `main`, `errors`, `parts_dir`, `error_parts_dir`, `log_dir` (all `Path`); `generate_synthetic.generate(..., split='train')`; `compare_generations.main(..., rewards=False)` and `score_run(run, rewards=False)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_generate_validation.py`:

```python
# Review Focus 4
def test_infer_base_model_resolves_a_container_path_under_models(tmp_path, monkeypatch):
    models = tmp_path / 'models'
    (models / 'qwen3-sft-merged').mkdir(parents=True)
    monkeypatch.setattr(gv, 'MODELS_DIR', models)
    adapter = tmp_path / 'adapter'
    adapter.mkdir()
    (adapter / 'adapter_config.json').write_text(
        json.dumps({'base_model_name_or_path': '/workspace/models/qwen3-sft-merged'}))
    assert gv._infer_base_model(adapter) == str(models / 'qwen3-sft-merged')


def test_infer_base_model_keeps_hub_ids(tmp_path):
    (tmp_path / 'adapter_config.json').write_text(
        json.dumps({'base_model_name_or_path': 'Qwen/Qwen3-4B'}))
    assert gv._infer_base_model(tmp_path) == 'Qwen/Qwen3-4B'


def test_run_label_for_checkpoints_and_local_merged_base(tmp_path):
    ckpt = tmp_path / 'qwen3-grpo' / 'checkpoint-200'
    ckpt.mkdir(parents=True)
    assert gv._run_label(ckpt, None) == 'qwen3-grpo-checkpoint-200'
    merged = tmp_path / 'qwen3-sft-merged'
    merged.mkdir()
    assert gv._run_label(None, str(merged)) == 'qwen3-sft-merged'
    assert gv._run_label(None, 'Qwen/Qwen3-4B') == 'base-Qwen3-4B'
```

Append to `tests/test_merge_synthetic_parts.py`:

```python
# Review Focus 5
def test_val_split_never_routes_to_the_sft_training_file():
    import generate_synthetic as gs
    import merge_synthetic_parts as msp
    val, train = msp.paths_for_split('val'), msp.paths_for_split('train')
    assert train['main'] == gs.OUTPUT_PATH
    assert val['main'].name == 'teacher_val.csv'
    assert val['main'] != gs.OUTPUT_PATH and val['errors'] != gs.ERRORS_PATH
    assert val['parts_dir'] != train['parts_dir']
```

Append to `tests/test_generate_synthetic.py`:

```python
def test_generate_uses_and_records_the_requested_split(monkeypatch, tmp_path):
    import generate_synthetic as gs
    seen = {}

    def fake_sample(n, seed, split='train'):
        seen['split'] = split
        return [{'poem_id': '5', 'author': 'А', 'song_title': 'Н', 'song_text': 'т'}]

    monkeypatch.setattr(gs, 'select_target_authors', lambda n, explicit=None: ['Б'])
    monkeypatch.setattr(gs, 'sample_source_poems', fake_sample)
    monkeypatch.setattr(gs.lst, 'run_transfer', lambda *a, **k: {
        'source_poem_id': '5', 'source_author': 'А', 'source_title': 'Н', 'target_author': 'Б',
        'generated_poem': 'п', 'model': 'm', 'backend': 'b', 'log_path': '', 'timestamp': '',
        'structural_fit': {'delta_lines': 0, 'delta_stanzas': 0, 'delta_tokens_per_line': 0,
                           'delta_lines_per_stanza': 0}})
    out = tmp_path / 'o.csv'
    gs.generate(num_target_authors=1, poems_per_author=1, split='val',
                output_path=out, errors_path=tmp_path / 'e.csv')
    assert seen['split'] == 'val'
    assert 'val' in out.read_text(encoding='utf-8').splitlines()[1].split(',')
```

Create or append `tests/test_compare_generations.py`:

```python
import compare_generations as cg


def test_rewards_flag_adds_reward_columns(monkeypatch):
    import grpo_rewards as gr
    monkeypatch.setattr(cg, '_poems_by_id', lambda: {'1': {'song_text': 'извор ред'}})
    monkeypatch.setattr(gr, 'score_batch', lambda s, g, t, cfg=None: [
        {'R': 0.4, 'R_style': 0.5, 'R_content': 0.3, 'R_style_g': 0.5,
         'R_content_g': 0.3, 'gate/copy_novelty': 1.0} for _ in g])
    # a real profiled author: score_run's structure_fit needs its style profile
    run = {'label': 'x', 'rows': [{'key': ('1', 'Блаже Конески', '0'), 'source_poem_id': '1',
                                   'target_author': 'Блаже Конески', 'split': 'val',
                                   'generated_poem': 'нова песна'}]}
    row = cg.score_run(run, rewards=True)[0]
    assert row['R'] == 0.4 and row['copy_novelty'] == 1.0
    assert cg.score_run(run, rewards=False)[0].get('R') is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/test_generate_validation.py tests/test_merge_synthetic_parts.py tests/test_generate_synthetic.py tests/test_compare_generations.py -v -m "not slow"`
Expected: the new tests FAIL (`MODELS_DIR`, `paths_for_split`, `split` kwarg, `rewards` kwarg missing).

- [ ] **Step 3: `generate_validation.py` changes**

Add `MODELS_DIR = ROOT / 'models'` next to `DEFAULT_ADAPTER_DIR`. Replace `_infer_base_model` and `_run_label`:

```python
def _infer_base_model(adapter_dir: Path) -> str:
    """base_model_name_or_path out of the adapter's own adapter_config.json,
    so --base-model only needs overriding when that record is missing.

    A GRPO adapter records the SFT-merged model as its base -- as the path it
    was trained with, which inside the container is /workspace/models/....
    When that path doesn't exist here, the same directory name under this
    repo's models/ is used instead."""
    config_path = adapter_dir / 'adapter_config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding='utf-8'))
        base = config.get('base_model_name_or_path')
        if base:
            if Path(base).exists():
                return base
            local = MODELS_DIR / Path(base).name
            if local.exists():
                return str(local)
            return base
    log.warning('could not infer base model from %s -- falling back to %s',
               config_path, DEFAULT_MODEL_NAME)
    return DEFAULT_MODEL_NAME


def _run_label(adapter_dir: Path, base_model: str) -> str:
    """Filename stem / `adapter` column value identifying which model produced
    a row: the adapter's directory name (checkpoint-N prefixed with its run),
    a local base model's directory name (the SFT-merged model), or
    base-<model> for a hub base with --no-adapter."""
    if adapter_dir is not None:
        adapter_dir = Path(adapter_dir)
        if adapter_dir.name.startswith('checkpoint-'):
            return f'{adapter_dir.parent.name}-{adapter_dir.name}'
        return adapter_dir.name
    if base_model and Path(base_model).is_dir():
        return Path(base_model).name
    return 'base-' + (base_model or DEFAULT_MODEL_NAME).split('/')[-1]
```

- [ ] **Step 4: `generate_synthetic.py --split`**

Add `split: str = 'train'` to `generate(...)`'s signature; change `sample_source_poems(poems_per_author, seed=seed)` → `sample_source_poems(poems_per_author, seed=seed, split=split)`, and the row's `'split': 'train'` → `'split': split`. In the CLI add

```python
    parser.add_argument('--split', default='train', choices=['train', 'val'],
                       help="source-poem split. 'val' is only for the teacher comparison "
                            "run -- write it to data/synthetic/teacher_val.csv, never to "
                            "synthetic_dataset.csv (train_sft.py reads that file)")
```

and pass `split=args.split` into `generate(...)`. Refuse the dangerous combination at the top of `generate`:

```python
    if split != 'train' and Path(output_path).resolve() == OUTPUT_PATH.resolve():
        raise ValueError(f'split={split!r} rows must not go to {OUTPUT_PATH.name} '
                         '(the SFT training file) -- pass --output-path')
```

- [ ] **Step 5: `merge_synthetic_parts.paths_for_split` + CLI**

Add to `src/merge_synthetic_parts.py`:

```python
def paths_for_split(split: str) -> dict:
    """Where the parallel generator writes for a split. 'val' (the Gemma
    teacher comparison run) gets its own main file and parts dir, so a val row
    can never be merged into synthetic_dataset.csv, which train_sft.py reads."""
    synthetic = OUTPUT_PATH.parent
    if split == 'train':
        main, errors, parts = OUTPUT_PATH, ERRORS_PATH, synthetic / 'parts'
    elif split == 'val':
        main, errors = synthetic / 'teacher_val.csv', synthetic / 'teacher_val_errors.csv'
        parts = synthetic / 'parts_val'
    else:
        raise ValueError(f'no synthetic output location for split {split!r}')
    return {'main': main, 'errors': errors, 'parts_dir': parts,
            'error_parts_dir': parts / 'errors', 'log_dir': parts / 'logs'}
```

and a CLI subcommand next to `seed`/`merge`:

```python
    paths_p = sub.add_parser('paths', help='print KEY=path lines for a split (shell use)')
    paths_p.add_argument('--split', default='train')
```

handled as:

```python
    elif args.command == 'paths':
        for key, path in paths_for_split(args.split).items():
            print(f'{key.upper()}={path.relative_to(ROOT) if path.is_absolute() else path}')
```

(Use the subparser variable name and `ROOT` as they already exist in the file; add `ROOT = Path(__file__).resolve().parent.parent` if absent.)

- [ ] **Step 6: `scripts/generate_synthetic_parallel.sh` split routing**

Change the usage comment to `[poems-per-author] [num-target-authors] [samples-per-pair] [seed] [split]` and add an example `scripts/generate_synthetic_parallel.sh 50 10 1 42 val   # Gemma teacher on the val pairs -> data/synthetic/teacher_val.csv`. Replace the fixed paths with:

```bash
SPLIT="${5:-train}"

eval "$(uv run python src/merge_synthetic_parts.py paths --split "$SPLIT")"
PARTS_DIR="$PARTS_DIR"; ERROR_PARTS_DIR="$ERROR_PARTS_DIR"; LOG_DIR="$LOG_DIR"
echo "Split: $SPLIT -> $MAIN"
mkdir -p "$PARTS_DIR" "$ERROR_PARTS_DIR" "$LOG_DIR"
```

In `merge_all`, add `--output-path "$MAIN"` to the dataset merge and `--output-path "$ERRORS"` to the errors merge. In the loop, pass `--main "$MAIN"` to `merge_synthetic_parts.py seed` and `--split "$SPLIT"` to `generate_synthetic.py`.

- [ ] **Step 7: `compare_generations.py --rewards`**

Add to `SUMMARY_COLUMNS` after `('line_novelty', 'line-novelty')`:

```python
    ('copy_novelty', 'copy-novelty'),
    ('R_style', 'R-style'),
    ('R_content', 'R-content'),
    ('R_style_g', 'R-style-g'),
    ('R_content_g', 'R-content-g'),
    ('R', 'R'),
```

Change `score_run(run)` to `score_run(run, rewards: bool = False)`; after the loop that builds `scored`, add:

```python
    if rewards:
        import grpo_rewards as gr  # classla + sentence-transformers: only when asked
        idx = [i for i, row in enumerate(run['rows']) if sources[i]]
        results = gr.score_batch([sources[i] for i in idx],
                                 [run['rows'][i]['generated_poem'] for i in idx],
                                 [run['rows'][i]['target_author'] for i in idx])
        for i, res in zip(idx, results):
            scored[i].update({k: res[k] for k in
                              ('R', 'R_style', 'R_content', 'R_style_g', 'R_content_g')})
            scored[i]['copy_novelty'] = res['gate/copy_novelty']
```

where `sources` is a list filled inside the existing loop (`sources.append(source_text)` right after `source_text` is resolved; initialise `sources = []` before the loop). Thread `rewards` through `main(..., rewards=False)` → `score_run(run, rewards)`, and add `parser.add_argument('--rewards', action='store_true', help='add GRPO reward columns (loads classla + embedder)')`, passing `rewards=args.rewards`.

- [ ] **Step 8: gitignore + `slurm/eval_grpo.slurm`**

Append to `.gitignore` next to `data/synthetic/parts/`:

```gitignore
data/synthetic/parts_val/
data/synthetic/teacher_val_errors.csv
```

`slurm/eval_grpo.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=qwen3-grpo-eval
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Before/after GRPO on the same 468 val pairs, back to back on one GPU (same
# reasoning as compare_base_lora.slurm): the SFT-merged model (GRPO's start
# policy) then the GRPO adapter, then compare_generations.py --rewards over
# base / SFT / SFT-merged / GRPO / Gemma-on-val (if teacher_val.csv exists).
#
# Usage:
#   sbatch slurm/eval_grpo.slurm
#   ADAPTER=models/qwen3-grpo/checkpoint-200 sbatch slurm/eval_grpo.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/eval_grpo.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

export ADAPTER="${ADAPTER:-models/qwen3-grpo}"
[[ -d "$ADAPTER" ]] || { echo "Missing $ADAPTER -- run train_grpo.slurm first." >&2; exit 1; }
if [[ "$(basename "$ADAPTER")" == checkpoint-* ]]; then
    export LABEL="$(basename "$(dirname "$ADAPTER")")-$(basename "$ADAPTER")"
else
    export LABEL="$(basename "$ADAPTER")"
fi

run_in_container '
    G=data/validation_generations
    $PY src/merge_sft_adapter.py
    $PY src/generate_validation.py --no-adapter --base-model models/qwen3-sft-merged --split val "$@"
    $PY src/generate_validation.py --adapter "$ADAPTER" --split val "$@"
    RUNS=($G/base-Qwen3-4B_val.csv $G/qwen3-lora-sft_val.csv $G/qwen3-sft-merged_val.csv $G/${LABEL}_val.csv)
    LABELS=(base sft sft-merged "$LABEL")
    if [[ -f data/synthetic/teacher_val.csv ]]; then
        RUNS+=(data/synthetic/teacher_val.csv); LABELS+=(gemma)
    fi
    $PY src/compare_generations.py --rewards --corpus-reference \
        --runs "${RUNS[@]}" --labels "${LABELS[@]}" \
        --rows-out $G/grpo_comparison_rows.csv --report-out $G/grpo_comparison_summary.csv
' "$@"
```

- [ ] **Step 9: Run tests and checks**

```bash
chmod +x slurm/eval_grpo.slurm && bash -n slurm/eval_grpo.slurm && bash -n scripts/generate_synthetic_parallel.sh
uv run python src/merge_synthetic_parts.py paths --split val
uv run pytest tests/test_generate_validation.py tests/test_merge_synthetic_parts.py tests/test_generate_synthetic.py tests/test_compare_generations.py -v -m "not slow"
uv run python src/compare_generations.py --rewards --runs data/validation_generations/qwen3-lora-sft_val.csv --labels sft --rows-out /private/tmp/claude-502/-Users-Aleks-Documents-Programming-Python-NLP/60b53d13-c3a0-44c3-9970-6ac90a540361/scratchpad/rows.csv --report-out /private/tmp/claude-502/-Users-Aleks-Documents-Programming-Python-NLP/60b53d13-c3a0-44c3-9970-6ac90a540361/scratchpad/summary.csv
```
Expected: `paths` prints `MAIN=data/synthetic/teacher_val.csv` etc.; tests PASS; the compare run prints the table with filled `R-*` rows for the existing SFT val CSV (the first real, full-scale reward numbers — note them for the report).

- [ ] **Step 10: Commit**

```bash
git add src/generate_validation.py src/generate_synthetic.py src/merge_synthetic_parts.py \
        scripts/generate_synthetic_parallel.sh src/compare_generations.py .gitignore \
        slurm/eval_grpo.slurm tests/
git commit -m "Evaluation for GRPO: merged/checkpoint loading, Gemma on val, reward columns

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Documentation — `CLAUDE.md` and `data/report.tex`

**Files:**
- Modify: `CLAUDE.md` (new section after "Qwen3 LoRA SFT training pipeline"; also fix that section's "reward/GRPO stage is not implemented yet" sentence)
- Modify: `data/report.tex` (new `\section` before `\section{Reproducibility}`; add GRPO artifacts to Reproducibility)
- Modify: `docs/superpowers/specs/2026-09-27-grpo-stage-design.md` (record the planning deviations)

- [ ] **Step 1: `CLAUDE.md`**

In the SFT section's intro, replace "This is stage 1 of a larger plan; the reward/GRPO stage is not implemented yet — `src/style_metrics.py` (step 8) is the piece it would plug into…" with "Stage 2 (GRPO) is described in the next section; `src/style_metrics.py` (step 8) supplies its gate metrics unchanged." Then add a section `## GRPO stage (fourth pipeline, downstream of SFT)` with numbered entries in the SFT section's style covering, in order:

1. `src/style_features.py` — per-poem features shared with `style_profiler.py` (profile output verified byte-identical after the extraction).
2. `src/build_reward_calibration.py` → `data/reward_calibration.json` (tracked, deterministic): train-split-only per-author feature mean/std, leave-one-out distribution self-similarity, TF-IDF p90, content baseline.
3. `src/content_similarity.py` + `src/grpo_rewards.py` — the reward formulas (style z-score Gaussian with σ floor, one-sided distribution scores, groups and default weights; line-level embedding F1 rescaled by the baseline; gates `validity`, `line_uniqueness`, `copy_novelty`; `R = α·R_style_g + β·R_content_g`), and that `tagging.tag_poem` (whole-poem, corpus-equivalent) is what tags generated text.
4. `src/evaluate_reward.py` — single / `--fixtures` / `--groups` modes, with the exact commands.
5. `src/merge_sft_adapter.py` → `models/qwen3-sft-merged/` (gitignored) and **why**: TRL's KL reference is the adapter-off model, so merging makes both the start policy and the reference exactly SFT.
6. `src/grpo_data.py`, `src/train_grpo.py` — defaults table (copy from Global Constraints), `--smoke`, output `models/qwen3-grpo/` (gitignored) with `run_config.json` and `reward_components.jsonl`.
7. Slurm: `grpo_common.sh`, `grpo_smoke.slurm`, `reward_groups.slurm`, `train_grpo.slurm`, `eval_grpo.slurm`; the classla bind and the one-time `classla.download('mk')`.
8. Evaluation: Gemma on val via `scripts/generate_synthetic_parallel.sh 50 10 1 42 val` → `data/synthetic/teacher_val.csv` (never `synthetic_dataset.csv`); `compare_generations.py --rewards`.

- [ ] **Step 2: `data/report.tex`**

Add `\section{Reward-driven fine-tuning (GRPO): reward design and validation}` before `\section{Reproducibility}` with subsections:

- *Why the SFT model is merged before GRPO* (KL reference argument).
- *Style reward* — per-poem features from the profile definitions (state that the refactor left `author_style_profiles.csv` byte-identical, so the profile definitions are unchanged), per-author per-poem calibration on train poems only, the Gaussian/one-sided formulas as display equations, the group table with default weights.
- *Content reward* — line-level embedding F1 against the source poem, baseline rescaling with the measured `content_baseline` value from `reward_calibration.json`.
- *Gates* — motivated by the numbers already in the report's SFT evaluation section (untuned base `line_novelty` 0.22, i.e. copying); `copy_novelty` containment rule and why Jaccard was rejected (one-word edits on ~5-token lines).
- *Validation before training* — the fixture table from `evaluate_reward.py --fixtures` (paste the printed numbers) and the full-scale reward columns for the existing SFT val run from Task 12 Step 9.
- *Known limitations* — classifier fit on the full corpus (val/test included) and TF-IDF palette likewise; Cyrillic-script paraphrase in another language passes the gates; per-poem TTR length dependence.

In `\section{Reproducibility}`, add the GRPO commands (`build_reward_calibration.py`, the four Slurm jobs) and artifacts.

Compile: `cd data && xelatex report.tex && xelatex report.tex` — Expected: PDF builds without errors (Cyrillic via `fontspec`).

- [ ] **Step 3: Record planning deviations in the spec**

Append to the spec a `## 10. Changes made during planning` section listing the six items under "Deviations from the spec" at the top of this plan, verbatim.

- [ ] **Step 4: Full fast test suite**

Run: `uv run pytest -m "not slow" -q`
Expected: all PASS. Then `uv run pytest -q` (includes slow) — all PASS.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md data/report.tex data/report.pdf docs/superpowers/specs/2026-09-27-grpo-stage-design.md
git commit -m "Document the GRPO stage in CLAUDE.md and the thesis report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Commands for the user after implementation (not run automatically)

```bash
# reward smoke test (local, CPU)
uv run python src/evaluate_reward.py --fixtures

# tiny GRPO test (cluster, minutes)
sbatch slurm/grpo_smoke.slurm

# choose alpha/beta from within-group spread (cluster)
sbatch slurm/reward_groups.slurm

# Gemma teacher on the val pairs (local, Gemini API, ~2-3 h)
scripts/generate_synthetic_parallel.sh 50 10 1 42 val

# full GRPO run (cluster, ~10-12 h)
sbatch slurm/train_grpo.slurm

# before/after + three-way comparison (cluster)
sbatch slurm/eval_grpo.slurm
```

Outputs: `models/qwen3-sft-merged/`, `models/qwen3-grpo-smoke/`, `models/qwen3-grpo/{adapter,run_config.json,reward_components.jsonl,checkpoint-*/}`, `data/grpo/reward_groups.csv`, `data/synthetic/teacher_val.csv`, `data/validation_generations/{qwen3-sft-merged,qwen3-grpo}_val.csv`, `data/validation_generations/grpo_comparison_{rows,summary}.csv`.
