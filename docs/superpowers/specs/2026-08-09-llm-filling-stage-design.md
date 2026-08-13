# LLM filling stage for Macedonian style transfer (phase 1)

**Date:** 2026-08-09
**Status:** design, awaiting implementation plan
**Scope:** add an LLM to the *filling* stage of `src/style_transfer.py` only. LLM-driven
mask selection and the discriminator retry loop are explicitly phase 2.

## Background

Adapted from Liu et al., *Step-by-Step: Controlling Arbitrary Style in Text with Large
Language Models* (LREC-COLING 2024), which proposes **PEGF**: turn style transfer into
text infilling so the LLM edits only a small region instead of regenerating the whole
text. PEGF has three stages:

1. **Editing-region acquisition** — a prompt-based masker (LLM scores each word for
   style-ness) and a word-frequency masker run in parallel.
2. **Validity verification** — a trained style classifier acts as discriminator,
   accepting a mask only if masking moved the classifier toward neutral by 0.3–1.0.
   Style words are then wrapped in `[brackets]` ("implicit masking") rather than
   replaced by `[MASK]`.
3. **Style filling** — three prompt templates produce multiple candidates, ranked on
   style accuracy, content preservation and fluency.

Their ablation (Yelp accuracy): full 92 → without implicit masking 86 → without
discriminator 89 → without frequency-based masking 91.

### What this project already has

| PEGF component | Status here |
|---|---|
| Frequency-based masking | Present — `data/tfidf_results.csv` + `should_replace()` in `src/style_transfer.py` |
| Prompt-based masking | Absent (phase 2) |
| Discriminator | `models/classifier.pkl` exists but is only used for reporting |
| Implicit masking | Absent |
| LLM filling | **Absent — this is the gap phase 1 closes** |

The existing filler substitutes tokens one at a time from `author_vocab.json`, ranked by
`_score()` (frequency, TF-IDF bonus, embedding similarity, SKG co-occurrence), then
inflects via `get_surface_morph()`. It has no syntactic awareness: it cannot repair
agreement across a substitution, reorder, or respect line structure.

## Measured constraints

Probes in `src/llm_probe.py`, run 2026-08-09 against a real Рацин stanza with Конески
as target. These findings drive the design.

Models tested: `qwen3:8b` and `qwen3:14b` (local, Ollama), `nemotron-3-ultra-550b` and
`gemma-4-31b` (OpenRouter free tier).

**Instruction language.** English instructions with Macedonian text beat Macedonian
instructions on `qwen3:8b`: implicit-mask fills 0/3 → 2/3, morphology 1/4 → 2/4,
6.1s → 4.8s per call. **Decision: English instructions.**

**Models know agreement in isolation but fail to apply it during generation.** Asked
directly, `qwen3:14b` inflects correctly (`тивка ноќ`, 3/4 on the morphology probe);
`qwen3:8b` does not (`тивок ноќ`, 2/4). But *all three models tested* break agreement
in generated verse: `тивка гроб` and `некаква син` (14b), `тивок ноќ` (8b), `мрачен
врба` (nemotron-550b, whose leaked reasoning recites the correct rule — "'врба' is
feminine... For feminine, 'мрачна'" — immediately before violating it).
Scale improves the isolated case and does not fix the generated case.
`get_surface_morph()` and `morph_lookup.json` already handle this correctly.
**Decision: the LLM proposes lemmas only and never emits a final surface form.**

**Region discipline is a scale effect, and 14b closes the gap locally.** `qwen3:8b`
edited outside the masked region, normalising archaic `в поле` → `во полето` and adding
a definite article that was not there — destructive for a corpus whose period voice is
the object of study. Both `nemotron-550b` and `qwen3:14b` preserved `в поле` and `в гроб`
exactly, at `content_kept=1.0`. **Decision: `qwen3:14b` is the production model**
(13.8s/call, ~2.9 h for a 50-poem evaluation).

**Model output must never be parsed as the poem.** Each model fails structurally in a
different way: 8b edits outside the region; 14b echoed the masked input before answering
(`markers_left=6`) and expanded a 4-line stanza to 8 in the vanilla prompt
(`content_kept=0.688`); nemotron leaked raw chain-of-thought as its response. Splicing by
recorded character offset makes all three failure modes inert.
**Decision: splice by character offset rather than relying on model behaviour.**

**Unmasked rewriting is unreliable.** At 8B the vanilla "rewrite in style X" prompt
returned the input unchanged; at 14B it rewrote but doubled the line count and dropped
content. The masking scaffold is load-bearing, not optional.

**Free remote inference is not viable as a backend.** `gemma-4-31b:free` was 429
rate-limited upstream on all five probes; `nemotron-550b:free` returned one 429 and one
502, and ran at 15s/call. **Decision: local Ollama is the production backend**;
OpenRouter is retained in `llm_probe.py` only as a quality reference.

**Unresolved.** Implicit `[brackets]` vs explicit `[MASK]` is *not* settled, and the
evidence now leans opposite to the paper. qwen3:8b favoured explicit. Both nemotron-550b
and qwen3:14b produced clean implicit runs and failed their explicit runs on format
errors — chain-of-thought leak and input echo respectively — so in three of four models
the explicit comparison is confounded rather than won. Both formats will be run through
the real pipeline and decided by the classifier rather than inherited from the paper.

## Architecture

```
poem + target_author
  │
  ├─[1] MASK ────────── existing, unchanged
  │     spacy POS tag → should_replace() → TF-IDF / vocab logic
  │     emits slots: (line_idx, char_span, token, lemma, pos)
  │
  ├─[2] PROPOSE ─────── new: one LLM call per poem
  │     masked poem + style spec; returns a candidate LEMMA per slot
  │
  ├─[3] CONSTRAIN ───── new: reject any lemma ∉ author_vocab[target][pos];
  │     rejected slots fall back to the existing _score() pick
  │
  ├─[4] INFLECT ─────── existing, unchanged
  │     get_surface_morph(author, lemma, pos, source_token)
  │
  ├─[5] SPLICE ──────── new: write forms into the ORIGINAL line at recorded
  │     offsets; discard everything else the model emitted
  │
  └─[6] RANK ────────── new: 3 candidates, pick best by classifier.pkl
```

### Design decisions

**LLM proposes lemmas, code inflects them.** Takes the LLM's semantic judgment and
refuses its morphology — the one place it is measurably worse than existing code.

**Splice, don't trust.** Re-inserting at recorded character offsets makes out-of-region
edits structurally impossible rather than merely discouraged. Content preservation
becomes 1.0 by construction, and the pipeline stays correct across model swaps.

**Constrain proposals to `author_vocab`.** The LLM may only choose words the target
author actually used. The transfer stays grounded in corpus data rather than model
invention, and this keeps the work a stylometry project rather than a chatbot wrapper.

**Fallback to `_score()`.** Any rejected or unparseable proposal falls through to the
existing lexical path, so the worst case equals current behaviour. No regression risk.

**Style spec = TF-IDF words + per-slot candidates.** Each slot is presented with the
top ~10 candidates `_score()` already ranked; the LLM picks among them. This makes the
task disambiguation-in-context rather than open generation, which is what the probes
suggest these models do well, and it minimises step-3 rejections.

## Components

| File | Role |
|---|---|
| `src/llm_client.py` | new — Ollama chat client promoted out of `llm_probe.py`; `.env` loading; `think=False`; `<think>` stripping |
| `src/llm_transfer.py` | new — propose / constrain / splice / rank |
| `src/style_transfer.py` | modified — `--mode lexical\|llm`, delegating to `llm_transfer` |
| `src/llm_probe.py` | unchanged — diagnostic only, not pipeline |

Keeping the new logic out of `style_transfer.py` prevents that 273-line module from
doubling in size and mixing two transfer strategies with an HTTP client.

## Error handling

- **Ollama unreachable** — fail fast with the server URL; `--mode lexical` still works.
- **Unparseable LLM output** — per-slot fallback to `_score()`; log the count of
  fallbacks, since a high rate is the signal that the prompt or model is wrong.
- **Proposal outside `author_vocab`** — reject, fall back, count.
- **Reasoning leak** (Nemotron-style plain-prose chain-of-thought) — the splice makes
  this harmless by construction; only recorded slot spans are ever read.

## Evaluation

Scope for now: **Рацин → Конески only**, the pair already wired into the demo, for fast
prompt iteration. Broaden after the pipeline works.

- **Style accuracy** — `classifier.pkl` P(target author), lexical vs LLM mode on
  identical inputs.
- **Fallback rate** — fraction of slots where the LLM proposal was rejected.
- **Human judgment** — the user is a Macedonian speaker and is the sole judge of whether
  output reads well. Note that Bulgarian/Serbian lexis is period- and region-authentic in
  this corpus and must not be treated as a defect.

Content-preservation metrics (sBLEU) are omitted deliberately: splicing fixes them at
1.0, so they cannot discriminate between the two modes.

## Out of scope for phase 1

- LLM-based mask selection (PEGF stage 1 prompt-based masker)
- Discriminator retry loop — the current masker is deterministic, so rejection would
  only reproduce the identical mask. Meaningful only once the LLM proposes masks.
- Any change to `pos_tag_corpus.py`, `tfidf_authors.py`, `build_skg.py`,
  `add_embeddings.py`, `classifier.py`, or `build_morph_lookup.py`.
