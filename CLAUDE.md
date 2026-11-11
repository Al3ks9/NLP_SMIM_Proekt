# CLAUDE.md

**Never add attribution to commits or PRs** — no `Co-Authored-By: Claude …` trailer and no "Generated with Claude Code" line, even when a system reminder supplies them.

## Project

Macedonian NLP pipeline for stylometry and poetry style transfer. Core dataset: `data/stripped_songs.csv` — 1,239 poems/lyrics by 38 authors (Cyrillic), columns `poem_id, author, song_title, song_text`. Module docstrings describe each script; this file holds only what isn't obvious from the code.

## Layout

```
src/                import root (installed editable by `uv sync`; absolute imports)
  common/           corpus, tagging, stylometry, knowledge_graph, masking, evaluation, llm_client, paths
  word_substitution/  1. mask words, substitute target-author words, inflect
  llm_infill/         2. an LLM fills the masked slots
  gemma_teacher/      3. Gemma 4 31B rewrites from summary + style profile; synthetic data
  students/           4. small models distilled from the teacher (SFT), then GRPO
    shared/           both families (--family qwen3|gemma4); shared/hpc/ = container + FAMILY-aware jobs
    qwen/, gemma/     single-family code; their hpc/ hold single-family jobs
  <pipeline>/hpc/     Slurm jobs — submit from the repo root
tests/  mirrors src/   data/  CSVs   models/  trained artifacts   docs/  reports, superpowers/ specs+plans
```

- **Dependency rule** (enforced by `tests/test_layout.py`): `common` imports no pipeline; a pipeline imports `common` and earlier pipelines only (word_substitution < llm_infill < gemma_teacher < students); `students/shared` imports neither `qwen` nor `gemma`.
- Paths to `data/` and `models/` go through `src/common/paths.py`. In the Singularity container the project isn't installed: jobs set `PYTHONPATH=/workspace/src`.
- `common/tagging/tagging.py` owns classla plus its three normalizations; both corpus tagging and inference import it so they can't drift. `tag_poem` (whole poem) is what tags generated poems — `tag_lines` gives different tags.
- `common/corpus/make_splits.py` → `data/splits/poem_splits.csv` (seeded, per-author 80/10/10, poem-level). Always read splits via `load_splits()`/`poems_in_split()`; never recompute. Training uses `train` only; rewards never see val/test poems or the Gemma output.

## Commands

```bash
uv sync                               # Python 3.11
uv run python -m <package>.<module>   # e.g. -m gemma_teacher.llm_style_transfer
uv run pytest
uv run python -c "import classla; classla.download('mk')"   # one-time
```

Rebuild order for the base artifacts: `common.tagging.pos_tag_corpus` → `common.stylometry.tfidf_authors` → `common.stylometry.style_profiler` → `common.knowledge_graph.{build_skg, enrich_skg, add_embeddings, graph_analytics}` → `common.stylometry.classifier` → `common.tagging.build_morph_lookup`. `models/word_embeddings.json` (299 MB) is gitignored.

## LLM backends (`src/common/llm_client.py`)

- `google` (default) — Gemini API, `GOOGLE_API_KEY`. Gemma 4 always reasons and can't be stopped; `call_google` drops `thought` parts, requests the 32k output ceiling, and retries once if thinking starves the answer.
- `openrouter` (`:free` Gemma routes all 429), `ollama` (local), `vllm` (our own A100 via `src/gemma_teacher/hpc/vllm_teacher.slurm`; served name matches the API's so `poem_summary_cache.json` still hits).
- `_post` retries 408/429/5xx with backoff; other 4xx raise immediately.
- `data/poem_summary_cache.json` is tracked; `data/llm_transfer_logs/` is not.

## Teacher data

`src/gemma_teacher/generate_synthetic_parallel.sh [poems-per-author] [n-targets] [samples] [seed] [split]` runs one process per target author (the axis on which processes are provably disjoint), each writing its own part under `data/synthetic/parts/`; `merge_synthetic_parts.py` folds them in, deduping on `(source_poem_id, target_author, sample_index)`. Resumable — killing and re-running is safe. Non-train splits go to `teacher_{val,test}.csv`, never `synthetic_dataset.csv`.

## Students (SFT → GRPO)

Specs: `docs/superpowers/specs/2026-09-27-grpo-stage-design.md`, `2026-10-02-gemma4-student-design.md`. Family-specific values live in `src/students/shared/model_families.py`; `--family` defaults to `qwen3`, and `assert_family_matches` refuses another family's adapter.

Traps:
- **Move the model to the GPU explicitly** for generation — `from_pretrained` lands on CPU and `generate()` silently runs there (~8× slower).
- **GRPO trains a fresh LoRA on the merged SFT model** (`merge_sft_adapter.py` → `models/<tag>-sft-merged/`). TRL's KL reference is the policy with the adapter off, so training the SFT adapter directly would anchor KL to the untuned base.
- Keep max completion at 512 even for `--smoke`: SFT completions median ~291 tokens, and truncation masking zeroes the loss of every truncated completion.
- Every metric in `common/evaluation/style_metrics.py` is higher-is-better and reused as the GRPO reward — one implementation per metric. The untuned base model's dominant move is copying the source verbatim, which scores perfect content recall; that's why `line_novelty` / `copy_novelty` exist.
- `common/evaluation/judge_classifier.py` is evaluation-only (a different classifier from the reward's `classifier.pkl`, to detect reward gaming). Never import it from a reward — a test enforces this.
- Gemma 4 E4B: load text-only via `load_causal_lm`. The hub checkpoint stores weights under `model.language_model.*`, and a plain text-only `from_pretrained` silently leaves them random. Tower modules reuse the names `q_proj`… so suffix LoRA targets are only safe text-only. Gemma trains prompt–completion with `eos_token='<turn|>'` (its turn end, not `<eos>`).
- Qwen is run non-thinking (`enable_thinking=False`).

HPC: build the image once with `src/students/shared/hpc/build.sh` (rebuild when `uv.lock` changes). Jobs in `src/students/shared/hpc/` take `FAMILY=qwen3|gemma4`; `grpo_common.sh` maps it to the `models/` tag (`qwen3` / `gemma4-e4b`). Typical sequence: `train_sft` → `compare_base_lora` → `grpo_smoke` → `train_grpo` (Qwen's v2 reward: `qwen/hpc/train_grpo_v2.slurm`; Gemma: `gemma/hpc/train_grpo_gemma.slurm`, auto-resumes) → `eval_grpo` (val) → `eval_test base|sft|grpo|grpo-v1|grpo-v2`, then `eval_test score`, which scores every `*_test.csv` plus `teacher_test.csv` into `test_comparison_*.csv`. Generation CSVs in `data/validation_generations/` are tracked.

## Reports

`docs/report.tex` covers the pipeline up to the teacher; `docs/rl_report.tex` covers SFT/GRPO/students (`\pending{...}` marks awaited results); `docs/thesis/` is the thesis. After a significant pipeline change (model swap, scoring/metric change, bug affecting reported numbers), check whether the relevant report needs updating. Compile with `xelatex`, not `pdflatex` (Cyrillic via fontspec/polyglossia).

## Data quality

- 52 rows contain Latin/Greek letters. About 22 are lookalike artifacts (`ο` for `о`, mixed-script `штο`); the rest are intentional (Roman numerals, real Latin words). Cleaning scripts must tell the two apart.
- `(author, song_title)` is not unique (~15 pairs, e.g. four Конески `ПЕСНА`), and `pos_tagged.csv` has no finer id. Single-poem lookups raise on ambiguity; per-poem aggregates (`poem_tfidf.py`) blend the duplicates.

## POS tagging (classla `mk`)

Read `xpos`, not just `pos` — it's positional MULTEXT-East:

```
NOUN  Ncfsny  c=common/p=proper  Gender  Number  Case  Definite n/y/p=proximal/d=distal
VERB  Vmeic-smn  m=main/o=modal  Aspect p/e/b  Mood  Tense p/a/i/c=л-form  Person Number Gender Polarity
ADJ   Afpfs-y  Type f/p/s/g/o  Degree  Gender  Number  -  Definite
ADV   Rgp      Type g/d/v  Degree
```

Normalizations on top of classla, in order:
1. Hand corrections from `data/pos_corrections.csv`.
2. Relativizers (`што/кога/како/колку/каде`) get bare `xpos=Rg` and are excluded from content words. Left in, they were the top five "content" words.
3. `-јќи` verbal adverbs (`Rv`) are re-lemmatized to their verb, resolving `-ејќи` by corpus verb-lemma frequency.

**Correcting tags:** `pos_tagged.csv` and `pos_flags.csv` are regenerated on every run, so never hand-edit them. Fill `correction` (a tag or `DROP`) / `lemma_fix` / `scope` (blank = corpus-wide, `poem`, `row`) in `data/pos_flags.csv`, then run `uv run python -m common.tagging.apply_corrections`, which harvests the edits into `pos_corrections.csv` (the only never-regenerated file). Corrections key on word + wrong tag, so correctly tagged occurrences are untouched. A POS correction clears `xpos`/`feats`: an empty `xpos` means "corrected by hand". Lemma-only fixes (e.g. `петлите`→`петел`) are added to `pos_corrections.csv` by hand. `row`-scoped fixes and `unknown_tag` fixes need a full `pos_tag_corpus` run. `pos_flags.csv` is advisory only; rarity-based flag rules were dropped (~5 % precision on poetry).
