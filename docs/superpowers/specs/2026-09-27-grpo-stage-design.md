# GRPO stage: reward-driven fine-tuning of the SFT Qwen3-4B style-transfer model

Date: 2026-09-27
Status: design approved in conversation, pending written-spec review

## 1. Goal and success criteria

Second stage of the Qwen training pipeline. Starting from the SFT model
(`models/qwen3-lora-sft`), optimise Qwen3-4B toward the target author's style
while preserving the source poem's content, using GRPO (TRL `GRPOTrainer`) with
two independently computed rewards:

```
source poem + target author ─► SFT policy ─► G candidates
                                               │
                         ┌─────────────────────┴─────────────────────┐
                    style reward                                content reward
                         └───────────► R = α·R_style + β·R_content ◄─┘
                                               │
                                             GRPO ─► updated LoRA
```

Success for this stage is **a demonstrated `generate → reward → GRPO update →
generate again` loop**, backed by a reward that has been inspected on
hand-built cases before any long run — not a finished model.

### Research constraints (binding)

- Train split only. Val/test poems never enter GRPO prompts, reward
  calibration, or in-training evaluation.
- Gemma output is never a reward target or reference. Content is measured
  against the **source poem**; style against the **target author's real
  poems**.
- Rewards are functions of (generated poem, source poem, target author) only.
  No hidden answer information.
- Source and target author stay distinct (self-transfer pairs are skipped).
- Style-profile definitions are not changed. The refactor in §3.1 is verified
  to reproduce `author_style_profiles.csv` byte-for-byte.
- Every reward component and gate is logged separately.
- Reproducible: seeded, with provenance (git SHA, SFT adapter, calibration
  hash, TRL version) written alongside every run.
- No large training run is started automatically.

## 2. Decisions taken during design

| # | Decision | Chosen | Rejected alternatives |
|---|---|---|---|
| D1 | POS-based profile features in the reward | Run classla (via `tagging.py`) on every candidate | Untagged features only; both behind a flag |
| D2 | Guarding against copy / repetition / format hacks | Multiplicative gates inside the two rewards | Additive third reward; no guards |
| D3 | Initial policy and KL reference | Merge SFT adapter into the base weights, train a **fresh** LoRA on top; KL reference = merged SFT | Continue training the SFT adapter (KL ref would be untuned Qwen); two adapters (no clean TRL path) |
| D4 | Three-way evaluation set | Run Gemma on the same 468 val pairs as the Qwen runs | Gemma on train only (unpaired); 100-pair subset |
| D5 | Code structure | Extract per-poem features into `style_features.py`, rewards in `grpo_rewards.py` built on `style_metrics.py` | Grow `style_metrics.py`; re-implement features in the reward |
| D6 | Reward calibration data | Train-split poems only | All poems (as the existing profiles do) |
| D7 | `classifier.pkl` | Reused as is (fit on full corpus); leak documented in report | Retrain on train split now |

## 3. Components

### 3.1 `src/style_features.py` (new) + `src/style_profiler.py` (refactor)

`style_profiler.py` is currently a flat script. Its per-poem computations move
into importable functions:

- `poem_features(tokens, text) -> dict` where `tokens` is a list of
  `(lemma, pos)` for the poem's tagged tokens (the same rows
  `pos_tagged.csv` holds; the reward path produces them via `tagging.py`) and
  `text` is the raw poem. The reward path must hand it exactly the token set
  `pos_tag_corpus.py` writes to `pos_tagged.csv` (same corrections, same
  filtering), so shared token selection is factored out of
  `pos_tag_corpus.py` if it is not already reachable from `tagging.py`.
  Returns per-poem:
  `pct_NOUN, pct_VERB, pct_ADJ, pct_ADV, pct_PROPN`, `lexical_diversity`
  (`mattr()` over lemmas — below the 200-token window it falls back to TTR,
  exactly as the profiler's function does), `avg_tokens_per_line`,
  `num_lines`, `num_stanzas`, `avg_lines_per_stanza`, `pos_bigrams`
  (Counter), `rhyme_endings` (Counter, via `line_final_syllable`).
- `line_final_syllable`, `mattr`, `CONTENT_POS`, `MATTR_WINDOW` move here and
  are re-exported from `style_profiler` for existing importers.

`style_profiler.py` keeps its aggregation (author-level totals, not averages of
per-poem values) and its output. **Verification:** regenerate
`author_style_profiles.csv` and diff against the committed file — must be
identical.

### 3.2 `src/build_reward_calibration.py` (new) → `data/reward_calibration.json` (tracked)

Runs over **train-split** real poems (`make_splits.poems_in_split('train')`),
reading tags from `pos_tagged.csv` (no classla needed). For each author with a
style profile:

- per scalar feature: `mean`, `std` over that author's train poems (one value
  per poem);
- `pos_bigram_dist`, `rhyme_dist`: the author's aggregate normalised
  distributions;
- for each distribution feature: mean and std of each real train poem's cosine
  to its own author's distribution (one-sided calibration, §4.1);
- `tfidf_hit_p90`: 90th percentile of `style_metrics.style_match(...)['tfidf_hit_rate']`
  over the author's train poems.

Corpus-wide: per-feature `std` over all train poems (for the σ floor), and the
content baseline `b` (§4.2): mean raw line-F1 over 2,000 seeded random pairs of
train poems by different authors.

Also records: seed, split file hash, embedding model name, creation time.
Deterministic given the seed.

### 3.3 `src/grpo_rewards.py` (new)

- `score(source_text, generated_text, target_author) -> dict` — every feature
  score, group score, gate, and the three rewards, flat. This is what
  `evaluate_reward.py` and `compare_generations.py --rewards` call.
- `score_batch(sources, generations, targets) -> list[dict]` — one batched
  classla call and one batched embedding call.
- `style_reward(prompts, completions, source_text, target_author, **kw)` and
  `content_reward(...)` — TRL reward-function signatures, both backed by one
  shared per-batch `score_batch` result (memoised on the completion list
  identity so classla runs once per step, not twice). Each appends one JSON
  record per completion to a configurable `reward_components.jsonl`.
- Lazy singletons: classla pipeline (GPU if available), sentence-transformers
  model, calibration JSON. Source-line embedding cache keyed by line text.
- Configuration via a `RewardConfig` dataclass: `style_weights`,
  `alpha`, `beta`, `copy_jaccard_threshold` (0.7), `sigma_floor_frac` (0.5),
  `content_model`, `calibration_path`.

### 3.4 `src/style_metrics.py` (additive change)

Add `copy_novelty(source_text, text)`: share of generated lines whose token
Jaccard with every source line is < `copy_jaccard_threshold`. Existing
`line_novelty` is left unchanged so previously reported numbers keep their
meaning.

### 3.5 `src/evaluate_reward.py` (new) — standalone reward inspector

Modes:

- single: `--source-poem-id ID | --source-file F`, `--target-author A`,
  `--generated-file F | --generations-csv CSV --row N` → prints
  ```
  Style reward:   0.xxx   (gated: 0.xxx)
  Content reward: 0.xxx   (gated: 0.xxx)
  Total reward:   0.xxx
  ```
  plus gate values and a per-feature table: raw value, author μ/σ, z, score,
  group, weight.
- `--fixtures`: runs every case in `tests/fixtures/reward_cases.json`, prints
  a compact table, and exits non-zero if an ordering expectation fails.
- `--groups --adapter/--merged-model ... --num-prompts 20 --num-generations G`
  (GPU): samples G completions per train prompt from the SFT policy and
  reports each component's mean, between-prompt std and **mean within-group
  std** — the quantity that decides which component dominates GRPO's
  group-normalised advantage. Used to set α/β before the long run.

### 3.6 `tests/fixtures/reward_cases.json` + `tests/test_grpo_rewards.py` (new)

For 3–4 (source poem from train, target author ≠ source author) pairs, fixed
candidates:

1. a real poem by the target author (different content)
2. the source poem verbatim
3. the source with 1–2 words per line changed (near-copy)
4. Gemma's output for the pair (from `synthetic_dataset.csv`, where available)
5. SFT output for the pair (where available)
6. a looping repetition (one line × 12)
7. an English/markdown answer about the poem
8. empty string

Asserted orderings (not exact values):

- `content_gated` of 2 and 3 == 0 (copy gate)
- `style` of 1 > `style` of 2 (target voice beats source voice)
- `content` raw of 2 > `content` raw of 1 (the content scorer itself works
  before gating)
- total of 6, 7, 8 < total of 4
- total of 7 and 8 == 0 (validity gate)
- every component in [0, 1]

Plus unit tests: z-scoring and σ floor, one-sided distribution calibration,
content baseline rescaling and clipping, `copy_novelty`, `score_batch` equals
per-item `score`.

### 3.7 `src/merge_sft_adapter.py` (new) → `models/qwen3-sft-merged/` (gitignored)

Base model + SFT adapter → `merge_and_unload()` → `save_pretrained` in bf16,
with tokenizer and `merge_info.json` (base model, adapter path, adapter
`run_config.json` hash). Idempotent; refuses to overwrite a merge from a
different adapter without `--force`.

### 3.8 `src/grpo_data.py` (new)

- Sources: `poems_in_split('train')` only; targets:
  `sft_data.select_target_authors(10)` (or `--target-authors`); pairs via
  `generate_synthetic.build_pairs` (self-transfer skipped); seeded shuffle;
  `--num-prompts` cap.
- Columns: `prompt` (pre-rendered by `sft_data.render_inference_prompt`,
  thinking disabled — identical to SFT training and validation),
  `source_poem_id`, `source_text`, `target_author`.
- Prompts longer than `--max-prompt-length` tokens (default 1024) are
  **dropped, not truncated** (a truncated source would make the content reward
  compare against text the model never saw); count logged.
- Returns an HF `Dataset`.

### 3.9 `src/train_grpo.py` (new)

- Loads `models/qwen3-sft-merged` (bf16, sdpa, gradient checkpointing), checks
  `merge_info.json` exists, attaches a fresh `LoraConfig` (same target modules
  as `train_sft.DEFAULT_TARGET_MODULES`).
- `GRPOTrainer(model, reward_funcs=[style_reward, content_reward],
  args=GRPOConfig(reward_weights=[α, β], ...), train_dataset=..., peft_config=...)`.
- A `TrainerCallback` adds per-step component means (`style/<group>`,
  `gate/<name>`, `content/raw_f1`) to the logged metrics.
- Saves adapter + tokenizer + `run_config.json` (all args, git SHA, SFT
  adapter path, calibration file hash, TRL/transformers/peft versions,
  resolved `GRPOConfig`) to `--output-dir`.
- `--smoke`: 4 prompts, G=4, 3 steps, 128-token completions; asserts (a)
  rewards finite and non-constant within ≥ 1 group, (b) loss finite, (c)
  LoRA weight norm moved off its init, (d) greedy generations on 2 fixed
  prompts differ before vs after. Non-zero exit on any failure.

Defaults:

| Parameter | Default |
|---|---|
| `--sft-merged` | `models/qwen3-sft-merged` |
| LoRA r / alpha / dropout | 16 / 32 / 0.0 (dropout off: keeps sampling-time and scoring-time log-probs consistent) |
| `--learning-rate` | 1e-5, constant with 10 warmup steps |
| `--num-generations` | 4 |
| `--per-device-train-batch-size` / `--gradient-accumulation-steps` | 8 / 4 (32 completions = 8 prompts per optimiser step) |
| `--max-prompt-length` | 1024 (filter, §3.8) |
| `--max-completion-length` | 512, `mask_truncated_completions=True` |
| `--temperature` | 0.9 |
| `--beta` (KL) | 0.04 |
| `--loss-type`, `--scale-rewards`, `--num-iterations` | TRL defaults (`dapo`, `group`, 1), recorded |
| `--style-weight` (α) / `--content-weight` (β) | 0.5 / 0.5 (not `--alpha`/`--beta`, which would clash with the KL `--beta`) |
| `--style-weights` | JSON, §4.1 defaults |
| `--max-steps` | 500 |
| `--logging-steps` / `--save-steps` / `--save-total-limit` | 1 / 50 / 5 |
| `--seed` | 42 |
| `--bf16` | on |
| `--output-dir` | `models/qwen3-grpo` |

No in-training evaluation (it would feed val rewards into the loop and is
expensive); checkpoints are evaluated afterwards (§5).

### 3.10 Modified: `generate_validation.py`, `generate_synthetic.py`, `scripts/generate_synthetic_parallel.sh`, `compare_generations.py`

- `generate_validation._infer_base_model`: if the recorded base path does not
  exist (e.g. the container path `/workspace/models/qwen3-sft-merged`),
  resolve its basename under `ROOT/models`.
- `generate_validation._run_label`: `checkpoint-N` adapters are labelled
  `<parent>-checkpoint-N`.
- `generate_synthetic.py --split {train,val}` (default `train`); rows carry
  that split.
- `generate_synthetic_parallel.sh`: optional 5th positional arg `split`. For
  `val`, parts go under `data/synthetic/parts_val/` and merge into
  `data/synthetic/teacher_val.csv`, so val rows cannot reach
  `synthetic_dataset.csv` (which `train_sft.py` reads).
- `compare_generations.py --rewards`: adds `R_style`, `R_content`, `R` and
  gate columns via `grpo_rewards.score_batch`. Off by default so the plain
  path stays CPU-seconds.

## 4. Reward definition

All component scores are in [0, 1], higher is better.

### 4.1 Style

Scalar features (§3.1): for target author *a* and feature *f*,

```
σ = max(σ_a,f, sigma_floor_frac · σ_corpus,f)
z = (x_f − μ_a,f) / σ
s_f = exp(−½ z²)
```

Distribution features (`pos_bigrams`, `rhyme_endings`): `c = cos(poem_dist,
author_dist)`; one-sided calibration against the author's own poems:
`s = 1` if `c ≥ μ_c`, else `exp(−½ ((c − μ_c)/σ_c)²)`.

Reused from `style_metrics`: `clf_target_prob` (raw), and
`tfidf = min(1, tfidf_hit_rate / tfidf_hit_p90_a)`.

Groups and default weights (`--style-weights`, 0 removes a group):

| Group | Members (scores averaged within group) | Weight |
|---|---|---|
| `pos_profile` | `pct_NOUN, pct_VERB, pct_ADJ, pct_ADV, pct_PROPN` | 1.0 |
| `lexical_diversity` | `lexical_diversity` | 1.0 |
| `structure` | `avg_tokens_per_line, num_lines, num_stanzas, avg_lines_per_stanza` | 1.0 |
| `pos_bigrams` | `pos_bigrams` | 1.0 |
| `classifier` | `clf_target_prob` | 1.0 |
| `rhyme` | `rhyme_endings` | 0.5 |
| `tfidf` | `tfidf` | 0.5 |

`R_style = Σ w_g · s_g / Σ w_g`.

### 4.2 Content

Line-level embedding F1 (BERTScore-style), `paraphrase-multilingual-mpnet-base-v2`
by default (`--content-model`), normalised embeddings:

```
S = cos(source_lines, generated_lines)       # |src| × |gen|
recall    = mean over source lines of row-max
precision = mean over generated lines of column-max
F1        = 2PR / (P + R)
R_content = clip((F1 − b) / (1 − b), 0, 1)   # b from calibration
```

Line-level avoids the model's 128-token truncation and stops a poem that covers
only the first stanza from scoring full recall.

### 4.3 Gates and combination

```
validity    = is_cyrillic · no_markup · is_nonempty          (style_metrics)
R_style_g   = validity · line_uniqueness · R_style
R_content_g = validity · copy_novelty    · R_content
R           = α · R_style_g + β · R_content_g                (α = β = 0.5)
```

Implemented as two TRL reward functions with `reward_weights=[α, β]`
(`validity` applied inside each is equivalent to gating the total).

α/β are revisited after `evaluate_reward.py --groups` (§3.5): under
`scale_rewards='group'` the component with the larger within-group spread
dominates the advantage regardless of its mean.

### 4.4 Known limitations (documented in the report)

- A paraphrase in another Cyrillic language (Serbian, Russian) passes the
  validity gate and scores well on multilingual content similarity. Not
  addressed now; not observed in SFT output.
- `classifier.pkl` was fit on the full corpus (val/test included). It is
  author-level stylometry and never sees the target content, but it is a leak
  path; retraining on train only is a follow-up.
- Per-poem TTR (below the MATTR window) depends on poem length; z-scoring
  against the same author's per-poem distribution keeps it comparable, but
  the structure group's `num_lines` interacts with it.

## 5. Evaluation

All runs over the same 468 val pairs (50 val poems × 10 target authors,
self-transfer skipped, seed 42) with the existing decoding (temperature 0.7,
top-p 0.9, 512 new tokens, per-row `torch.manual_seed(seed + i)`), on the A100.

| Run | Source | Output |
|---|---|---|
| Untuned base | existing | `data/validation_generations/base-Qwen3-4B_val.csv` |
| SFT (LoRA) | existing | `data/validation_generations/qwen3-lora-sft_val.csv` |
| SFT merged (GRPO's start and KL reference) | `generate_validation.py --no-adapter --base-model models/qwen3-sft-merged` | `data/validation_generations/qwen3-sft-merged_val.csv` |
| SFT + GRPO | `generate_validation.py --adapter models/qwen3-grpo` | `data/validation_generations/qwen3-grpo_val.csv` |
| Gemma | `generate_synthetic_parallel.sh 50 10 1 42 val` | `data/synthetic/teacher_val.csv` |

SFT-merged vs SFT-LoRA also checks that the bf16 merge did not shift
behaviour (metrics should agree within sampling noise). The SFT-merged and
GRPO runs go back to back in one job (`eval_grpo.slurm`), matching
`compare_base_lora.slurm`'s reasoning.

Comparison: `compare_generations.py --rewards --runs <5 csvs> --labels ...`;
the paired view keys on `(source_poem_id, target_author, sample_index)`.

The test split is untouched; the final test-split command is documented but
not run.

## 6. Slurm / HPC

All jobs follow the existing pattern (`singularity exec --nv`, HF cache bind,
`$PWD → /workspace`, `/opt/venv/bin/python`, forwarded args), single A100,
bf16. New bind: `$HOME/classla_resources → /root/classla_resources`.
Pre-flight fails fast if the `.sif`, SFT adapter, classla `mk` models or
`data/reward_calibration.json` are missing.

| Script | Does |
|---|---|
| `slurm/grpo_smoke.slurm` | merge if missing → `evaluate_reward.py --fixtures` → `train_grpo.py --smoke` |
| `slurm/reward_groups.slurm` | merge if missing → `evaluate_reward.py --groups` |
| `slurm/train_grpo.slurm` | merge if missing → `train_grpo.py "$@"` |
| `slurm/eval_grpo.slurm` | SFT-merged + GRPO val generation back to back → `compare_generations.py --rewards`; `ADAPTER=` env selects a checkpoint |

Nothing is submitted automatically.

## 7. Verification

- Local (macOS, CPU): profile byte-identity; calibration build; reward unit and
  fixture tests; `evaluate_reward.py` on rows of the existing val CSVs;
  `grpo_data` split-isolation test (every `source_poem_id` ∈ train); the
  parallel-script split-routing test; a pytest plumbing test of `train_grpo`
  (2 steps, tiny random-init Qwen3 config, CPU) that checks reward functions,
  dataset columns, callback and save path wire together — not a substitute for
  the real smoke test.
- Cluster (by the user): merge, `grpo_smoke.slurm`, `reward_groups.slurm`, the
  full run, `eval_grpo.slurm`.

## 8. Docs

- `CLAUDE.md`: new "GRPO stage" section (file roles, merge + fresh LoRA
  rationale, reward definition, gates, commands, gitignored artifacts
  `models/qwen3-sft-merged/`, `models/qwen3-grpo/`, `data/synthetic/parts_val/`).
- `data/report.tex`: reward formulation and calibration; gates and why
  (base-model copying); the `style_profiler` refactor (definitions unchanged,
  verified byte-identical); train-only calibration; the classifier leak; the
  Cyrillic-paraphrase blind spot.

## 9. Implementation order

1. `style_features.py` refactor + profile byte-identity check
2. `build_reward_calibration.py`
3. Style reward
4. Content reward
5. Gates + combination (`grpo_rewards.py` complete, `copy_novelty`)
6. `evaluate_reward.py` + fixtures + tests
7. `merge_sft_adapter.py`, `grpo_data.py`
8. `train_grpo.py` + local plumbing test
9. `grpo_smoke.slurm`, `reward_groups.slurm`, `train_grpo.slurm`
10. Evaluation changes, Gemma val path, `eval_grpo.slurm`
11. `CLAUDE.md`, `report.tex`
