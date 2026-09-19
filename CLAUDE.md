# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Macedonian-language NLP pipeline for stylometric analysis and style transfer. The core dataset is `data/stripped_songs.csv` — 2,711 Macedonian poems and song lyrics (Cyrillic text) with columns `author`, `song_title`, and `song_text`. The pipeline covers POS tagging, TF-IDF profiling, knowledge graph construction, word embeddings, author classification, and style transfer.

## Folder structure

```
src/        Python scripts — pipeline steps and shared tagging/morph modules
data/       CSV files — corpus input and analysis outputs
models/     Trained artifacts — classifier.pkl, *.json lookups, skg_final.gexf
lib/        JS assets used by pyvis when generating skg_visual.html
```

All scripts resolve paths via `pathlib.Path(__file__)` so they work from any working directory.

`src/tagging.py` owns the classla pipeline and the three normalizations applied on top of it: hand corrections from `data/pos_corrections.csv`, bare-`Rg` relativizer filtering, and `-јќи` re-lemmatization. Both `pos_tag_corpus.py` and the inference path import it, so corpus tags and inference tags cannot drift.

`src/morph.py` owns feats parsing, the relaxation order, and `surface_form()`.

## Environment & commands

This project uses `uv` for dependency management (Python 3.10+).

```bash
uv sync                             # install dependencies
uv run python src/<script.py>       # run a script in the venv
uv add <package>                    # add a dependency

# one-time: fetch the classla Macedonian models used by pos_tag_corpus.py
uv run python -c "import classla; classla.download('mk')"
```

## Pipeline order

Run scripts in this order to rebuild all artifacts from scratch:

1. `src/pos_tag_corpus.py` → `data/pos_tagged.csv`, `data/pos_flags.csv` (applies `data/pos_corrections.csv`; see "Correcting tags by hand")
2. `src/tfidf_authors.py` → `data/tfidf_results.csv`
3. `src/style_profiler.py` → `data/author_style_profiles.csv`
4. `src/build_skg.py` → `models/skg.gexf`
5. `src/enrich_skg.py` → `models/skg_enriched.gexf`, `models/author_vocab.json`, `models/pos_transitions.json`
6. `src/add_embeddings.py` → `models/skg_final.gexf` (also writes `models/word_embeddings.json`, excluded from git — 299 MB)
7. `src/graph_analytics.py` → `data/graph_analytics.csv`
8. `src/classifier.py` → `models/classifier.pkl`
9. `src/build_morph_lookup.py` → `models/morph_lookup.json`
10. `src/style_transfer.py` — interactive demo, requires all of the above

## LLM-based generative style transfer (separate pipeline)

A second, independent style-transfer approach alongside the token-substitution
pipeline above (`candidate_selection.py` / `style_transfer.py`): instead of
splicing candidate words into masked slots, it sends one LLM prompt combining
an LLM-written summary of a source poem, its content keywords, the target
author's style profile in prose, a relevance-filtered vocabulary palette, tone
exemplar lines, and explicit structural targets, and asks the model to write a
new poem from scratch. Requires steps 1-3 above (`pos_tagged.csv`,
`tfidf_results.csv`, `author_style_profiles.csv`) plus `models/word_embeddings.json`.

1. `src/llm_client.py` — shared call wrapper (`call()`) over three backends, also used by `src/llm_probe.py`:
   - `google` — Gemini API direct, needs `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in the environment or `.env`. **The default.** Gemma 4 always reasons and can't be stopped (`thinkingBudget` is a 400), so `call_google` filters the `thought=true` response part, requests the full 32,768-token output ceiling, and retries once when thinking starves the answer — thinking length swings between ~2k and ~16k tokens for the same prompt.
   - `openrouter` — needs `OPENROUTER_API_KEY`. Every `:free` Gemma route currently 429s from a shared upstream pool, which is why `google` exists.
   - `ollama` — local, no key. `--backend ollama --model qwen3:14b` reproduces the pre-Gemma logs.
   - `_post` (shared by all three) retries transient HTTP failures (408/429/5xx) with exponential backoff plus jitter, up to `MAX_RETRIES` (5) attempts, before raising — Gemini's `google` backend in particular throws intermittent 500 "Internal error"/503 "high demand" under load. Non-transient 4xx (bad request, bad key) raise immediately rather than retry. This is separate from `call_google`'s own thinking-starved retry above, which fires on a 200 response with no answer part, not an HTTP error.
2. `src/poem_tfidf.py` → `data/poem_tfidf_results.csv` — per-poem TF-IDF (distinct from `tfidf_results.csv`'s author-level, surface-word TF-IDF: this one's document is a single poem and its tokens are `pos_tagged.csv` lemmas)
3. `src/style_narrator.py` — `describe_style(author)`: turns a profile row into corpus-relative prose (percentile rank per numeric column, binned into tertiles, plus `top_pos_bigrams`/`top_rhyme_endings` translated to sentences)
4. `src/exemplar_selection.py` — `select_exemplars(author)`: k-means over line-embedding centroids of TF-IDF-qualifying lines, returns the lines closest to each cluster centroid. Line texts are de-duplicated author-wide (refrains and the corpus's near-duplicate poems otherwise put the same line in the prompt twice), and the final selection takes at most one line per source poem, globally across clusters
5. `src/llm_style_transfer.py` — prompt assembly, generation, and structural-fit validation; entry point:
   ```bash
   uv run python src/llm_style_transfer.py --source-author "..." --source-title "..." --target-author "..."
   ```
   Defaults to `gemma-4-31b-it` on the `google` backend. `--source-poem-id` is the only way to reach a poem whose (author, title) pair is ambiguous; `--source-author`/`--source-title` are overwritten from the poem's own record when it is given.
   Poem summaries are cached in `data/poem_summary_cache.json` (tracked — deterministic per model+prompt-version+text). Per-run audit records (prompt, generated poem, structural fit, metadata) are written to `data/llm_transfer_logs/*.json` (gitignored — reproducible from a run, not meant to accumulate in git history).

## Qwen3 LoRA SFT training pipeline (third, downstream pipeline)

Distills the LLM style-transfer pipeline above into a small fine-tuned model:
Gemma's heavily-scaffolded (style profile prose, vocabulary palette, exemplar
lines, structural targets) prompt produces training examples for a much
smaller model that performs the same transfer given just a source poem and a
target author name. Requires steps 1-3 of the original pipeline
(`pos_tagged.csv`, `tfidf_results.csv`, `author_style_profiles.csv`) plus
`models/word_embeddings.json`, same as the LLM style-transfer pipeline it
consumes. This is stage 1 of a larger plan; the reward/GRPO stage is not
implemented yet (see `src/generate_validation.py`'s docstring for the
intended handoff).

1. `src/make_splits.py` → `data/splits/poem_splits.csv` — per-author 80/10/10 train/val/test split, seeded (`SEED=42`), poem-level (not `(author, song_title)` — see Data quality context below). Authors below `MIN_POEMS_FOR_3WAY=3` poems (8 of the corpus's 38 authors have 1-2 poems) can't be cut three ways and go entirely to `train`; every author with >= 3 poems gets >= 1 poem in each split. `load_splits()`/`poems_in_split()` are what every later stage reads — never recompute the split elsewhere.
2. `src/sft_data.py` — shared infrastructure so the training and inference paths can't drift on the task format:
   - `select_target_authors(n)`: the `n` authors with the most poems among those with a style profile at all, ties broken alphabetically. Not an author-diversity algorithm — "distinct target styles" falls out of picking real, different, well-profiled poets.
   - `build_messages(source_poem, target_author, generated_poem=None)`: the compact `[system, user]` (or `[system, user, assistant]`) chat format the SFT task is actually trained on — deliberately NOT the full Gemma prompt. `render_inference_prompt`/`render_chat` render it through the tokenizer's own chat template with `enable_thinking=False` (Qwen3 is treated as non-reasoning for this task, matching `llm_client.py`'s existing `think=False`/`_strip_think` handling).
   - `load_synthetic_rows()` / `build_hf_dataset()` load `generate_synthetic.py`'s output into an HF `Dataset` with a `messages` column (not a pre-rendered string — TRL's `assistant_only_loss` needs per-role structure to mask the loss, see `train_sft.py`).
3. `src/generate_synthetic.py` → `data/synthetic/synthetic_dataset.csv` — calls `llm_style_transfer.run_transfer()` unmodified, in a loop over sampled (source poem, target author) pairs from the **train** split only. Resumable (the output CSV is the "already done" record) and per-pair failures land in `data/synthetic/synthetic_errors.csv` (gitignored) rather than aborting the batch. Example:
   ```bash
   uv run python src/generate_synthetic.py --num-target-authors 10 --poems-per-author 100 --samples-per-pair 1 --seed 42
   ```
   `--poems-per-author` is the count of source poems sampled once and reused across every target author, not resampled per author (see the script's own docstring before changing this).

   Gemma's `google` backend is the throughput bottleneck (a few requests/minute, ~3 min/pair including the reasoning overhead), so `scripts/generate_synthetic_parallel.sh [poems-per-author] [num-target-authors] [samples-per-pair] [seed]` runs one `generate_synthetic.py` process per target author in parallel instead of one process looping over all pairs serially — the dedup key is `(source_poem_id, target_author, sample_index)`, and target_author is the one axis two processes are provably disjoint on. Each process gets its own `--output-path`/`--errors-path` under `data/synthetic/parts/` (gitignored) — concurrent appends from multiple processes to one shared CSV aren't safe in general (a generated poem can push a single row past the OS's atomic write size). `src/merge_synthetic_parts.py` folds the per-author parts back into the main `synthetic_dataset.csv`/`synthetic_errors.csv`, deduping on that same key, and seeds each part from the main file's already-done rows before its process starts so a prior serial (or interrupted parallel) run's successes aren't regenerated. The wrapper script merges leftover parts before reseeding too, so killing it mid-run and re-invoking it is safe the same way the serial script is.
4. `src/train_sft.py` — LoRA SFT of `Qwen/Qwen3-4B` via TRL's `SFTTrainer` + PEFT's `LoraConfig`, trained on `split=train` rows only; `split=val` rows (if present) are `SFTTrainer`'s `eval_dataset` for loss monitoring only (never backpropagated). LoRA targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` — confirmed against transformers' actual Qwen3 modeling source, not assumed from a generic Llama-family default (`q_norm`/`k_norm` are RMSNorm layers, deliberately excluded). Loss is masked to the assistant turn via `assistant_only_loss=True` (TRL's `get_training_chat_template` explicitly patches `{% generation %}` markers onto Qwen3's template for this). Saves the adapter + tokenizer + `run_config.json` to `models/qwen3-lora-sft/` (gitignored — regenerate via the Slurm job below).
5. `singularity/qwen-sft.def` + `singularity/build.sh` — the GPU runtime environment, built straight from this project's own `pyproject.toml`/`uv.lock` (`uv sync --frozen` at build time) so it can't drift from `uv run` outside a container. No CUDA/PyTorch base image needed: PyPI's default Linux `torch` wheel bundles its own CUDA runtime as ordinary pip packages (`nvidia-cuda-runtime-cu12`, `nvidia-cudnn-cu12`, etc.), so a plain `python:3.11-slim` base plus `uv sync` is enough — only the NVIDIA *driver* has to come from the host, via `--nv`. Build once with `./singularity/build.sh` (redirects Singularity's tmp/cache dirs onto the NFS-backed home filesystem first — this cluster's local `/tmp` is a few GB, nowhere near enough for the pulled layers + synced venv); rebuild whenever `pyproject.toml`/`uv.lock` changes. Gitignored (`singularity/*.sif`) — a build artifact, not source.
6. `slurm/train_sft.slurm` — single-GPU job (`sbatch slurm/train_sft.slurm`) that runs `train_sft.py` inside `singularity/qwen-sft.sif`, matching this cluster's other Singularity jobs: `singularity exec --nv` with explicit `-B` binds for the HF cache (`$HOME/.cache/huggingface` → `/root/.cache/huggingface`) and the repo (`$PWD` → `/workspace`, one mount covering `src/`+`data/`+`models/` — unlike a job that binds separate `./data`/`./code` host trees, this project keeps them in one repo), then a `bash -c` that `cd`s into `/workspace` and calls `/opt/venv/bin/python` directly (the image's baked venv, no conda `source activate` needed). Extra args after the script path are forwarded straight to `train_sft.py` via the `bash -c '...' _ "$@"` passthrough idiom. Fails fast with a clear message if the `.sif` hasn't been built yet.
7. `src/generate_validation.py` — loads the trained adapter and generates from the **val** split (never `train`), using the exact same `sft_data` chat format as training so there's no prompt skew, and reuses `llm_style_transfer.structural_targets`/`validate_structure` for the fit metrics so Qwen's outputs are directly comparable to Gemma's. Output → `data/validation_generations/<adapter-name>_<split>.csv` (tracked — curated, non-deterministic-to-regenerate model output, same reasoning as `poem_summary_cache.json`). Also needs a GPU — run the same way `train_sft.slurm` does (see above).

## Thesis report

`data/report.tex` is the thesis-facing write-up of pipeline methodology and validation findings (tagger migration, correctness bugs found and fixed, metric changes, community-detection stability, and similar). **Whenever you make a significant change to the pipeline** — swapping a model/tagger, changing a scoring or graph-construction method, fixing a bug that affects reported numbers, adding or redefining a metric — **check whether `data/report.tex` needs a corresponding update**, and update it if so. Compile with `xelatex report.tex` (not `pdflatex` — the document needs native Cyrillic via `fontspec`/`polyglossia`, and `T1`/`T2A` font-encoding setups aren't fully installed in this environment).

## Data quality context

Latin and Greek character artifacts exist inside Macedonian (Cyrillic) text — 231 rows affected out of 2,711. These are single characters (e.g. Latin `a`, `e`, `j`, `c`) that visually resemble Cyrillic letters but are the wrong Unicode codepoint.

`(author, song_title)` is not a unique poem key. ~15 pairs (e.g. Блаже Конески has 4 distinct poems titled `ПЕСНА`) are shared by multiple physical poems, and `pos_tagged.csv` carries no finer-grained id — this is `pos_tag_corpus.py`'s schema, inherited by every downstream file keyed the same way (`tfidf_authors.py`, `style_profiler.py`, `candidate_selection.py`'s `corpus_token_stream`, `poem_tfidf.py`). Consumers that need one specific poem (`llm_style_transfer.py`'s `load_poem_text`) raise on an ambiguous title rather than silently picking one; consumers that aggregate per-poem tokens (`poem_tfidf.py`) silently blend the duplicates' vocabulary together for those ~15 pairs. A real fix needs a poem-level id threaded through `pos_tag_corpus.py` and every script that consumes its output.

When writing data-cleaning or preprocessing scripts, distinguish between:
- **Intentional Latin/Greek** (foreign words, author names, titles that legitimately use Latin script)
- **Artifact characters** (Cyrillic lookalikes from OCR or copy-paste errors)

### POS tagging quality

Tagging uses **classla** (`mk` models), not spaCy — classla lemmatizes inflection correctly and adds MULTEXT-East `xpos` plus morphological `feats` columns to `pos_tagged.csv`. Hand-checking 70 random tokens against their source lines put UPOS accuracy around 96%.

**Read `xpos`, not just `pos`.** It is positional, and each slot carries a distinction `pos` flattens:

```
NOUN  Ncfsny  N  c=common/p=proper  Gender f/m/n  Number s/p/t=count  Case n/v  Definite n/y/p/d
VERB  Vmeic-smn  V  m=main/o=modal  Aspect p/e/b=biaspectual  Mood i/m
                    Tense p=pres/a=aorist/i=imperf/c=л-form  Person  Number  Gender  Polarity
ADJ   Afpfs-y  A  Type f=qual/p=participial/s=possessive/g=general/o=ordinal  Degree p/c/s  Gender  Number  -  Definite n/y
ADV   Rgp      R  Type g=general/d=modal/v=verbal  Degree p/c/s
```

Noun definiteness has four values, not two: `y` = neutral `-от/-та`, `p` = proximal `-ов/-ва`, `d` = distal `-он/-не`. `build_morph_lookup.py` keys on the `feats` column directly — see `docs/superpowers/specs/2026-08-13-feats-morph-lookup-design.md`.

Three corrections `pos_tag_corpus.py` applies on top of classla's output:

- **Hand corrections from `data/pos_corrections.csv`** are applied first — see "Correcting tags by hand" below.
- **Relativizers are dropped.** classla tags `што`/`кога`/`како`/`колку`/`каде` as ADV, but they are relativizers ("Зборовите **што** ти ги дадов"). They carry bare `xpos=Rg` (no degree slot) where real adverbs get `Rgp`/`Rgc`/`Rgs`, so `is_content()` filters on that. Left in, they were the five most frequent "content" words in the corpus — 1,478 tokens, 19% of all ADV — and swamped every POS ratio built on it.
- **`-јќи` verbal adverbs (`xpos=Rv`) are re-lemmatized to their base verb.** classla leaves them unlemmatized (`барајќи` → `барајќи`). `gerund_lemma()` rebuilds the verb, resolving the ambiguous `-ејќи` ending (both `велејќи`→`вели` and `знаејќи`→`знае` are possible) against corpus verb-lemma *frequencies* — a bare membership test picks up single mistagged tokens and yields `бране` over `брани`.

### Correcting tags by hand

`pos_tagged.csv` and `pos_flags.csv` are both rebuilt from scratch on every tagging run, so **never hand-edit them expecting the change to stick**. Corrections live in `data/pos_corrections.csv`, the one file here that is never regenerated.

```bash
# 1. review data/pos_flags.csv, filling in the correction / lemma_fix / scope columns
# 2. harvest those edits and patch pos_tagged.csv
uv run python src/apply_corrections.py
# 3. corrections are re-applied automatically on every later tagging run
uv run python src/pos_tag_corpus.py
```

`correction` takes a POS tag or `DROP` (removes the token — for junk like `01:47`, `navigare`, bare Greek letters that no tag fits). `lemma_fix` sets the lemma. `scope` is blank for corpus-wide, or `poem` / `row` to narrow it.

Either column alone is enough. A **lemma-only fix** (blank `correction`, `lemma_fix` set) is the path for a bad lemma under a correct tag — `apply_to` then leaves `xpos`/`feats` intact, since no POS judgement was overruled. No flag rule catches lemma instability, so these have to be added to `pos_corrections.csv` by hand rather than harvested from `pos_flags.csv`; `apply_corrections.py` picks them up on its next run regardless. Example: classla lemmatizes `петлите` to `петлин` while getting `петел`/`петелот`/`петли` right, splitting one lemma across two spellings.

A correction keys on the word **and the tag that was wrong**, so it only touches tokens tagged the way you judged incorrect — correctly-tagged occurrences of the same word are never rewritten. Global scope is the default because the flags collapse hard: 1,969 flag rows are only 1,291 distinct (word, wrong-tag) pairs, and `гора`/PROPN alone is 30 rows. Use `row` for the context-dependent `unstable_tag` cases, where the same form is legitimately a noun in one line and a verb in another.

Notes:
- A corrected POS clears `xpos`/`feats`, since classla derived them alongside the tag you just overruled. **An empty `xpos` in `pos_tagged.csv` means "corrected by hand".**
- Corrections are applied before the flag rules run, so a token you have fixed drops out of `pos_flags.csv` and the review queue shrinks as you work.
- `row`-scoped fixes, and fixes to tokens classla gave no content tag (the `unknown_tag` rows), need a full `pos_tag_corpus.py` run — `apply_corrections.py` stores them and says so.
- One bad value rejects the whole batch rather than half-applying it. To undo a correction, delete its row from `pos_corrections.csv`.

`pos_tag_corpus.py` also writes `data/pos_flags.csv`, an advisory audit of suspicious tags (~1.4% of tokens) grouped by `reason`. Nothing filters on it:

| reason | what it catches |
|---|---|
| `upos_xpos_clash` | `upos` and `xpos` disagree on the basic category (`upos=NOUN`, `xpos=I`). classla runs the two taggers separately, so a clash means neither reading is trustworthy — high precision |
| `unstable_tag` | the form is tagged differently elsewhere in the corpus, and this is the minority reading (thresholds: `MIN_OCC`, `MAX_SHARE`) |
| `lowercase_propn` | PROPN assigned to a lowercase token — usually a common noun |
| `unknown_tag` | classla returned no `upos` / `xpos=X` — mostly dialectal clippings (`ко`, `сал`, `та`) |
| `non_cyrillic`, `mixed_script`, `bad_lemma` | script artifacts reaching the tagger |

Rules that fire on *rarity* rather than *evidence* were tried and dropped: this corpus is poetry, so rare nouns and verbless lines are normal, and rarity-based rules ran at roughly 5% precision.
