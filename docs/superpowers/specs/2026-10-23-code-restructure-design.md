# Code restructure: one folder per end-to-end pipeline

## Intent

`src/` is a flat directory of ~48 modules spanning four pipelines plus the
corpus/analysis foundation they share, with their HPC jobs in a separate flat
`slurm/` and helpers scattered across `scripts/` and `singularity/`. Reading
the repo, you can't tell which pipeline a file belongs to. The thesis presents
the work as a sequence of style-transfer approaches, and the code should be
laid out the same way, so that each approach can be read end to end in one place.

## Goals

- Code is grouped as a shared foundation plus one folder per pipeline, in the
  thesis's order:
  1. **word substitution** — mask words, substitute target-author words, inflect them;
  2. **LLM infill** — an LLM picks the words for the masked slots;
  3. **Gemma teacher** — Gemma 4 31B rewrites from a summary + style profile;
  4. **students** — small models distilled from the teacher, then GRPO;
     Qwen3-4B and Gemma 4 E4B each have their own folder, and a `shared/`
     folder holds the training/eval code both use.
- Each pipeline's HPC jobs live with that pipeline.
- `tests/` mirrors `src/`.
- Behaviour is unchanged: the same commands produce the same outputs, read
  and written at the same `data/` and `models/` paths.

## Non-goals

- **`data/` and `models/` are not reorganised.** Every path a script reads or
  writes under them stays exactly the same. The user will decide on these
  separately, later.
- No refactoring beyond what the move forces: no renaming of modules, functions
  or CLI flags, and no logic changes.
- No changes to pipeline results or to tracked data files.

## Anti-goals (a failure even if it "works")

- Any change in output: a moved script that writes to a different path,
  reads a different file, or produces different numbers.
- Breaking the HPC workflow: the `.sif` images already built on the cluster
  must keep working without a rebuild, and a job must still be submitted from
  the repo root with a single `sbatch <path>`.
- Per-script `sys.path` hacks to make imports work.

## Target layout

```
src/
├── common/
│   ├── corpus/            add_poem_ids, strip_lines, make_splits
│   ├── tagging/           tagging, morph, corrections, apply_corrections,
│   │                      pos_tag_corpus, build_morph_lookup, classla_test
│   ├── stylometry/        tfidf_authors, style_features, style_profiler, classifier
│   ├── stylometry/targets.py   load_profile_rows, select_target_authors (from sft_data)
│   ├── knowledge_graph/   build_skg, enrich_skg, add_embeddings, graph_analytics,
│   │                      visualize_skg
│   ├── masking/           candidate_selection (+ load_target_words from llm_probe)
│   ├── evaluation/        style_metrics, content_similarity, judge_classifier
│   │                      (+ structural_targets, validate_structure)
│   └── llm_client.py
├── word_substitution/     style_transfer
├── llm_infill/            llm_probe, probe_qwen, probe_gemma, probe_nemotron,
│                          probe_google_api, show_slots (candidate_selection's demo)
├── gemma_teacher/         llm_style_transfer, style_narrator, exemplar_selection,
│   │                      poem_tfidf, generate_synthetic, merge_synthetic_parts,
│   │                      generate_synthetic_parallel.sh
│   └── hpc/               generate_synthetic.slurm, vllm_teacher.slurm, build_vllm.sh
└── students/
    ├── shared/            sft_data, model_families, train_sft, merge_sft_adapter,
    │   │                  grpo_data, grpo_rewards, build_reward_calibration,
    │   │                  evaluate_reward, train_grpo, generate_validation,
    │   │                  compare_generations
    │   └── hpc/           qwen-sft.def, build.sh, grpo_common.sh
    ├── qwen/              diagnose_merge, plot_grpo_curves
    │   └── hpc/           train_sft, grpo_smoke, reward_groups, train_grpo,
    │                      train_grpo_v2, eval_grpo, eval_test, compare_base_lora,
    │                      decoding_ablation, generate_validation, diagnose_merge
    └── gemma/             gemma_compat
        └── hpc/           gemma_compat, train_grpo_gemma
tests/                     same tree as src/ (fixtures/ stays at tests/fixtures/)
docs/                      report.tex, report.pdf, rl_report.tex, rl_report.pdf
                           move here from data/; specs/ and plans/ stay put
```

The top-level `slurm/`, `scripts/` and `singularity/` folders disappear, except
for `singularity/*.sif` (see Constraints). `songs.csv` and
`artifact_lines_report.txt` move from the repo root into `data/`, and the code
that reads them (`strip_lines`) is updated to match.

**`lib/` stays at the repo root** *(designer's call)*. pyvis copies its JS
assets into `./lib` next to `skg_visual.html` whenever that folder is
missing, and the HTML's output location is fixed by the Non-goals. Moving
`lib/` would only make pyvis recreate it at the root.

**`docs/rl_report.tex`:** after the move, its `\includegraphics{figures/...}`
must still resolve to `data/figures/`.

Two cross-pipeline links must be cut so the dependency direction is always
pipeline → `common/` (never pipeline → pipeline, and never `common/` → pipeline):

- `style_metrics.py` (common/evaluation, the GRPO reward) imports
  `structural_targets` and `validate_structure` from `llm_style_transfer.py`
  (gemma_teacher). Those two functions, and whatever they depend on, move into
  `common/evaluation/`. `llm_style_transfer` imports them from there.
- `candidate_selection.py` is used by pipelines 1, 2 and 3, which is why it
  sits in `common/masking/`.

The students' dependency on `gemma_teacher` is real and stays: they train on
its output. Specifically, `generate_validation` imports `generate_synthetic`'s
pair sampling and `grpo_data` imports `build_pairs`. A dependency from a later
pipeline on an earlier one is acceptable. A dependency on a later pipeline, or
from `common/` on any pipeline, is not.

The review found more imports that break that rule. Each is fixed by moving
code verbatim, with no logic changes:

- `sft_data.load_profile_rows` and `sft_data.select_target_authors` are used by
  `generate_synthetic` (gemma_teacher), `judge_classifier` and
  `compare_generations` (common), and `build_reward_calibration` (students).
  They move to `common/stylometry/targets.py`. `sft_data` imports them from
  there, so existing callers of `sft_data.select_target_authors` keep working.
- `compare_generations` imports `grpo_rewards` for `--rewards`, and its job is
  comparing student generations with the teacher and real poems. It moves to
  `students/shared/`, not `common/evaluation/`.
- `candidate_selection` lazily imports from `llm_probe`:
  - `load_target_words` is used by the legacy-fallback tier. It moves to
    `common/masking/`, and `llm_probe` imports it from there.
  - The `if __name__ == '__main__':` demo uses the probe's `MASK_WORDS`,
    `SOURCE_AUTHOR`, `TARGET_AUTHOR` and `load_stanza`. That block moves
    verbatim to a new `llm_infill/show_slots.py`.

  *(Builder's call, settled by the designer.)*

The final check enforces the dependency direction: a script greps the
imports and fails on any `common` → pipeline import or any pipeline →
later-pipeline import.

## Decided

- **Folder names, not numbers.** Python can't import a package whose name
  starts with a digit. The pipeline order is documented in `CLAUDE.md` and a
  short `README.md` in `src/`.
- **One import root, `python -m`.** `src/` is the import root. Every folder is
  a package (`__init__.py`). Imports are absolute from that root (`from
  common.evaluation import style_metrics`). Scripts run as
  `uv run python -m students.shared.train_sft …` instead of
  `uv run python src/train_sft.py …`.
- **Students layout:** `students/qwen/` and `students/gemma/` hold what is
  specific to that model, and everything the `--family` flag already
  parametrises lives once in `students/shared/`.
- **Code first.** `data/` and `models/` paths are untouched (Non-goals).

## Constraints

- **Paths.** 38 modules compute the repo root as
  `Path(__file__).resolve().parent.parent`, which breaks at a deeper nesting.
  Replace them with one shared definition (e.g. `common/paths.py` exporting
  `ROOT`, `DATA`, `MODELS`) so the depth assumption lives in one place.
- **Containers.** `qwen-sft.def` builds its venv with `uv sync --frozen` and
  the jobs call `/opt/venv/bin/python` inside it, with the repo bound at
  `/workspace`. Inside the container the import root must be reachable as
  `/workspace/src` (e.g. `SINGULARITYENV_PYTHONPATH`/`--env` in
  `grpo_common.sh` and the other job wrappers), so **the existing
  `qwen-sft.sif` and `vllm.sif` keep working unrebuilt**. If the chosen import
  mechanism changes what `uv sync` installs, the `.def` must still build. The
  built images stay at `singularity/*.sif` (gitignored), so that the cluster's
  copies don't need to be moved or rebuilt.
- **Slurm.** Jobs are still submitted from the repo root (`sbatch
  src/students/qwen/hpc/train_grpo_v2.slurm`). Each job's "find the repo root"
  preamble (`cd "$SLURM_SUBMIT_DIR"; [[ -f slurm/x ]] || cd ..`) must be updated
  to the new depth. `logs/` stays at the repo root.
- **Cached pickle** *(designer's call)*. `models/judge_classifier.pkl`
  stores a reference to `style_metrics.tokens` by module path, so after the
  move it fails to load with `ModuleNotFoundError` rather than refitting.
  - The fix is a documented step: delete the cached file and refit (`--rebuild`,
    about 2 s), on the laptop and on the cluster.
  - Before deleting, record the judge's numbers with the old code
    (`data/judge_classifier_eval.json`, and the judge columns for one
    generations CSV), and confirm the refit reproduces them exactly.
  - No alias module.
  - `classifier.pkl` holds only sklearn and numpy objects and is unaffected.
- **Moves are `git mv`** so that history follows each file, with renames kept
  separate from content edits where practical, so rename detection holds.
- **Tests.** The full suite (364 tests on `grpo-v2`) passes after the move,
  with the same count. `pyproject.toml`'s pytest `pythonpath = ["src"]` already
  matches the new import root.
- **Text references.** `CLAUDE.md`, the docstrings' `Usage:` lines, slurm
  comments, and `report.tex`/`rl_report.tex` mention `src/<file>.py` paths.
  `CLAUDE.md` and the run commands must be updated. Paths quoted inside the
  report should be updated where they name a runnable command.
- **No commits.** The user commits; leave the work uncommitted, or as WIP
  commits on a branch if the plan needs checkpoints and the user agrees.

## Sequencing

1. `main` already contains `grpo-v2` (fast-forwarded locally, not yet pushed).
2. The restructure happens directly on `main`, in the main checkout (user's choice).
3. `local-gemma-backend` (the vLLM teacher work, committed as `af07a26` but
   untested on the HPC and not merged) is merged in afterwards; git follows the renames. Its edits touch
   `llm_client.py`, `generate_synthetic.py`, `merge_synthetic_parts.py` and
   `generate_synthetic_parallel.sh`, which are all moved files, and it adds
   `vllm_teacher.slurm` and `build_vllm.sh`, which belong in
   `gemma_teacher/hpc/`.

## Left to the builder

- The exact import mechanism behind "`src/` is the import root": a
  `PYTHONPATH`/`uv` configuration or making the project installable. Any
  mechanism is acceptable if it meets the constraints above and adds no
  per-script path hacks.
- Whether to keep thin compatibility shims at old paths. The default is no.
- Order of the moves, and how to verify each step (beyond the full suite
  passing at the end, plus a dry check that every slurm job's commands point
  at modules that exist).
