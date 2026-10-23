# Code Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regroup the flat `src/`, `slurm/`, `scripts/` and `singularity/` into a shared `common/` foundation plus one folder per pipeline (word substitution → LLM infill → Gemma teacher → students), without changing any behaviour, output path or number.

**Architecture:** `src/` becomes the single import root, holding five top-level packages (`common`, `word_substitution`, `llm_infill`, `gemma_teacher`, `students`). These are installed editable via hatchling, so `uv run python -m <pkg>.<module>` works from anywhere. Inside the container, jobs get `PYTHONPATH=/workspace/src`, so the existing `.sif` images keep working. The first task cuts the cross-pipeline links in the *flat* layout and verifies them with the existing suite. A second task then does the mechanical move. That keeps logic edits and renames apart.

**Tech Stack:** Python 3.11, uv, hatchling (build backend only), pytest, bash/Slurm, Singularity.

**Spec:** `docs/superpowers/specs/2026-10-23-code-restructure-design.md`

## Global Constraints

- Every path read or written under `data/` and `models/` stays byte-for-byte the same. The only exceptions are `songs.csv` and `artifact_lines_report.txt` moving into `data/`, and `report*.tex/pdf` moving into `docs/`.
- No renaming of modules, functions or CLI flags, and no logic changes. Moved functions move verbatim.
- No per-script `sys.path` hacks. The one in `scripts/diagnose_merge.py` and the one in `generate_synthetic_parallel.sh` are removed.
- Imports are absolute from `src/` (`from common.evaluation import style_metrics`).
- Dependency direction:
  - `common` imports no pipeline.
  - A pipeline imports `common` and earlier pipelines only. The order is `word_substitution` < `llm_infill` < `gemma_teacher` < `students`.
  - `students.shared` imports neither `students.qwen` nor `students.gemma`, and those two don't import each other.
- Slurm jobs are submitted from the repo root. `logs/` stays at the root. Built images stay at `singularity/*.sif` (gitignored), along with `singularity/.build_tmp` and `singularity/.build_cache`.
- `lib/` stays at the repo root.
- Moves use `git mv`.
- The user commits by default. Each task ends in a checkpoint. Commit it only if the user has agreed to per-task commits on `main`, and otherwise leave it staged and move on.

## Module map (old flat name → new dotted path)

| old | new |
|---|---|
| add_poem_ids, strip_lines, make_splits | `common.corpus.<same>` |
| tagging, morph, corrections, apply_corrections, pos_tag_corpus, build_morph_lookup, classla_test | `common.tagging.<same>` |
| tfidf_authors, style_features, style_profiler, classifier | `common.stylometry.<same>` |
| *new* targets (Task 1) | `common.stylometry.targets` |
| build_skg, enrich_skg, add_embeddings, graph_analytics, visualize_skg | `common.knowledge_graph.<same>` |
| candidate_selection | `common.masking.candidate_selection` |
| style_metrics, content_similarity, judge_classifier, *new* structure (Task 1) | `common.evaluation.<same>` |
| llm_client | `common.llm_client` |
| style_transfer | `word_substitution.style_transfer` |
| llm_probe, probe_qwen, probe_gemma, probe_nemotron, probe_google_api, *new* show_slots (Task 1) | `llm_infill.<same>` |
| llm_style_transfer, style_narrator, exemplar_selection, poem_tfidf, generate_synthetic, merge_synthetic_parts | `gemma_teacher.<same>` |
| sft_data, model_families, train_sft, merge_sft_adapter, grpo_data, grpo_rewards, build_reward_calibration, evaluate_reward, train_grpo, generate_validation, compare_generations | `students.shared.<same>` |
| scripts/diagnose_merge.py, scripts/plot_grpo_curves.py | `students.qwen.<same>` |
| gemma_compat | `students.gemma.gemma_compat` |

Test `tests/test_<m>.py` moves to `tests/<same folders as m>/test_<m>.py`, for example `tests/common/evaluation/test_style_metrics.py`. `tests/fixtures/` stays where it is.

HPC files:
- `slurm/generate_synthetic.slurm` → `src/gemma_teacher/hpc/`
- `scripts/generate_synthetic_parallel.sh` → `src/gemma_teacher/`
- `singularity/qwen-sft.def`, `singularity/build.sh`, `slurm/grpo_common.sh` → `src/students/shared/hpc/`
- `slurm/gemma_compat.slurm`, `slurm/train_grpo_gemma.slurm` → `src/students/gemma/hpc/`
- every other `slurm/*.slurm` → `src/students/qwen/hpc/`

## Review Focus

1. **Patch targets that silently stop patching.** A test that monkeypatches `lst.structural_targets` keeps passing while patching the wrong module. Callers must call `structure.structural_targets(...)` through the module, and the tests must patch `common.evaluation.structure` (Task 1).
2. **Lazy, in-function imports** (`judge_classifier.py:118`, `compare_generations.py:182`, `candidate_selection`) are invisible to a top-of-file grep. The layout test (Task 3) parses every `import` node with `ast`, not just module-level ones.
3. **Root resolution at the new depth.** Any `Path(__file__)…parent.parent` left behind resolves to `src/` or `src/common/`, and writes outputs there instead of `data/`. Task 3's grep must find zero, and the test in Task 2 pins `paths.ROOT`.
4. **Container jobs on the existing `qwen-sft.sif`.** That image was built without the project installed, so `python -m students…` fails unless `PYTHONPATH=/workspace/src` is set inside every container body (Task 4's check script).
5. **The cached `judge_classifier.pkl`** pickles `style_metrics.tokens` by module path, and raises `ModuleNotFoundError` after the move. Task 5 refits it, and checks the numbers are identical.

---

### Task 0: Baseline

**Files:** none in the repo. Write outputs to `$SCRATCH=/private/tmp/claude-502/restructure-baseline/`.

- [ ] **Step 1:** `uv run pytest -q 2>&1 | tail -1 > $SCRATCH/pytest.txt`. Expected: `N passed` with no failures. Record N; on `grpo-v2` it was 364.
- [ ] **Step 2:** `cp data/judge_classifier_eval.json $SCRATCH/`. Then record the judge columns for one CSV: `uv run python src/compare_generations.py data/validation_generations/qwen3-grpo-v2_val.csv --rows-out $SCRATCH/judge_rows.csv`. Check the real flag name with `--help` and use it. The goal is to save the per-row `judge-prob`/`judge-top1` values.
- [ ] **Step 3:** `git status --short > $SCRATCH/status.txt`. Two untracked `judge_comparison_*.csv` files belong to the user; don't touch them.

### Task 1: Cut the cross-pipeline links (flat layout)

Moves code verbatim into three new flat modules, so that Task 3 only renames.

**Files:**
- Create: `src/targets.py`, `src/structure.py`, `src/show_slots.py`
- Modify:
  - `src/sft_data.py:92-…`: move `load_profile_rows` and `select_target_authors` out, then import them back.
  - `src/llm_style_transfer.py:151-215`: move `load_profile_rows`, `STRUCTURAL_COLUMNS`, `structural_targets`, `_tokens` and `validate_structure` out, then import them back. `format_structural_instruction` stays.
  - `src/style_metrics.py:152-153` and `src/generate_validation.py:242-243`: call `structure.structural_targets` / `structure.validate_structure`. Drop the `lst` import from `style_metrics`.
  - `src/candidate_selection.py`: receives `load_target_words`, and loses its `__main__` block.
  - `src/llm_probe.py:126-141`.
  - `src/generate_synthetic.py:42`, `src/judge_classifier.py:118`, `src/compare_generations.py:37`, `src/build_reward_calibration.py:35`, `src/gemma_compat.py:178`: import from `targets`.
- Test: `tests/test_targets.py`, `tests/test_structure.py`, and edits to `tests/test_generate_validation.py:76,108`.

**Interfaces (Produces):**
- `targets.load_profile_rows() -> list` and `targets.select_target_authors(n: int, explicit: list = None) -> list`. These are verbatim from `sft_data`, which re-exports both names, so `sft_data.select_target_authors` keeps working.
- `structure.load_profile_rows() -> dict` (verbatim from `llm_style_transfer`, with its own `_CACHE`/`_cached` copy). Also `structure.STRUCTURAL_COLUMNS`, `structure.structural_targets(target_author: str) -> dict` and `structure.validate_structure(generated_text: str, targets: dict) -> dict`. `llm_style_transfer` re-exports all four names.
- `candidate_selection.load_target_words(n: int, author: str) -> list[str]`. `author` is required and there is no `TARGET_AUTHOR` default. The body is otherwise verbatim, including the hard-coded fallback list. `llm_probe.load_target_words(n=15, author=None)` stays as a wrapper: `return candidate_selection.load_target_words(n, author or TARGET_AUTHOR)`.
- `show_slots.py` is `candidate_selection`'s old `__main__` block, verbatim, under its own `if __name__ == '__main__':`. It imports `build_slots` from `candidate_selection` and the probe constants from `llm_probe`.

- [ ] **Step 1: Write failing tests.**
  - `tests/test_targets.py::test_sft_data_reexports_targets` asserts `sft_data.select_target_authors is targets.select_target_authors` and the same for `load_profile_rows`.
  - `tests/test_structure.py::test_lst_reexports_structure` asserts the same identity for `structural_targets`, `validate_structure` and `STRUCTURAL_COLUMNS` between `llm_style_transfer` and `structure`.
  - `tests/test_structure.py::test_style_metrics_does_not_import_teacher` asserts `'llm_style_transfer' not in Path(style_metrics.__file__).read_text()`.
  - `tests/test_candidate_selection.py::test_legacy_fallback_needs_no_probe` asserts `'llm_probe' not in` the source of `candidate_selection`.
- [ ] **Step 2:** `uv run pytest tests/test_targets.py tests/test_structure.py tests/test_candidate_selection.py -q`. Expected: FAIL (ImportError / assertion).
- [ ] **Step 3: Make the moves listed above.**
  - In `tests/test_generate_validation.py`, change `monkeypatch.setattr(lst, 'structural_targets', …)` to `monkeypatch.setattr(structure, 'structural_targets', …)` (two places).
  - `candidate_selection.tier_legacy_fallback` calls the local `load_target_words(author=target_author)` without the lazy import, and its comment about circularity goes.
- [ ] **Step 4:** `uv run pytest -q`. Expected: N + 4 passed and 0 failed.
- [ ] **Step 5:** `uv run python src/show_slots.py | head -5`. Expected: the same JSON that `uv run python src/candidate_selection.py | head -5` printed before Step 3. Capture that output before Step 3 to compare.
- [ ] **Step 6: Checkpoint.** `git add -A src tests`, then commit `Move shared helpers out of pipeline modules before restructure` if per-task commits are agreed.

### Task 2: Packaging and the shared root path

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (by `uv lock`), `singularity/qwen-sft.def:55`
- Create: `src/common/__init__.py`, `src/common/paths.py`, `src/README.md`, and an empty `__init__.py` in every package directory of the target tree:
  - `common/{corpus,tagging,stylometry,knowledge_graph,masking,evaluation}`
  - `word_substitution`, `llm_infill`, `gemma_teacher`
  - `students/{,shared,qwen,gemma}`

  No `__init__.py` in the `hpc/` folders.
- Test: `tests/common/test_paths.py`

**Interfaces (Produces):** `common.paths.ROOT: Path` (the repo root, `Path(__file__).resolve().parents[2]`), `common.paths.DATA = ROOT / 'data'`, `common.paths.MODELS = ROOT / 'models'`.

- [ ] **Step 1: Write the failing test.** `tests/common/test_paths.py::test_root_is_repo_root` asserts `(paths.ROOT / 'pyproject.toml').is_file()`, `paths.DATA == paths.ROOT / 'data'` and `paths.MODELS == paths.ROOT / 'models'`.
- [ ] **Step 2:** `uv run pytest tests/common/test_paths.py -q`. Expected: FAIL (ModuleNotFoundError).
- [ ] **Step 3: Edit `pyproject.toml`.**
  - Add `[build-system] requires = ["hatchling"]`, `build-backend = "hatchling.build"`.
  - Add `[tool.hatch.build.targets.wheel] packages = ["src/common", "src/word_substitution", "src/llm_infill", "src/gemma_teacher", "src/students"]`.
  - Keep `[tool.pytest.ini_options] pythonpath = ["src"]`.
  - Run `uv lock && uv sync`.
  - In `qwen-sft.def`, change the line to `uv sync --frozen --no-install-project --python 3.11`, because `src/` isn't in the build context. The comment above that line should say that jobs reach the code through `PYTHONPATH`.
- [ ] **Step 4: Write `src/README.md`.** One short paragraph per package, in pipeline order, with the dependency rule, and how to run a module (`uv run python -m gemma_teacher.llm_style_transfer …`).
- [ ] **Step 5:** `uv run pytest tests/common/test_paths.py -q && uv run python -c "import common.paths, students"` from `/tmp`. Expected: PASS, with no error from `/tmp`, which shows the editable install works outside the repo.
- [ ] **Step 6:** `uv run pytest -q`. Expected: same count as Task 1 + 1, 0 failed.
- [ ] **Step 7: Checkpoint.** `Make src/ an installable import root`.

### Task 3: Move the Python modules and tests

One atomic change: the imports can't be half-moved.

**Files:** every `src/*.py` and `tests/test_*.py`, plus `scripts/diagnose_merge.py` and `scripts/plot_grpo_curves.py`, moved per the Module map. Create `tests/test_layout.py`.

**Interfaces (Consumes):** `common.paths` (Task 2), and the Task 1 modules `targets`, `structure`, `show_slots`.

- [ ] **Step 1: Write the failing layout test.** Write `tests/test_layout.py::test_dependency_direction`. It walks `src/**/*.py`, parses each file with `ast`, and collects every `Import`/`ImportFrom` at any depth. It maps each import's first dotted component to a rank: `common`=0, `word_substitution`=1, `llm_infill`=2, `gemma_teacher`=3, `students`=4. It asserts, for every file:
  - an import's rank ≤ the file's own rank;
  - a `common` file never imports rank > 0;
  - `students/shared` never imports `students.qwen`/`students.gemma`;
  - `students/qwen` and `students/gemma` never import each other.

  It also needs `test_no_flat_imports`: no import's first component equals any old flat module name from the map, which catches a missed rewrite. Run it. Expected: FAIL, because the flat modules are still imported.
- [ ] **Step 2: Move the files.** `git mv` each file to its mapped location, including `targets.py` → `common/stylometry/`, `structure.py` → `common/evaluation/`, and `show_slots.py` → `llm_infill/`.
- [ ] **Step 3: Rewrite imports.** Use a throwaway codemod in the scratchpad, not committed, driven by the Module map. Apply it to `src/` and `tests/`:
  - `import X` → `from <pkg> import X`
  - `import X as Y` → `from <pkg> import X as Y`
  - `from X import a, b` → `from <pkg>.X import a, b`

  It applies at any indentation. Multi-line parenthesised imports keep their bodies.
- [ ] **Step 4: Replace root resolution.**
  - In every moved module, replace `ROOT = Path(__file__).resolve().parent.parent` with `from common.paths import ROOT`, and likewise `DATA`/`MODELS` where a module defines them as `ROOT / 'data'` / `ROOT / 'models'`. Module-specific names like `MODELS_DIR` and `CACHE_PATH` keep their definitions, built on the imported names.
  - Delete the `sys.path.insert` in `diagnose_merge.py`.
  - In tests:
    - `test_model_families.py:9` and `test_tagging.py:154` use `paths.ROOT`;
    - `test_morph.py:235` uses `paths.MODELS`;
    - `test_model_families.py:11`'s `FIXTURE` resolves via `paths.ROOT / 'tests' / 'fixtures' / …`.
  - In `test_judge_classifier.py:7,62`, replace the `SRC / name` lookups with the new files: `students/shared/grpo_rewards.py`, `students/shared/train_grpo.py` and `common/evaluation/style_metrics.py`.
  - `strip_lines.py` reads `DATA / 'songs.csv'`. `git mv artifact_lines_report.txt data/` and `mv songs.csv data/` (untracked).
- [ ] **Step 5:** `grep -rn "parent\.parent\|sys\.path" src tests`. Expected: no output.
- [ ] **Step 6:** `uv run pytest -q`. Expected:
  - The same count as Task 2 plus the 2 layout tests, and 0 failed.
  - Any `judge_classifier` failure from loading the cached pickle is expected here. Note it, and leave it for Task 5.
- [ ] **Step 7: Smoke the entry points.** Run each from the repo root. Expected: each prints argparse help, or for the modules without argparse, starts without an ImportError (interrupt the interactive one).
  - `uv run python -m students.shared.train_grpo --help`
  - `uv run python -m students.shared.compare_generations --help`
  - `uv run python -m gemma_teacher.llm_style_transfer --help`
  - `uv run python -m gemma_teacher.merge_synthetic_parts --help`
  - `uv run python -m students.gemma.gemma_compat --help`
  - `uv run python -m word_substitution.style_transfer < /dev/null`
- [ ] **Step 8:** `rm -rf src/__pycache__ tests/__pycache__`. These are stale and untracked.
- [ ] **Step 9: Checkpoint.** `Restructure src/ into common + per-pipeline packages`.

### Task 4: Move the HPC and shell files

**Files:** per the HPC map. Modify each moved `.slurm`/`.sh`, and `tests/students/shared/test_model_families.py:93,101`.

- [ ] **Step 1: Write a failing test.** Add `tests/test_layout.py::test_hpc_scripts_reference_live_modules`. For every `src/**/hpc/*` and `src/**/*.sh` it checks:
  - every `-m a.b.c` names a module where `importlib.util.find_spec` is not None;
  - no text matches `\bsrc/\w+\.py`, `\bslurm/`, `\bscripts/` or `singularity/(qwen-sft\.def|build\.sh)`;
  - every file that contains `singularity exec`, or sources `grpo_common.sh`, contains `PYTHONPATH=/workspace/src`.

  Run it. Expected: FAIL, because there are no files there yet.
- [ ] **Step 2: Move the files.** `git mv` per the HPC map.
- [ ] **Step 3: Edit each moved file.**
  - `$PY src/X.py` and `/opt/venv/bin/python src/X.py` become `… -m <dotted X>`.
  - `uv run python src/X.py` becomes `uv run python -m <dotted X>`.
  - The parallel script's inline `python -c` drops `sys.path.insert` and imports `from common.stylometry.targets import select_target_authors`.
  - The preamble `[[ -f slurm/<job>.slurm ]] || cd ..` becomes `[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch $0" >&2; exit 1; }`.
  - `source slurm/grpo_common.sh` becomes `source src/students/shared/hpc/grpo_common.sh`.
  - Add `export PYTHONPATH=/workspace/src` next to the existing `export PYTHONUNBUFFERED…` in `grpo_common.sh`'s `run_in_container`, and in every job that calls `singularity exec` itself: `train_sft`, `compare_base_lora`, `decoding_ablation`, `generate_validation`.
  - Update the "sbatch slurm/…", "./singularity/build.sh" and "uv run python src/…" strings in error messages and comments to the new paths.
  - `build.sh` resolves the root with `cd "$(dirname "$0")/../../../.."` and builds from `src/students/shared/hpc/qwen-sft.def`. The `.def`'s `%files` paths (`pyproject.toml`, `uv.lock`) are relative to the cwd, which is the root, so they're unchanged. The output stays `singularity/qwen-sft.sif`.
  - In `test_model_families.py`, `source slurm/grpo_common.sh` becomes the new path.
- [ ] **Step 4:** `for f in $(git ls-files 'src/**/hpc/*' 'src/**/*.sh'); do bash -n "$f" || echo "BAD $f"; done`. Expected: no `BAD`.
- [ ] **Step 5:** `uv run pytest -q`. Expected: previous count + 1, and 0 failed (the Task 5 judge caveat still applies).
- [ ] **Step 6:** `ls slurm scripts singularity`. Expected:
  - `slurm` and `scripts` don't exist, apart from untracked leftovers, which you should report to the user rather than delete;
  - `singularity/` holds only `*.sif`/build caches, if any.
- [ ] **Step 7: Checkpoint.** `Move Slurm/Singularity/shell files next to their pipelines`.

### Task 5: Refit the judge, and verify it's unchanged

**Files:** `models/judge_classifier.pkl` (gitignored), `data/judge_classifier_eval.json` (tracked, must end unchanged).

- [ ] **Step 1:** `rm models/judge_classifier.pkl && uv run python -m common.evaluation.judge_classifier --rebuild`. Expected: it writes the pickle and the eval json.
- [ ] **Step 2:** `git diff --exit-code data/judge_classifier_eval.json`. Expected: exit 0.
- [ ] **Step 3:** Rerun Task 0 Step 2's command as `-m students.shared.compare_generations`, into `$SCRATCH/judge_rows_after.csv`. Then compare: `diff <(cut -d, -f<judge cols> $SCRATCH/judge_rows.csv) <(… _after.csv)`. Expected: no differences.
- [ ] **Step 4:** `uv run pytest -q`. Expected: everything passes, and the count matches Task 4.
- [ ] **Step 5:** Tell the user: on the cluster, after pulling, run `rm models/judge_classifier.pkl` once. It refits on first use.

### Task 6: Reports, `CLAUDE.md`, docstrings

**Files:**
- `git mv data/report.tex data/report.pdf data/rl_report.tex data/rl_report.pdf docs/`
- Modify:
  - `docs/rl_report.tex:265`
  - `docs/report.tex` and `docs/rl_report.tex`: path mentions
  - `CLAUDE.md`
  - every moved module's docstring `Usage:` lines

- [ ] **Step 1: Fix the figure path.** In `docs/rl_report.tex`, add `\graphicspath{{../data/figures/}}` to the preamble and leave line 265 as it is.
  - Run `cd docs && xelatex -interaction=nonstopmode rl_report.tex && xelatex -interaction=nonstopmode report.tex`. Expected: both PDFs build, with no `File … not found`.
  - Remove the `.aux/.log/.out` byproducts unless the repo already tracks them.
- [ ] **Step 2: Rewrite path text.** With a throwaway sed or codemod driven by both maps, apply to `CLAUDE.md`, `docs/*.tex` and `src/**/*.py` docstrings/comments:
  - `uv run python src/X.py` → `uv run python -m <dotted>`;
  - `src/X.py` → `src/<new path>/X.py`;
  - `slurm/X.slurm`, `scripts/X`, `singularity/qwen-sft.def`, `singularity/build.sh` and `slurm/grpo_common.sh` → their new paths.
  - Leave `singularity/*.sif` / `singularity/vllm.sif` mentions as they are.
  - In `CLAUDE.md`, rewrite the "Folder structure" section to the new tree, the "All scripts resolve paths via…" line to point at `common/paths.py`, and the `data/report.tex` references to `docs/report.tex`. Also change "Python 3.10+" to 3.11, matching `.python-version`.
- [ ] **Step 3:** `grep -rnE "\bsrc/[a-z_]+\.py|\bslurm/|\bscripts/|data/(rl_)?report\.(tex|pdf)" CLAUDE.md docs/*.tex src`. Expected: no output, except dated specs/plans under `docs/superpowers/`, which are historical and are not rewritten.
- [ ] **Step 4:** `uv run pytest -q`. Expected: everything passes.
- [ ] **Step 5: Checkpoint.** `Update docs and CLAUDE.md for the new layout`.

### Task 7: Final check and handover notes

- [ ] **Step 1:** `uv run pytest -q`. Expected: Task 0's N + 4 (Task 1) + 1 (Task 2) + 3 (layout) passed, and 0 failed.
- [ ] **Step 2:** `git status --short`. Expected: no stray files apart from the user's two untracked `judge_comparison_*.csv` files.
- [ ] **Step 3: Report to the user.** Cover:
  - the new commands, e.g. `sbatch src/students/qwen/hpc/train_grpo_v2.slurm`;
  - the cluster's one-time steps: `git pull`, then `uv sync` on the login node, then `rm models/judge_classifier.pkl`;
  - **merging `local-gemma-backend`:**
    - git follows the renames for `llm_client.py`, `generate_synthetic.py`, `merge_synthetic_parts.py` and `generate_synthetic_parallel.sh`;
    - its two new files land at the old `slurm/vllm_teacher.slurm` and `singularity/build_vllm.sh`, and must be `git mv`'d to `src/gemma_teacher/hpc/` and edited per Task 4 Step 3 (preamble, `-m`, `PYTHONPATH`, `scripts/` path). `test_hpc_scripts_reference_live_modules` will catch anything missed;
    - its `CLAUDE.md` hunks will conflict and must be re-applied by hand.
