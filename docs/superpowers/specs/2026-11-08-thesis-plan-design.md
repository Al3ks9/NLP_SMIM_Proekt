# Bachelor's thesis: plan and design

**Date:** 2026-11-08
**Status:** approved 2026-11-08
**Title:** Способноста на мали јазични модели за промена на стил на македонски
автори со примена на учење со поттикнување
(*The ability of small language models to transfer the style of Macedonian
authors using reinforcement learning*)

## Intent

A bachelor's thesis (FCSE Skopje) whose one take-away is: **small language
models are not good at writing Macedonian poetry in a given author's style, and
even a large model (the Gemma 4 31B teacher) is not great at it. The limit is
data, not the setup.** The setup is careful and sound, and the thesis has to
make that visible, because a negative result is only convincing if the
measurement can be trusted.

The thesis is a new text, not the two reports glued together. `docs/report.tex`
and `docs/rl_report.tex` are the source of numbers, findings and wording.

## Constraints

- Bachelor's thesis, no faculty template: standard one-column LaTeX, abstract,
  XeLaTeX (Cyrillic via fontspec/polyglossia, as in the reports).
- **20–30 pages** of body text, appendices not counted.
- Written in English first; the author translates to Macedonian chapter by
  chapter as each one is finished. The final thesis is entirely in Macedonian, abstract included; there is no
  English abstract.
- **About two weeks.** Writing happens while GPU jobs run, whenever they can.
- The usual HPC is fully reserved. The Gemma 4 E4B student **must** be in the
  thesis; the author is looking for another cluster.
- Two author counts, both correct, used deliberately: the corpus and the splits
  cover **1,239 poems by 38 authors**; everything stylometric (tagged corpus,
  profiles, TF-IDF, classifier, author vocabulary) covers the **24 authors with
  ≥ 5 poems** (`MIN_POEMS = 5`; 1,207 poems in `pos_tagged.csv`). Chapter 3
  introduces both and the filter; chapters 4–5 say 24. `report.tex`'s "2,711
  poems" is stale. `rl_report.tex` line 66 ("1,239 poems of the 38 profiled
  authors") is wrong (only 24 are profiled) and gets fixed in passing.

## Decisions made with the author

- The earlier approaches (word substitution, LLM infilling) appear briefly, as
  the path that led to the teacher–student setup, with the verdict that they
  do not work. They also get scored with the same automatic metrics, so the
  verdict is a number.
- Both students (Qwen3-4B and Gemma 4 E4B) are required.
- Human evaluation is in. The author rates; more raters if available. Layout
  "A": one output at a time, six 1–5 scales. 20 test pairs × 6 systems
  (teacher, Qwen SFT, Qwen GRPO-v2, Gemma SFT, Gemma GRPO, **real poem
  control**) = 120 items. Each item has a collapsible panel with three real
  poems by the target author.
- LLM infilling is described from the probes with one example, not scored.
  Word substitution is scored on the test pairs.
- "The limit is data" is **argued** in the thesis from existing evidence, with a
  CPU analysis behind it (B6). A data-scaling run (training on 50 % of the
  synthetic data, no new teacher generations) is **decided after the Gemma
  student has trained** (R8), since training is fast.
- Every pipeline stage contributes an example to the thesis; this document says
  how each one is produced.

## File layout

```
docs/thesis/
  main.tex              preamble, title page, abstract, \input of chapters
  ch1_introduction.tex  … ch10_conclusion.tex, app_*.tex
  references.bib
  figures/              generated figures copied or symlinked here
  mk/                   Macedonian versions, same file names, swapped in by main.tex
```

`main.tex` switches language with one macro so the English and Macedonian
chapter files can be compiled from the same preamble.

## Outline (page budget ≈ 27)

Status key: **write now** (material exists), **partial** (part waits on a run),
**wait** (needs a result that doesn't exist yet), **build** (needs code first).

| # | Chapter | Pages | Content | Sources | Status |
|---|---|---|---|---|---|
| — | Abstract (MK only) | 1 | problem, method, the negative result with 3–4 numbers | everything | last |
| 1 | Introduction | 2 | why Macedonian, why poetry, why small models; research question: *can a small LM, given only a source poem and an author's name, rewrite it in that author's voice, and does RL help?*; contributions; chapter map | — | write now |
| 2 | Background | 3 | text style transfer (supervised, unsupervised, prompting, PEGF infilling); stylometry and authorship attribution; knowledge distillation; LoRA; RL for LMs (RLHF/PPO → GRPO); Macedonian NLP (CLASSLA) | literature (citations to collect and verify) | write now |
| 3 | Data | 2.5 | the corpus (1,239 poems, 38 authors), cleaning (script artifacts, poem ids), tagging with CLASSLA and the three corrections, hand corrections, splits (987/126/126), **how little data there is** (poems per author distribution; ~115 teacher examples per target) | CLAUDE.md, report.tex §1, `pos_flags.csv`, `poem_splits.csv` | write now |
| 4 | Stylometric foundation | 2.5 | style profiles (POS shares, MATTR, structure, POS bigrams, rhyme), TF-IDF palettes, the knowledge graph (one paragraph + figure), the reward classifier vs. the independent judge (why two), validation fixes in brief (tagger migration, rhyme bug, community artifact) | report.tex, `style_features.py`, `judge_classifier_eval.json` | write now |
| 5 | The path here | 2 | (a) word substitution with morphological realisation; (b) LLM infilling probes (PEGF), small local models fail at Macedonian; (c) the scaffolded one-shot Gemma prompt, which works and becomes the teacher. Each with one example and, for (a), test-pair scores | report.tex §§5–6, probe logs, new eval (below) | partial (build: old-pipeline eval) |
| 6 | Method | 4 | teacher prompt and synthetic data; task format; SFT (both students); reward (style, content, gates, weights from within-group spread); GRPO (merge-then-LoRA, hyperparameters, v1 → v2); the Gemma student's differences (text-only load, loss masking, eos) | rl_report §§2–3, CLAUDE.md Gemma section | write now (Gemma details partial) |
| 7 | Evaluation setup | 2 | protocol (pairs, seeds, decoding), metric families, judge, decoding baseline, test-split discipline, human-evaluation protocol | rl_report §1, this doc | write now |
| 8 | Results | 6 | 8.1 earlier approaches vs. students (one table); 8.2 Qwen val → test; 8.3 per-author identification; 8.4 decoding baseline; 8.5 teacher; 8.6 Gemma student vs. Qwen; 8.7 human ratings; example outputs | rl_report §4, new runs | partial (Qwen written now; teacher-test, Gemma, human, old pipelines wait) |
| 9 | Discussion | 2 | why it fails: data scale, style collapse onto the two largest authors, nothing for GRPO's group advantage to amplify, reward-classifier gaming, the teacher's own limits; threats to validity (one seed, automatic author identity, one rater) | rl_report §4.3–4.4 | partial |
| 10 | Conclusion | 1 | answer to the research question; future work (more data, per-author data augmentation, preference data from human ratings) | — | last |
| A | Appendix: examples | — | one worked example per stage (table below) | runs below | partial |
| B | Appendix: prompts and hyperparameters | — | teacher prompt, student prompt, SFT/GRPO configs | `llm_transfer_logs`, `run_config.json` | write now |
| C | Appendix: reproducibility | — | the command table below, condensed | this doc | write now |

## Results and examples: how each one is produced

All commands from the repo root. "Where" = CPU (laptop), API (Gemini key in
`.env`), GPU (cluster). Outputs under `data/` are tracked unless noted.

### Already produced

| Result | Command | Output | Where |
|---|---|---|---|
| Qwen val comparison (base, SFT, GRPO v1/v2, teacher, real poems) | `uv run python -m students.shared.compare_generations --rewards --corpus-reference --runs …_val.csv … data/synthetic/teacher_val.csv` | `data/validation_generations/judge_v2_comparison_*.csv` | CPU |
| Qwen test comparison | `sbatch src/students/shared/hpc/eval_test.slurm score` (CPU-equivalent: same `compare_generations` call) | `data/validation_generations/test_comparison_*.csv` | done |
| Decoding baseline, checkpoint 300 | — | `data/validation_generations/judge_comparison_*.csv` | done |
| GRPO training curves | `uv run --with matplotlib python -m students.qwen.plot_grpo_curves` | `data/figures/grpo_training_curves.pdf` | CPU (reads `models/reward_components_grpo_v1.jsonl` / `_v2.jsonl`, present locally) |
| Judge held-out accuracy | `uv run python -m common.evaluation.judge_classifier --rebuild` | `data/judge_classifier_eval.json` | CPU |
| Reward within-group spread | `sbatch src/students/shared/hpc/reward_groups.slurm` | `data/grpo/reward_groups.csv` | done |

### To run

| # | Result | Command | Output | Where | Blocks |
|---|---|---|---|---|---|
| R1 | Teacher on test (**running**, started by the author) | `caffeinate -i src/gemma_teacher/generate_synthetic_parallel.sh 50 10 1 42 test` (resumes: ~100 of 466 rows already sit in `data/synthetic/parts_test/`; rerun until the count stops growing). **API only** (my call): same backend as `teacher_val.csv`, no vLLM rows mixed in. Quota-bound, so it may take several sessions over days 1–5 | `data/synthetic/teacher_test.csv` | API | 8.5, human eval |
| R2 | Test comparison incl. teacher and earlier approaches | `compare_generations --rewards --corpus-reference --runs <4 Qwen test CSVs> data/synthetic/teacher_test.csv <R4 CSV> --labels …` | `test_comparison_*.csv` (overwrites) | CPU | 8.1, 8.5 |
| R3 | Gemma student: fix merge check, then the HPC order in CLAUDE.md (compat → SFT → base-vs-SFT → GRPO → eval), plus test split | see "Gemma student" below | `data/validation_generations/gemma4-e4b-*` | GPU | 6, 8.6, human eval |
| R4 | Word substitution on the 466 test pairs | new script (build B1) | `data/validation_generations/word-substitution_test.csv` | CPU | 5, 8.1 |
| R5 | LLM-infilling example(s) | `uv run python -m llm_infill.probe_qwen` (Ollama) and/or `uv run python -m llm_infill.probe_google_api --model gemma-4-31b-it` (`probe_gemma` goes through OpenRouter's `:free` route, which 429s); `uv run python -m llm_infill.show_slots` for the slot candidates | console output → saved to `data/examples/infill_probe_<model>.txt` | CPU / API | 5 |
| R6 | Human-evaluation page | build B2, then rate | `data/human_eval/ratings_<name>.json` | browser | 8.7 |
| R7 | Human-evaluation summary | build B3 | `data/human_eval/summary.csv` | CPU | 8.7 |
| R8 | *Conditional:* data-scaling run — SFT (and GRPO if time) on a seeded 50 % of `synthetic_dataset.csv` per target, evaluated on the test pairs. Which student, and whether GRPO too, decided after R3 | `train_sft.py` on a subset file (small addition: `--train-fraction` or a pre-split CSV) | `data/validation_generations/<family>-sft-half_test.csv` | GPU | 9 (turns the data argument into a measurement) |

### Example per stage (Appendix A, and inline where noted)

One source poem threads through as many stages as possible: Кочо Рацин,
„Балада за непознатиот“ → Блаже Конески, already used in `report.tex`, plus
one test pair (Ацо Караманов, „И можеби ти ќе дојдеш тогаш“ → Блаже Конески)
for the model outputs.

| Stage | Example | How to get it |
|---|---|---|
| Tagging | 2–3 lines with UPOS, xpos, lemma, one correction | rows of `data/pos_tagged.csv` for that poem; one row from `data/pos_corrections.csv` |
| Style profile | Конески's profile as numbers and as prose | row of `data/author_style_profiles.csv`; `uv run python -c "from gemma_teacher.style_narrator import describe_style; print(describe_style('Блаже Конески'))"` |
| Knowledge graph | one figure (subgraph around an author) | `uv run python -m common.knowledge_graph.visualize_skg` → `skg_visual.html`, screenshot; or a static networkx/matplotlib subgraph (build, optional) |
| Word substitution | source → output, substituted words marked | the same `transfer_style(text, target, source_author)` call B1 makes (the 2-argument CLI omits `source_author`, which changes eligible tokens; the example must use B1's setting) — B4 prints it from B1's CSV |
| LLM infilling | masked stanza → small-model fill vs. Gemma fill | R5 |
| Teacher prompt | the full scaffolded prompt (shortened) + Gemma's poem | the **latest** of the three `data/llm_transfer_logs/*Кочо_Рацин__Блаже_Конески.json` (2026-08-31, matches the current prompt version; check `metadata` before use) |
| Base / SFT / GRPO v1 / v2 | the same test pair from each system | `data/validation_generations/{base-Qwen3-4B,qwen3-sft-merged,qwen3-grpo,qwen3-grpo-v2}_test.csv`, row `source_poem_id=43, target_author=Блаже Конески` |
| Gemma student | same pair | R3 outputs |
| Teacher on test | same pair | R1 |
| Failure modes | one degenerate (repetition) output, one copied-line output, one wrong-voice output | filter `test_comparison_rows.csv` on `line_uniqueness < 0.8`, `copy_novelty < 1`, `clf_predicted == 'Блаже Конески'` with another target |

A small helper (`src/students/shared/pick_examples.py`, build B4, optional)
prints all of the above for a given (poem id, target) so the examples can be
regenerated; otherwise they are pulled by hand with the commands listed.

## Builds

### B1. Word-substitution evaluation (earlier approach, quantitative)

`src/word_substitution/generate_test.py`: for each of the 466 test pairs
(`grpo_data`/`generate_validation` pair logic, same 50 poems × 10 targets,
self pairs skipped), call `transfer_style(source_text, target, source_author)`
and write a CSV with the same columns `compare_generations` reads
(`source_poem_id, source_author, source_title, target_author, split,
generated_poem`, …). Deterministic, CPU. Then include it in R2 with label
`word-substitution`. Expected outcome: high content recall and line-copying
(it edits the source in place), low author identification, which is the
point.

LLM infilling has no whole-poem pipeline (only the five-call probe harness on
one demo stanza), so it is reported qualitatively from the probes, not scored
on the test pairs. *(Open question for the author, see below.)*

### B2. Human-evaluation page

One self-contained HTML file, built by
`src/students/shared/human_eval.py build` → `data/human_eval/rating_page.html`
(tracked). No server; opens from disk; can be sent to other raters.

- **Items:** 20 test pairs, two per target author, chosen with a fixed seed
  among pairs that all five systems produced. Five outputs per pair plus the
  real-poem control → 120 items.
- **Real-poem control (retrieval baseline):** for each pair, the target
  author's train-split poem closest to the source by the content reward's
  line-level embedding F1 (`common.evaluation.content_similarity`), excluding
  the three poems shown in that author's panel (the rater would recognise
  them). It is shown exactly like a generated poem, with all six scales, so
  it can't be told apart by the form; the *keeps the source's content* score
  is excluded from its analysis (my call).
  Item ids are opaque; the system label lives only in the builder's key file
  (`data/human_eval/items_key.json`, tracked), never in the page.
- **Order:** start screen asks for the rater's name; order is a seeded shuffle
  of the rater's name, with the constraint that two outputs of the same pair are
  never adjacent.
- **Item screen (layout A):** header "Item n / 120 · target: <author>" and a
  progress bar; source poem (author, title) left, generated poem right; six
  1–5 scales: *correct Macedonian, coherent, poetic quality, keeps the source's
  content, sounds like <target>, usable as-is*; optional comment; Back / Next.
  Keys 1–5 set the focused scale, Tab/arrows move between scales, Enter = Next.
  Next is enabled once all six are set.
- **Author panel:** collapsible "About <author>" section, closed by default,
  with the same three real poems by the target for every item of that author.
  Chosen from the author's **train-split** poems with a fixed seed, preferring
  poems of 8–30 lines (builder's call).
- **Saving:** every answer goes to `localStorage` immediately (keyed by rater
  name), so the tab can be closed and resumed. An Export button (always visible,
  and prominent at the end) downloads `ratings_<name>.json`:
  `{rater, page_version, started, finished, ratings: [{item_id, scores:{…}, comment, seconds}]}`.
  The rater drops it into `data/human_eval/`.
- **Language of the page:** Macedonian labels (the raters are Macedonian
  speakers); English glosses in the thesis.
- **Staged build:** the page can be built before R1/R3 finish with the three
  Qwen-and-teacher systems that exist, but the **final** page needs all five;
  ratings from a partial page are not mixed with the final one.

### B3. Human-evaluation aggregation

`src/students/shared/human_eval.py summarize`: reads every
`data/human_eval/ratings_*.json`, joins with `items_key.json`, writes
`data/human_eval/summary.csv` (mean and bootstrap 95 % interval per system ×
scale) and prints the LaTeX table rows. With ≥ 2 raters, also Krippendorff's
α per scale. Tests for both B2 (item selection, no system leak in the HTML,
adjacency constraint) and B3 (aggregation on a fixture).

### B5. `eval_test.slurm` for both families

Honour `FAMILY` (as `eval_grpo.slurm` does): systems `base|sft|grpo` resolve
to the family's `models/` dirs and labels; `score` takes every `*_test.csv`
present (both families, teacher, word substitution). Qwen's existing labels
and outputs unchanged (test in `tests/test_layout.py` style).

### B6. Data-limit analysis (CPU)

`src/students/shared/data_limit.py`: per target author, number of corpus /
train poems and teacher examples against judge and classifier top-1 for every
system (val and test), plus each system's share of outputs read as each
author. One scatter figure (`data/figures/identification_vs_poems.pdf`) and a
table. Supports chapter 9's argument that the students only learn the authors
the corpus is rich in.

### B4. Example picker (optional)

`src/students/shared/pick_examples.py --poem-id 43 --target "Блаже Конески"`:
prints that pair's output from every system CSV it finds, plus the per-row
metrics. Saves copying by hand.

## Gemma student (R3) — the critical path

`data/gemma_compat_report.json`: load, LoRA, template, lengths and generation
pass; **`sft_merge` fails** (merged model's log-prob differs from
adapter-on by 0.136/token, tolerance 0.02). If the merge is wrong, GRPO would
start from, and be KL-anchored to, a model that is not the SFT student. This
has to be diagnosed **before** any cluster time is spent on SFT/GRPO.

0. `models/gemma4-compat/` is not on the laptop; on a new cluster re-run
   `gemma_compat.slurm` first to recreate the 3-step adapter.
1. Run `src/students/qwen/diagnose_merge.py --family gemma4` (it separates
   bf16 rounding, a wrong PEFT merge, and bf16 forward noise). Needs a GPU
   (E4B fits a single 24 GB+ card, e.g. a Colab/Kaggle-class GPU).
2. Branch on the diagnosis (my call): if it is bf16 rounding of the tiny
   compat adapter, confirm on the **real** SFT adapter right after Gemma SFT
   (`diagnose_merge` with `GEMMA_ADAPTER=models/gemma4-e4b-lora-sft`) before
   GRPO starts; if it is a real merge bug, fix the merge (e.g. merge in fp32).
   Rule: GRPO does not start until merged ≈ adapter-on for the real SFT adapter,
   so its start policy and KL anchor are the SFT student, as for Qwen.
3. Then, on whatever cluster is available: `FAMILY=gemma4 sbatch
   src/students/shared/hpc/train_sft.slurm` → `FAMILY=gemma4 sbatch
   src/students/shared/hpc/compare_base_lora.slurm` → `sbatch
   src/students/gemma/hpc/train_grpo_gemma.slurm` → `FAMILY=gemma4 sbatch
   src/students/shared/hpc/eval_grpo.slurm`, and a test-split eval.
   `eval_test.slurm` is Qwen-only (hard-codes `FAMILY=qwen3`, the systems and
   the `qwen3-*` paths): **build B5** extends it to honour `FAMILY` like
   `eval_grpo.slurm` does.
4. GPU budget estimate (from Qwen): SFT ~15 min, GRPO ~8 h, each eval ~2 h.

## Schedule (two weeks, writing in parallel with runs)

| Days | Runs / builds | Writing |
|---|---|---|
| 1 | R1 running; Gemma merge diagnosis → SFT → GRPO (R3, expected today); B5; B1 + R4; skeleton `docs/thesis/` | ch. 3 Data |
| 2 | R3 evals; R2 (with teacher + word substitution); B6 | ch. 4 Stylometric foundation |
| 3 | B2 + B3 (dry run with the systems that exist); decide R8 | ch. 5 Path here (with R4, R5) |
| 4–5 | R8 if chosen | ch. 6 Method, ch. 7 Evaluation setup |
| 6–7 | | ch. 8.1–8.5 (everything Qwen + teacher + earlier approaches), ch. 2 Background |
| 8–9 | R3 eval; final page with 5 systems + real-poem control; **rate** (~2.5 h) | ch. 1 Introduction, appendices |
| 10–11 | B3 summary | ch. 8.6 Gemma, 8.7 human, ch. 9 Discussion |
| 12–13 | | ch. 10, abstract, full read-through, translation catch-up |
| 14 | buffer | submit |

Fallback (expected not to be needed): the final rating page is built on day 9
at the latest with whatever Gemma outputs exist (SFT only, or none → 4
systems + control); Gemma automatic results still go in the thesis.

## Non-goals

- No new modelling experiments beyond what is listed (no new reward, no third
  student, no more data collection).
- No interactive results dashboard; the tables live in the thesis.
- No translation by Claude unless asked; the author translates.

## Left to the builder

- LaTeX preamble details, bibliography style, figure styling.
- Exact CSV column handling in B1 (match whatever `compare_generations` needs).
- Page styling of B2 within the layout above.

## Calls made by Claude (check while reading)

- Teacher test outputs come from the API only, no vLLM rows mixed in.
- GRPO for Gemma starts only once merged ≈ adapter-on for the real SFT adapter.
- Real-poem control is blind (all six scales shown); its content score is
  excluded in analysis.
- The three author-panel poems and the control poem never coincide.
- Examples follow one test pair (poem 43 → Блаже Конески) plus Рацин →
  Конески for the pre-student stages.
- Page in Macedonian; panel poems prefer 8–30 lines.
