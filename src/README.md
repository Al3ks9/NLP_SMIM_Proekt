# src/

`src/` is the import root: five packages, installed editable by `uv sync`.
Run any module from anywhere with `uv run python -m <package>.<module>`, e.g.
`uv run python -m gemma_teacher.llm_style_transfer --help`.

The pipelines are the thesis's successive approaches to style transfer, in order:

- **`common/`**: the foundation every pipeline uses. Corpus preparation and
  splits (`corpus/`), POS tagging and its corrections (`tagging/`), author
  TF-IDF, style profiles, target-author selection and the author classifier
  (`stylometry/`), the semantic knowledge graph and word embeddings
  (`knowledge_graph/`), per-slot candidate selection for masked words
  (`masking/`), evaluation metrics shared with the GRPO reward
  (`evaluation/`), the LLM client, and `paths.py` (repo-root paths).
- **`word_substitution/`** (1): mask words in the source poem, substitute
  target-author words, inflect them with the morph lookup. No LLM.
- **`llm_infill/`** (2): an LLM picks the words for the masked slots (probes).
- **`gemma_teacher/`** (3): Gemma 4 31B rewrites the poem from a summary,
  style profile, vocabulary palette and exemplar lines. Run at scale, it
  produces the students' training data.
- **`students/`** (4): small models distilled from the teacher (SFT), then
  GRPO. `shared/` holds everything both families use (`--family`);
  `qwen/` and `gemma/` hold what is specific to Qwen3-4B and Gemma 4 E4B.

Each pipeline's Slurm/Singularity files live in its `hpc/` folder. Submit
jobs from the repo root: `sbatch src/students/qwen/hpc/train_grpo_v2.slurm`.

Dependency rule (enforced by `tests/test_layout.py`): `common` imports no
pipeline; a pipeline imports `common` and earlier pipelines only;
`students/shared` imports neither `qwen` nor `gemma`.
