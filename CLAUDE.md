# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Macedonian-language NLP pipeline for stylometric analysis and style transfer. The core dataset is `data/stripped_songs.csv` — 2,711 Macedonian poems and song lyrics (Cyrillic text) with columns `author`, `song_title`, and `song_text`. The pipeline covers POS tagging, TF-IDF profiling, knowledge graph construction, word embeddings, author classification, and style transfer.

## Folder structure

```
src/        Python scripts (one per pipeline step)
data/       CSV files — corpus input and analysis outputs
models/     Trained artifacts — classifier.pkl, *.json lookups, skg_final.gexf
lib/        JS assets used by pyvis when generating skg_visual.html
```

All scripts resolve paths via `pathlib.Path(__file__)` so they work from any working directory.

## Environment & commands

This project uses `uv` for dependency management (Python 3.10+).

```bash
uv sync                             # install dependencies
uv run python src/<script.py>       # run a script in the venv
uv add <package>                    # add a dependency
```

## Pipeline order

Run scripts in this order to rebuild all artifacts from scratch:

1. `src/pos_tag_corpus.py` → `data/pos_tagged.csv`
2. `src/tfidf_authors.py` → `data/tfidf_results.csv`
3. `src/style_profiler.py` → `data/author_style_profiles.csv`
4. `src/build_skg.py` → `models/skg.gexf`
5. `src/enrich_skg.py` → `models/skg_enriched.gexf`, `models/author_vocab.json`, `models/pos_transitions.json`
6. `src/add_embeddings.py` → `models/skg_final.gexf` (also writes `models/word_embeddings.json`, excluded from git — 299 MB)
7. `src/graph_analytics.py` → `data/graph_analytics.csv`
8. `src/classifier.py` → `models/classifier.pkl`
9. `src/build_morph_lookup.py` → `models/morph_lookup.json`
10. `src/style_transfer.py` — interactive demo, requires all of the above

## Data quality context

Latin and Greek character artifacts exist inside Macedonian (Cyrillic) text — 231 rows affected out of 2,711. These are single characters (e.g. Latin `a`, `e`, `j`, `c`) that visually resemble Cyrillic letters but are the wrong Unicode codepoint.

When writing data-cleaning or preprocessing scripts, distinguish between:
- **Intentional Latin/Greek** (foreign words, author names, titles that legitimately use Latin script)
- **Artifact characters** (Cyrillic lookalikes from OCR or copy-paste errors)
