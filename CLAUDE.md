# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Macedonian-language NLP dataset project. The core asset is `songs.csv` — a collection of 2,711 Macedonian poems and song lyrics (Cyrillic text) with columns `author`, `song_title`, and `song_text`. The project involves data collection, cleaning, and preparation for NLP tasks using HuggingFace Transformers.

## Environment & commands

This project uses `uv` for dependency management (Python 3.10+).

```bash
uv sync                        # install dependencies
uv run python <script.py>      # run a script in the venv
uv add <package>               # add a dependency
```

The only declared dependency is `transformers>=5.9.0`.

## Data quality context

`artifact_lines_report.txt` documents Latin and Greek character artifacts found inside Macedonian (Cyrillic) text — 231 rows affected out of 2,711. These are single characters (e.g. Latin `a`, `e`, `j`, `c`) that visually resemble Cyrillic letters but are the wrong Unicode codepoint. This is a known data quality issue that must be accounted for in any preprocessing or tokenization work.

When writing data-cleaning or preprocessing scripts, distinguish between:
- **Intentional Latin/Greek** (foreign words, author names, titles that legitimately use Latin script)
- **Artifact characters** (Cyrillic lookalikes from OCR or copy-paste errors, as catalogued in the report)
