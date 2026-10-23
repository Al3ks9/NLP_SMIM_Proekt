"""
Probe google/gemma-4-31b-it:free on Macedonian style-transfer tasks.

Gemma is the strongest small multilingual family on the free tier, so this is
the likeliest candidate for the filling stage.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    uv run python src/probe_gemma.py

Costs 5 requests against the OpenRouter free quota (50/day without credits).
"""

import sys

from llm_probe import run

MODEL = 'google/gemma-4-31b-it:free'

if __name__ == '__main__':
    # English instructions by default — measured better than Macedonian ones on qwen3:8b.
    run(MODEL, backend='openrouter',
        instr_lang='mk' if '--mk' in sys.argv else 'en')
