"""
Probe nvidia/nemotron-3-ultra-550b-a55b:free on Macedonian style-transfer tasks.

The largest model on the free tier — the test is whether raw scale beats Gemma's
multilingual tuning on a low-resource language like Macedonian.

Usage:
    export OPENROUTER_API_KEY=sk-or-...
    uv run python src/probe_nemotron.py

Costs 5 requests against the OpenRouter free quota (50/day without credits).
"""

import sys

from llm_probe import run

MODEL = 'nvidia/nemotron-3-ultra-550b-a55b:free'

if __name__ == '__main__':
    # English instructions by default — measured better than Macedonian ones on qwen3:8b.
    run(MODEL, backend='openrouter',
        instr_lang='mk' if '--mk' in sys.argv else 'en')
