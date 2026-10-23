"""
Probe qwen3:8b running locally under Ollama on Macedonian style-transfer tasks.

This is the primary candidate for the filling stage: no usage limits, so both
the lean and the full PEGF pipeline are affordable. The open question is whether
an 8B model writes Macedonian verse well enough — that is what this measures.

Usage:
    ollama pull qwen3:8b
    uv run python src/probe_qwen.py                    # English instructions (default)
    uv run python src/probe_qwen.py --mk               # Macedonian instructions
    uv run python src/probe_qwen.py --model qwen3:14b  # any local ollama tag

English instructions are the default: measured better than Macedonian ones on
qwen3:8b (implicit-mask fills 0/3 -> 2/3, morphology 1/4 -> 2/4). Use --mk to
re-check that on a different model — the Macedonian text under test is identical
either way, so the flag isolates instruction-following from language ability.

No quota cost. Prints per-call timing, since local throughput determines how
expensive a full multi-call PEGF run over the corpus would be.
"""

import sys

from llm_probe import run

MODEL = 'qwen3:14b'  # 14b beat 8b on region discipline at content_kept 0.769 -> 1.0

if __name__ == '__main__':
    model = MODEL
    if '--model' in sys.argv:
        model = sys.argv[sys.argv.index('--model') + 1]

    run(model, backend='ollama',
        instr_lang='mk' if '--mk' in sys.argv else 'en')
