#!/bin/bash
#SBATCH --job-name=gemma31b-teacher
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# The Gemma 4 31B teacher on our own A100 instead of Google's API: starts a
# vLLM server (singularity/vllm.sif) on this node, checks one answer, then runs
# src/gemma_teacher/generate_synthetic_parallel.sh against it with BACKEND=vllm. Same
# resumable, per-author-parallel generation as generate_synthetic.slurm
# -- only the place the model runs changes (llm_client.call_vllm).
#
# Why: the API's quota (429) and intermittent 500s/timeouts left the val
# teacher run 12 pairs short and make a 468-pair test run impractical.
#
# Fit on one 80 GB A100: bf16 weights are 62.5 GB (31.3B params). 50 of the 60
# layers use a 1024-token sliding window, so the KV cache per sequence stays
# small (~2.4 GB at the full 32k), and ~9 GB is left at 0.92 utilisation --
# room for a few long reasoning traces at once; vLLM queues the rest. The
# server runs text-only (image/audio limits 0) and at 32768 tokens, the API's
# output ceiling. If it runs out of memory at startup, try
# EXTRA_VLLM_ARGS="--enforce-eager" (drops CUDA graphs, ~2 GB) before
# "--kv-cache-dtype fp8" (which changes numerics slightly).
#
# The first run downloads the weights (~62 GB) into the HF cache.
#
# Usage (same positional args as generate_synthetic_parallel.sh):
#   ./src/gemma_teacher/hpc/build_vllm.sh                          # once
#   sbatch src/gemma_teacher/hpc/vllm_teacher.slurm 50 10 1 42 val       # finish teacher_val.csv
#   sbatch src/gemma_teacher/hpc/vllm_teacher.slurm 50 10 1 42 test      # teacher on the test pairs -> teacher_test.csv

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }
mkdir -p logs

VLLM_SIF="${VLLM_SIF:-singularity/vllm.sif}"
if [[ ! -f "$VLLM_SIF" ]]; then
    echo "Missing $VLLM_SIF -- build it first: ./src/gemma_teacher/hpc/build_vllm.sh" >&2
    exit 1
fi

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"
PORT="${PORT:-8000}"
SERVER_LOG="logs/vllm_${SLURM_JOB_ID:-local}.log"

echo "Host: $(hostname)  server log: $SERVER_LOG"

SINGULARITYENV_HF_HOME=/root/.cache/huggingface \
singularity exec --nv \
    -B "$HF_CACHE:/root/.cache/huggingface" \
    "$VLLM_SIF" \
    vllm serve google/gemma-4-31B-it \
        --served-model-name gemma-4-31b-it \
        --max-model-len 32768 \
        --gpu-memory-utilization 0.92 \
        --max-num-seqs 10 \
        --reasoning-parser gemma4 \
        --limit-mm-per-prompt '{"image": 0, "audio": 0}' \
        --seed 42 \
        --host 127.0.0.1 --port "$PORT" \
        ${EXTRA_VLLM_ARGS:-} \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

# Up to an hour: the first start downloads the weights.
for _ in $(seq 360); do
    if curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "vLLM exited during startup -- see $SERVER_LOG" >&2
        tail -n 40 "$SERVER_LOG" >&2
        exit 1
    fi
    sleep 10
done
curl -sf "http://127.0.0.1:$PORT/health" > /dev/null || { echo "vLLM not healthy after 1 h" >&2; exit 1; }
echo "vLLM up."

export VLLM_HOST="http://127.0.0.1:$PORT"

# One real answer before committing hours to the batch: non-empty, Cyrillic,
# no reasoning text leaking through.
uv run python - <<'PY'
from common import llm_client as lc
out = lc.call_vllm('gemma-4-31b-it', 'Напиши два стиха на македонски јазик за есента. Само стиховите.')
print('self-check answer:', repr(out))
cyr = sum('Ѐ' <= c <= 'ӿ' for c in out)
assert out and cyr > len(out) // 3, 'answer empty or not Cyrillic'
assert '<|channel>' not in out and not out.lower().startswith('thought'), 'reasoning leaked into the answer'
PY

BACKEND=vllm src/gemma_teacher/generate_synthetic_parallel.sh "$@"
