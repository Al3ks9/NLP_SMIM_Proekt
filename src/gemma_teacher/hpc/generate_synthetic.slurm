#!/bin/bash
#SBATCH --job-name=synthetic-gen
#SBATCH --partition=openlab-queue
#SBATCH --nodelist=gpgpu01
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%A_%a.out
#
# Generates the SFT synthetic dataset (Qwen SFT pipeline stage 3, see
# CLAUDE.md) by calling scripts/generate_synthetic_parallel.sh, which itself
# runs one src/generate_synthetic.py process per target author.
#
# No GPU is requested, on purpose: generate_synthetic.py -> run_transfer()
# -> llm_client.call() hits the `google` backend, a plain HTTPS call to
# Gemini's API -- the actual model runs on Google's servers, so there is no
# local inference for a GPU to accelerate (unlike train_sft.slurm, which
# loads Qwen3-4B locally). Skipping --gres=gpu also means this can run
# alongside a real GPU job (e.g. train_sft.slurm) on the same node without
# contending with it for the GPU. The bottleneck here is Gemini's API
# throughput (~3 min/pair per llm_client.py's docs), which is why this uses
# the parallel wrapper (one process per target author) instead of a single
# serial generate_synthetic.py run.
#
# Runs directly via `uv run`, not Singularity -- this script makes network
# calls, not GPU/CUDA-dependent local inference, so it doesn't need the
# reproducible CUDA runtime train_sft.slurm's container exists for.
#
# Needs GOOGLE_API_KEY (or GEMINI_API_KEY) reachable by llm_client.py's
# _key_from_env -- either exported in the submitting shell (sbatch exports it
# by default) or in a `.env` file at the repo root (gitignored; see
# src/llm_client.py). The latter is more robust across resubmits/requeues.
#
# Usage:
#   sbatch slurm/generate_synthetic.slurm [poems-per-author] [num-target-authors] [samples-per-pair] [seed]
#   sbatch slurm/generate_synthetic.slurm 100 10 1 42

set -euo pipefail

# See slurm/train_sft.slurm for why $0 can't be used here: Slurm copies the
# script into a per-job spool dir before running it, so dirname "$0" would
# resolve to that spool path, not this file's real location.
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/generate_synthetic.slurm ]] || cd ..
mkdir -p logs

if [[ -z "${GOOGLE_API_KEY:-}${GEMINI_API_KEY:-}" && ! -f .env ]]; then
    echo "No GOOGLE_API_KEY/GEMINI_API_KEY in the environment and no .env file at the repo root -- set one before submitting (see src/llm_client.py's _key_from_env)." >&2
    exit 1
fi

echo "Host: $(hostname)"

scripts/generate_synthetic_parallel.sh "$@"
