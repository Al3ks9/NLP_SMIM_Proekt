#!/bin/bash
#SBATCH --job-name=qwen3-lora-sft
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Single-GPU LoRA SFT of Qwen/Qwen3-4B on the synthetic style-transfer
# dataset. No multi-GPU / distributed setup -- one process, one device,
# matching this project's single-GPU target.
#
# Runs inside the Singularity image built from singularity/qwen-sft.def
# (build once with ./singularity/build.sh) rather than the bare `uv`
# environment -- see that file for why: it's built straight from this
# project's own pyproject.toml/uv.lock, so it can't drift from `uv run`
# outside a container, and it carries its own CUDA runtime (via torch's
# bundled nvidia-*-cu12 wheels), needing only the host's NVIDIA driver
# (bound in via --nv).
#
# Usage:
#   sbatch slurm/train_sft.slurm
#   sbatch slurm/train_sft.slurm --num-train-epochs 5 --learning-rate 1e-4   # extra args forwarded to train_sft.py

set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

SIF="${SIF:-singularity/qwen-sft.sif}"
if [[ ! -f "$SIF" ]]; then
    echo "Missing $SIF -- build it first: ./singularity/build.sh" >&2
    exit 1
fi

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

echo "Host: $(hostname)"

# Matches this cluster's other singularity jobs: exec --nv with explicit -B
# binds for the HF model/dataset cache and the repo, then a bash -c that cds
# in and runs. No conda `source activate` needed here (unlike an
# NGC/conda-based image like medgemma.sif) -- singularity/qwen-sft.def bakes
# a plain venv at /opt/venv, invoked by its absolute path below so this
# doesn't depend on %environment's PATH tweak being sourced by `exec`.
#
# The repo (src/, data/, models/) is bound as a single /workspace, unlike
# medgemma's separate ./data and ./code mounts -- this project keeps code and
# data in one repo rather than two host trees, so one bind covers both.
singularity exec --nv \
    -B "$HF_CACHE:/root/.cache/huggingface" \
    -B "$PWD:/workspace" \
    "$SIF" \
    bash -c '
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME=/root/.cache/huggingface
        cd /workspace
        /opt/venv/bin/python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available(), \"--\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU visible\")"
        /opt/venv/bin/python src/train_sft.py \
            --num-train-epochs 3 \
            --per-device-train-batch-size 4 \
            --per-device-eval-batch-size 4 \
            --gradient-accumulation-steps 4 \
            --learning-rate 2e-4 \
            --lora-r 16 \
            --lora-alpha 32 \
            --lora-dropout 0.05 \
            --max-seq-length 1024 \
            --bf16 \
            "$@"
    ' _ "$@"
