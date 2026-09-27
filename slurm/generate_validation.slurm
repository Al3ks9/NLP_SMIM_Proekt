#!/bin/bash
#SBATCH --job-name=qwen3-validation-gen
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Validation generation (Qwen SFT pipeline stage 6, see CLAUDE.md): loads the
# base Qwen/Qwen3-4B model plus the trained LoRA adapter from
# models/qwen3-lora-sft (src/train_sft.py's output) and runs style transfer
# over real poems from the val split, for manual inspection. Unlike
# generate_synthetic.slurm (which just makes HTTPS calls to Gemini and needs
# no GPU), this loads the model locally via transformers/peft, so it needs
# the same GPU + Singularity/CUDA runtime as train_sft.slurm -- see that
# script and singularity/qwen-sft.def for why.
#
# Usage:
#   sbatch slurm/generate_validation.slurm
#   sbatch slurm/generate_validation.slurm --split test --num-source-poems 100   # extra args forwarded to generate_validation.py

set -euo pipefail

# $0 is unreliable here -- Slurm copies the script into a spool dir on the
# compute node before running it, so dirname "$0" resolves to that spool
# path, not this file's real location. $SLURM_SUBMIT_DIR is the directory
# sbatch was invoked from, and is correct regardless of the spool copy.
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/generate_validation.slurm ]] || cd ..
mkdir -p logs

SIF="${SIF:-singularity/qwen-sft.sif}"
if [[ ! -f "$SIF" ]]; then
    echo "Missing $SIF -- build it first: ./singularity/build.sh" >&2
    exit 1
fi

if [[ ! -d models/qwen3-lora-sft ]]; then
    echo "Missing models/qwen3-lora-sft -- run train_sft.slurm first." >&2
    exit 1
fi

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

echo "Host: $(hostname)"

# Same layout as train_sft.slurm: exec --nv with binds for the HF cache and
# the repo, then a bash -c that cd's in and runs the image's baked venv.
singularity exec --nv \
    -B "$HF_CACHE:/root/.cache/huggingface" \
    -B "$PWD:/workspace" \
    "$SIF" \
    bash -c '
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME=/root/.cache/huggingface
        cd /workspace
        /opt/venv/bin/python src/generate_validation.py "$@"
    ' _ "$@"
