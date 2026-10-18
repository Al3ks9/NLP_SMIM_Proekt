#!/bin/bash
#SBATCH --job-name=gemma4-compat
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Gemma 4 E4B compatibility check -- run before any Gemma training
# (src/gemma_compat.py; spec §4). Loads the model, checks LoRA targets, the
# chat template and a sample generation, measures train-example lengths, runs
# a 3-step SFT + merge round-trip, then a 3-step GRPO smoke on that merge.
#
# Writes data/gemma_compat_report.json (push it back: it decides the loader
# and Gemma's --max-seq-length). Scratch in models/gemma4-compat/ (gitignored).
#
# Usage:
#   sbatch slurm/gemma_compat.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/gemma_compat.slurm ]] || cd ..
FAMILY=gemma4
REQUIRE_SFT=0   # no Gemma SFT exists yet; this job makes a throwaway one
source slurm/grpo_common.sh
grpo_preflight

run_in_container '
    $PY src/gemma_compat.py --work-dir models/gemma4-compat
    $PY src/train_grpo.py --family gemma4 --smoke \
        --sft-merged models/gemma4-compat/merged --output-dir models/gemma4-compat/grpo-smoke
'
