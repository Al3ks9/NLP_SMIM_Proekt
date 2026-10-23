#!/bin/bash
#SBATCH --job-name=gemma4-grpo
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# GRPO of the Gemma 4 E4B student with exactly GRPO-v2's reward and
# hyperparameters (slurm/train_grpo_v2.slurm), so Qwen-v2 vs Gemma isolates
# the student model. Spec: docs/superpowers/specs/2026-10-02-gemma4-student-design.md.
#
# Needs models/gemma4-e4b-lora-sft (FAMILY=gemma4 sbatch slurm/train_sft.slurm).
# Output: models/gemma4-e4b-grpo/. Evaluate with
#   FAMILY=gemma4 sbatch slurm/eval_grpo.slurm
# Extra args are appended (and override the ones below).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/train_grpo_gemma.slurm ]] || cd ..
FAMILY=gemma4
source slurm/grpo_common.sh
grpo_preflight

run_in_container '
    $PY src/merge_sft_adapter.py --family gemma4
    $PY src/train_grpo.py --family gemma4 \
        --style-weight 0.3 --content-weight 0.7 \
        --style-weights "{\"classifier\": 5.6}" \
        --copy-gate-power 3 \
        "$@"
' "$@"
