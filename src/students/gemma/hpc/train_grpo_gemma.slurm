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
# hyperparameters (src/students/qwen/hpc/train_grpo_v2.slurm), so Qwen-v2 vs Gemma isolates
# the student model. Spec: docs/superpowers/specs/2026-10-02-gemma4-student-design.md.
#
# Needs models/gemma4-e4b-lora-sft (FAMILY=gemma4 sbatch src/students/qwen/hpc/train_sft.slurm).
# Output: models/gemma4-e4b-grpo/. Evaluate with
#   FAMILY=gemma4 sbatch src/students/qwen/hpc/eval_grpo.slurm
# Extra args are appended (and override the ones below).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }
FAMILY=gemma4
source src/students/shared/hpc/grpo_common.sh
grpo_preflight

run_in_container '
    $PY -m students.shared.merge_sft_adapter --family gemma4
    $PY -m students.shared.train_grpo --family gemma4 \
        --style-weight 0.3 --content-weight 0.7 \
        --style-weights "{\"classifier\": 5.6}" \
        --copy-gate-power 3 \
        "$@"
' "$@"
