#!/bin/bash
#SBATCH --job-name=qwen3-reward-groups
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# G samples per train prompt from the SFT policy, scored by the GRPO reward;
# reports each component's within-group spread (what dominates GRPO's
# group-normalised advantage) -> data/grpo/reward_groups.csv. Run before the
# long job to choose --style-weight / --content-weight.
#
# Usage: sbatch src/students/qwen/hpc/reward_groups.slurm [--num-prompts 20 --num-generations 4]

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }
source src/students/shared/hpc/grpo_common.sh
grpo_preflight

run_in_container '
    $PY -m students.shared.merge_sft_adapter --family "$FAMILY"
    $PY -m students.shared.evaluate_reward --family "$FAMILY" --groups "$@"
' "$@"
