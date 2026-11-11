#!/bin/bash
#SBATCH --job-name=student-grpo-smoke
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Minutes-long end-to-end check before any real GRPO run: merge the SFT
# adapter (if not done), score the hand-built reward fixtures, then 3 tiny
# GRPO steps with the real rewards on the real model. Exits non-zero if the
# fixtures' ordering expectations or the smoke assertions fail.
#
# Usage: sbatch src/students/shared/hpc/grpo_smoke.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }
source src/students/shared/hpc/grpo_common.sh
grpo_preflight

run_in_container '
    $PY -m students.shared.merge_sft_adapter --family "$FAMILY"
    $PY -m students.shared.evaluate_reward --fixtures
    $PY -m students.shared.train_grpo --family "$FAMILY" --smoke "$@"
' "$@"
