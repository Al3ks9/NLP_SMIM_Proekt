#!/bin/bash
#SBATCH --job-name=qwen3-grpo-smoke
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
# Usage: sbatch slurm/grpo_smoke.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/grpo_smoke.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

run_in_container '
    $PY src/merge_sft_adapter.py
    $PY src/evaluate_reward.py --fixtures
    $PY src/train_grpo.py --smoke "$@"
' "$@"
