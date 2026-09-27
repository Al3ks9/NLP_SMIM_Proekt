#!/bin/bash
#SBATCH --job-name=qwen3-grpo
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Full single-A100 GRPO run: SFT-merged policy + fresh LoRA, style/content
# rewards (src/grpo_rewards.py). Output: models/qwen3-grpo/ (adapter,
# run_config.json, reward_components.jsonl, checkpoint-*/). Run
# grpo_smoke.slurm first. Extra args go straight to train_grpo.py.
#
# Usage:
#   sbatch slurm/train_grpo.slurm
#   sbatch slurm/train_grpo.slurm --max-steps 200 --style-weight 0.6 --content-weight 0.4

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/train_grpo.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

run_in_container '
    $PY src/merge_sft_adapter.py
    $PY src/train_grpo.py "$@"
' "$@"
