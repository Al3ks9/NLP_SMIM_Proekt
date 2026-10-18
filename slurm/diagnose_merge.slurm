#!/bin/bash
#SBATCH --job-name=merge-diagnose
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Diagnoses gemma_compat's failed sft_merge check (scripts/diagnose_merge.py):
# the Gemma compat adapter vs its merge, then Qwen's real SFT adapter vs its
# merge as the control. ~15 min. Push back the log.
#
# Usage:
#   sbatch slurm/diagnose_merge.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/diagnose_merge.slurm ]] || cd ..
REQUIRE_SFT=0
source slurm/grpo_common.sh
grpo_preflight
for d in models/gemma4-compat/sft models/gemma4-compat/merged models/qwen3-lora-sft models/qwen3-sft-merged; do
    [[ -d $d ]] || { echo "Missing $d" >&2; exit 1; }
done

run_in_container '
    $PY scripts/diagnose_merge.py --family gemma4 \
        --adapter models/gemma4-compat/sft --merged models/gemma4-compat/merged
    $PY scripts/diagnose_merge.py --family qwen3 \
        --adapter models/qwen3-lora-sft --merged models/qwen3-sft-merged
'
