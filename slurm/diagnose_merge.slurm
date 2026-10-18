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
# merge as the control. Builds the Gemma merge first if it is missing
# (merge_sft_adapter.py is a no-op for an existing one). ~15 min. Push back the log.
#
# Usage:
#   sbatch slurm/diagnose_merge.slurm                      # compat adapter + Qwen control
#   GEMMA_ADAPTER=models/gemma4-e4b-lora-sft GEMMA_MERGED=models/gemma4-e4b-sft-merged \
#     QWEN=0 sbatch slurm/diagnose_merge.slurm             # Gemma's real SFT merge only

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/diagnose_merge.slurm ]] || cd ..
REQUIRE_SFT=0
source slurm/grpo_common.sh
grpo_preflight
export GEMMA_ADAPTER="${GEMMA_ADAPTER:-models/gemma4-compat/sft}"
export GEMMA_MERGED="${GEMMA_MERGED:-models/gemma4-compat/merged}"
export QWEN="${QWEN:-1}"
dirs=("$GEMMA_ADAPTER")   # the merge is (re)built below; idempotent
[[ "$QWEN" == 1 ]] && dirs+=(models/qwen3-lora-sft models/qwen3-sft-merged)
for d in "${dirs[@]}"; do
    [[ -d $d ]] || { echo "Missing $d" >&2; exit 1; }
done

run_in_container '
    $PY src/merge_sft_adapter.py --family gemma4 --adapter "$GEMMA_ADAPTER" --output-dir "$GEMMA_MERGED"
    $PY scripts/diagnose_merge.py --family gemma4 --adapter "$GEMMA_ADAPTER" --merged "$GEMMA_MERGED"
    if [[ "$QWEN" == 1 ]]; then
        $PY scripts/diagnose_merge.py --family qwen3 \
            --adapter models/qwen3-lora-sft --merged models/qwen3-sft-merged
    fi
'
