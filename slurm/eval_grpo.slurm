#!/bin/bash
#SBATCH --job-name=qwen3-grpo-eval
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Before/after GRPO on the same 468 val pairs, back to back on one GPU (same
# reasoning as compare_base_lora.slurm): the SFT-merged model (GRPO's start
# policy) then the GRPO adapter, then compare_generations.py --rewards over
# base / SFT / SFT-merged / GRPO / Gemma-on-val (if teacher_val.csv exists).
#
# Usage:
#   sbatch slurm/eval_grpo.slurm
#   ADAPTER=models/qwen3-grpo/checkpoint-200 sbatch slurm/eval_grpo.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/eval_grpo.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

export ADAPTER="${ADAPTER:-models/qwen3-grpo}"
[[ -d "$ADAPTER" ]] || { echo "Missing $ADAPTER -- run train_grpo.slurm first." >&2; exit 1; }
if [[ "$(basename "$ADAPTER")" == checkpoint-* ]]; then
    export LABEL="$(basename "$(dirname "$ADAPTER")")-$(basename "$ADAPTER")"
else
    export LABEL="$(basename "$ADAPTER")"
fi

run_in_container '
    G=data/validation_generations
    $PY src/merge_sft_adapter.py
    $PY src/generate_validation.py --no-adapter --base-model models/qwen3-sft-merged --split val "$@"
    $PY src/generate_validation.py --adapter "$ADAPTER" --split val "$@"
    RUNS=($G/base-Qwen3-4B_val.csv $G/qwen3-lora-sft_val.csv $G/qwen3-sft-merged_val.csv $G/${LABEL}_val.csv)
    LABELS=(base sft sft-merged "$LABEL")
    if [[ -f data/synthetic/teacher_val.csv ]]; then
        RUNS+=(data/synthetic/teacher_val.csv); LABELS+=(gemma)
    fi
    $PY src/compare_generations.py --rewards --corpus-reference \
        --runs "${RUNS[@]}" --labels "${LABELS[@]}" \
        --rows-out $G/grpo_comparison_rows.csv --report-out $G/grpo_comparison_summary.csv
' "$@"
