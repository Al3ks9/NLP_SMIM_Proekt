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
#   FAMILY=gemma4 sbatch slurm/eval_grpo.slurm      # Gemma 4 E4B student: gemma4-e4b-* paths
#
# The SFT-merged generations are reused if they already exist (the node is
# pinned and the seeds fixed, so a rerun reproduces the same CSV);
# REGEN_BASELINE=1 forces a fresh one. Every GRPO run already under
# data/validation_generations (final adapter and checkpoints) is included in
# the comparison.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/eval_grpo.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

export REGEN_BASELINE="${REGEN_BASELINE:-0}"
export ADAPTER="${ADAPTER:-models/$TAG-grpo}"
# Qwen keeps its original output names; other families get a TAG prefix.
if [[ "$FAMILY" == qwen3 ]]; then export CMP=grpo_comparison; else export CMP=${TAG}_grpo_comparison; fi
[[ -d "$ADAPTER" ]] || { echo "Missing $ADAPTER -- run train_grpo.slurm first." >&2; exit 1; }
if [[ "$(basename "$ADAPTER")" == checkpoint-* ]]; then
    export LABEL="$(basename "$(dirname "$ADAPTER")")-$(basename "$ADAPTER")"
else
    export LABEL="$(basename "$ADAPTER")"
fi

run_in_container '
    G=data/validation_generations
    $PY src/merge_sft_adapter.py --family "$FAMILY"
    if [[ "${REGEN_BASELINE:-0}" == 1 || ! -f $G/$TAG-sft-merged_val.csv ]]; then
        $PY src/generate_validation.py --family "$FAMILY" --no-adapter --base-model models/$TAG-sft-merged --split val "$@"
    else
        echo "Reusing $G/$TAG-sft-merged_val.csv (REGEN_BASELINE=1 to regenerate)"
    fi
    $PY src/generate_validation.py --family "$FAMILY" --adapter "$ADAPTER" --split val "$@"
    RUNS=() LABELS=()
    for pair in "$BASE_LABEL:base" "$TAG-lora-sft:sft" "$TAG-sft-merged:sft-merged"; do
        if [[ -f $G/${pair%%:*}_val.csv ]]; then RUNS+=("$G/${pair%%:*}_val.csv"); LABELS+=("${pair##*:}"); fi
    done
    shopt -s nullglob   # no GRPO CSV yet -> empty loop, not a literal pattern
    for f in $G/$TAG-grpo*_val.csv; do
        RUNS+=("$f"); LABELS+=("$(basename "$f" _val.csv)")
    done
    shopt -u nullglob
    if [[ -f data/synthetic/teacher_val.csv ]]; then
        RUNS+=(data/synthetic/teacher_val.csv); LABELS+=(gemma)
    fi
    $PY src/compare_generations.py --rewards --corpus-reference \
        --runs "${RUNS[@]}" --labels "${LABELS[@]}" \
        --rows-out $G/${CMP}_rows.csv --report-out $G/${CMP}_summary.csv
' "$@"
