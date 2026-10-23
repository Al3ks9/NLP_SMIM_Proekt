#!/bin/bash
#SBATCH --job-name=qwen3-test
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Final Qwen results on the untouched TEST split: 50 test poems x 10 targets
# (466 pairs, self pairs skipped), same prompt, seeds and plain sampling as
# every val run. One system per job, so the four can run in parallel (same
# GPU model throughout; the CPU-vs-GPU RNG caveat in compare_base_lora.slurm
# does not apply between A100s). ~2 h per system.
#
# Usage:
#   sbatch slurm/eval_test.slurm base          # untuned Qwen3-4B
#   sbatch slurm/eval_test.slurm sft           # SFT (merged = GRPO's start policy)
#   sbatch slurm/eval_test.slurm grpo-v1
#   sbatch slurm/eval_test.slurm grpo-v2
#   sbatch slurm/eval_test.slurm score         # after all four: scores + rewards -> test_comparison_*.csv
#
# Writes data/validation_generations/<label>_test.csv (tracked). Push back the
# CSVs (and the score step's two comparison CSVs).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/eval_test.slurm ]] || cd ..
FAMILY=qwen3
source slurm/grpo_common.sh
grpo_preflight

export SYSTEM="${1:?usage: sbatch slurm/eval_test.slurm base|sft|grpo-v1|grpo-v2|score}"
case "$SYSTEM" in
    base|score) ;;
    sft)     [[ -d models/qwen3-sft-merged ]] || { echo "Missing models/qwen3-sft-merged" >&2; exit 1; } ;;
    grpo-v1) [[ -d models/qwen3-grpo ]]       || { echo "Missing models/qwen3-grpo" >&2; exit 1; } ;;
    grpo-v2) [[ -d models/qwen3-grpo-v2 ]]    || { echo "Missing models/qwen3-grpo-v2" >&2; exit 1; } ;;
    *) echo "Unknown system $SYSTEM" >&2; exit 1 ;;
esac

run_in_container '
    G=data/validation_generations
    GEN="$PY src/generate_validation.py --family qwen3 --split test"
    case "$SYSTEM" in
        base)    $GEN --no-adapter ;;
        sft)     $GEN --no-adapter --base-model models/qwen3-sft-merged ;;
        grpo-v1) $GEN --adapter models/qwen3-grpo ;;
        grpo-v2) $GEN --adapter models/qwen3-grpo-v2 ;;
        score)
            $PY src/compare_generations.py --rewards --corpus-reference \
                --runs $G/base-Qwen3-4B_test.csv $G/qwen3-sft-merged_test.csv \
                       $G/qwen3-grpo_test.csv $G/qwen3-grpo-v2_test.csv \
                --labels base sft grpo-v1 grpo-v2 \
                --rows-out $G/test_comparison_rows.csv --report-out $G/test_comparison_summary.csv ;;
    esac
'
