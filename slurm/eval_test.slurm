#!/bin/bash
#SBATCH --job-name=test-eval
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Final results on the untouched TEST split: 50 test poems x 10 targets
# (466 pairs, self pairs skipped), same prompt, seeds and plain sampling as
# every val run. One system per job, so they can run in parallel (same
# GPU model throughout; the CPU-vs-GPU RNG caveat in compare_base_lora.slurm
# does not apply between A100s). ~2 h per system.
#
# Usage (Qwen3-4B, the default):
#   sbatch slurm/eval_test.slurm base          # untuned Qwen3-4B
#   sbatch slurm/eval_test.slurm sft           # SFT (merged = GRPO's start policy)
#   sbatch slurm/eval_test.slurm grpo-v1
#   sbatch slurm/eval_test.slurm grpo-v2
#   sbatch slurm/eval_test.slurm score         # after all four -> test_comparison_*.csv
#
# Gemma 4 E4B (one GRPO run, the v2 reward, at models/gemma4-e4b-grpo):
#   FAMILY=gemma4 sbatch slurm/eval_test.slurm base|sft|grpo
#   FAMILY=gemma4 sbatch slurm/eval_test.slurm score   # -> gemma4-e4b_test_comparison_*.csv
#
# Writes data/validation_generations/<label>_test.csv (tracked). Push back the
# CSVs (and the score step's two comparison CSVs).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/eval_test.slurm ]] || cd ..
source slurm/grpo_common.sh
grpo_preflight

export SYSTEM="${1:?usage: sbatch slurm/eval_test.slurm base|sft|grpo|grpo-v1|grpo-v2|score}"
need() { [[ -d "$1" ]] || { echo "Missing $1" >&2; exit 1; }; }
case "$FAMILY:$SYSTEM" in
    *:base|*:score) ;;
    *:sft)           need models/$TAG-sft-merged ;;
    qwen3:grpo-v1)   need models/qwen3-grpo ;;
    qwen3:grpo-v2)   need models/qwen3-grpo-v2 ;;
    gemma4:grpo)     need models/gemma4-e4b-grpo ;;
    *) echo "Unknown system $SYSTEM for FAMILY=$FAMILY" >&2; exit 1 ;;
esac

run_in_container '
    G=data/validation_generations
    GEN="$PY src/generate_validation.py --family $FAMILY --split test"
    case "$SYSTEM" in
        base)    $GEN --no-adapter ;;
        sft)     $GEN --no-adapter --base-model models/$TAG-sft-merged ;;
        grpo)    $GEN --adapter models/$TAG-grpo ;;
        grpo-v1) $GEN --adapter models/qwen3-grpo ;;
        grpo-v2) $GEN --adapter models/qwen3-grpo-v2 ;;
        score)
            if [[ $FAMILY == qwen3 ]]; then
                RUNS="$G/base-Qwen3-4B_test.csv $G/qwen3-sft-merged_test.csv $G/qwen3-grpo_test.csv $G/qwen3-grpo-v2_test.csv"
                LABELS="base sft grpo-v1 grpo-v2"; OUT=$G/test_comparison
            else
                RUNS="$G/${BASE_LABEL}_test.csv $G/$TAG-sft-merged_test.csv $G/$TAG-grpo_test.csv"
                LABELS="base sft grpo"; OUT=$G/${TAG}_test_comparison
            fi
            $PY src/compare_generations.py --rewards --corpus-reference \
                --runs $RUNS --labels $LABELS \
                --rows-out ${OUT}_rows.csv --report-out ${OUT}_summary.csv ;;
    esac
'
