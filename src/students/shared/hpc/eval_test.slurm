#!/bin/bash
#SBATCH --job-name=student-test
#SBATCH --partition=cybermacs-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Final results on the untouched TEST split: 50 test poems x 10 targets
# (466 pairs, self pairs skipped), same prompt, seeds and plain sampling as
# every val run. One system per job, so they can run in parallel (same GPU
# model throughout; the CPU-vs-GPU RNG caveat in compare_base_lora.slurm does
# not apply between A100s). ~2 h per system. FAMILY (qwen3 | gemma4, default
# qwen3) picks the student, as in eval_grpo.slurm.
#
# Usage:
#   sbatch src/students/shared/hpc/eval_test.slurm base          # untuned base model
#   sbatch src/students/shared/hpc/eval_test.slurm sft           # SFT (merged = GRPO's start policy)
#   sbatch src/students/shared/hpc/eval_test.slurm grpo          # models/$TAG-grpo
#   sbatch src/students/shared/hpc/eval_test.slurm grpo-v1       # Qwen only: models/qwen3-grpo
#   sbatch src/students/shared/hpc/eval_test.slurm grpo-v2       # Qwen only: models/qwen3-grpo-v2
#   FAMILY=gemma4 sbatch src/students/shared/hpc/eval_test.slurm base|sft|grpo
#   sbatch src/students/shared/hpc/eval_test.slurm score         # after the systems: scores + rewards -> test_comparison_*.csv
#
# Writes data/validation_generations/<label>_test.csv (tracked). `score`
# scores every *_test.csv there (both students, word substitution) plus the
# teacher's data/synthetic/teacher_test.csv if present; labels are the file
# stems without _test. Push back the CSVs (and the score step's two comparison
# CSVs).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }

export SYSTEM="${1:?usage: sbatch src/students/shared/hpc/eval_test.slurm base|sft|grpo|grpo-v1|grpo-v2|score}"
# base and score need no SFT adapter
case "$SYSTEM" in base|score) export REQUIRE_SFT=0 ;; esac
source src/students/shared/hpc/grpo_common.sh
grpo_preflight

case "$SYSTEM" in
    base|score) ;;
    sft)     [[ -d models/$TAG-sft-merged ]] || { echo "Missing models/$TAG-sft-merged" >&2; exit 1; } ;;
    grpo)    [[ -d models/$TAG-grpo ]]       || { echo "Missing models/$TAG-grpo" >&2; exit 1; } ;;
    grpo-v1|grpo-v2)
        [[ "$FAMILY" == qwen3 ]] || { echo "$SYSTEM is a Qwen run; use grpo for FAMILY=$FAMILY" >&2; exit 1; }
        dir=models/qwen3-grpo; [[ "$SYSTEM" == grpo-v2 ]] && dir=models/qwen3-grpo-v2
        [[ -d $dir ]] || { echo "Missing $dir" >&2; exit 1; } ;;
    *) echo "Unknown system $SYSTEM" >&2; exit 1 ;;
esac

run_in_container '
    G=data/validation_generations
    GEN="$PY -m students.shared.generate_validation --family "$FAMILY" --split test"
    case "$SYSTEM" in
        base)    $GEN --no-adapter ;;
        sft)     $GEN --no-adapter --base-model models/$TAG-sft-merged ;;
        grpo)    $GEN --adapter models/$TAG-grpo ;;
        grpo-v1) $GEN --adapter models/qwen3-grpo ;;
        grpo-v2) $GEN --adapter models/qwen3-grpo-v2 ;;
        score)
            runs=(); labels=()
            for f in $G/*_test.csv; do runs+=("$f"); labels+=("$(basename "$f" _test.csv)"); done
            T=data/synthetic/teacher_test.csv
            if [[ -f $T ]]; then runs+=("$T"); labels+=(gemma-teacher); fi
            $PY -m students.shared.compare_generations --rewards --corpus-reference \
                --runs "${runs[@]}" --labels "${labels[@]}" \
                --rows-out $G/test_comparison_rows.csv --report-out $G/test_comparison_summary.csv ;;
    esac
'
