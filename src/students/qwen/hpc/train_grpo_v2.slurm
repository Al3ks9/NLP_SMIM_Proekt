#!/bin/bash
#SBATCH --job-name=qwen3-grpo-v2
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# GRPO-v2: the v1 run (same start policy, hyperparameters and alpha/beta =
# 0.3/0.7) with the two reward fixes from report.tex "GRPO results /
# Implications", so v1 vs v2 isolates the reward change:
#
#   - classifier group weight 5.6 (others 1.0, rhyme/tfidf 0.5): the inverse
#     of its within-group spread in data/grpo/reward_groups.csv (0.051,
#     against 0.288 mean for lexical_diversity and pos_bigrams), so it steers
#     the group-normalised advantage as much as those two groups do.
#   - copy gate copy_novelty**3: one copied line of 20 costs ~14 % of the
#     content reward instead of 5 %.
#
# Output: models/qwen3-grpo-v2/. Evaluate with
#   ADAPTER=models/qwen3-grpo-v2 sbatch src/students/shared/hpc/eval_grpo.slurm
# Extra args are appended (and override the ones below).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }
FAMILY=qwen3   # this run is the Qwen v1-vs-v2 comparison
source src/students/shared/hpc/grpo_common.sh
grpo_preflight

run_in_container '
    $PY -m students.shared.merge_sft_adapter --family qwen3
    $PY -m students.shared.train_grpo \
        --output-dir models/qwen3-grpo-v2 \
        --style-weight 0.3 --content-weight 0.7 \
        --style-weights "{\"classifier\": 5.6}" \
        --copy-gate-power 3 \
        "$@"
' "$@"
