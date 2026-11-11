#!/bin/bash
#SBATCH --job-name=gemma4-grpo
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# GRPO of the Gemma 4 E4B student with exactly GRPO-v2's reward and
# hyperparameters (src/students/qwen/hpc/train_grpo_v2.slurm), so Qwen-v2 vs Gemma isolates
# the student model. Spec: docs/superpowers/specs/2026-10-02-gemma4-student-design.md.
#
# Needs models/gemma4-e4b-lora-sft (FAMILY=gemma4 sbatch src/students/qwen/hpc/train_sft.slurm).
# Output: models/gemma4-e4b-grpo/. Evaluate with
#   FAMILY=gemma4 sbatch src/students/qwen/hpc/eval_grpo.slurm
# Extra args are appended (and override the ones below).
#
# Resumes automatically from the newest models/gemma4-e4b-grpo/checkpoint-N
# (model, optimizer, scheduler, RNG and data position), so a timed-out job is
# continued by just resubmitting it. RESUME=0 starts from scratch instead.
# On resume, reward_components.jsonl is cut back to the records logged before
# step N, so the steps the resumed run redoes are not logged twice.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[[ -f pyproject.toml ]] || { echo "Submit from the repo root: sbatch <path to this .slurm>" >&2; exit 1; }
FAMILY=gemma4
source src/students/shared/hpc/grpo_common.sh
grpo_preflight

OUT_DIR="models/$TAG-grpo"
RESUME_ARGS=()
if [[ "${RESUME:-1}" == 1 ]]; then
    LAST_CKPT=$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
    if [[ -n "$LAST_CKPT" && -f "$LAST_CKPT/trainer_state.json" ]]; then
        echo "Resuming from $LAST_CKPT"
        RESUME_ARGS=(--resume-from-checkpoint "$LAST_CKPT")
        STEP="${LAST_CKPT##*-}"
        # Keep the first contiguous run of records (a step decrease means a
        # later run restarted from scratch into this dir), then drop step >= N.
        python3 - "$OUT_DIR/reward_components.jsonl" "$STEP" <<'EOF'
import json, os, sys
path, step = sys.argv[1], int(sys.argv[2])
if os.path.exists(path):
    kept, prev, total = [], -1, 0
    with open(path, encoding='utf-8') as f:
        for line in f:
            total += 1
            s = json.loads(line).get('step')
            s = -1 if s is None else s
            if s < prev:
                break
            prev = s
            if s < step:
                kept.append(line)
    if len(kept) != total:
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            f.writelines(kept)
        os.replace(tmp, path)
    print(f'reward_components.jsonl: kept {len(kept)} of {total} records (step < {step})')
EOF
    else
        echo "No checkpoint in $OUT_DIR -- starting from scratch."
    fi
fi

run_in_container '
    $PY -m students.shared.merge_sft_adapter --family gemma4
    $PY -m students.shared.train_grpo --family gemma4 \
        --style-weight 0.3 --content-weight 0.7 \
        --style-weights "{\"classifier\": 5.6}" \
        --copy-gate-power 3 \
        "$@"
' "${RESUME_ARGS[@]}" "$@"
