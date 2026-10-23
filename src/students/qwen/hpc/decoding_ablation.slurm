#!/bin/bash
#SBATCH --job-name=qwen3-decode-ablation
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# Decoding ablation for the LoRA-tuned model: same adapter, same val pairs,
# three decoding settings. It answers the one question that has to be settled
# before spending GPU time on GRPO -- whether the repeated lines in the tuned
# output are a decoding artifact (plain sampling looping) or are in the
# model's distribution (in which case only more training can move them).
#
# A smaller --num-source-poems than the full run on purpose: this is a
# direction check, not a headline number.
#
# Usage:
#   sbatch slurm/decoding_ablation.slurm
#   sbatch slurm/decoding_ablation.slurm 20     # source poems per setting

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/decoding_ablation.slurm ]] || cd ..
mkdir -p logs

SIF="${SIF:-singularity/qwen-sft.sif}"
[[ -f "$SIF" ]] || { echo "Missing $SIF -- build it first: ./singularity/build.sh" >&2; exit 1; }
[[ -d models/qwen3-lora-sft ]] || { echo "Missing models/qwen3-lora-sft." >&2; exit 1; }

N_POEMS="${1:-6}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"
OUT=data/validation_generations/ablation
mkdir -p "$OUT"

echo "Host: $(hostname)  source poems per setting: $N_POEMS"

singularity exec --nv \
    -B "$HF_CACHE:/root/.cache/huggingface" \
    -B "$PWD:/workspace" \
    "$SIF" \
    bash -c '
        set -euo pipefail
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=false
        export HF_HOME=/root/.cache/huggingface
        cd /workspace
        N="$1"
        OUT=data/validation_generations/ablation
        run () {  # run <name> <extra generate_validation.py args...>
            local name="$1"; shift
            echo "=== $name ==="
            /opt/venv/bin/python src/generate_validation.py \
                --split val --num-source-poems "$N" \
                --output "$OUT/$name.csv" "$@"
        }
        run sample-baseline
        run reppen-1.1  --repetition-penalty 1.1
        run reppen-1.2  --repetition-penalty 1.2
        run norepeat-4  --no-repeat-ngram-size 4
    ' _ "$N_POEMS"
