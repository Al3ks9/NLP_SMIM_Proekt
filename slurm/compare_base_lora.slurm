#!/bin/bash
#SBATCH --job-name=qwen3-base-vs-lora
#SBATCH --partition=openlab-queue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --nodelist=gpgpu01
#SBATCH --output=logs/%A_%a.out
#
# The A/B generation run behind src/compare_generations.py: the untuned base
# model and the LoRA-tuned model over the SAME val pairs, prompt format, seeds
# and decoding settings, back to back in one job on one GPU, so the only
# difference between the two output CSVs is the adapter.
#
# Both runs go through generate_validation.py -- --no-adapter for the baseline,
# the default adapter path for the tuned model. The tuned run is repeated here
# (rather than reusing the earlier CSV) because that earlier one was generated
# on CPU: device changes the sampling RNG stream, so a CPU-vs-GPU pair would
# confound the comparison it is supposed to isolate.
#
# Usage:
#   sbatch slurm/compare_base_lora.slurm
#   sbatch slurm/compare_base_lora.slurm --num-source-poems 25   # forwarded to both runs

set -euo pipefail

# $0 is unreliable here -- see generate_validation.slurm's note on the spool copy.
cd "$SLURM_SUBMIT_DIR"
[[ -f slurm/compare_base_lora.slurm ]] || cd ..
mkdir -p logs

SIF="${SIF:-singularity/qwen-sft.sif}"
if [[ ! -f "$SIF" ]]; then
    echo "Missing $SIF -- build it first: ./singularity/build.sh" >&2
    exit 1
fi

if [[ ! -d models/qwen3-lora-sft ]]; then
    echo "Missing models/qwen3-lora-sft -- run train_sft.slurm first." >&2
    exit 1
fi

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

echo "Host: $(hostname)"

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
        /opt/venv/bin/python -c "import torch; print(\"CUDA available:\", torch.cuda.is_available(), \"--\", torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"no GPU visible\")"
        echo "=== baseline: untuned base model ==="
        /opt/venv/bin/python src/generate_validation.py --no-adapter --split val "$@"
        echo "=== tuned: LoRA adapter ==="
        /opt/venv/bin/python src/generate_validation.py --split val "$@"
    ' _ "$@"
