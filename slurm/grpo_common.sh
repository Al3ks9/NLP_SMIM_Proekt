# Shared by the GRPO Slurm jobs (sourced, not submitted). Same container
# pattern as train_sft.slurm, plus a classla model bind -- the GRPO style
# reward POS-tags every generated poem (tagging.tag_poem).
#
# classla's mk models must be fetched once on a node with internet:
#   uv run python -c "import classla; classla.download('mk')"

SIF="${SIF:-singularity/qwen-sft.sif}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CLASSLA_DIR="${CLASSLA_RESOURCES_DIR:-$HOME/classla_resources}"

grpo_preflight() {
    [[ -f "$SIF" ]] || { echo "Missing $SIF -- build it first: ./singularity/build.sh" >&2; exit 1; }
    [[ -d models/qwen3-lora-sft ]] || { echo "Missing models/qwen3-lora-sft -- run train_sft.slurm first." >&2; exit 1; }
    [[ -d "$CLASSLA_DIR/mk" ]] || { echo "Missing classla mk models in $CLASSLA_DIR -- see slurm/grpo_common.sh" >&2; exit 1; }
    [[ -f data/reward_calibration.json ]] || { echo "Missing data/reward_calibration.json -- run: uv run python src/build_reward_calibration.py" >&2; exit 1; }
    mkdir -p "$HF_CACHE" logs
    echo "Host: $(hostname)"
}

# run_in_container 'shell body using "$@"' [args forwarded to the body...]
run_in_container() {
    local body="$1"; shift
    singularity exec --nv \
        -B "$HF_CACHE:/root/.cache/huggingface" \
        -B "$CLASSLA_DIR:/root/classla_resources" \
        -B "$PWD:/workspace" \
        "$SIF" \
        bash -c "
            set -euo pipefail
            export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
            export HF_HOME=/root/.cache/huggingface CLASSLA_RESOURCES_DIR=/root/classla_resources
            cd /workspace
            nvidia-smi -L || true
            PY=/opt/venv/bin/python
            $body
        " _ "$@"
}
