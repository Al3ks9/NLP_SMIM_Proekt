# Shared by the GRPO Slurm jobs (sourced, not submitted). Same container
# pattern as train_sft.slurm, plus a classla model bind -- the GRPO style
# reward POS-tags every generated poem (tagging.tag_poem).
#
# classla's mk models must be fetched once on a node with internet:
#   uv run python -c "import classla; classla.download('mk')"

SIF="${SIF:-singularity/qwen-sft.sif}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
CLASSLA_DIR="${CLASSLA_RESOURCES_DIR:-$HOME/classla_resources}"

# FAMILY (qwen3 | gemma4, default qwen3) picks the student model; every job
# forwards it as --family. TAG / BASE_LABEL mirror src/model_families.py's
# tag and base_label() -- tests/test_model_families.py keeps them in sync.
FAMILY="${FAMILY:-qwen3}"
export FAMILY

family_vars() {
    case "$FAMILY" in
        qwen3)  TAG=qwen3;      BASE_LABEL=base-Qwen3-4B ;;
        gemma4) TAG=gemma4-e4b; BASE_LABEL=base-gemma-4-E4B-it ;;
        *) echo "Unknown FAMILY=$FAMILY (known: qwen3, gemma4)" >&2; return 1 ;;
    esac
    export TAG BASE_LABEL
}

# REQUIRE_SFT=0 skips the SFT-adapter check (gemma_compat.slurm runs before
# any Gemma SFT exists).
grpo_preflight() {
    family_vars || exit 1
    [[ -f "$SIF" ]] || { echo "Missing $SIF -- build it first: ./singularity/build.sh" >&2; exit 1; }
    if [[ "${REQUIRE_SFT:-1}" == 1 && ! -d models/$TAG-lora-sft ]]; then
        echo "Missing models/$TAG-lora-sft -- run FAMILY=$FAMILY sbatch slurm/train_sft.slurm first." >&2
        exit 1
    fi
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
