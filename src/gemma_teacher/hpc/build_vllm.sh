#!/bin/bash
# Pulls singularity/vllm.sif: the official vLLM OpenAI-compatible server image,
# used by src/gemma_teacher/hpc/vllm_teacher.slurm to serve the Gemma 4 31B teacher locally.
#
# A separate image from qwen-sft.sif on purpose: vLLM pins its own torch/CUDA
# build, which would fight this project's uv.lock if synced into one venv. The
# client side (generate_synthetic.py) keeps running from the project's own
# environment and talks to this server over HTTP.
#
# Pinned to a released tag (Gemma 4 + the gemma4 reasoning parser landed in
# 0.19.1). Bump deliberately -- a different vLLM can sample differently.
# Same tmp/cache redirection as src/students/shared/hpc/build.sh: local /tmp is a few GB, the image
# is ~9 GB compressed.
#
# Usage:
#   ./src/gemma_teacher/hpc/build_vllm.sh

set -euo pipefail
cd "$(dirname "$0")/../../.."

VLLM_IMAGE="${VLLM_IMAGE:-docker://vllm/vllm-openai:v0.30.0}"

export SINGULARITY_TMPDIR="$PWD/singularity/.build_tmp"
export SINGULARITY_CACHEDIR="$PWD/singularity/.build_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

echo "Pulling $VLLM_IMAGE -> singularity/vllm.sif"
singularity pull --force singularity/vllm.sif "$VLLM_IMAGE"
ls -lh singularity/vllm.sif
