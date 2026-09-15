#!/bin/bash
# Builds singularity/qwen-sft.sif from qwen-sft.def.
#
# Needs --fakeroot (or root) privileges for `singularity build`; on this
# cluster --fakeroot has worked for a plain user account (confirmed with a
# throwaway `singularity build --fakeroot test.sif docker://alpine` before
# committing to the real, much larger build).
#
# Redirects Singularity's temp/cache dirs onto this project's NFS-backed home
# filesystem instead of the node's local /tmp -- local /tmp on this cluster's
# login node has only a few GB free, nowhere near enough for the layers this
# build pulls plus the synced venv (torch + its bundled CUDA libraries alone
# are several GB). Matches the singularity_tmp/singularity_cache convention
# already used elsewhere on this cluster.
#
# Usage:
#   ./singularity/build.sh

set -euo pipefail
cd "$(dirname "$0")/.."

export SINGULARITY_TMPDIR="$PWD/singularity/.build_tmp"
export SINGULARITY_CACHEDIR="$PWD/singularity/.build_cache"
mkdir -p "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

echo "Building singularity/qwen-sft.sif (tmpdir=$SINGULARITY_TMPDIR, cachedir=$SINGULARITY_CACHEDIR)"
singularity build --fakeroot singularity/qwen-sft.sif singularity/qwen-sft.def

echo "Built singularity/qwen-sft.sif:"
ls -lh singularity/qwen-sft.sif
