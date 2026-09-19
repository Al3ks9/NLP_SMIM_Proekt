#!/usr/bin/env bash
# Runs one generate_synthetic.py process per target author in parallel, then
# merges the per-author part files back into the canonical
# data/synthetic/synthetic_dataset.csv / synthetic_errors.csv.
#
# Why split on target_author specifically: generate_synthetic.py's dedup key
# is (source_poem_id, target_author, sample_index), and target_author is the
# one axis two processes can be provably disjoint on -- no coordination
# needed beyond giving each process its own --output-path/--errors-path.
# Concurrent appends from multiple processes to ONE shared CSV are not safe
# in general (a generated poem can make a single row exceed the OS's atomic
# write size, corrupting the file on interleaved writes), so each process
# gets an isolated file under data/synthetic/parts/ and merge_synthetic_parts.py
# folds them back together afterward.
#
# Safe to re-run or resume after Ctrl-C / a killed process: any part files
# left over from an earlier (possibly interrupted) run are merged into the
# main dataset FIRST, before being reseeded, so no already-generated row is
# ever discarded or regenerated.
#
# Usage:
#   scripts/generate_synthetic_parallel.sh [poems-per-author] [num-target-authors] [samples-per-pair] [seed]
#   scripts/generate_synthetic_parallel.sh 20 10 1 42

set -eo pipefail
shopt -s nullglob
# NOT set -u: macOS's stock bash is 3.2 (last GPLv2 release), which has a
# long-standing bug where `${arr[@]}` on a zero-element array is treated as
# unbound under `set -u`, even after `arr=()`. Safer to skip nounset than to
# hand-guard every array expansion in this script for a 12-year-old shell.

cd "$(dirname "$0")/.."

POEMS_PER_AUTHOR="${1:-20}"
NUM_TARGET_AUTHORS="${2:-10}"
SAMPLES_PER_PAIR="${3:-1}"
SEED="${4:-42}"

PARTS_DIR="data/synthetic/parts"
ERROR_PARTS_DIR="data/synthetic/parts/errors"
LOG_DIR="data/synthetic/parts/logs"
mkdir -p "$PARTS_DIR" "$ERROR_PARTS_DIR" "$LOG_DIR"

merge_all() {
    local dataset_parts=("$PARTS_DIR"/*.csv)
    if [ ${#dataset_parts[@]} -gt 0 ]; then
        uv run python src/merge_synthetic_parts.py merge --kind dataset "${dataset_parts[@]}"
    fi
    local error_parts=("$ERROR_PARTS_DIR"/*.csv)
    if [ ${#error_parts[@]} -gt 0 ]; then
        uv run python src/merge_synthetic_parts.py merge --kind errors "${error_parts[@]}"
    fi
}

echo "Merging any part files left over from a previous run..."
merge_all

echo "Resolving $NUM_TARGET_AUTHORS target authors..."
AUTHORS=()
while IFS= read -r author; do
    AUTHORS+=("$author")
done < <(uv run python -c "
import sys; sys.path.insert(0, 'src')
from sft_data import select_target_authors
for a in select_target_authors($NUM_TARGET_AUTHORS):
    print(a)
")
echo "Target authors: ${AUTHORS[*]}"

pids=()
for author in "${AUTHORS[@]}"; do
    slug=$(echo "$author" | tr ' ' '_')
    part="$PARTS_DIR/${slug}.csv"
    errors="$ERROR_PARTS_DIR/${slug}.csv"
    log="$LOG_DIR/${slug}.log"

    uv run python src/merge_synthetic_parts.py seed --author "$author" --part "$part"

    echo "Launching $author -> $part (log: $log)"
    uv run python src/generate_synthetic.py \
        --target-authors "$author" \
        --poems-per-author "$POEMS_PER_AUTHOR" \
        --samples-per-pair "$SAMPLES_PER_PAIR" \
        --seed "$SEED" \
        --output-path "$part" \
        --errors-path "$errors" \
        > "$log" 2>&1 &
    pids+=("$!")
done

echo "Waiting on ${#pids[@]} processes (tail -f ${LOG_DIR}/*.log to watch progress)..."
fail=0
for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
done

echo "Merging results back into the main dataset..."
merge_all

if [ "$fail" -ne 0 ]; then
    echo "One or more per-author processes exited non-zero -- check $LOG_DIR" >&2
    exit 1
fi
echo "Done."
