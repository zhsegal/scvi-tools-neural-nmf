#!/usr/bin/env bash
# Submit a four-way benchmark notebook to the GPU queue via bsub.
# Usage:
#   ./run_benchmark_on_gpu.sh four_way_benchmark_bcell.ipynb [MEM_MB]
#   ./run_benchmark_on_gpu.sh four_way_benchmark_cd8.ipynb 64000
set -euo pipefail

NB="${1:?usage: $0 <notebook.ipynb> [mem_mb]}"
MEM_MB="${2:-32000}"
WALL_H="${WALL_H:-72}"
NB_DIR="/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks"

[[ "$NB" = /* ]] || NB="$NB_DIR/$NB"
[[ -f "$NB" ]] || { echo "missing: $NB" >&2; exit 1; }

stem="$(basename "$NB" .ipynb)"
log="$NB_DIR/${stem}.bsub.log"
rm -f "$log"

bsub \
    -q gsla_high_gpu \
    -gpu "num=1:j_exclusive=yes" \
    -R "rusage[mem=${MEM_MB}]" \
    -W "${WALL_H}:00" \
    -J "${stem}" \
    -o "$log" \
    -e "$log" \
    "$NB_DIR/_run_nb.sh" "$NB"
