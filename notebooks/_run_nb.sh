#!/usr/bin/env bash
# Inner script run inside bsub-allocated GPU node.
# Usage: _run_nb.sh <notebook.ipynb>
set -eu

NB="${1:?missing notebook}"
NB_DIR="/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks"
[[ "$NB" = /* ]] || NB="$NB_DIR/$NB"

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

cd "$NB_DIR"
echo "[$(date)] host=$(hostname) nb=$NB"
echo "python: $(which python)"
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

exec jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    "$NB"
