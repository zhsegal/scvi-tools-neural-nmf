#!/usr/bin/env bash
# Inner script: download + preprocess Perez SLE (mono, CD4) and COMBAT (mono, CD8).
set -eu

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

NB=/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks

echo "[$(date)] host=$(hostname)"
free -g | head -2

python -u "$NB/_download_immune_datasets.py" perez_sle_mono_clean
python -u "$NB/_download_immune_datasets.py" perez_sle_cd4_clean
python -u "$NB/_download_combat.py"

echo "[$(date)] all downloads done"
ls -lh "$NB"/perez_sle_*_clean.h5ad "$NB"/combat_*_clean.h5ad
