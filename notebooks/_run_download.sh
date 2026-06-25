#!/usr/bin/env bash
# Inner script for CD8 download bsub job.
set -eu

source /home/projects/nyosef/zvise/.local/share/mamba/etc/profile.d/mamba.sh
mamba activate neural_nmf_env

echo "[$(date)] host=$(hostname)"
free -g | head -2

exec python -u /home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks/_download_immune_datasets.py zheng_cd8_clean
