"""Repair cNMF caches whose usage matrix W is all-NaN.

Pre-fix train_cnmf.py reindexed cNMF's usage (parsed back as int64 for numeric cell
names) against string obs_names -> all-NaN W (broke modalities 2 & 5). This rebuilds W
from the EXISTING consensus output (no re-factorization) and rewrites the .pkl. Spectra
H / loadings were unaffected, so they are kept as-is.

Run:  python notebooks/_repair_cnmf_usage.py
Scans notebooks/.model_cache_*/**/cnmf_k10.pkl with a sibling cnmf_k10_run/ dir.
"""
from __future__ import annotations
import pickle
from pathlib import Path

import anndata as ad
import numpy as np

NB_DIR = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks")
DENSITY_THRESHOLD = 0.5


def _repair(pkl_path: Path):
    run_dir = pkl_path.with_name("cnmf_k10_run")
    counts_fn = run_dir / "cnmf_k10_counts.h5ad"
    if not (run_dir.exists() and counts_fn.exists()):
        print(f"  skip {pkl_path} (no cnmf_k10_run/counts)", flush=True)
        return
    w = pickle.load(open(pkl_path, "rb"))
    W = np.asarray(w.W)
    if not np.isnan(W).any():
        print(f"  OK   {pkl_path.relative_to(NB_DIR)} (W has no NaN)", flush=True)
        return

    from cnmf import cNMF
    k = int(w.H.shape[0])
    obj = cNMF(output_dir=str(run_dir), name="cnmf_k10")
    usage, _scores, _tpm, _ = obj.load_results(
        K=k, density_threshold=DENSITY_THRESHOLD, norm_usage=True)
    obs_names = ad.read_h5ad(counts_fn, backed="r").obs_names.astype(str)
    usage.index = usage.index.astype(str)
    usage = usage.reindex(obs_names)
    W_fixed = usage.to_numpy(dtype=float)
    if np.isnan(W_fixed).any():
        print(f"  FAIL {pkl_path.relative_to(NB_DIR)} (still NaN after rebuild)", flush=True)
        return
    w.W = W_fixed
    pickle.dump(w, open(pkl_path, "wb"))
    print(f"  FIXED {pkl_path.relative_to(NB_DIR)}  W {W_fixed.shape}", flush=True)


def main():
    pkls = sorted(NB_DIR.glob(".model_cache_*/**/cnmf_k10.pkl"))
    print(f"scanning {len(pkls)} cnmf_k10.pkl cache(s)", flush=True)
    for p in pkls:
        _repair(p)


if __name__ == "__main__":
    main()
