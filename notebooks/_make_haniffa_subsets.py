"""Subset the Haniffa COVID-19 PBMC atlas into clean per-population h5ads (CD4 T,
classical monocytes, NK, B) with the SAME schema as haniffa_cd8_clean.h5ad
(var: feature_id [Ensembl], feature_name [symbol]; .X = raw counts) so the
modalities benchmark can train on each directly.

Generalizes _preprocess_haniffa_cd8.py to a dict of population -> full_clustering
states. Loads the raw counts CSR once, then slices/maps/HVG-subsets per population.

Reads:  notebooks/haniffa_covid_pbmc_raw.h5ad
Writes: notebooks/haniffa_{cd4,mono,nk,b}_clean.h5ad

Run:    python notebooks/_make_haniffa_subsets.py [cd4 mono nk b]
        (no args -> all four; pass names to build a subset)
"""
from __future__ import annotations
import gc
import sys
import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import cellxgene_census
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.sparse import csr_matrix

NB_DIR = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks")
IN_PATH = NB_DIR / "haniffa_covid_pbmc_raw.h5ad"
HVG_N = 4000
CENSUS_VERSION = "2025-11-08"

# population -> (output stem, full_clustering states). Mirrors CD8_STATES in
# _preprocess_haniffa_cd8.py (CD8 already built; not repeated here).
POPULATIONS = {
    "cd4": ("haniffa_cd4_clean", [
        "CD4.Naive", "CD4.CM", "CD4.EM", "CD4.IL22", "CD4.Tfh",
        "CD4.Th1", "CD4.Th2", "CD4.Th17", "CD4.Prolif", "Treg",
    ]),
    "mono": ("haniffa_mono_clean", [
        "CD14_mono", "CD83_CD14_mono",  # classical only (exclude CD16/non-classical)
    ]),
    "nk": ("haniffa_nk_clean", [
        "NK_16hi", "NK_56hi", "NK_prolif",  # exclude NKT / ILC
    ]),
    "b": ("haniffa_b_clean", [
        "B_naive", "B_switched_memory", "B_non-switched_memory", "B_immature",
        "B_exhausted", "Plasmablast", "Plasma_cell_IgG", "Plasma_cell_IgA",
        "Plasma_cell_IgM",  # B -> plasma trajectory (exclude B_malignant)
    ]),
}

OBS_CARRY = [
    ("patient_id", "donor_id"),
    ("Status", "Status"),
    ("Status_on_day_collection_summary", "Status_on_day_collection_summary"),
    ("Site", "Site"),
    ("Sex", "Sex"),
]


def _decode(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    return arr


def _attr_str(group, key, default):
    v = group.attrs.get(key, default)
    return v.decode() if isinstance(v, bytes) else v


def _read_categorical(f, col):
    codes = f[f"obs/{col}"][:]
    cats = _decode(f[f"obs/__categories/{col}"][:])
    return cats[codes]


def _symbol_to_ensembl():
    print(f"fetching cellxgene gene table (census {CENSUS_VERSION})", flush=True)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        gt = (census["census_data"]["homo_sapiens"].ms["RNA"].var.read()
              .concat().to_pandas())
    gt = gt[["feature_id", "feature_name"]].dropna()
    gt = gt.drop_duplicates(subset="feature_name", keep="first")
    d = dict(zip(gt["feature_name"].astype(str), gt["feature_id"].astype(str)))
    print(f"  census gene table: {len(d)} symbol->Ensembl entries", flush=True)
    return d


def _build_one(stem, states, raw_full, full_clust, obs_all, symbols, sym2ens):
    row_mask = np.isin(full_clust, states)
    print(f"\n[{stem}] mask: {int(row_mask.sum())} / {row_mask.size} cells "
          f"({len(states)} states)", flush=True)
    if row_mask.sum() == 0:
        print(f"[{stem}] WARNING: no cells matched {states}; skipping", flush=True)
        return

    raw = raw_full[row_mask].tocsr().astype(np.float32)
    obs_df = pd.DataFrame(
        {"cell_type": full_clust[row_mask],
         **{dst: obs_all[src][row_mask] for src, dst in obs_all["_carry"]}},
        index=pd.Index(obs_all["_index"][row_mask], name="cell_id"),
    )
    var_df = pd.DataFrame({"feature_name": symbols}, index=symbols)

    feature_id = np.array([sym2ens.get(s, "") for s in symbols])
    mapped = feature_id != ""
    print(f"[{stem}] mapped {int(mapped.sum())} / {len(symbols)} symbols to Ensembl", flush=True)
    raw = raw[:, mapped].tocsr()
    var_df = var_df.iloc[mapped].copy()
    var_df["feature_id"] = feature_id[mapped]
    var_df.index = var_df["feature_id"].values

    adata = ad.AnnData(X=raw, obs=obs_df, var=var_df)
    sc.pp.filter_genes(adata, min_cells=5)
    print(f"[{stem}] after filter_genes(min_cells=5): {adata.shape}", flush=True)
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=HVG_N, flavor="seurat_v3",
                                    batch_key="donor_id", subset=True)
        print(f"[{stem}] HVG: batched seurat_v3 succeeded", flush=True)
    except ValueError as exc:
        print(f"[{stem}] HVG: batched failed ({exc}); unbatched fallback", flush=True)
        sc.pp.highly_variable_genes(adata, n_top_genes=HVG_N, flavor="seurat_v3", subset=True)
    adata.var = adata.var[["feature_id", "feature_name"]]

    out_path = NB_DIR / f"{stem}.h5ad"
    print(f"[{stem}] final: {adata.shape}", flush=True)
    print(f"[{stem}] cell_type counts:\n{adata.obs['cell_type'].value_counts()}", flush=True)
    adata.write_h5ad(out_path, compression="gzip")
    print(f"[{stem}] wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)
    del adata, raw, var_df, obs_df
    gc.collect()


def main():
    if not IN_PATH.exists():
        sys.exit(f"missing input: {IN_PATH}")
    names = sys.argv[1:] or list(POPULATIONS)
    bad = [n for n in names if n not in POPULATIONS]
    if bad:
        sys.exit(f"unknown population(s) {bad}; choose from {list(POPULATIONS)}")

    print(f"reading metadata + raw counts from {IN_PATH}", flush=True)
    with h5py.File(IN_PATH, "r") as f:
        var_index_col = _attr_str(f["var"], "_index", "_index")
        obs_index_col = _attr_str(f["obs"], "_index", "_index")
        symbols = _decode(f[f"var/{var_index_col}"][:])
        n_genes = len(symbols)
        full_clust = _read_categorical(f, "full_clustering")
        n_cells = full_clust.size
        print(f"  cells={n_cells}, genes={n_genes}", flush=True)

        obs_all = {"_index": _decode(f[f"obs/{obs_index_col}"][:]), "_carry": []}
        for src, dst in OBS_CARRY:
            if src in f["obs"]:
                obs_all[src] = _read_categorical(f, src)
                obs_all["_carry"].append((src, dst))

        print("loading layers/raw (raw counts CSR) ...", flush=True)
        indptr = f["layers/raw/indptr"][:].astype(np.int64)
        indices = f["layers/raw/indices"][:].astype(np.int32, copy=False)
        data = f["layers/raw/data"][:].astype(np.float32, copy=False)

    raw_full = csr_matrix((data, indices, indptr), shape=(n_cells, n_genes))
    del data, indices, indptr
    gc.collect()
    print(f"  raw_full: {raw_full.shape}, nnz={raw_full.nnz}", flush=True)

    sym2ens = _symbol_to_ensembl()
    for n in names:
        stem, states = POPULATIONS[n]
        _build_one(stem, states, raw_full, full_clust, obs_all, symbols, sym2ens)


if __name__ == "__main__":
    main()
