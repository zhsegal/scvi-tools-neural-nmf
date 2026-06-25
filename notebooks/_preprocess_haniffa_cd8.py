"""Preprocess the Haniffa COVID-19 PBMC atlas to a CD8-only h5ad with the same
schema (var: feature_id [Ensembl], feature_name [symbol]; .X = raw counts) as
zheng_cd8_clean.h5ad so the four-way SemanticSCVI benchmark notebook can train
on it directly.

Strategy (to stay within ~24 GB memcg cap):
  * Read obs columns + var symbols directly via h5py (cheap).
  * Build the CD8 row mask before touching any matrix.
  * Bulk-load layers["raw"] as a scipy CSR (raw counts, ~12 GB) — we never load
    the normalized .X (saves ~12 GB).
  * Slice CSR by CD8 rows, drop the full matrix to free memory.

Reads:  notebooks/haniffa_covid_pbmc_raw.h5ad  (haniffa21.processed.h5ad)
Writes: notebooks/haniffa_cd8_clean.h5ad
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
OUT_PATH = NB_DIR / "haniffa_cd8_clean.h5ad"

CD8_STATES = ["CD8.EM", "CD8.TE", "CD8.Naive", "CD8.Prolif"]
HVG_N = 4000
CENSUS_VERSION = "2025-11-08"


def _decode(arr):
    if arr.dtype.kind in ("S", "O"):
        return np.array([x.decode() if isinstance(x, bytes) else x for x in arr])
    return arr


def _attr_str(group, key, default):
    v = group.attrs.get(key, default)
    if isinstance(v, bytes):
        v = v.decode()
    return v


def _read_categorical(f, col):
    codes = f[f"obs/{col}"][:]
    cats = _decode(f[f"obs/__categories/{col}"][:])
    return cats[codes]


def _symbol_to_ensembl():
    print(f"fetching cellxgene gene table (census {CENSUS_VERSION})", flush=True)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        gt = (
            census["census_data"]["homo_sapiens"]
            .ms["RNA"]
            .var.read()
            .concat()
            .to_pandas()
        )
    gt = gt[["feature_id", "feature_name"]].dropna()
    gt = gt.drop_duplicates(subset="feature_name", keep="first")
    d = dict(zip(gt["feature_name"].astype(str), gt["feature_id"].astype(str)))
    print(f"  census gene table: {len(d)} symbol→Ensembl entries", flush=True)
    return d


def main():
    if not IN_PATH.exists():
        sys.exit(f"missing input: {IN_PATH}")

    print(f"reading metadata from {IN_PATH}", flush=True)
    with h5py.File(IN_PATH, "r") as f:
        var_index_col = _attr_str(f["var"], "_index", "_index")
        obs_index_col = _attr_str(f["obs"], "_index", "_index")
        symbols = _decode(f[f"var/{var_index_col}"][:])
        n_genes = len(symbols)
        full_clust = _read_categorical(f, "full_clustering")
        n_cells = full_clust.size
        print(f"  cells={n_cells}, genes={n_genes}", flush=True)
        row_mask = np.isin(full_clust, CD8_STATES)
        print(f"  CD8 mask: {int(row_mask.sum())} / {n_cells}", flush=True)

        obs_dict = {"cell_type": full_clust[row_mask]}
        for src, dst in (
            ("patient_id", "donor_id"),
            ("Status", "Status"),
            ("Status_on_day_collection_summary", "Status_on_day_collection_summary"),
            ("Site", "Site"),
            ("Sex", "Sex"),
        ):
            if src in f["obs"]:
                obs_dict[dst] = _read_categorical(f, src)[row_mask]
        obs_index = _decode(f[f"obs/{obs_index_col}"][:])[row_mask]

        print("loading layers/raw (raw counts CSR) ...", flush=True)
        n_obs = n_cells
        indptr = f["layers/raw/indptr"][:].astype(np.int64)
        print(f"  indptr loaded ({indptr.nbytes/1e6:.0f} MB)", flush=True)
        indices = f["layers/raw/indices"][:].astype(np.int32, copy=False)
        print(f"  indices loaded ({indices.nbytes/1e6:.0f} MB)", flush=True)
        data = f["layers/raw/data"][:].astype(np.float32, copy=False)
        print(f"  data loaded ({data.nbytes/1e6:.0f} MB)", flush=True)

    raw_full = csr_matrix((data, indices, indptr), shape=(n_obs, n_genes))
    del data, indices, indptr
    gc.collect()
    print(f"  raw_full: {raw_full.shape}, nnz={raw_full.nnz}", flush=True)

    raw = raw_full[row_mask].tocsr().astype(np.float32)
    del raw_full
    gc.collect()
    print(f"  raw (CD8 only): {raw.shape}, nnz={raw.nnz}", flush=True)
    samp = raw.data[: min(50_000, raw.data.size)]
    if samp.size > 0 and not np.allclose(samp, np.round(samp)):
        print(f"WARNING: layers/raw not integer; sample {samp[:5]}", flush=True)

    obs_df = pd.DataFrame(obs_dict, index=pd.Index(obs_index, name="cell_id"))
    var_df = pd.DataFrame({"feature_name": symbols}, index=symbols)

    sym2ens = _symbol_to_ensembl()
    feature_id = np.array([sym2ens.get(s, "") for s in symbols])
    mapped = feature_id != ""
    print(f"mapped {int(mapped.sum())} / {len(symbols)} symbols to Ensembl", flush=True)

    raw = raw[:, mapped].tocsr()
    var_df = var_df.iloc[mapped].copy()
    var_df["feature_id"] = feature_id[mapped]
    var_df.index = var_df["feature_id"].values

    adata = ad.AnnData(X=raw, obs=obs_df, var=var_df)
    print(f"after Ensembl mapping: {adata.shape}", flush=True)

    sc.pp.filter_genes(adata, min_cells=5)
    print(f"after filter_genes(min_cells=5): {adata.shape}", flush=True)

    # seurat_v3 fits a LOESS per batch. With Haniffa's heterogeneous donor sizes
    # the per-donor LOESS becomes ill-conditioned even after filtering tiny
    # donors. Use unbatched HVG instead (n=104k cells, plenty for stable LOESS).
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=HVG_N, flavor="seurat_v3",
            batch_key="donor_id", subset=True,
        )
        print("  HVG: batched seurat_v3 succeeded", flush=True)
    except ValueError as exc:
        print(f"  HVG: batched seurat_v3 failed ({exc}); falling back to unbatched", flush=True)
        sc.pp.highly_variable_genes(
            adata, n_top_genes=HVG_N, flavor="seurat_v3", subset=True,
        )
        print("  HVG: unbatched seurat_v3 succeeded", flush=True)
    adata.var = adata.var[["feature_id", "feature_name"]]
    print(f"final shape: {adata.shape}", flush=True)
    print(f"cell_type counts:\n{adata.obs['cell_type'].value_counts()}", flush=True)
    print(f"donor counts (top 10):\n{adata.obs['donor_id'].value_counts().head(10)}", flush=True)

    adata.write_h5ad(OUT_PATH, compression="gzip")
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
