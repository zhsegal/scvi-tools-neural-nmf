"""Download + preprocess the COMBAT blood atlas (Cell 2022) CITE-seq object into
classical-monocyte and CD8-T h5ads with the same schema as the CELLxGENE-sourced
datasets (var: feature_id [Ensembl], feature_name [symbol]; .X = raw counts).

COMBAT is NOT in CELLxGENE Census — the processed CITE-seq object lives on Zenodo
(record 6120249, ~unzipped multi-GB `COMBAT-CITESeq-DATA.h5ad`). Its design factor is
`Source` (disease group: healthy / COVID mild·severe·critical / sepsis / influenza).

Outputs (NB_DIR):
  combat_mono_clean.h5ad  — Annotation_major_subset == cMono  (classical monocytes)
  combat_cd8_clean.h5ad   — Annotation_major_subset == CD8    (CD8 T cells)

Each: raw counts in .X (float32); obs cell_type (= minor subset), donor_id, Source,
Pool_ID; HVG=4000 seurat_v3 (batched on donor_id, unbatched fallback).

Run:  python notebooks/_download_combat.py            # both
      python notebooks/_download_combat.py combat_cd8_clean
"""
from __future__ import annotations
import sys
import urllib.request
import warnings
warnings.filterwarnings("ignore")

import anndata as ad
import cellxgene_census
import numpy as np
import scanpy as sc
from pathlib import Path
from scipy.sparse import issparse

NB_DIR = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks")
SRC_H5AD = NB_DIR / "COMBAT-CITESeq-DATA.h5ad"
SRC_URL = "https://zenodo.org/record/6120249/files/COMBAT-CITESeq-DATA.h5ad?download=1"
HVG_N = 4000
CENSUS_VERSION = "2025-11-08"

# Annotation_major_subset value -> output stem. cMono = classical monocytes.
DATASETS = {
    "combat_mono_clean": "cMono",
    "combat_cd8_clean": "CD8",
}

# obs column candidates (first present wins). COMBAT field names confirmed at runtime
# via the column print below.
DONOR_CANDIDATES = ["COMBAT_participant_timepoint_ID", "scRNASeq_sample_ID", "COMBAT_ID"]
BATCH_CANDIDATES = ["Pool_ID", "GEX_pool", "scRNASeq_sample_ID"]
MAJOR_COL = "Annotation_major_subset"
MINOR_COL = "Annotation_minor_subset"
SOURCE_COL = "Source"


def _download():
    if SRC_H5AD.exists():
        print(f"source present: {SRC_H5AD} ({SRC_H5AD.stat().st_size / 1e9:.1f} GB)", flush=True)
        return
    print(f"downloading COMBAT CITE-seq object -> {SRC_H5AD}", flush=True)
    urllib.request.urlretrieve(SRC_URL, SRC_H5AD)
    print(f"  done ({SRC_H5AD.stat().st_size / 1e9:.1f} GB)", flush=True)


def _symbol_to_ensembl():
    print(f"fetching cellxgene gene table (census {CENSUS_VERSION})", flush=True)
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        gt = (census["census_data"]["homo_sapiens"].ms["RNA"].var.read()
              .concat().to_pandas())
    gt = gt[["feature_id", "feature_name"]].dropna().drop_duplicates(
        subset="feature_name", keep="first")
    d = dict(zip(gt["feature_name"].astype(str), gt["feature_id"].astype(str)))
    print(f"  census gene table: {len(d)} symbol->Ensembl entries", flush=True)
    return d


def _pick(cols, candidates, what):
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(f"none of {candidates} found for {what}; available: {list(cols)}")


def _raw_counts_layer(adata):
    """Return a (matrix, source-name) of integer counts, checking layers/.raw/.X."""
    for name, mat in [("layers['counts']", adata.layers.get("counts")),
                      ("layers['raw']", adata.layers.get("raw"))]:
        if mat is not None:
            return mat, name
    if adata.raw is not None:
        return adata.raw.X, ".raw.X"
    return adata.X, ".X"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    targets = list(DATASETS.items()) if name is None else [(name, DATASETS[name])]
    if any(not (NB_DIR / f"{s}.h5ad").exists() for s, _ in targets):
        _download()
        print(f"loading {SRC_H5AD.name} (backed) ...", flush=True)
        full = sc.read_h5ad(SRC_H5AD, backed="r")
        print(f"  full: {full.shape}; obs cols: {list(full.obs.columns)}", flush=True)
        print(f"  {MAJOR_COL} values: {full.obs[MAJOR_COL].value_counts().to_dict()}", flush=True)
        sym2ens = _symbol_to_ensembl()

    for stem, major in targets:
        out_path = NB_DIR / f"{stem}.h5ad"
        if out_path.exists():
            print(f"[{stem}] exists; skipping", flush=True)
            continue
        print(f"\n=== [{stem}] subset {MAJOR_COL} == {major} ===", flush=True)
        mask = (full.obs[MAJOR_COL] == major).values
        adata = full[mask].to_memory()
        print(f"[{stem}] cells: {adata.n_obs}", flush=True)

        mat, src = _raw_counts_layer(adata)
        print(f"[{stem}] counts from {src}", flush=True)
        X = mat.tocsr() if issparse(mat) else mat
        samp = (X.data if issparse(X) else np.asarray(X).ravel())[:50000]
        if samp.size and not np.allclose(samp, np.round(samp)):
            print(f"WARNING: {src} not integer; sample {samp[:5]}", flush=True)
        adata.X = X.astype(np.float32)
        adata.layers.clear()
        adata.raw = None

        donor = _pick(adata.obs.columns, DONOR_CANDIDATES, "donor_id")
        batch = _pick(adata.obs.columns, BATCH_CANDIDATES, "batch")
        obs = adata.obs
        adata.obs = obs.assign(
            cell_type=obs[MINOR_COL].astype(str).values if MINOR_COL in obs else major,
            donor_id=obs[donor].astype(str).values,
            Source=obs[SOURCE_COL].astype(str).values if SOURCE_COL in obs else "NA",
            Pool_ID=obs[batch].astype(str).values,
        )[["cell_type", "donor_id", "Source", "Pool_ID"]]

        # map symbols -> Ensembl (skip if var already Ensembl)
        symbols = adata.var_names.astype(str).values
        if not symbols[0].startswith("ENSG"):
            feature_id = np.array([sym2ens.get(s, "") for s in symbols])
            mapped = feature_id != ""
            print(f"[{stem}] mapped {int(mapped.sum())}/{len(symbols)} symbols to Ensembl", flush=True)
            adata = adata[:, mapped].copy()
            adata.var = adata.var.assign(feature_name=symbols[mapped],
                                         feature_id=feature_id[mapped])
            adata.var_names = adata.var["feature_id"].values
        else:
            adata.var = adata.var.assign(feature_id=symbols,
                                         feature_name=adata.var.get("feature_name", symbols))

        sc.pp.filter_genes(adata, min_cells=5)
        try:
            sc.pp.highly_variable_genes(adata, n_top_genes=HVG_N, flavor="seurat_v3",
                                        batch_key="donor_id", subset=True)
        except ValueError as exc:
            print(f"[{stem}] batched HVG failed ({exc}); unbatched fallback", flush=True)
            sc.pp.highly_variable_genes(adata, n_top_genes=HVG_N, flavor="seurat_v3", subset=True)
        adata.var = adata.var[["feature_id", "feature_name"]]

        print(f"[{stem}] final: {adata.shape}", flush=True)
        print(f"[{stem}] cell_type:\n{adata.obs['cell_type'].value_counts()}", flush=True)
        print(f"[{stem}] Source:\n{adata.obs['Source'].value_counts()}", flush=True)
        adata.write_h5ad(out_path, compression="gzip")
        print(f"[{stem}] wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
