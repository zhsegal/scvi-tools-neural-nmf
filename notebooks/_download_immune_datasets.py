"""Download + preprocess CD8 (Gong 2025) and tonsil B (King 2021) datasets from
CELLxGENE Census; output h5ads with the same var/obs schema as monocytes_clean.h5ad.

Each output:
- raw counts in .X (float32)
- var: feature_id, feature_name, feature_type, feature_length, soma_joinid, nnz, n_measured_obs
- obs: cell_type, disease, donor_id (plus assay for B-cell since 2 chemistries)
- HVG=4000 via seurat_v3 (operates on raw counts) with batch_key=donor_id
"""
from __future__ import annotations
import sys
import warnings
warnings.filterwarnings("ignore")

import cellxgene_census
import numpy as np
import scanpy as sc
from pathlib import Path

OUT_DIR = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks")
HVG_N = 4000
CENSUS_VERSION = "2025-11-08"

DATASETS = {
    "zheng_cd8_clean": {
        "dataset_id": "b90d4a42-714c-4590-b548-6e75e6c081e8",
        "label": "CD8",
        # CD8 atlas has γδ + MAIT alongside CD8 subsets — drop non-CD8 for clean CD8 atlas
        "cell_type_filter": [
            "effector memory CD8-positive, alpha-beta T cell",
            "naive thymus-derived CD8-positive, alpha-beta T cell",
            "central memory CD8-positive, alpha-beta T cell",
            "CD8aa(I) thymocyte",
            "CD8-positive, alpha-beta memory T cell",
        ],
    },
    "tonsil_b_clean": {
        "dataset_id": "482954b2-0456-4901-b379-b62f99c0ab2d",
        "label": "BCELL",
        "cell_type_filter": None,  # use all B-lineage cells
    },
    # Perez et al. 2022 SLE lupus (Science) — disease (SLE/normal) + ancestry factor.
    # Single primary dataset (1.26M PBMC, 261 donors); filter by dataset_id alone.
    "perez_sle_mono_clean": {
        "dataset_id": "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
        "label": "MONO",
        "cell_type_filter": ["classical monocyte"],
        "primary": None,
    },
    "perez_sle_cd4_clean": {
        "dataset_id": "218acb0f-9f2f-4f76-b90b-15a4b7c7f629",
        "label": "CD4",
        "cell_type_filter": ["CD4-positive, alpha-beta T cell"],
        "primary": None,
    },
}

KEEP_VAR = ["feature_id", "feature_name", "feature_type", "feature_length",
            "soma_joinid", "nnz", "n_measured_obs"]
KEEP_OBS = ["cell_type", "disease", "donor_id", "assay", "tissue",
            "self_reported_ethnicity", "sex"]


def fetch(dataset_id: str, cell_type_filter: list[str] | None, primary: bool | None = False):
    val = f"dataset_id == '{dataset_id}'"
    if primary is not None:
        val += f" and is_primary_data == {primary}"
    if cell_type_filter is not None:
        ct_list = ", ".join(f"'{c}'" for c in cell_type_filter)
        val += f" and cell_type in [{ct_list}]"
    with cellxgene_census.open_soma(census_version=CENSUS_VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_value_filter=val,
        )
    return adata


def preprocess(adata, hvg_n: int):
    # CELLxGENE puts raw counts in adata.X already; cast for safety
    adata.X = adata.X.astype(np.float32)
    # var_names default from census is soma_joinid (integer string). Switch to
    # Ensembl IDs so they match monocytes_clean and feed Geneformer / id_map.
    adata.var_names = adata.var["feature_id"].astype(str).values
    # Drop unmeasured/zero genes first (helps HVG numerics)
    sc.pp.filter_genes(adata, min_cells=5)
    # seurat_v3 fits a per-batch LOESS; with many heterogeneous donors (e.g. Perez's
    # 261) the batched fit can become ill-conditioned — fall back to unbatched HVG.
    try:
        sc.pp.highly_variable_genes(
            adata, n_top_genes=hvg_n, flavor="seurat_v3",
            batch_key="donor_id", subset=True,
        )
    except ValueError as exc:
        print(f"  HVG: batched seurat_v3 failed ({exc}); falling back to unbatched", flush=True)
        sc.pp.highly_variable_genes(
            adata, n_top_genes=hvg_n, flavor="seurat_v3", subset=True,
        )
    # Keep only the var columns we want, in the same order monocyte has
    keep_cols = [c for c in KEEP_VAR if c in adata.var.columns]
    adata.var = adata.var[keep_cols]
    keep_obs = [c for c in KEEP_OBS if c in adata.obs.columns]
    adata.obs = adata.obs[keep_obs]
    return adata


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name is None:
        targets = list(DATASETS.items())
    else:
        targets = [(name, DATASETS[name])]
    for stem, cfg in targets:
        out_path = OUT_DIR / f"{stem}.h5ad"
        if out_path.exists():
            print(f"[{stem}] exists at {out_path}; skipping")
            continue
        print(f"\n=== [{stem}] downloading dataset {cfg['dataset_id']} ===", flush=True)
        adata = fetch(cfg["dataset_id"], cfg["cell_type_filter"], cfg.get("primary", False))
        print(f"[{stem}] raw shape: {adata.shape}", flush=True)
        print(f"[{stem}] obs preview: {adata.obs.columns.tolist()}", flush=True)
        adata = preprocess(adata, HVG_N)
        print(f"[{stem}] after HVG ({HVG_N}): {adata.shape}", flush=True)
        print(f"[{stem}] cell_type counts:\n{adata.obs['cell_type'].value_counts()}", flush=True)
        print(f"[{stem}] donor counts:\n{adata.obs['donor_id'].value_counts().head()}", flush=True)
        adata.write_h5ad(out_path, compression="gzip")
        print(f"[{stem}] wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
