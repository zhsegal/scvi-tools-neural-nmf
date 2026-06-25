# Immune single-population datasets for SemanticSCVI

Two clean, well-annotated human immune populations to drive gene-program discovery. Both ship with raw counts (required for NB likelihood). Pick **Zheng CD8** for exhaustion/effector program biology, **tonsil B** for differentiation-trajectory programs.

Save all downloaded `.h5ad` files under `notebooks/data/` (gitignored).

---

## 1. Zheng pan-cancer CD8⁺ T cells (Science 2021)

~400k tumor-infiltrating T cells, 316 donors, 21 cancer types. Subset CD8 → ~200k cells.

### Download (CELLxGENE — easiest)

```bash
mkdir -p notebooks/data && cd notebooks/data
# Pan-cancer T cell atlas, Zheng et al. 2021 (collection on CELLxGENE)
# Direct h5ad: open the dataset page and copy the "Download .h5ad" URL.
# Fallback: GEO super-series GSE156728 has per-cancer count matrices.
wget -O zheng_tcells.h5ad "<paste CELLxGENE h5ad URL here>"
```

If CELLxGENE link is gone, scrape it programmatically:

```python
import cellxgene_census
census = cellxgene_census.open_soma()
adata = cellxgene_census.get_anndata(
    census,
    organism="Homo sapiens",
    obs_value_filter=(
        "dataset_id == '<zheng_dataset_id>' and "
        "cell_type in ['CD8-positive, alpha-beta T cell']"
    ),
)
adata.write_h5ad("notebooks/data/zheng_cd8.h5ad")
```

### Subset + preprocess

```python
import scanpy as sc
adata = sc.read_h5ad("notebooks/data/zheng_tcells.h5ad")

# Subset to CD8 (column name varies — check adata.obs.columns first)
adata = adata[adata.obs["cell_type"].str.contains("CD8", case=False)].copy()

# Keep raw counts in .layers["counts"]; SemanticSCVI needs them
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="donor_id", subset=True)
adata.X = adata.layers["counts"]   # restore raw counts
adata.write_h5ad("notebooks/data/zheng_cd8_hvg.h5ad")
```

### Key obs columns

`donor_id`, `cancer_type`, `tissue` (tumor / normal-adjacent / blood), `cell_type` (subcluster: Tex, Tem, Trm, Tn, Tcm, MAIT…).

### Expected gene programs

Cytotoxicity (GZMB/PRF1/NKG7), IFN-α/γ response, terminal exhaustion (LAG3/TIGIT/HAVCR2/TOX), proliferation (MKI67/STMN1), tissue-resident memory (ITGAE/ZNF683), naive (CCR7/SELL/TCF7).

---

## 2. Tonsil B cells (Massoni-Badosa, Immunity 2024)

~120k B cells across naive → GC → memory → plasma. Cleanest known B cell trajectory.

### Download via `HCATonsilData` (R, recommended)

```r
# install.packages("BiocManager"); BiocManager::install("HCATonsilData")
library(HCATonsilData)
sce <- HCATonsilData(assayType = "RNA", cellType = "All")    # or cellType = "NBC_MBC", "GCBC", "PC"
# Export to h5ad via zellkonverter
zellkonverter::writeH5AD(sce, "notebooks/data/tonsil_all.h5ad")
```

### Or: direct h5ad from CELLxGENE

```bash
# "An atlas of cells in the human tonsil" collection
wget -O notebooks/data/tonsil_atlas.h5ad "<paste CELLxGENE h5ad URL here>"
```

### Subset to B-lineage + preprocess

```python
import scanpy as sc
adata = sc.read_h5ad("notebooks/data/tonsil_atlas.h5ad")

# B-lineage compartments in this atlas
b_types = ["NBC_MBC", "GCBC", "PC", "Activated_NBC", "Preplasmablast"]
adata = adata[adata.obs["annotation_level_1"].isin(b_types)].copy()

adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, batch_key="donor_id", subset=True)
adata.X = adata.layers["counts"]
adata.write_h5ad("notebooks/data/tonsil_b_hvg.h5ad")
```

### Key obs columns

`donor_id`, `annotation_level_1` / `annotation_20220215` (fine subcluster), `assay` (mostly 10x 3'), `age_group`.

### Expected gene programs

BCR/CD40 signaling, GC dark-zone proliferation (MKI67/AICDA), light-zone selection (BCL6/CD83), class-switch (AID/UNG), ER/UPR + unfolded protein response (in plasma cells: XBP1/MZB1/DERL3), memory survival (TNFRSF13B).

---

## 3. Train SemanticSCVI

Both datasets plug into the existing pipeline once you set `labels_key` to the fine subcluster column.

```python
import scvi
from scvi.model import SemanticSCVI

adata = sc.read_h5ad("notebooks/data/zheng_cd8_hvg.h5ad")     # or tonsil_b_hvg.h5ad
SemanticSCVI.setup_anndata(
    adata,
    layer="counts",
    batch_key="donor_id",
    labels_key="cell_type",          # or "annotation_level_1" for tonsil
)
model = SemanticSCVI(
    adata,
    n_semantic_factors=20,
    n_shared_latent=5,
    n_private_latent=3,
    n_labels_source=adata.obs["cell_type"].nunique(),
)
model.train(max_epochs=200, warmup_epochs=20, warmup_schedule="cosine")
```

Then benchmark with `scripts/benchmarking.py::SemanticBenchmark.run_all(hallmark_gmt=..., semantic_map=...)`.

---

## Quick smoke test before full training

Run `scripts/quick_test.py` against a 5k-cell subsample first — verifies the obs columns wire up correctly and the loss decreases.

```python
adata_small = adata[adata.obs.sample(n=5000, random_state=0).index].copy()
adata_small.write_h5ad("notebooks/data/zheng_cd8_5k.h5ad")
```
