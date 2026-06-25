"""Add semantic_geom-derived denoised expression + dispersions (+ W) to the
existing haniffa_cd8 projections h5ad, without retraining.

Mirrors the geom branch of four_way_benchmark_haniffa_cd8.ipynb Cell 11 exactly,
but loads the cached Site-batch geom model against the already-written projections
adata (same cells/genes/order) and only ADDS the geom keys — genept keys untouched.
"""
import os
import sys
import types
from pathlib import Path

import numpy as np
import scanpy as sc

NB_DIR = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks")
sys.path.insert(0, str(NB_DIR))

from scvi.model._semantic_scvi import SemanticSCVI  # noqa: E402

H5AD = NB_DIR / "haniffa_cd8_clean_projections.h5ad"
GEOM_CACHE = NB_DIR / ".model_cache_haniffa_cd8" / "semantic_scvi" / "375e77e432"

print(f"Loading {H5AD} ...", flush=True)
adata = sc.read_h5ad(H5AD)
print(f"  adata: {adata.shape}", flush=True)
assert "X_semantic_geom" in adata.obsm, "missing obsm['X_semantic_geom']"

print(f"Registering + loading geom model from {GEOM_CACHE} ...", flush=True)
SemanticSCVI.setup_anndata(adata, layer=None, labels_key="cell_type", batch_key="Site")
model = SemanticSCVI.load(str(GEOM_CACHE), adata=adata)
print("  loaded.", flush=True)

# Sanity: recomputed latent must match the stored X_semantic_geom (row alignment).
z_new = np.asarray(model.get_latent_representation(adata))
z_old = np.asarray(adata.obsm["X_semantic_geom"])
max_abs = float(np.abs(z_new - z_old).max())
corr = float(np.corrcoef(z_new.ravel(), z_old.ravel())[0, 1])
print(f"  latent check: max|Δ|={max_abs:.4g}  corr={corr:.6f}", flush=True)
assert corr > 0.999 and max_abs < 1e-2, (
    f"latent mismatch (corr={corr}, max|Δ|={max_abs}) — adata/model misaligned; aborting."
)

# Patch transform_batch (mirrors Cell 11) for get_normalized_expression.
def _fixed_get_transform_batch_gen_kwargs(self, batch):
    return {"transform_batch": batch}

model._get_transform_batch_gen_kwargs = types.MethodType(
    _fixed_get_transform_batch_gen_kwargs, model
)

print("Computing geom denoised expression (library_size=10_000) ...", flush=True)
denoised_geom = np.asarray(
    model.get_normalized_expression(adata, library_size=10_000)
)
adata.layers["denoised_gamma_geom"] = denoised_geom
print(f"  layers['denoised_gamma_geom'] = {denoised_geom.shape}", flush=True)

try:
    px_r_geom = model.module.px_r.detach().exp().cpu().numpy()
    adata.uns["model_dispersions_geom"] = px_r_geom
    print(f"  uns['model_dispersions_geom'] = {px_r_geom.shape}", flush=True)
except AttributeError:
    print("  WARN: could not extract px_r dispersions from geom model", flush=True)

W_geom = model.get_loadings()
W_geom = W_geom.reindex(adata.var_names)
adata.varm["W_semantic_geom"] = W_geom.values
adata.uns["W_semantic_geom_columns"] = list(W_geom.columns)
print(f"  varm['W_semantic_geom'] = {adata.varm['W_semantic_geom'].shape}", flush=True)

tmp = H5AD.with_suffix(".h5ad.tmp")
print(f"Writing {tmp} ...", flush=True)
adata.write_h5ad(tmp)
os.replace(tmp, H5AD)
print(f"DONE: atomically replaced {H5AD}", flush=True)
print(f"  obsm : {list(adata.obsm)}", flush=True)
print(f"  varm : {list(adata.varm)}", flush=True)
print(f"  layers: {list(adata.layers)}", flush=True)
print(f"  uns  : {list(adata.uns)}", flush=True)
