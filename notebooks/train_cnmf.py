"""Consensus NMF (cNMF, Kotliar et al. 2019) wrapper for the factor benchmark.

Drop-in replacement for the plain sklearn-NMF baseline. Exposes the same interface
as ``NMFWrapper`` / ``SCHPFWrapper`` (``get_latent_representation`` -> cells x K,
``get_loadings`` -> genes x K DataFrame) so the rest of the pipeline is unchanged.

cNMF runs many NMF replicates, clusters their components into consensus gene-expression
programs, filters outlier replicates, and refits usage. We force cNMF onto the same gene
set the other models use (``genes_file``) so the comparison is on identical genes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


class CNMFWrapper:
    """Holds only arrays (no live cNMF object / file handles) so it pickles cleanly."""

    def __init__(self, model, W, H, feature_names):
        self.model = model            # kept None; cNMF state lives on disk
        self.W = np.asarray(W)        # cells x n_factors (usage)
        self.H = np.asarray(H)        # n_factors x genes (spectra)
        self.feature_names = list(feature_names)

    def get_latent_representation(self, adata=None):
        return self.W

    def get_loadings(self):
        return pd.DataFrame(
            self.H.T,
            index=self.feature_names,
            columns=[f"Factor_{i}" for i in range(self.H.shape[0])],
        )


def train_cnmf_model(
    adata,
    n_factors=10,
    output_dir=".",
    name="cnmf_k10",
    n_iter=20,
    density_threshold=0.5,
    num_highvar_genes=None,
    seed=42,
    beta_loss="frobenius",
    init="random",
    loadings="score",
):
    """Run cNMF end-to-end for a single K and return a ``CNMFWrapper``.

    Parameters
    ----------
    adata : AnnData
        Raw counts in ``.X``; ``var_names`` are the gene IDs the rest of the pipeline
        uses. cNMF normalizes (TPM + variance scaling) internally.
    n_factors : int
        Number of consensus programs (K).
    output_dir : path
        cNMF working directory (intermediates are written under ``output_dir/name``).
    n_iter : int
        Number of NMF replicates pooled into the consensus.
    density_threshold : float
        Consensus outlier-spectra filter (passed to ``consensus`` and ``load_results``).
    num_highvar_genes : int or None
        If None, force cNMF onto ALL ``adata.var_names`` via ``genes_file`` (fair
        comparison with the other models). If an int, let cNMF pick that many HVGs.
    loadings : {"score", "tpm"}
        Which spectra become the loadings: ``gene_spectra_score`` (z-scored, canonical
        for marker genes) or ``gene_spectra_tpm`` (non-negative TPM units).
    """
    from cnmf import cNMF

    k = int(n_factors)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Training cNMF (K={k}, n_iter={n_iter}, density_threshold={density_threshold}) ---")

    # 1. Build a non-negative counts matrix and DROP zero-variance genes.
    # cNMF's refit_usage=True step divides spectra/counts by each gene's TPM std; a gene
    # with zero variance (e.g. all-zero in this cell subset) -> divide-by-0 -> NaN usages.
    X = adata.X
    Xc = (X.tocsr() if sp.issparse(X) else sp.csr_matrix(np.asarray(X))).astype(np.float64)
    Xc.data = np.maximum(Xc.data, 0)
    n = Xc.shape[0]
    mean = np.asarray(Xc.sum(0)).ravel() / n
    var = np.asarray(Xc.multiply(Xc).sum(0)).ravel() / n - mean ** 2
    keep = var > 0
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  dropping {n_drop} zero-variance genes before cNMF (loadings 0-filled later)")
    kept_names = adata.var_names[keep].astype(str)
    counts = sc.AnnData(
        X=Xc[:, keep],
        obs=pd.DataFrame(index=adata.obs_names.astype(str)),
        var=pd.DataFrame(index=kept_names),
    )
    counts_fn = output_dir / f"{name}_counts.h5ad"
    counts.write_h5ad(counts_fn)

    # 2. Force the same gene space as the other models unless an HVG count is requested.
    genes_file = None
    prepare_kwargs = {}
    if num_highvar_genes is None:
        genes_file = output_dir / f"{name}_genes.txt"
        genes_file.write_text("\n".join(kept_names) + "\n")
        prepare_kwargs["genes_file"] = str(genes_file)
    else:
        prepare_kwargs["num_highvar_genes"] = int(num_highvar_genes)

    # 3. prepare -> factorize -> combine -> consensus.
    obj = cNMF(output_dir=str(output_dir), name=name)
    obj.prepare(
        str(counts_fn), components=[k], n_iter=n_iter, seed=seed,
        beta_loss=beta_loss, init=init, **prepare_kwargs,
    )
    obj.factorize(worker_i=0, total_workers=1)
    obj.combine(components=[k])
    obj.consensus(
        k, density_threshold=density_threshold,
        show_clustering=False, close_clustergram_fig=True,
    )

    # 4. Load consensus results: usage (cells x K), spectra (genes x K after internal .T).
    usage, scores, tpm, _ = obj.load_results(
        K=k, density_threshold=density_threshold, norm_usage=True,
    )
    if np.isnan(usage.to_numpy()).any():
        raise ValueError(
            "cNMF returned NaN usages — usually a zero-variance gene reaching the "
            "refit_usage step. Genes are filtered upstream; if this fires, check "
            f"the counts written to {counts_fn}."
        )
    spectra = scores if loadings == "score" else tpm

    # 5. Align to the pipeline's cell/gene order (cNMF may reorder/drop). Coerce both
    # indices to str FIRST: cNMF parses purely-numeric cell/gene names (e.g. CELLxGENE
    # "0","1",... barcodes) back as int64, so a string reindex would match nothing and
    # silently yield an all-NaN usage matrix.
    spectra.index = spectra.index.astype(str)
    usage.index = usage.index.astype(str)
    spectra = spectra.reindex(adata.var_names.astype(str)).fillna(0.0)
    usage = usage.reindex(adata.obs_names.astype(str))
    if np.isnan(usage.to_numpy()).any():
        raise ValueError(
            "cNMF usage has NaN after aligning to adata.obs_names — cell-name mismatch "
            "between cNMF output and the pipeline adata (check obs_names dtypes)."
        )

    # 6. Wrap: W = usage (cells x K), H = spectra.T (K x genes).
    return CNMFWrapper(
        model=None,
        W=usage.to_numpy(dtype=float),
        H=spectra.to_numpy(dtype=float).T,
        feature_names=adata.var_names,
    )
