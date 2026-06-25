"""Five-modality factor-model comparison (companion to ``four_way_benchmark_*``).

Self-contained metrics + plots + HTML report for comparing factor models
(SemanticSCVI, LDVAE+, scHPF, cNMF, EXPIMAP) along five interpretable axes:

  1. gene sparsity        — few important genes per factor (good) vs all genes
  2. program independence — pairwise factor collinearity (some allowed, not too much)
  3. stability            — retrain on fewer cells -> matched factors recur
  4. MSigDB best-program  — fraction of factors whose best library program is
                            significant (BH q) AND large-effect (fold enrichment)
  5. factor<->metadata     — factors track biological metadata, not batch

Reuses ``benchmarking._split_hallmark`` for the H/C2 library split and the
training wrappers in ``benchmark_helpers`` / ``train_schpf`` for stability
retrains. Every adapter exposes ``.get_loadings()`` (genes x factors DataFrame
indexed by ``adata.var_names``) and ``.get_latent_representation()`` (cells x K).
"""
from __future__ import annotations

import base64
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

LIB_ORDER = ("H", "C2_immune", "C7")
LIB_TITLES = {"H": "H (Hallmark)", "C2_immune": "C2 (immune)", "C7": "C7 (cell-type sig.)"}


# --------------------------------------------------------------------------- #
# Figure export — hybrid SVG (editable text + rasterized data points)
# --------------------------------------------------------------------------- #
def _rasterize(*axes):
    """Rasterize the data-point artists (scatter/strip collections, heatmaps) on each ax,
    so a hybrid SVG keeps axes/text as vector paths but pixelates dense point clouds."""
    for ax in axes:
        for coll in ax.collections:
            coll.set_rasterized(True)


def _save(fig, png_path, svg_path=None, dpi_png=150, dpi_svg=300):
    """Save PNG (for the HTML report) and, if requested, a paper-ready hybrid SVG:
    text stays editable (``svg.fonttype='none'``), rasterized artists render at high dpi."""
    fig.savefig(png_path, dpi=dpi_png, bbox_inches="tight")
    if svg_path is not None:
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig.savefig(svg_path, format="svg", dpi=dpi_svg, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Shared extraction
# --------------------------------------------------------------------------- #
def get_loadings_dict(models):
    """{name: genes x factors DataFrame} from each adapter's ``.get_loadings()``."""
    return {name: m.get_loadings() for name, m in models.items()}


def get_latent_dict(models):
    """{name: cells x K ndarray} from each adapter's ``.get_latent_representation()``."""
    return {name: np.asarray(m.get_latent_representation()) for name, m in models.items()}


def symbol_map(adata, gene_mapping):
    """Return (id->symbol dict, universe set of symbols) from ``adata.var``.

    ``gene_mapping = (source_col, target_col)`` e.g. ("feature_id", "feature_name").
    Loadings are indexed by ``adata.var_names``; map those to symbols.
    """
    src, tgt = gene_mapping
    var = adata.var
    if src in var and (var.index == var[src]).all():
        idmap = dict(zip(var.index, var[tgt]))
    elif src in var:
        idmap = dict(zip(var[src], var[tgt]))
    else:
        idmap = dict(zip(var.index, var[tgt])) if tgt in var else {}
    universe = set(idmap.values()) if idmap else set(var.index)
    return idmap, universe


# --------------------------------------------------------------------------- #
# Modality 1 — gene sparsity
# --------------------------------------------------------------------------- #
def _gini(w):
    x = np.sort(np.abs(np.asarray(w, float)))
    n = x.size
    s = x.sum()
    if n == 0 or s == 0:
        return np.nan
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * x) / (n * s))


def _eff_genes(w):
    """Inverse participation ratio of normalized |loadings| (lower = sparser)."""
    x = np.abs(np.asarray(w, float))
    s = x.sum()
    if s == 0:
        return np.nan
    p = x / s
    return float(1.0 / np.sum(p ** 2))


def _genes_to_mass(w, frac):
    x = np.sort(np.abs(np.asarray(w, float)))[::-1]
    s = x.sum()
    if s == 0:
        return np.nan
    c = np.cumsum(x) / s
    return int(np.searchsorted(c, frac) + 1)


def compute_sparsity(loadings_dict, mass_frac=0.9, eps=1e-9):
    """Per-factor sparsity. Higher gini / lower eff_genes / lower genes_to_mass = sparser.

    Two extra columns are robust to *structural* zeros (e.g. masked decoders like EXPIMAP,
    whose loadings are forced to 0 outside each program's gene set — which inflates the
    global Gini without any learned concentration):

    - ``frac_used``    = fraction of genes with |loading| > eps (how many genes a factor
                         actually touches; a masked factor ≈ program_size / n_genes).
    - ``gini_support`` = Gini computed ONLY over the nonzero support (|loading| > eps) —
                         "given the genes you use, do you concentrate?". A masked factor
                         that loads its whole pathway ~uniformly scores LOW here, while a
                         dense model's support ≈ all genes so it ≈ the global Gini.
    """
    rows = []
    for model, L in loadings_dict.items():
        for f in L.columns:
            w = L[f].to_numpy()
            sup = w[np.abs(w) > eps]
            rows.append(dict(
                Model=model, Factor=str(f),
                gini=_gini(w), gini_support=_gini(sup),
                frac_used=float(np.mean(np.abs(w) > eps)),
                eff_genes=_eff_genes(w),
                genes_to_mass=_genes_to_mass(w, mass_frac), n_genes=len(w),
            ))
    return pd.DataFrame(rows)


def plot_sparsity(df, out_path, model_order, mass_frac=0.9, svg_path=None):
    metrics = [
        ("gini_support", "Gini within used genes  (↑ concentrates)"),
        ("eff_genes", "effective #genes  (↓ sparser)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(9.5, 4.2))
    for ax, (col, title) in zip(axes, metrics):
        sns.boxplot(data=df, x="Model", y=col, order=model_order, ax=ax,
                    hue="Model", legend=False, palette="tab10", showfliers=False)
        sns.stripplot(data=df, x="Model", y=col, order=model_order, ax=ax,
                      color="0.2", size=3, alpha=0.5)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("Gene sparsity per factor — good models concentrate on few genes "
                 "(global Gini omitted: inflated by structural zeros in masked models)",
                 y=1.03, fontsize=11)
    fig.tight_layout()
    _rasterize(*axes)
    _save(fig, out_path, svg_path)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Modality 2 — program independence / collinearity
# --------------------------------------------------------------------------- #
def compute_collinearity(latent_dict, threshold=0.5):
    """Pairwise |Pearson| between latent factors (cell program usage)."""
    summary, mats = [], {}
    for model, Z in latent_dict.items():
        Z = np.asarray(Z, float)
        C = np.corrcoef(Z, rowvar=False)
        mats[model] = C
        k = C.shape[0]
        iu = np.triu_indices(k, 1)
        offs = np.abs(C[iu])
        summary.append(dict(
            Model=model,
            mean_abs_offdiag=float(np.nanmean(offs)),
            max_abs_offdiag=float(np.nanmax(offs)),
            frac_redundant=float(np.nanmean(offs > threshold)),
            n_pairs=int(len(offs)),
        ))
    return pd.DataFrame(summary), mats


def plot_collinearity(mats, summary, out_path, model_order, threshold=0.5, svg_path=None):
    n = len(model_order)
    fig = plt.figure(figsize=(3.2 * n, 6.4))
    gs = fig.add_gridspec(2, n, height_ratios=[1.1, 1.0])
    for ci, model in enumerate(model_order):
        ax = fig.add_subplot(gs[0, ci])
        C = np.abs(mats[model])
        sns.heatmap(C, vmin=0, vmax=1, cmap="rocket_r", square=True, cbar=(ci == n - 1),
                    xticklabels=False, yticklabels=False, ax=ax)
        r = summary.loc[summary.Model == model].iloc[0]
        ax.set_title(f"{model}\nmean|r|={r.mean_abs_offdiag:.2f} max={r.max_abs_offdiag:.2f}",
                     fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    long = []
    for model in model_order:
        C = mats[model]
        iu = np.triu_indices(C.shape[0], 1)
        for v in np.abs(C[iu]):
            long.append(dict(Model=model, abs_r=v))
    long = pd.DataFrame(long)
    sns.boxplot(data=long, x="Model", y="abs_r", order=model_order, ax=ax,
                hue="Model", legend=False, palette="tab10", showfliers=False)
    sns.stripplot(data=long, x="Model", y="abs_r", order=model_order, ax=ax,
                  color="0.2", size=3, alpha=0.5)
    ax.axhline(threshold, ls="--", lw=1, color="#b00")
    ax.set_ylabel("|off-diagonal Pearson r|")
    ax.set_xlabel("")
    ax.set_title("Program collinearity — near-0 = over-orthogonal, moderate = acceptable, "
                 f"red line = redundancy threshold ({threshold})", fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    fig.tight_layout()
    _rasterize(*fig.axes)
    _save(fig, out_path, svg_path)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Modality 3 — stability under cell subsampling
# --------------------------------------------------------------------------- #
def match_factors(L_ref, L_sub):
    """Hungarian-matched mean |corr| between reference and subsample loadings.

    Matches on the shared gene index; one-to-one factor assignment maximizing
    absolute Pearson correlation. Returns (mean_matched_abs_corr, corr_matrix).
    """
    common = L_ref.index.intersection(L_sub.index)
    A = L_ref.loc[common].to_numpy(float)
    B = L_sub.loc[common].to_numpy(float)
    A = A - A.mean(0)
    B = B - B.mean(0)
    A /= (np.linalg.norm(A, axis=0) + 1e-12)
    B /= (np.linalg.norm(B, axis=0) + 1e-12)
    C = np.abs(A.T @ B)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean()), C


def _canonical_corrs(A, B):
    """Canonical correlations between the column-spaces of ``A`` and ``B``.

    ``A``/``B`` share rows (genes or cells) and have K columns. Columns are
    mean-centered, orthonormalized via economy QR, then the singular values of
    ``Qa.T @ Qb`` give the cosines of the principal angles (= canonical
    correlations) in ``[0, 1]``. Invariant to any rotation/permutation/sign of the
    K columns. Returns ``(mean, vector)``; ``(nan, [])`` if either is rank-empty.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    # Drop rows non-finite in either matrix (e.g. a model that failed to converge on a
    # subsample). If a whole factor column is NaN the mask empties -> nan, not a crash.
    keep = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[keep], B[keep]
    A = A - A.mean(0, keepdims=True)
    B = B - B.mean(0, keepdims=True)
    if A.shape[0] < 2 or A.shape[1] == 0 or B.shape[1] == 0:
        return np.nan, np.array([])
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        return np.nan, np.array([])
    try:
        Qa, _ = np.linalg.qr(A)
        Qb, _ = np.linalg.qr(B)
        M = Qa.T @ Qb
        if not np.isfinite(M).all():
            return np.nan, np.array([])
        s = np.linalg.svd(M, compute_uv=False)
    except np.linalg.LinAlgError:
        return np.nan, np.array([])
    s = np.clip(s, 0.0, 1.0)
    return float(s.mean()), s


def subspace_stability(L_ref, L_sub):
    """Rotation-invariant stability of gene-loading subspaces (full vs subsample).

    Mean canonical correlation between the K-dim column-spaces of the two
    ``genes x K`` loading matrices, on their shared genes.
    """
    common = L_ref.index.intersection(L_sub.index)
    if len(common) < 2:
        return np.nan
    return _canonical_corrs(L_ref.loc[common].to_numpy(float),
                            L_sub.loc[common].to_numpy(float))[0]


def embedding_stability(Z_ref_df, Z_sub_df):
    """Rotation-invariant stability of cell embeddings on the shared cells.

    Mean canonical correlation (CCA) between the two ``cells x K`` latent matrices
    restricted to the cells present in both (subsample cells ⊆ full cells).
    """
    common = Z_ref_df.index.intersection(Z_sub_df.index)
    if len(common) < 2:
        return np.nan
    return _canonical_corrs(Z_ref_df.loc[common].to_numpy(float),
                            Z_sub_df.loc[common].to_numpy(float))[0]


_STABILITY_TITLES = {
    "subspace_corr": "Loading-subspace CCA (rotation-inv., ↑ better)",
    "embed_corr": "Cell-embedding CCA (rotation-inv., ↑ better)",
    "matched_corr": "Per-axis Hungarian |corr| (legacy, ↑ better)",
}


def plot_stability(df, out_path, model_order,
                   value_cols=("embed_corr", "matched_corr"), svg_path=None):
    """One error-bar panel per metric in ``value_cols`` present in ``df``.

    ``df`` columns: Model, fraction, seed, + any of the stability metric columns.
    """
    cols = [c for c in value_cols if c in df.columns]
    if not cols:
        cols = ["matched_corr"]
    palette = dict(zip(model_order, sns.color_palette("tab10", len(model_order))))
    fig, axes = plt.subplots(1, len(cols), figsize=(5.0 * len(cols), 4.2),
                             squeeze=False)
    for ax, col in zip(axes[0], cols):
        agg = df.groupby(["Model", "fraction"])[col].agg(["mean", "std"]).reset_index()
        for model in model_order:
            g = agg[agg.Model == model].sort_values("fraction")
            if g.empty:
                continue
            ax.errorbar(g["fraction"], g["mean"], yerr=g["std"].fillna(0),
                        marker="o", capsize=3, color=palette[model], label=model)
        ax.set_xlabel("subsample fraction of cells")
        ax.set_ylabel(col)
        ax.set_ylim(0, 1.02)
        ax.set_title(_STABILITY_TITLES.get(col, col), fontsize=9)
    axes[0][-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Stability — factor/representation recovery under fewer cells", y=1.02)
    fig.tight_layout()
    _save(fig, out_path, svg_path)
    plt.show()
    plt.close(fig)


def train_all_models(adata, maps, cfg, cache_dir, seed=0, force=False):
    """Train/load all models on ``adata``; return ``{name: adapter}``.

    ``maps`` = {"geneformer": tensor, "genept": tensor} aligned to ``adata.var``
    (gene order; unaffected by cell subsampling). ``cfg`` carries hyperparameters
    (see the notebook config cell). ``cache_dir`` is the per-run cache root.
    """
    import scvi

    from benchmark_helpers import (
        _ExpimapAdapter, _ScviAdapter,
        build_expimap_mask_gmt, train_or_load_expimap,
        train_or_load_nonneg_ldvae, train_or_load_pickle,
        train_or_load_semantic_scvi,
    )
    from train_cnmf import train_cnmf_model
    from train_schpf import train_schpf_model

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scvi.settings.seed = seed
    bk = cfg["BATCH_KEY"]
    models = {}

    a = adata.copy()
    geom = train_or_load_semantic_scvi(
        a, maps["geneformer"], cache_dir=cache_dir / "semantic_geom",
        force_train=force, max_epochs=cfg["SEM_GEOM_MAX_EPOCHS"],
        warmup_epochs=cfg["SEM_GEOM_WARMUP"], n_epochs_kl_warmup=cfg["SEM_GEOM_KL_WARMUP"],
        batch_key=bk, **cfg["SEM_GEOM_KWARGS"],
    )
    models["semantic_geom"] = _ScviAdapter(geom, a)

    a = adata.copy()
    genept = train_or_load_semantic_scvi(
        a, maps["genept"], cache_dir=cache_dir / "semantic_genept",
        force_train=force, max_epochs=cfg["SEM_GENEPT_MAX_EPOCHS"],
        warmup_epochs=cfg["SEM_GENEPT_WARMUP"], n_epochs_kl_warmup=cfg["SEM_GENEPT_KL_WARMUP"],
        batch_key=bk, **cfg["SEM_GENEPT_KWARGS"],
    )
    models["semantic_genept"] = _ScviAdapter(genept, a)

    a = adata.copy()
    ldvae = train_or_load_nonneg_ldvae(
        a, cache_dir=cache_dir / "ldvae_nn", force_train=force,
        max_epochs=cfg["LDVAE_MAX_EPOCHS"], batch_key=bk, **cfg["LDVAE_KWARGS"],
    )
    models["ldvae_nn"] = _ScviAdapter(ldvae, a)

    models["schpf_k10"] = train_or_load_pickle(
        "scHPF", lambda: train_schpf_model(adata, n_factors=cfg["N_FACTORS"]),
        cache_path=cache_dir / "schpf_k10.pkl", force_train=force,
    )

    models["cnmf_k10"] = train_or_load_pickle(
        "cNMF",
        lambda: train_cnmf_model(
            adata, n_factors=cfg["N_FACTORS"], output_dir=cache_dir / "cnmf_k10_run",
            name="cnmf_k10", n_iter=cfg["CNMF_N_ITER"],
            density_threshold=cfg["CNMF_DENSITY_THRESHOLD"],
            num_highvar_genes=cfg.get("CNMF_NUM_HIGHVAR_GENES"), seed=seed,
            beta_loss=cfg["CNMF_BETA_LOSS"], init=cfg["CNMF_INIT"],
            loadings=cfg["CNMF_LOADINGS"],
        ),
        cache_path=cache_dir / "cnmf_k10.pkl", force_train=force,
    )

    mask_gmt = Path(cfg["EXPIMAP_MASK_GMT"])
    if not mask_gmt.exists():
        build_expimap_mask_gmt(adata, cfg["EXPIMAP_SOURCE_GMT"],
                               cfg["EXPIMAP_MASK_TERMS"], mask_gmt)
    a = adata.copy()
    expimap = train_or_load_expimap(
        a, gmt_path=mask_gmt, cache_dir=cache_dir / "expimap_k10", force_train=force,
        n_epochs=cfg["EXPIMAP_N_EPOCHS"], alpha_kl=cfg["EXPIMAP_ALPHA_KL"],
        alpha=cfg["EXPIMAP_ALPHA"], alpha_epoch_anneal=cfg["EXPIMAP_ALPHA_EPOCH_ANNEAL"],
        **cfg["EXPIMAP_KWARGS"],
    )
    models["expimap_k10"] = _ExpimapAdapter(expimap, a)
    return models


# --------------------------------------------------------------------------- #
# Modality 4 — MSigDB best-program-per-factor
# --------------------------------------------------------------------------- #
def _load_gmt(path):
    gene_sets = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = set(g for g in parts[2:] if g)
    return gene_sets


def load_libraries(lib1_gmt, lib2_gmt, exclude_terms=None):
    """{"H":..., "C2_immune":..., "C7":...} (H/C2 split from lib1; C7 = lib2).

    ``exclude_terms`` drops gene sets by name from every library (held-out evaluation).
    Use it to remove an annotation-supervised model's mask terms (e.g. EXPIMAP's HALLMARK
    programs) so its modality-4 enrichment isn't circular — applied uniformly to all models.
    """
    from benchmarking import _split_hallmark
    libs = {}
    if lib1_gmt:
        h, c2 = _split_hallmark(_load_gmt(lib1_gmt))
        libs["H"], libs["C2_immune"] = h, c2
    if lib2_gmt:
        libs["C7"] = _load_gmt(lib2_gmt)
    if exclude_terms:
        drop = set(exclude_terms)
        for label in libs:
            before = len(libs[label])
            libs[label] = {k: v for k, v in libs[label].items() if k not in drop}
            removed = before - len(libs[label])
            if removed:
                print(f"  held-out: removed {removed} term(s) from {label} "
                      f"({before} -> {len(libs[label])})")
    return libs


def _hg_rows(p_genes, library, universe):
    """ER + one-sided hypergeometric p for each gene set (mirrors benchmarking._hg_enrichment_rows)."""
    f = set(p_genes) & universe
    G = len(universe)
    n_f = len(f)
    if n_f == 0 or G == 0:
        return []
    out = []
    for s_name, s_genes in library.items():
        c = s_genes & universe
        n_c = len(c)
        if n_c == 0:
            continue
        overlap = len(f & c)
        er = (overlap / n_c) / (n_f / G)
        pval = float(hypergeom.sf(overlap - 1, G, n_c, n_f))
        out.append(dict(gene_set=s_name, overlap=overlap, set_size=n_c,
                        factor_size=n_f, G=G, ER=er, pvalue=pval))
    return out


def best_program_proportions(loadings_dict, libraries, adata, gene_mapping,
                             n_top=30, q_thresh=0.05, er_thresh=2.0, best_by="pval"):
    """Per (model, library): best program per factor + proportion sig / sig-strong.

    For each factor, top-``n_top`` genes are tested (one-sided HG) vs every set in
    the library; **BH (fdr_bh) correction is applied across all factor×geneset tests
    within each (model, library) block**. Each factor's "best fitting" program is the
    smallest-p set (``best_by="pval"``) or the largest-ER set (``best_by="ER"``).
    A factor is a hit if its best program has q < ``q_thresh`` AND ER > ``er_thresh``.
    """
    idmap, universe = symbol_map(adata, gene_mapping)
    detail = []
    for model, L in loadings_dict.items():
        for lib_label, library in libraries.items():
            flat, ranges = [], []
            for f in L.columns:
                top = L[f].nlargest(n_top).index.tolist()
                genes = set(idmap.get(g, g) for g in top)
                recs = _hg_rows(genes, library, universe)
                start = len(flat)
                flat.extend(r["pvalue"] for r in recs)
                ranges.append((f, recs, start, len(flat)))
            if not flat:
                continue
            _, q_all, _, _ = multipletests(flat, method="fdr_bh")
            for f, recs, s, e in ranges:
                if not recs:
                    continue
                qs = q_all[s:e]
                if best_by == "ER":
                    bi = max(range(len(recs)), key=lambda i: recs[i]["ER"])
                else:
                    bi = min(range(len(recs)), key=lambda i: recs[i]["pvalue"])
                best = recs[bi]
                q = float(qs[bi])
                detail.append(dict(
                    Model=model, Library=lib_label, Factor=str(f),
                    gene_set=best["gene_set"], ER=best["ER"], pvalue=best["pvalue"],
                    qvalue=q, overlap=best["overlap"], set_size=best["set_size"],
                    factor_size=best["factor_size"],
                    significant=bool(q < q_thresh),
                    sig_strong=bool(q < q_thresh and best["ER"] > er_thresh),
                ))
    detail = pd.DataFrame(detail)
    # ``prop_sig_strong`` saturates at 1.0 for any coherent model (a best-of-library match
    # almost always exists), so it can't discriminate the top. Add graded columns:
    #   median_ER     — median fold-enrichment of each factor's best match (effect size).
    #   median_ER_sig — same, restricted to factors whose best match is BH-significant.
    # These stay graded (no ceiling), so SemanticSCVI vs cNMF vs EXPIMAP separate cleanly.
    summary = (detail.groupby(["Library", "Model"])
               .agg(n_factors=("Factor", "size"),
                    prop_sig=("significant", "mean"),
                    prop_sig_strong=("sig_strong", "mean"),
                    median_ER=("ER", "median")).reset_index())
    sig_med = (detail[detail["significant"]].groupby(["Library", "Model"])["ER"]
               .median().rename("median_ER_sig").reset_index())
    summary = summary.merge(sig_med, on=["Library", "Model"], how="left")
    return detail, summary


def plot_best_program(detail, out_path, model_order, q_thresh, er_thresh, lib_order=None,
                      svg_path=None):
    """4-panel modality-4 view (grouped bars per model × library throughout):

      (1) median best-match fold-enrichment (effect size),
      (2) median best-match −log10(BH q) (significance),
      (3) % factors with best program q<q_thresh AND ER>er_thresh,
      (4) % factors with best program q<q_thresh only (p-cutoff).
    """
    d = detail.copy()
    libs = ([l for l in LIB_ORDER if l in d["Library"].unique()]
            if lib_order is None else list(lib_order))
    d["nlq"] = -np.log10(d["qvalue"].to_numpy(dtype=float) + 1e-300)
    pal = sns.color_palette("Set2", len(libs))
    x = np.arange(len(model_order))
    w = 0.8 / len(libs)

    def _bars(ax, piv, ylabel, title, hline=None, ymax=None):
        piv = piv.reindex(model_order)[libs]
        for i, lib in enumerate(libs):
            ax.bar(x + i * w, piv[lib].to_numpy(dtype=float), width=w,
                   color=pal[i], label=LIB_TITLES.get(lib, lib))
        if hline is not None:
            ax.axhline(hline, ls="--", lw=1, color="#444")
        ax.set_xticks(x + 0.4 - w / 2)
        ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
        if ymax is not None:
            ax.set_ylim(0, ymax)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, title="library")

    er_med = d.groupby(["Model", "Library"])["ER"].median().unstack("Library")
    nlq_med = d.groupby(["Model", "Library"])["nlq"].median().unstack("Library")
    sig = d["qvalue"].to_numpy(dtype=float) < q_thresh
    strong = d["ER"].to_numpy(dtype=float) > er_thresh
    p_ss = d.assign(_m=sig & strong).groupby(["Model", "Library"])["_m"].mean().mul(100).unstack("Library")
    p_s = d.assign(_m=sig).groupby(["Model", "Library"])["_m"].mean().mul(100).unstack("Library")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    _bars(axes[0, 0], er_med, "median fold-enrichment", "(1) Enrichment effect size",
          hline=er_thresh)
    _bars(axes[0, 1], nlq_med, "median −log10(BH q)", "(2) Adjusted p-value (significance)",
          hline=-np.log10(q_thresh))
    _bars(axes[1, 0], p_ss, "% factors", f"(3) % factors: q<{q_thresh:g} AND ER>{er_thresh:g}",
          ymax=100)
    _bars(axes[1, 1], p_s, "% factors", f"(4) % factors: q<{q_thresh:g}  (p-cutoff only)",
          ymax=100)
    fig.suptitle("MSigDB best-program per factor — effect size, significance, and cutoffs",
                 y=1.01, fontsize=12)
    fig.tight_layout()
    _save(fig, out_path, svg_path)
    plt.show()
    plt.close(fig)


def plot_best_program_effect_size(detail, out_path, model_order, er_thresh, lib_order=None,
                                  svg_path=None):
    """Single-panel modality-4 view: median best-match fold-enrichment (effect size only),
    grouped bars per model × library. Companion to the full 4-panel ``plot_best_program``."""
    d = detail.copy()
    libs = ([l for l in LIB_ORDER if l in d["Library"].unique()]
            if lib_order is None else list(lib_order))
    pal = sns.color_palette("Set2", len(libs))
    x = np.arange(len(model_order))
    w = 0.8 / len(libs)
    er_med = d.groupby(["Model", "Library"])["ER"].median().unstack("Library").reindex(model_order)[libs]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, lib in enumerate(libs):
        ax.bar(x + i * w, er_med[lib].to_numpy(dtype=float), width=w,
               color=pal[i], label=LIB_TITLES.get(lib, lib))
    ax.axhline(er_thresh, ls="--", lw=1, color="#444")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("median fold-enrichment")
    ax.set_title("MSigDB best-program enrichment effect size", fontsize=11)
    ax.legend(fontsize=7, title="library")
    fig.tight_layout()
    _save(fig, out_path, svg_path)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Modality 5 — factor <-> metadata association
# --------------------------------------------------------------------------- #
def _eta2(values, labels):
    """Correlation ratio η² (categorical->continuous), SS_between/SS_total."""
    d = pd.DataFrame({"v": np.asarray(values, float), "g": np.asarray(labels)}).dropna()
    if d.empty:
        return np.nan
    grand = d["v"].mean()
    ss_tot = float(((d["v"] - grand) ** 2).sum())
    if ss_tot == 0:
        return np.nan
    ss_between = float(d.groupby("g")["v"].apply(
        lambda x: len(x) * (x.mean() - grand) ** 2).sum())
    return ss_between / ss_tot


def compute_metadata_assoc(latent_dict, obs, bio_vars, batch_vars):
    """Per (model, var): max η² over factors. Good model = high bio, low batch."""
    rows = []
    groups = {"bio": bio_vars, "batch": batch_vars}
    for model, Z in latent_dict.items():
        Z = np.asarray(Z, float)
        for grp, vars_ in groups.items():
            for v in vars_:
                labels = obs[v].to_numpy()
                etas = [_eta2(Z[:, k], labels) for k in range(Z.shape[1])]
                valid = [e for e in etas if e == e]
                rows.append(dict(
                    Model=model, group=grp, var=v,
                    max_eta2=float(np.nanmax(etas)) if valid else np.nan,
                    mean_top3=float(np.mean(sorted(valid, reverse=True)[:3])) if valid else np.nan,
                ))
    detail = pd.DataFrame(rows)
    summary = (detail.groupby("Model").apply(lambda g: pd.Series({
        "bio_mean": g.loc[g.group == "bio", "max_eta2"].mean(),
        "batch_mean": g.loc[g.group == "batch", "max_eta2"].mean(),
    }), include_groups=False).reset_index())
    summary["bio_minus_batch"] = summary["bio_mean"] - summary["batch_mean"]
    return detail, summary


def plot_metadata(detail, out_path, model_order, svg_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    palette = dict(zip(model_order, sns.color_palette("tab10", len(model_order))))
    for ax, grp, title in [(axes[0], "bio", "Biological vars (↑ better)"),
                           (axes[1], "batch", "Technical/batch vars (↓ better)")]:
        sub = detail[detail.group == grp]
        sns.barplot(data=sub, x="var", y="max_eta2", hue="Model", hue_order=model_order,
                    palette=palette, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("max η² across factors")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend_.remove() if ax.legend_ else None
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=7, title="model")
    fig.suptitle("Factor↔metadata association — best factor per variable", y=1.02)
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    _save(fig, out_path, svg_path)
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# HTML report
# --------------------------------------------------------------------------- #
def _embed(path, alt):
    path = Path(path)
    if not path.exists():
        return f"<p><em>(missing: {alt})</em></p>"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<div class="fig"><img src="data:image/png;base64,{b64}" alt="{alt}"></div>'


def _table(df):
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else f"{v:.3f}"
        return str(v)
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(r[c])}</td>" for c in df.columns) + "</tr>"
                   for _, r in df.iterrows())
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def build_modalities_report(out_dir, model_names, adata_shape, knobs, tables, notes=""):
    """Render ``out_dir/report_modalities.html`` from the five modality PNGs + tables.

    ``knobs`` = dict of config knobs; ``tables`` = dict with optional keys
    ``leaderboard``, ``sparsity``, ``collinearity``, ``best_program``,
    ``metadata``, ``stability`` (each a DataFrame).
    """
    out_dir = Path(out_dir)
    knob_tbl = _table(pd.DataFrame({"knob": list(knobs), "value": [str(v) for v in knobs.values()]}))
    sections = []

    def sec(title, png, alt, key, blurb):
        t = tables.get(key)
        tbl = _table(t) if isinstance(t, pd.DataFrame) and not t.empty else ""
        return (f'<section><h2>{title}</h2><p class="blurb">{blurb}</p>'
                f'{_embed(out_dir / png, alt)}{tbl}</section>')

    if "leaderboard" in tables:
        sections.append(f'<section><h2>Leaderboard</h2>{_table(tables["leaderboard"])}</section>')
    sections.append(sec("1 · Gene sparsity", "mod1_sparsity.png", "sparsity", "sparsity",
                        "How concentrated each factor's loadings are. Good models load few genes "
                        "per factor (low effective-gene-count / genes-to-mass). <b>Note:</b> global "
                        "Gini is inflated by <i>structural</i> zeros for mask-based models like "
                        "EXPIMAP (its decoder is forced to 0 outside each pathway), so we report "
                        "<i>within-support</i> Gini (concentration among the genes a factor actually "
                        "uses) and <i>fraction of genes used</i> instead."))
    sections.append(sec("2 · Program independence", "mod2_collinearity.png", "collinearity",
                        "collinearity",
                        "Pairwise |Pearson r| between latent programs (cell usage). Want mostly "
                        "independent but some collinearity is acceptable; near-0 is over-orthogonal."))
    sections.append(sec("3 · Stability", "mod3_stability.png", "stability", "stability",
                        "Retrained on subsampled cells; Hungarian-matched |corr| of subsample "
                        "factors vs the full-data model. Higher = more reproducible."))
    sections.append(sec("4 · MSigDB best-program", "mod4_best_program.png", "best_program",
                        "best_program",
                        "For each factor, the best-fitting library program (BH-corrected per "
                        "model×library). Four panels, grouped bars per model × library: "
                        "(1) median best-match fold-enrichment (effect size); (2) median best-match "
                        "−log10(BH q) (significance); (3) % factors passing both q and effect-size "
                        "cutoffs; (4) % factors passing the q cutoff only. <b>Held-out:</b> EXPIMAP's mask "
                        "programs are removed from H/C2 for all models (its factors ARE those "
                        "HALLMARK terms); C7 (lib2) is mask-disjoint. Note EXPIMAP is "
                        "annotation-supervised, so its enrichment reflects construction, not discovery."))
    sections.append(sec("5 · Factor↔metadata", "mod5_metadata.png", "metadata", "metadata",
                        "Max η² of any factor with each metadata variable. Good models track "
                        "biological variables and avoid tracking batch."))

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Five-modality factor benchmark</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
h1{{border-bottom:2px solid #333}} h2{{margin-top:2.2rem;color:#1a4d7a}}
.blurb{{color:#555;font-size:.92rem}} .fig img{{max-width:100%;border:1px solid #ddd;border-radius:4px}}
table{{border-collapse:collapse;margin:1rem 0;font-size:.85rem}}
th,td{{border:1px solid #ccc;padding:3px 8px;text-align:right}} th{{background:#f0f4f8}}
td:first-child,th:first-child{{text-align:left}} .meta{{color:#666;font-size:.85rem}}
</style></head><body>
<h1>Five-modality factor-model benchmark</h1>
<p class="meta">Models: {', '.join(model_names)} · data {adata_shape[0]}×{adata_shape[1]}</p>
<p class="meta">{notes}</p>
<h2>Knobs</h2>{knob_tbl}
{''.join(sections)}
</body></html>"""
    path = out_dir / "report_modalities.html"
    path.write_text(html)
    return path
