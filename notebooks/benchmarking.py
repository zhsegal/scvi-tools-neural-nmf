"""Benchmarking pipeline for SemanticSCVI and SemanticSCVIVA models.

Provides :class:`SemanticBenchmark` (core SemanticSCVI) and
:class:`SemanticSpatialBenchmark` (SemanticSCVIVA — adds niche-aware metrics on
top of the core suite). Both expose the same ``run_all()`` orchestrator used by
``scripts/run.py benchmark``.

OpenAI GPT-based scoring methods (:meth:`SemanticBenchmark.gpt_score_programs`
and the GPT block in :meth:`SemanticBenchmark.benchmark_gene_modules`) require
``OPENAI_API_KEY`` in the environment; without it they short-circuit to no-ops.
"""

from __future__ import annotations

import os
import pickle
from itertools import combinations
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import seaborn as sns
import torch
import umap
from openai import OpenAI
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy, fisher_exact, hypergeom, kurtosis, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, silhouette_score

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")
    multipletests = None

_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=_api_key) if _api_key else None


def _safe_box(df, ax, ycol, title, color_map):
    """Boxplot + strip; renders an empty placeholder when the column has no values.

    Avoids the seaborn ``boxprops`` UnboundLocalError that fires on all-NaN data.
    """
    if ycol not in df.columns or not df[ycol].notna().any():
        ax.text(0.5, 0.5, f"{title}\n(no data)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    data = df.dropna(subset=[ycol])
    sns.boxplot(
        data=data, x="Model", y=ycol, ax=ax,
        palette=color_map, hue="Model", legend=False,
    )
    sns.stripplot(
        data=data, x="Model", y=ycol, ax=ax,
        color="black", size=4, alpha=0.6,
    )
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)


_LIB_LABELS = {"lib1": "C2/H (immune)", "lib2": "C7 (monocyte)"}


def _shorten_program(name, lib_label=None, max_len=42):
    """Compact display form for an MSigDB program name.

    When ``lib_label`` is provided, prepends a collection tag so users can tell
    at a glance which atlas the match came from:
    - ``lib1`` (Hallmark + C2 immune-related): ``H:`` for Hallmark, ``C2:R``
      for Reactome, ``C2:K`` for KEGG legacy/medicus, ``C2:W`` for
      WikiPathways, ``C2:`` for the rest (CGP).
    - ``lib2`` (C7 IMMUNESIGDB filtered): ``C7:`` always.
    """
    if name is None or (isinstance(name, float) and name != name):  # NaN
        return ""
    raw = str(name)
    prefix = ""
    body = raw
    if lib_label == "lib1":
        if raw.startswith("HALLMARK_"):
            prefix = "H: "
            body = raw[len("HALLMARK_"):]
        elif raw.startswith("REACTOME_"):
            prefix = "C2:R: "
            body = raw[len("REACTOME_"):]
        elif raw.startswith("KEGG_") or raw.startswith("KEGG_MEDICUS_") or raw.startswith("KEGG_LEGACY_"):
            prefix = "C2:K: "
            body = raw.split("_", 1)[1] if "_" in raw else raw
        elif raw.startswith("WP_"):
            prefix = "C2:W: "
            body = raw[len("WP_"):]
        elif raw.startswith("PID_"):
            prefix = "C2:P: "
            body = raw[len("PID_"):]
        elif raw.startswith("BIOCARTA_"):
            prefix = "C2:B: "
            body = raw[len("BIOCARTA_"):]
        else:
            prefix = "C2: "
    elif lib_label == "lib2":
        prefix = "C7: "
    body = body.replace("_", " ")
    out = prefix + body
    if len(out) > max_len:
        out = out[: max_len - 1] + "…"
    return out


def _plot_spec_sens_scatter(df, ax, label, color_map, models, id_col, id_prefix, title, top_n=3):
    """Scatter ``lib<X>_spec`` vs ``lib<X>_sens`` colored by Model, with the
    top-``top_n`` points (by spec, per-Model) annotated with
    ``"<id_prefix><id>: <best_program>"`` (best_program shortened).
    """
    sx = f"{label}_spec"
    sy = f"{label}_sens"
    bp_col = f"{label}_best_program"
    if sx not in df.columns or sy not in df.columns:
        ax.text(0.5, 0.5, f"{label}: missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return

    annot_rows = []
    for m in models:
        sub = df[df["Model"] == m].dropna(subset=[sx, sy])
        if sub.empty:
            continue
        ax.scatter(sub[sx], sub[sy], color=color_map[m], label=m, s=60, alpha=0.75, edgecolor="black")
        top = sub.sort_values(sx, ascending=False).head(top_n)
        annot_rows.append(top)

    ax.set_xlabel("Specificity = |p ∩ s| / |p|")
    ax.set_ylabel("Sensitivity = |p ∩ s| / |s|")
    ax.set_title(f"{title}  [{_LIB_LABELS.get(label, label)}]")
    ax.set_xlim(-0.02, 1.02)
    sy_max = df[sy].max() if df[sy].notna().any() else 0.5
    ax.set_ylim(-0.02, max(0.5, sy_max * 1.1))
    ax.legend(loc="best", fontsize=8)

    if annot_rows:
        annot_df = pd.concat(annot_rows, ignore_index=True)
        for _, row in annot_df.iterrows():
            ident = f"{id_prefix}{row[id_col]}"
            prog = _shorten_program(row.get(bp_col), lib_label=label) if bp_col in row else ""
            txt = f"{ident}: {prog}" if prog else ident
            ax.annotate(
                txt,
                (row[sx], row[sy]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, alpha=0.85,
            )


def _render_text_panel(ax, sections, title, max_lines_per_panel=60):
    """Render a vertical text panel with ``[(header, [lines])]`` sections.

    Headers are bold and slightly larger; lines are monospace for alignment.
    Lines beyond ``max_lines_per_panel`` are truncated with a ``... (N more)``
    summary so the panel stays legible.
    """
    ax.set_title(title)
    ax.axis("off")

    flat = []
    for header, lines in sections:
        flat.append(("hdr", header))
        for ln in lines:
            flat.append(("ln", ln))
        flat.append(("sp", ""))
    if len(flat) > max_lines_per_panel:
        keep = flat[: max_lines_per_panel - 1]
        more = sum(1 for kind, _ in flat[max_lines_per_panel - 1:] if kind == "ln")
        flat = keep + [("ln", f"… ({more} more rows omitted)")]

    n = max(len(flat), 1)
    dy = 1.0 / n
    y = 1.0 - dy * 0.5
    for kind, txt in flat:
        if kind == "hdr":
            ax.text(0.0, y, txt, transform=ax.transAxes, fontsize=9,
                    family="DejaVu Sans", weight="bold", va="center")
        elif kind == "ln":
            ax.text(0.0, y, txt, transform=ax.transAxes, fontsize=8,
                    family="monospace", va="center")
        # else "sp" — skip; advance y for spacing
        y -= dy


def _format_interpretation_lines(df, score_col, program_col, id_col, id_prefix):
    """Per Model, emit ``[(model_header, [lines])]`` showing each row's id +
    LLM score + LLM-provided program label, sorted by score descending."""
    out = []
    for model in df["Model"].unique():
        sub = df[df["Model"] == model].copy()
        sub = sub.sort_values(score_col, ascending=False, na_position="last")
        lines = []
        for _, row in sub.iterrows():
            ident = f"{id_prefix}{row[id_col]}"
            score = row.get(score_col)
            prog = row.get(program_col, "")
            score_txt = f"{score:>5.1f}" if score == score else "  NaN"
            prog = "" if prog is None else str(prog)
            if len(prog) > 60:
                prog = prog[:59] + "…"
            lines.append(f"  {ident:<8s} [{score_txt}]  {prog}")
        out.append((f"━━ {model} ━━", lines))
    return out


def _format_top10_lines(df, lib_label, id_col, id_prefix, top_n=10):
    """Per Model, top-``top_n`` rows by ``<lib>_spec`` descending. Returns
    sections suitable for ``_render_text_panel``."""
    sx = f"{lib_label}_spec"
    sy = f"{lib_label}_sens"
    bp_col = f"{lib_label}_best_program"
    out = []
    for model in df["Model"].unique():
        sub = df[df["Model"] == model].dropna(subset=[sx]).sort_values(sx, ascending=False).head(top_n)
        if sub.empty:
            continue
        lines = []
        for _, row in sub.iterrows():
            ident = f"{id_prefix}{row[id_col]}"
            sp = row[sx]
            sn = row.get(sy, float("nan"))
            prog = _shorten_program(row.get(bp_col), lib_label=lib_label, max_len=60)
            sp_t = f"{sp:.2f}" if sp == sp else "  NaN"
            sn_t = f"{sn:.3f}" if sn == sn else "  NaN"
            lines.append(f"  {ident:<8s} spec={sp_t} sens={sn_t}  {prog}")
        out.append((f"━━ {model} ━━ ({_LIB_LABELS.get(lib_label, lib_label)})", lines))
    return out


# MSigDB-style libraries used by the HG-enrichment metric. Order is the
# plotting order (one column per library).
_HG_LIB_ORDER = ("H", "C2_immune", "C7")
_HG_LIB_TITLES = {
    "H": "H (Hallmark)",
    "C2_immune": "C2 (immune-related)",
    "C7": "C7 (monocytes)",
}


def _split_hallmark(gene_sets):
    """Split a GMT-derived ``{name: set(genes)}`` into ``(hallmark, rest)``."""
    h, c2 = {}, {}
    for name, genes in gene_sets.items():
        if str(name).startswith("HALLMARK_"):
            h[name] = genes
        else:
            c2[name] = genes
    return h, c2


def _plot_enrichment_distribution(long_df, ax, lib_label, color_map, models, q_thresh=0.05):
    """Boxplot + strip of ER among significant (q<``q_thresh``) gene-sets per Model.

    ``long_df`` has columns ``Model``, ``Library``, ``ER``, ``qvalue``.
    Falls back to a placeholder if no significant rows for this library.
    """
    sub = long_df[(long_df["Library"] == lib_label) & (long_df["qvalue"] < q_thresh)]
    sub = sub.dropna(subset=["ER"])

    title = f"{_HG_LIB_TITLES.get(lib_label, lib_label)}\nER of significant gene sets (q<{q_thresh})"

    if sub.empty:
        ax.text(0.5, 0.5, "(no significant gene sets)", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xticks([])
        return

    sub = sub.copy()
    sub["Model"] = pd.Categorical(sub["Model"], categories=models, ordered=True)
    sns.boxplot(
        data=sub, x="Model", y="ER", ax=ax,
        palette=color_map, hue="Model", legend=False, showfliers=False,
    )
    sns.stripplot(
        data=sub, x="Model", y="ER", ax=ax,
        color="black", size=3, alpha=0.5,
    )

    if sub["ER"].max() > 50:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("fold enrichment (ER)")
    ax.tick_params(axis="x", rotation=30)

    # n-significant annotations above each box
    counts = sub.groupby("Model", observed=False).size()
    y_top = ax.get_ylim()[1]
    for i, m in enumerate(models):
        n = int(counts.get(m, 0))
        ax.text(i, y_top, f"n={n}", ha="center", va="bottom", fontsize=8, color="#333")


class SemanticBenchmark:
    def __init__(
        self,
        models_dict,
        adata,
        pathway_index,
        gene_mapping=None,
        out_dir="benchmark_results",
        cell_type_context="the cells under study",
        judges=None,
    ):
        """
        models_dict: Dictionary of {name: model_object}
        adata: AnnData object
        pathway_index: The dictionary returned by build_pathway_index() OR path to .pkl file.
        gene_mapping: Tuple (source_col, target_col) to map gene IDs to Symbols if needed.
        cell_type_context: Free-text description of the cells being analyzed; passed
            to every LLM judge so prompts mention the right biology (e.g. "human
            tissue CD8 T cells (Haniffa COVID-19 atlas)").
        judges: Optional list of ``(persona_id, ScorerClass, kwargs)`` tuples that
            replace :meth:`_default_judges`. Each ScorerClass must follow the
            ``llm_scorers._BaseScorer`` interface.
        """
        self.models = models_dict
        self.adata = adata
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.gene_mapping = gene_mapping
        self.cell_type_context = cell_type_context
        self._judges = judges if judges is not None else self._default_judges()
        
        # Initialize Cache for Clustering
        self.cluster_cache = {}

        # 1. Load Pre-computed Index
        if isinstance(pathway_index, (str, Path)):
            print(f"Loading pathway index from file: {pathway_index}")
            with open(pathway_index, 'rb') as f:
                self.index_data = pickle.load(f)
        else:
            self.index_data = pathway_index
            
        self.pathways = self.index_data["pathways"]
        self.gene_to_pathways = self.index_data["gene_to_pathways"]
        self.M_pathways = self.index_data["M"]
        
        # 2. Pre-calculate Library Size
        if scipy.sparse.issparse(self.adata.X):
            self.lib_size = np.array(self.adata.X.sum(axis=1)).flatten()
        else:
            self.lib_size = np.array(self.adata.X.sum(axis=1)).flatten()
            
        # 3. Build ID Map
        if self.gene_mapping:
            source_col, target_col = self.gene_mapping
            self.id_map = dict(zip(self.adata.var[source_col], self.adata.var[target_col]))
        else:
            self.id_map = None

    def _get_loadings(self, model):
        """Robustly extract loadings matrix."""
        if hasattr(model, 'get_loadings'):
            loadings = model.get_loadings()
        elif hasattr(model.module, 'decoder'):
            try:
                fc = model.module.decoder.factor_regressor.fc_layers[0][0]
                w = fc.weight
                if hasattr(model.module.decoder, 'weights_positive') and model.module.decoder.weights_positive:
                    w = torch.nn.functional.softplus(w)
                loadings = pd.DataFrame(w.detach().cpu().numpy().T, index=self.adata.var_names)
            except Exception:
                return None
        else:
            return None
        
        # Align indices if mapping provided
        if self.gene_mapping:
            source_col, _ = self.gene_mapping
            if len(loadings) == len(self.adata):
                loadings.index = self.adata.var[source_col]
        return loadings

    # =========================================================================
    # SHARED CACHED CLUSTERING HELPER
    # =========================================================================
    def _get_gene_clusters(self, model_name, model, n_top=500, resolution=1.0, top_per_cluster=None):
        """
        Performs Leiden clustering on gene loadings.
        Returns the AnnData object containing the clusters.
        Caches the result based on (Model, TopN, Resolution, TopPerCluster) so benchmarks sharing settings don't recompute.

        ``top_per_cluster``: if set, after Leiden clustering keep only the top-N
        genes per cluster by ``max_loading``. UMAP coordinates are preserved
        from the full-pool embedding; only the points carried forward are gated.
        """
        cache_key = f"{model_name}_{n_top}_{resolution}_{top_per_cluster}"

        if cache_key in self.cluster_cache:
            return self.cluster_cache[cache_key]

        print(f"  > Computing gene clusters for {model_name} (Res={resolution})...")
        loadings = self._get_loadings(model)
        if loadings is None: return None

        # 1. Select Top Genes
        max_vals = loadings.abs().max(axis=1)
        top_genes_idx = max_vals.sort_values(ascending=False).head(n_top).index
        X_genes = loadings.loc[top_genes_idx].values

        # 2. Map IDs -> Symbols
        if self.id_map:
            gene_names = [self.id_map.get(g, g) for g in top_genes_idx]
        else:
            gene_names = top_genes_idx.tolist()

        # 3. Create Gene AnnData
        adata_genes = sc.AnnData(X=X_genes)
        adata_genes.obs_names = gene_names
        adata_genes.obs_names_make_unique()

        # Store metadata for later use (e.g. GPT scoring needs weights)
        adata_genes.obs['original_id'] = top_genes_idx.tolist()
        adata_genes.obs['max_loading'] = max_vals.loc[top_genes_idx].values

        # 4. Cluster (Compute Once!)
        sc.pp.neighbors(adata_genes, use_rep='X', n_neighbors=15, metric='cosine')
        sc.tl.leiden(adata_genes, resolution=resolution, key_added='leiden')
        sc.tl.umap(adata_genes)

        # 5. Gate: keep only top-X genes per cluster by max_loading
        if top_per_cluster is not None:
            keep_mask = (
                adata_genes.obs
                .groupby('leiden')['max_loading']
                .rank(method='first', ascending=False)
                .le(top_per_cluster)
            )
            adata_genes = adata_genes[keep_mask.values].copy()

        # Save to cache
        self.cluster_cache[cache_key] = adata_genes
        return adata_genes

    def _get_hierarchical_clusters(self, model_name, model, n_top=500, max_k=20):
        """Hierarchical (correlation-distance, average linkage) clustering of top genes.

        ``k`` is chosen by silhouette score over ``range(2, max_k+1)``. Result
        DataFrame is cached on ``self.cluster_cache`` separately from the
        Leiden cache so the two methods don't clobber each other.

        Returns a DataFrame indexed by gene IDs with columns ``Cluster``
        (1..k), ``max_loading``, and ``Symbol`` (mapped to gene symbol when
        ``id_map`` is available, else equal to the gene ID).
        """
        cache_key = f"hier_{model_name}_{n_top}_{max_k}"
        if cache_key in self.cluster_cache:
            return self.cluster_cache[cache_key]

        print(f"  > Computing hierarchical gene clusters for {model_name} (max_k={max_k})...")
        loadings = self._get_loadings(model)
        if loadings is None:
            return None

        max_vals = loadings.abs().max(axis=1)
        top_genes_idx = max_vals.sort_values(ascending=False).head(n_top).index
        subset = loadings.loc[top_genes_idx]

        # Drop zero-variance rows: pathway-mask models (EXPIMAP) produce
        # rows nonzero in a single GP -> std=0 -> NaN under 'correlation'.
        row_std = subset.values.std(axis=1)
        keep = row_std > 0
        if not keep.all():
            print(f"  > dropping {(~keep).sum()} zero-variance rows before correlation pdist")
            subset = subset.loc[keep]
            top_genes_idx = subset.index
        if subset.shape[0] < 2:
            print(f"  > too few rows ({subset.shape[0]}) after filtering for {model_name}; skipping")
            return None

        dists = pdist(subset.values, metric="correlation")
        dist_matrix = squareform(dists)
        Z = linkage(dists, method="average")

        best_k, best_score = 2, -1
        for k in range(2, max_k + 1):
            labels = fcluster(Z, t=k, criterion="maxclust")
            try:
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_k = k
        final_labels = fcluster(Z, t=best_k, criterion="maxclust")

        if self.id_map:
            symbols = [self.id_map.get(g, g) for g in top_genes_idx]
        else:
            symbols = list(top_genes_idx)

        df = pd.DataFrame(
            {
                "Cluster": final_labels,
                "max_loading": max_vals.loc[top_genes_idx].values,
                "Symbol": symbols,
            },
            index=top_genes_idx,
        )
        self.cluster_cache[cache_key] = df
        return df

    def _get_top_genes(self, loadings, n_top=100):
        top_genes = {}
        for factor_col in loadings.columns:
            ids = loadings[factor_col].sort_values(ascending=False).head(n_top).index.tolist()
            if self.id_map:
                genes = [self.id_map.get(uid, uid) for uid in ids]
            else:
                genes = ids
            top_genes[factor_col] = genes
        return top_genes

    # =========================================================================
    # 1. DEPTH (Summary)
    # =========================================================================
    def benchmark_depth(self):
        print("\n--- [1/8] Depth Correlation (Summary) ---")
        results = []
        
        for name, model in self.models.items():       
            try:
                scores = model.get_latent_representation(self.adata)
            except Exception as e:
                print(f"Skipping {name}: Could not extract latent rep. Error: {e}")
                continue
        
            current_lib_size = self.lib_size
            if len(scores) != len(current_lib_size):
                scores = scores[:len(current_lib_size)]
                
            for i in range(scores.shape[1]):
                r, _ = pearsonr(current_lib_size, scores[:, i])
                results.append({"Model": name, "Factor": i, "Abs_R": abs(r)})
                
        df = pd.DataFrame(results)
        if df.empty: 
            print("No data to plot.")
            return

        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, x="Model", y="Abs_R")
        plt.xticks(rotation=45, ha="right")
        plt.title("Depth Correlation Summary (Abs R)")
        plt.ylabel("Absolute Correlation with Library Size")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.out_dir / "1_depth_correlation_summary.png")
        plt.show()

    

    # =========================================================================
    # 3. ORTHOGONALITY
    # =========================================================================
    def benchmark_orthogonality(self):
        print("\n--- [3/8] Orthogonality of Latent Factors (All Models) ---")
        
        n_models = len(self.models)
        if n_models == 0: return

        # 1. Determine Grid Dimensions (e.g., 2 columns, flexible rows)
        n_cols = 3 if n_models >= 3 else n_models
        n_rows = (n_models + n_cols - 1) // n_cols  # Ceiling division
        
        # Create one big figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), constrained_layout=True)
        axes = np.array(axes).flatten()  # Flatten to make indexing easy (0, 1, 2...)

        # 2. Iterate and Plot
        idx = 0
        for name, model in self.models.items():
            ax = axes[idx]
            try:
                # Use Encoder Z
                scores = model.get_latent_representation(self.adata)
                z_df = pd.DataFrame(scores)
                
                # Compute Correlation Matrix (Pearson)
                corr_matrix = z_df.corr()
                
                # Plot Heatmap
                sns.heatmap(
                    corr_matrix, 
                    cmap="bwr", 
                    center=0, 
                    vmin=-1, 
                    vmax=1, 
                    annot=False, 
                    square=True,
                    cbar=(idx == n_cols-1), # Only show colorbar on the last plot of the row (cleaner)
                    ax=ax
                )
                ax.set_title(f"{name}", fontsize=12)
                ax.set_xlabel("Latent Factor")
                ax.set_ylabel("Latent Factor")
                
                idx += 1
                
            except Exception as e:
                print(f"Skipping {name}: Could not extract latent rep. Error: {e}")
                ax.axis('off') # Hide axis if failed
                idx += 1
                continue

        # 3. Clean up empty axes
        for i in range(idx, len(axes)):
            axes[i].axis('off')

        fig.suptitle("Latent Factor Orthogonality (Correlation)", fontsize=16)
        plt.savefig(self.out_dir / "3_orthogonality_summary.png")
        plt.show()

  
    # =========================================================================
    # 5. LATENT PROFILE (Ward Clustering)
    # =========================================================================
    def genes_clustermap(self, n_top=500):
        print(f"\n--- [5/8] Latent Factor Profile (Top {n_top}) ---")
        for name, model in self.models.items():
            loadings = self._get_loadings(model)
            if loadings is None: continue
            max_loadings = loadings.abs().max(axis=1)
            top_genes = max_loadings.sort_values(ascending=False).head(n_top).index
            subset = loadings.loc[top_genes]
            if self.id_map:
                new_index = subset.index.to_series().map(self.id_map).fillna(subset.index.to_series())
                subset.index = new_index
            g = sns.clustermap(
                subset, method='ward', metric='euclidean', z_score=None, 
                cmap="viridis", col_cluster=False, row_cluster=True,
                figsize=(10, 15), yticklabels=False
            )
            plt.title(f"{name}: Top {n_top} Genes Profile")
            plt.savefig(self.out_dir / f"5_latent_profile_{name}.png")
            plt.show()

    # =========================================================================
    # 6. BIOLOGICAL COHERENCE (Omega)
    # =========================================================================
    def program_omega(self, top_n=20):
        print("\n--- [6/8] Biological Coherence (Omega) ---")
        results = []
        for name, model in self.models.items():
            print(f"  > Processing {name}...")
            loadings = self._get_loadings(model)
            top_genes = self._get_top_genes(loadings, top_n)
            for factor, gene_list in top_genes.items():
                valid = [g for g in gene_list if g in self.gene_to_pathways]
                if len(valid) < 2: continue
                pairs = list(combinations(valid, 2))
                omegas = []
                for g1, g2 in pairs:
                    p1 = self.gene_to_pathways[g1]
                    p2 = self.gene_to_pathways[g2]
                    overlap = len(p1 & p2)
                    exp = (len(p1) * len(p2)) / self.M_pathways if self.M_pathways > 0 else 0
                    if exp > 0: omegas.append(overlap / exp)
                if omegas:
                    results.append({"Model": name, "Factor": factor, "Log2_Omega": np.log2(np.mean(omegas) + 1e-3)})
        df = pd.DataFrame(results)
        if df.empty: return
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="Model", y="Log2_Omega")
        plt.xticks(rotation=45, ha="right")
        plt.title("Per program omega score")
        plt.savefig(self.out_dir / "6_omega_coherence.png")
        plt.show()

    # =========================================================================
    # 7. JACCARD
    # =========================================================================
    def program_jaccard(self, top_n=100):
        print("\n--- [7/8] Jaccard Similarity ---")
        active_genes = set()
        for model in self.models.values():
            l = self._get_loadings(model)
            if l is not None:
                tg = self._get_top_genes(l, top_n)
                for gl in tg.values(): active_genes.update(gl)
        gmt_genes = set().union(*self.pathways.values())
        universe = sorted(list(active_genes & gmt_genes))
        if not universe: return
        g_idx = {g: i for i, g in enumerate(universe)}
        p_names = list(self.pathways.keys())
        mat = np.zeros((len(universe), len(p_names)), dtype=bool)
        for j, pname in enumerate(p_names):
            for g in self.pathways[pname]:
                if g in g_idx: mat[g_idx[g], j] = True
        sim_mat = 1 - squareform(pdist(mat, metric='jaccard'))
        sim_df_global = pd.DataFrame(sim_mat, index=universe, columns=universe)
        results = []
        for name, model in self.models.items():
            l = self._get_loadings(model)
            if l is None: continue
            top_genes = self._get_top_genes(l, top_n)
            for factor, gene_list in top_genes.items():
                valid = [g for g in gene_list if g in universe]
                if len(valid) < 2: continue
                pairs = list(combinations(valid, 2))
                sims = [sim_df_global.at[g1, g2] for g1, g2 in pairs]
                if sims:
                    results.append({"Model": name, "Factor": factor, "Mean_Jaccard": np.mean(sims)})
        df = pd.DataFrame(results)
        if df.empty: return
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="Model", y="Mean_Jaccard")
        plt.xticks(rotation=45, ha="right")

        plt.title("per program Jaccard Similarity")
        plt.savefig(self.out_dir / "7_jaccard_similarity.png")
        plt.show()

    # =========================================================================
    # 8. GENE MODULE UMAP
    # =========================================================================
    def benchmark_silhouette(self, n_top=500, resolution=1.0, top_per_cluster=None):
        print(f"\n--- [8/8] Gene Module UMAP & Separation Score ---")

        silhouette_results = []

        for name, model in self.models.items():
            # 1. Get Clusters (using the shared helper)
            adata_genes = self._get_gene_clusters(name, model, n_top, resolution, top_per_cluster=top_per_cluster)
            if adata_genes is None: continue
            
            # 2. PRINT GENE LISTS
            print(f"\n  > Gene Modules for {name}:")
            
            # Check if we can map IDs to Names (e.g. ENSG -> SYMBOL)
            if 'feature_name' in self.adata.var.columns:
                id_to_name = self.adata.var['feature_name'].to_dict()
                gene_labels = [id_to_name.get(g, g) for g in adata_genes.obs_names]
            else:
                gene_labels = adata_genes.obs_names.tolist()

            # Create a temporary dataframe to organize names and clusters
            df_genes = pd.DataFrame({
                'Gene': gene_labels, 
                'Cluster': adata_genes.obs['leiden'].values
            })

            # Iterate and print
            unique_clusters = sorted(df_genes['Cluster'].unique().astype(int))
            for cid in unique_clusters:
                genes = df_genes[df_genes['Cluster'] == str(cid)]['Gene'].tolist()
                # Print first 20 genes followed by "..." if list is long
                gene_str = ", ".join(genes[:60])
                suffix = "..." if len(genes) > 60 else ""
                print(f"    [Cluster {cid}] ({len(genes)} genes): {gene_str}{suffix}")

            # 3. Calculate Silhouette Score
            # We use the High-Dimensional Loadings (X) for accuracy.
            try:
                if len(unique_clusters) > 1:
                    score = silhouette_score(adata_genes.X, adata_genes.obs['leiden'])
                else:
                    score = 0.0 # Only 1 cluster found
            except Exception:
                score = 0.0
            
            silhouette_results.append({"Model": name, "Silhouette": score})

            # 4. Plot UMAP with Score in Title
            fig, ax = plt.subplots(figsize=(8, 6))
            sc.pl.umap(
                adata_genes, 
                color='leiden', 
                title=f"{name}\nSilhouette Score: {score:.3f} (Higher is Better)", 
                legend_loc='on data', 
                show=False, 
                ax=ax
            )
            plt.savefig(self.out_dir / f"8_gene_umap_{name}.png")
            plt.show()

        # Optional: Plot comparison of scores
        df_scores = pd.DataFrame(silhouette_results)
        if not df_scores.empty:
            plt.figure(figsize=(8, 6))
            sns.barplot(data=df_scores, x="Model", y="Silhouette", palette="viridis")
            plt.title("Gene Module Separation (Silhouette Score)")
            plt.ylabel("Silhouette Score (Higher = Better Defined Modules)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()

    # 
    # =========================================================================
    # 10. CLUSTER COHERENCE (Omega on Leiden Modules)
    # =========================================================================
    def omega_clusters(self, n_top=500, resolution=0.5):
        """
        1. Uses shared _get_gene_clusters.
        2. Calculates the Omega (Biological Coherence) for EACH cluster.
        3. Plots Mean +/- STD.
        """
        print(f"\n--- [10/10] Gene Cluster Coherence (Omega) ---")
        results = []
        from itertools import combinations

        for name, model in self.models.items():
            # USE HELPER
            adata_genes = self._get_gene_clusters(name, model, n_top, resolution)
            if adata_genes is None: continue

            clusters = adata_genes.obs['leiden'].unique()
            
            for cluster_id in clusters:
                cluster_genes = adata_genes.obs_names[adata_genes.obs['leiden'] == cluster_id].tolist()
                valid = [g for g in cluster_genes if g in self.gene_to_pathways]
                if len(valid) < 2: continue 

                pairs = list(combinations(valid, 2))
                omegas = []
                for g1, g2 in pairs:
                    p1 = self.gene_to_pathways[g1]
                    p2 = self.gene_to_pathways[g2]
                    overlap = len(p1 & p2)
                    exp = (len(p1) * len(p2)) / self.M_pathways if self.M_pathways > 0 else 0
                    if exp > 0: omegas.append(overlap / exp)

                if omegas:
                    results.append({
                        "Model": name,
                        "Cluster": cluster_id,
                        "Log2_Omega": np.log2(np.mean(omegas) + 1e-3)
                    })

        df = pd.DataFrame(results)
        if df.empty: return

        plt.figure(figsize=(10, 6))
        # PLOT: Mean + STD
        sns.barplot(
            data=df, 
            x="Model", 
            y="Log2_Omega", 
            errorbar='sd', 
            capsize=0.1, 
            alpha=0.8
        )
        
        plt.axhline(0, color='red', linestyle='--', label="Random Chance")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Omega Gene Clusters (Mean ± STD)")
        plt.ylabel("Log2(Omega)")
        plt.tight_layout()
        plt.savefig(self.out_dir / "10_cluster_omega_barplot.png")
        plt.show()

    # =========================================================================
    # 11. GPT COHERENCE SCORE
    # =========================================================================
    def gpt_score_programs(self, n_top=500, resolution=0.5, genes_per_cluster=10):
        print(f"\n--- [11/11] GPT Biological Coherence Score ---")
        results = []
        
        chat_prompt = """
        You score gene lists from 0–100 based on how coherent they are as a gene expression program.
        0 = random/unrelated. 100 = very coherent single biological module.
        Return ONLY the number. No words. be strict. cells are classical monoctyes
        """

        for name, model in self.models.items():
            print(f"  > Scoring {name}...")
            # USE HELPER
            adata_genes = self._get_gene_clusters(name, model, n_top, resolution)
            if adata_genes is None: continue

            clusters = adata_genes.obs['leiden'].unique()
            
            for cluster_id in clusters:
                cluster_mask = adata_genes.obs['leiden'] == cluster_id
                subset = adata_genes.obs[cluster_mask]
                
                # Sort by max_loading (which we saved in the helper!)
                top_core = subset.sort_values('max_loading', ascending=False).head(genes_per_cluster)
                core_genes_list = top_core.index.tolist()
                
                if len(core_genes_list) < 3: continue

                try:
                    gene_str = ", ".join(core_genes_list)
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": chat_prompt},
                            {"role": "user", "content": gene_str},
                        ],
                        temperature=0.0,
                        max_completion_tokens=10,
                    )
                    score = float(response.choices[0].message.content.strip())
                    
                    results.append({
                        "Model": name,
                        "Cluster": cluster_id,
                        "GPT_Score": score,
                        "Core_Genes": gene_str
                    })
                except Exception as e:
                    print(f"    GPT Error on C{cluster_id}: {e}")

        df = pd.DataFrame(results)
        if df.empty: return

        plt.figure(figsize=(8, 6))
        # PLOT: Mean + STD
        sns.barplot(
            data=df, 
            x="Model", 
            y="GPT_Score", 
            errorbar='sd', 
            capsize=0.1, 
            palette="viridis", 
            alpha=0.9
        )
        
        plt.xticks(rotation=45, ha="right")
        plt.title(f"GPT-4o Coherence Score (Mean ± STD)")
        plt.tight_layout()
        plt.savefig(self.out_dir / "11_gpt_coherence_barplot.png")
        plt.show()

    def pca_var(self):
        """
        Calculates and plots the Effective Dimensionality of the latent space.
        (How many PC components are needed to explain 90% of the variance in Z)
        """
        print("\n--- [Internal Quality] PCA Effective Dimensionality ---")
   
        results = []
        
        for name, model in self.models.items():
            metrics = {"Model": name}
            
            # --- EXTRACT DATA ---
            try:
                # Z: Cell Scores (n_cells, n_factors)
                # We only need Z for this metric
                Z = model.get_latent_representation(self.adata)
                
            except Exception as e:
                print(f"Skipping {name}: Data extraction failed ({e})")
                continue

            # --- METRIC: PCA EFFECTIVE DIMENSIONALITY ---
            # "How many components do we need to explain 90% of the variance?"
            try:
                # We fit PCA on the cell scores (Z)
                # If Z is noise/collapsed, 1 component explains everything (Low score).
                # If Z is rich/orthogonal, we need many components (High score).
                
                # Note: n_components=0.90 means "select enough components to explain 90% variance"
                pca = PCA(n_components=0.90) 
                pca.fit(Z)
                metrics["PCA_Dim_90pct"] = pca.n_components_
            except Exception as e:
                print(f"PCA calculation failed for {name}: {e}")
                metrics["PCA_Dim_90pct"] = 0

            results.append(metrics)

        # --- PLOTTING ---
        df = pd.DataFrame(results)
        if df.empty:
            print("No data to plot.")
            return

        plt.figure(figsize=(10, 8))
        sns.barplot(data=df, x="Model", y="PCA_Dim_90pct", palette="Blues")
        
        plt.title("Effective Dimensionality (PCA 90% Var)\n(Higher = Better Usage of Factors)", fontsize=16)
        
        # Optional: Add a reference line for your theoretical n_factors (e.g., 10 or 20)
        # You can adjust y=10 to match your actual model config if needed
        plt.axhline(y=10, color='red', linestyle='--', alpha=0.5, label="Target (Example: 10)")
        
        plt.xlabel("Model", fontsize=14)
        plt.ylabel("Components needed for 90% Var", fontsize=14)
        plt.xticks(rotation=45, ha="right", fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save and Show
        save_path = self.out_dir / "internal_quality_pca_dim.png"
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
        plt.show()

        
        
    def _plot_summary(self, data, y_col, title):
        """
        Helper function to plot bar charts for Omega and GPT scores.
        """
        df = pd.DataFrame(data)
        if df.empty: 
            print(f"No data to plot for {y_col}")
            return

        plt.figure(figsize=(8, 6))
        
        # Plot Mean with Standard Deviation error bars
        sns.barplot(
            data=df, 
            x="Model", 
            y=y_col, 
            errorbar='sd', 
            capsize=0.1, 
            alpha=0.9,
            palette="viridis"
        )
        
        # Add reference line for Omega
        if y_col == "Log2_Omega":
            plt.axhline(0, color='red', linestyle='--', label="Random Chance")
            plt.legend()
            
        plt.title(f"{title} (Mean ± STD)")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save and Show
        filename = f"benchmark_{y_col}.png"
        plt.savefig(self.out_dir / filename)
        plt.show()
    
    def benchmark_gene_modules(self, n_top=500, max_k=20):
        # --- Imports ---
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import pdist, squareform
        from sklearn.metrics import silhouette_score
        from itertools import combinations

        print(f"\n--- [Benchmark] Gene Module Detection & Scoring ---")
        
        omega_results = []
        gpt_results = []
        
        # GPT System Prompt
        chat_prompt = """
        You score gene lists from 0–100 based on how coherent they are as a gene expression program.
        0 = random/unrelated. 100 = very coherent single biological module.
        Return ONLY the number. No words. be strict. cells are classical monoctyes.
        """

        for name, model in self.models.items():
            print(f"\nProcessing {name}...")

            # --- 1. EXTRACT LOADINGS ---
            loadings = self._get_loadings(model)
            if loadings is None: continue

            # Filter to Top Genes
            max_vals = loadings.abs().max(axis=1)
            top_genes_idx = max_vals.sort_values(ascending=False).head(n_top).index
            subset = loadings.loc[top_genes_idx]

            # Drop zero-variance rows: pathway-mask models (EXPIMAP) produce
            # rows that are nonzero in a single GP, giving std=0 -> NaN under
            # 'correlation' metric -> linkage rejects the matrix.
            row_std = subset.values.std(axis=1)
            keep = row_std > 0
            if not keep.all():
                dropped = (~keep).sum()
                print(f"  > dropping {dropped} zero-variance rows before correlation pdist")
                subset = subset.loc[keep]
            if subset.shape[0] < 2:
                print(f"  > too few rows ({subset.shape[0]}) after filtering; skipping {name}")
                continue

            # --- 2. AUTO-DETECT SQUARES (Silhouette Optimization) ---
            dists = pdist(subset.values, metric='correlation')
            dist_matrix = squareform(dists)
            Z = linkage(dists, method='average')

            best_k = 2
            best_score = -1
            
            for k in range(2, max_k + 1):
                labels = fcluster(Z, t=k, criterion='maxclust')
                score = silhouette_score(dist_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_k = k
            
            print(f"  > Found {best_k} optimal gene modules (Silhouette: {best_score:.3f})")
            
            # Get Final Labels
            final_labels = fcluster(Z, t=best_k, criterion='maxclust')
            df_clusters = pd.DataFrame({'GeneID': subset.index, 'Cluster': final_labels})

            # --- MAP TO SYMBOLS (If available) ---
            # Checks for 'feature_name' in adata.var, or uses index
            if 'feature_name' in self.adata.var.columns:
                id_to_name = self.adata.var['feature_name'].to_dict()
                df_clusters['Name'] = df_clusters['GeneID'].map(lambda x: id_to_name.get(x, x))
            else:
                df_clusters['Name'] = df_clusters['GeneID']

            # --- 3. PRINT GENE LISTS ---
            print(f"  > Genes in clusters for {name}:")
            for cid in range(1, best_k + 1):
                # Get list of gene names for this cluster
                genes = df_clusters[df_clusters['Cluster'] == cid]['Name'].tolist()
                
                # Print concise summary
                gene_str = ", ".join(genes)
                print(f"    [Cluster {cid}] ({len(genes)} genes): {gene_str}")

            # --- 4. PLOT HEATMAP ---
            cluster_colors = pd.Series(final_labels, index=subset.index).map(
                dict(zip(range(1, best_k+1), sns.color_palette("tab20", best_k)))
            )
            
            g = sns.clustermap(
                subset.T.corr(), 
                row_linkage=Z, col_linkage=Z,
                row_colors=cluster_colors, col_colors=cluster_colors,
                cmap="vlag", center=0,
                xticklabels=False, yticklabels=False,
                figsize=(7, 7)
            )
            g.fig.suptitle(f"{name}: {best_k} Detected Modules", y=1.02)
            plt.savefig(self.out_dir / f"module_heatmap_{name}.png")
            plt.show()

            # --- 5. SCORE CLUSTERS (Omega & GPT) ---
            for cid in range(1, best_k + 1):
                # Use GeneID for Pathway lookup (Omega)
                gene_ids = df_clusters[df_clusters['Cluster'] == cid]['GeneID'].tolist()
                # Use Names for GPT (Better semantic understanding)
                gene_names = df_clusters[df_clusters['Cluster'] == cid]['Name'].tolist()
                
                if len(gene_ids) < 3: continue
                
                # A) OMEGA SCORE (Needs IDs usually)
                valid = [g for g in gene_ids if g in self.gene_to_pathways]
                if len(valid) >= 2:
                    pairs = list(combinations(valid, 2))
                    omegas = []
                    for g1, g2 in pairs:
                        p1 = self.gene_to_pathways[g1]
                        p2 = self.gene_to_pathways[g2]
                        overlap = len(p1 & p2)
                        exp = (len(p1) * len(p2)) / self.M_pathways if self.M_pathways > 0 else 0
                        if exp > 0: omegas.append(overlap / exp)
                    
                    if omegas:
                        omega_results.append({
                            "Model": name, 
                            "Cluster": f"C{cid}", 
                            "Log2_Omega": np.log2(np.mean(omegas) + 1e-3)
                        })

                # B) GPT SCORE (Needs Names)
                top_20_names = ", ".join(gene_names[:50])
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": chat_prompt},
                            {"role": "user", "content": top_20_names},
                        ],
                        temperature=0.0,
                        max_completion_tokens=10,
                    )
                    score = float(response.choices[0].message.content.strip())
                    gpt_results.append({
                        "Model": name, 
                        "Cluster": f"C{cid}", 
                        "GPT_Score": score
                    })
                except Exception as e:
                    pass

        # --- 6. PLOT COMPARISON ---
        self._plot_summary(omega_results, "Log2_Omega", "Omega Biological Coherence")
        self._plot_summary(gpt_results, "GPT_Score", "GPT-4o Semantic Coherence")

    # =========================================================================
    # ENCODER-DECODER PATHWAY COMPARISON (For Linear Models)
    # =========================================================================
    # =========================================================================
    # Decoupled entry points (preferred)
    # =========================================================================

    def run_figures(
        self,
        hallmark_gmt=None,
        gobp_gmt=None,
        semantic_map=None,
        lib1_gmt=None,
        lib2_gmt=None,
        per_projection_n_top=50,
        cluster_n_top=500,
        cluster_top_per_cluster=None,
        enrichment_n_top=100,
        quick_mode=False,
    ):
        """Generate every non-LLM benchmark + intermediate gene_groups JSON.

        Pair with :meth:`run_grading` for the LLM-dependent layer; that step
        is fully resumable from the on-disk cache, so this method should be
        run once per model and grading can be invoked any number of times.

        ``cluster_n_top`` sets the pool size of top-loaded genes fed into
        UMAP+Leiden (``benchmark_silhouette``, ``_get_gene_clusters``) and
        hierarchical (``benchmark_gene_modules``, ``_get_hierarchical_clusters``)
        clustering. ``cluster_top_per_cluster`` (optional) gates the Leiden
        pool to only the top-N genes per cluster (by max_loading) before
        silhouette and gene-group export. ``enrichment_n_top`` sizes the
        per-factor top-gene list used for the Hallmark / GO BP ORA enrichment.
        """
        if quick_mode:
            print("\n[quick_mode=True] skipping depth / silhouette / gene_modules / "
                  "enrichment / recovery / semantic_alignment")
        else:
            self.benchmark_depth()
            self.genes_clustermap()
            self.benchmark_silhouette(resolution=1.0, n_top=cluster_n_top, top_per_cluster=cluster_top_per_cluster)
            self.benchmark_gene_modules(n_top=cluster_n_top)
            if hallmark_gmt:
                print("\n" + "=" * 60)
                print("HALLMARK ENRICHMENT ANALYSIS")
                print("=" * 60)
                enrichment_hallmark = self.benchmark_enrichment(hallmark_gmt, n_top=enrichment_n_top)
                self.benchmark_program_recovery(enrichment_hallmark)
                self.get_factor_labels(enrichment_hallmark)
            if gobp_gmt:
                print("\n" + "=" * 60)
                print("GO BP ENRICHMENT ANALYSIS")
                print("=" * 60)
                enrichment_gobp = self.benchmark_enrichment(gobp_gmt, n_top=enrichment_n_top, plot=False)
                self.benchmark_program_recovery(enrichment_gobp)
            if semantic_map is not None:
                print("\n" + "=" * 60)
                print("SEMANTIC ALIGNMENT ANALYSIS")
                print("=" * 60)
                self.benchmark_semantic_alignment(semantic_map)

        # MSigDB-only per-projection + per-cluster (no LLM)
        if lib1_gmt is not None or lib2_gmt is not None:
            print("\n" + "=" * 60)
            print("PER-PROJECTION BIOLOGY (MSigDB only — figures stage)")
            print("=" * 60)
            self.benchmark_per_projection_biology(
                lib1_gmt=lib1_gmt,
                lib2_gmt=lib2_gmt,
                n_top=per_projection_n_top,
                enable_llm=False,
            )
            print("\n" + "=" * 60)
            print("PER-CLUSTER BIOLOGY (MSigDB only — figures stage)")
            print("=" * 60)
            self.benchmark_per_cluster_biology(
                lib1_gmt=lib1_gmt,
                lib2_gmt=lib2_gmt,
                n_top=cluster_n_top,
                n_top_per_cluster=per_projection_n_top,
                enable_llm=False,
            )

        # Persist gene groups so grading can be done later without reloading models.
        self._save_gene_groups(n_top=per_projection_n_top, cluster_n_top=cluster_n_top, top_per_cluster=cluster_top_per_cluster)

    def run_grading(
        self,
        lib1_gmt=None,
        lib2_gmt=None,
        per_projection_n_top=50,
        cluster_n_top=500,
        llm_cache_dir=None,
    ):
        """Add Sonnet + Haiku scores on top of the artifacts produced by
        :meth:`run_figures`. Fully resumable: anything already in the cache is
        reused and only missing gene-list hashes hit the API.

        ``cluster_n_top`` must match the value passed to ``run_figures`` so the
        cached clusters line up.
        """
        if lib1_gmt is None and lib2_gmt is None:
            print("  ! run_grading needs at least lib1_gmt or lib2_gmt; skipping.")
            return
        print("\n" + "=" * 60)
        print("PER-PROJECTION BIOLOGY (LLM + MSigDB — grading stage)")
        print("=" * 60)
        self.benchmark_per_projection_biology(
            lib1_gmt=lib1_gmt,
            lib2_gmt=lib2_gmt,
            n_top=per_projection_n_top,
            enable_llm=True,
            llm_cache_dir=llm_cache_dir,
        )
        print("\n" + "=" * 60)
        print("PER-CLUSTER BIOLOGY (LLM + MSigDB — grading stage)")
        print("=" * 60)
        self.benchmark_per_cluster_biology(
            lib1_gmt=lib1_gmt,
            lib2_gmt=lib2_gmt,
            n_top=cluster_n_top,
            n_top_per_cluster=per_projection_n_top,
            enable_llm=True,
            llm_cache_dir=llm_cache_dir,
        )

    def _save_gene_groups(self, n_top=50, cluster_n_top=500, top_per_cluster=None):
        """Persist per-model top genes (projections + 2-method clusters) as JSON.

        File: ``out_dir/<model_name>.gene_groups.json`` — consumed by
        ``run_grading`` and the combined-report builder.

        ``cluster_n_top`` sizes the top-gene pool used to build the UMAP+Leiden
        and hierarchical clusters before per-cluster top symbols are extracted.
        ``top_per_cluster`` forwards the Leiden per-cluster gate to
        ``_get_gene_clusters`` (hierarchical path is unchanged).
        """
        import json
        for name, model in self.models.items():
            projections = self._factor_top_symbols(model, n_top) or {}
            clusters = {}

            adata_genes = self._get_gene_clusters(name, model, n_top=cluster_n_top, resolution=1.0, top_per_cluster=top_per_cluster)
            if adata_genes is not None:
                cdf = self._leiden_to_cluster_df(adata_genes)
                if cdf is not None:
                    clusters["umap_leiden"] = self._top_symbols_per_cluster(
                        cdf, "Cluster", n_top_per_cluster=n_top, min_cluster_size=5
                    )

            cdf_h = self._get_hierarchical_clusters(name, model, n_top=cluster_n_top, max_k=20)
            if cdf_h is not None:
                clusters["hierarchical"] = self._top_symbols_per_cluster(
                    cdf_h, "Cluster", n_top_per_cluster=n_top, min_cluster_size=5
                )

            payload = {
                "model_name": name,
                "n_genes": n_top,
                "projections": {str(k): v for k, v in projections.items()},
                "clusters": clusters,
            }
            out_path = self.out_dir / f"{name}.gene_groups.json"
            with out_path.open("w") as fh:
                json.dump(payload, fh)
            print(f"  > Wrote {out_path}")

    # =========================================================================
    # Legacy combined entry point (kept for backward compatibility)
    # =========================================================================

    def run_all(
        self,
        hallmark_gmt=None,
        gobp_gmt=None,
        semantic_map=None,
        lib1_gmt=None,
        lib2_gmt=None,
        enable_llm=True,
        llm_cache_dir=None,
        per_projection_n_top=50,
        cluster_n_top=500,
        cluster_top_per_cluster=None,
        enrichment_n_top=100,
        quick_mode=False,
    ):
        """
        Run all benchmarks.

        Parameters
        ----------
        hallmark_gmt : str, optional
            Path to Hallmark GMT file for enrichment analysis
        gobp_gmt : str, optional
            Path to GO BP GMT file for detailed enrichment
        semantic_map : torch.Tensor, optional
            Semantic map for alignment scoring
        lib1_gmt, lib2_gmt : str, optional
            Paths to the two MSigDB libraries used by
            ``benchmark_per_projection_biology`` (immune-related and
            monocyte-specific, respectively).
        enable_llm : bool
            If True, run Claude (Opus + Sonnet) per-projection scoring.
        llm_cache_dir : str, optional
            On-disk cache directory for LLM scores. Defaults to
            ``out_dir/_llm_cache``.
        per_projection_n_top : int
            Top genes per factor used for LLM and MSigDB spec/sens.
        cluster_n_top : int
            Pool of top-loaded genes fed into UMAP+Leiden and hierarchical
            clustering (drives ``benchmark_silhouette``, ``benchmark_gene_modules``,
            and the gene-group clusters stored by ``_save_gene_groups``).
        enrichment_n_top : int
            Per-factor top-gene list size used for the Hallmark / GO BP ORA
            enrichment (``benchmark_enrichment``).
        """
        if quick_mode:
            print("\n[quick_mode=True] skipping depth / silhouette / gene_modules / "
                  "enrichment / recovery / semantic_alignment")
        else:
            # Matrix QC
            self.benchmark_depth()

            # Program QC
            self.genes_clustermap()
            self.benchmark_silhouette(resolution=1.0, n_top=cluster_n_top, top_per_cluster=cluster_top_per_cluster)
            self.benchmark_gene_modules(n_top=cluster_n_top)

            # BIOLOGICAL ENRICHMENT (if GMT files provided)
            if hallmark_gmt:
                print("\n" + "="*60)
                print("HALLMARK ENRICHMENT ANALYSIS")
                print("="*60)
                enrichment_hallmark = self.benchmark_enrichment(hallmark_gmt, n_top=enrichment_n_top)
                self.benchmark_program_recovery(enrichment_hallmark)
                self.get_factor_labels(enrichment_hallmark)

            if gobp_gmt:
                print("\n" + "="*60)
                print("GO BP ENRICHMENT ANALYSIS")
                print("="*60)
                enrichment_gobp = self.benchmark_enrichment(gobp_gmt, n_top=enrichment_n_top, plot=False)
                self.benchmark_program_recovery(enrichment_gobp)

            # SEMANTIC ALIGNMENT (if semantic map provided)
            if semantic_map is not None:
                print("\n" + "="*60)
                print("SEMANTIC ALIGNMENT ANALYSIS")
                print("="*60)
                self.benchmark_semantic_alignment(semantic_map)

        # PER-PROJECTION BIOLOGY (4 metrics per factor)
        if lib1_gmt is not None or lib2_gmt is not None or enable_llm:
            print("\n" + "="*60)
            print("PER-PROJECTION BIOLOGY (LLM + MSigDB)")
            print("="*60)
            self.benchmark_per_projection_biology(
                lib1_gmt=lib1_gmt,
                lib2_gmt=lib2_gmt,
                n_top=per_projection_n_top,
                enable_llm=enable_llm,
                llm_cache_dir=llm_cache_dir,
            )

        # PER-CLUSTER BIOLOGY (4 metrics × 2 clustering methods)
        if lib1_gmt is not None or lib2_gmt is not None or enable_llm:
            print("\n" + "="*60)
            print("PER-CLUSTER BIOLOGY (LLM + MSigDB)")
            print("="*60)
            self.benchmark_per_cluster_biology(
                lib1_gmt=lib1_gmt,
                lib2_gmt=lib2_gmt,
                n_top=cluster_n_top,
                n_top_per_cluster=per_projection_n_top,
                enable_llm=enable_llm,
                llm_cache_dir=llm_cache_dir,
            )

    # =========================================================================
    # BIOLOGICAL BENCHMARKING - ORA Enrichment & Program Recovery
    # =========================================================================

    def _load_gmt(self, gmt_path):
        """
        Parse a GMT file and return {gene_set_name: set(genes)}.
        """
        gene_sets = {}
        with open(gmt_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    name = parts[0]
                    genes = set(g for g in parts[2:] if g)
                    gene_sets[name] = genes
        return gene_sets

    def benchmark_enrichment(self, gmt_path, n_top=100, q_threshold=0.05, plot=True):
        """
        ORA (Overrepresentation Analysis) enrichment of factor top genes.

        For each model and factor, tests enrichment against each gene set
        using Fisher's exact test with BH correction.

        Parameters
        ----------
        gmt_path : str
            Path to GMT file (e.g., Hallmark, GO BP)
        n_top : int
            Number of top genes per factor to test
        q_threshold : float
            FDR threshold for significance (default 0.05)
        plot : bool
            Whether to generate enrichment heatmaps

        Returns
        -------
        dict : {model_name: DataFrame} where DF is GeneSet × Factor with -log10(q)
        """
        from scipy.stats import fisher_exact
        from statsmodels.stats.multitest import multipletests

        print(f"\n--- Enrichment Analysis (ORA) ---")
        print(f"  > Gene sets from: {gmt_path}")

        # Parse GMT
        gene_sets = self._load_gmt(gmt_path)
        print(f"  > Loaded {len(gene_sets)} gene sets")

        # Universe = all genes in adata (mapped to symbols if available)
        if self.id_map:
            universe = set(self.id_map.get(g, g) for g in self.adata.var_names)
        else:
            universe = set(self.adata.var_names)

        results = {}

        for name, model in self.models.items():
            print(f"  > Processing {name}...")
            loadings = self._get_loadings(model)
            if loadings is None:
                continue

            pvals = []
            for factor_col in loadings.columns:
                # Get top genes for this factor
                top_genes_raw = loadings[factor_col].nlargest(n_top).index.tolist()
                if self.id_map:
                    top_genes = set(self.id_map.get(g, g) for g in top_genes_raw)
                else:
                    top_genes = set(top_genes_raw)

                for gs_name, gs_genes in gene_sets.items():
                    # Restrict to universe
                    gs_in_universe = gs_genes & universe
                    if len(gs_in_universe) == 0:
                        continue

                    # 2x2 contingency table
                    a = len(top_genes & gs_in_universe)  # overlap
                    b = len(top_genes - gs_in_universe)  # in factor, not in set
                    c = len(gs_in_universe - top_genes)  # in set, not in factor
                    d = len(universe - top_genes - gs_in_universe)  # neither

                    _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
                    pvals.append({
                        'Factor': factor_col,
                        'GeneSet': gs_name,
                        'p': p,
                        'overlap': a,
                        'set_size': len(gs_in_universe)
                    })

            if not pvals:
                continue

            df = pd.DataFrame(pvals)

            # BH correction
            _, q, _, _ = multipletests(df['p'], method='fdr_bh')
            df['q'] = q
            df['-log10q'] = -np.log10(df['q'] + 1e-300)

            # Pivot for heatmap: GeneSet (rows) × Factor (columns)
            pivot = df.pivot(index='GeneSet', columns='Factor', values='-log10q')
            results[name] = pivot

            # Save full results
            df.to_csv(self.out_dir / f'enrichment_{name}_full.csv', index=False)

            if plot:
                # Filter to significant pathways for visualization
                sig_threshold = -np.log10(q_threshold)
                sig_rows = pivot.max(axis=1) > sig_threshold

                if sig_rows.sum() > 0:
                    plot_data = pivot.loc[sig_rows]

                    # Limit to top 50 pathways if too many
                    if len(plot_data) > 50:
                        max_scores = plot_data.max(axis=1).sort_values(ascending=False)
                        plot_data = plot_data.loc[max_scores.head(50).index]

                    fig_height = max(8, len(plot_data) * 0.3)
                    g = sns.clustermap(
                        plot_data,
                        cmap='Reds',
                        figsize=(10, fig_height),
                        col_cluster=False,
                        yticklabels=True,
                        xticklabels=True
                    )
                    g.fig.suptitle(f'{name}: Enrichment (-log10 q)', y=1.02)
                    plt.savefig(self.out_dir / f'enrichment_{name}.png', bbox_inches='tight', dpi=150)
                    plt.show()
                else:
                    print(f"    No significant enrichments (q < {q_threshold}) for {name}")

        return results

    def benchmark_program_recovery(self, enrichment_results, q_threshold=0.05):
        """
        Compute program recovery score: fraction of gene sets recovered by each model.

        A gene set is 'recovered' if at least one factor has significant
        enrichment (q < threshold) for that gene set.

        Parameters
        ----------
        enrichment_results : dict
            Output from benchmark_enrichment()
        q_threshold : float
            FDR threshold for significance

        Returns
        -------
        DataFrame with columns: Model, Recovered, Total, Rate
        """
        print(f"\n--- Program Recovery Score ---")

        recovery = []
        sig_threshold = -np.log10(q_threshold)

        for model_name, pivot in enrichment_results.items():
            # Count gene sets where at least one factor is significant
            max_per_set = pivot.max(axis=1)
            recovered = (max_per_set > sig_threshold).sum()
            total = len(pivot)
            rate = recovered / total if total > 0 else 0

            print(f"  > {model_name}: {recovered}/{total} programs recovered ({rate:.1%})")

            recovery.append({
                'Model': model_name,
                'Recovered': recovered,
                'Total': total,
                'Rate': rate
            })

        df = pd.DataFrame(recovery)

        # Plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df, x='Model', y='Rate', palette='viridis')

        # Add count labels on bars
        for i, row in df.iterrows():
            ax.text(i, row['Rate'] + 0.02, f"{row['Recovered']}/{row['Total']}",
                   ha='center', fontsize=10)

        plt.ylabel('Program Recovery Rate')
        plt.title(f'Fraction of Gene Sets Recovered (q < {q_threshold})')
        plt.xticks(rotation=45, ha='right')
        plt.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        plt.ylim(0, min(1.1, df['Rate'].max() + 0.15))
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.out_dir / 'program_recovery.png', dpi=150)
        plt.show()

        # Save
        df.to_csv(self.out_dir / 'program_recovery.csv', index=False)

        return df

    def benchmark_semantic_alignment(self, semantic_map, n_top=100):
        """
        Measure factor alignment with semantic map structure.

        For each factor, compute weighted centroid of top genes in semantic space,
        then compute average cosine similarity of those genes to the centroid.
        Models with semantic loss should score higher.

        Parameters
        ----------
        semantic_map : torch.Tensor
            Shape (n_genes, d_semantic) - gene embeddings in semantic space
        n_top : int
            Number of top genes per factor to evaluate

        Returns
        -------
        DataFrame with columns: Model, Semantic_Alignment
        """
        import torch.nn.functional as F

        print(f"\n--- Semantic Alignment Score ---")

        results = []
        S = semantic_map.float()  # (n_genes, d_semantic)

        # Build gene index mapping
        gene_list = list(self.adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_list)}

        for name, model in self.models.items():
            loadings = self._get_loadings(model)
            if loadings is None:
                continue

            factor_scores = []
            for factor_col in loadings.columns:
                # Get top genes and their weights
                top_series = loadings[factor_col].nlargest(n_top)
                top_genes = top_series.index.tolist()

                # Get indices in semantic map
                gene_indices = []
                weights = []
                for g in top_genes:
                    if g in gene_to_idx:
                        gene_indices.append(gene_to_idx[g])
                        weights.append(top_series[g])

                if len(gene_indices) < 5:
                    continue

                # Normalize weights
                w = np.array(weights)
                w = w / w.sum()
                w_tensor = torch.tensor(w, dtype=torch.float32).unsqueeze(1)

                # Get semantic embeddings for top genes
                S_sub = S[gene_indices]  # (n_genes_found, d_semantic)

                # Weighted centroid
                centroid = (w_tensor * S_sub).sum(dim=0, keepdim=True)  # (1, d_semantic)

                # Cosine similarity of each gene to centroid
                sims = F.cosine_similarity(S_sub, centroid.expand(len(gene_indices), -1), dim=1)

                factor_scores.append(sims.mean().item())

            if factor_scores:
                avg_score = np.mean(factor_scores)
                print(f"  > {name}: {avg_score:.4f}")
                results.append({'Model': name, 'Semantic_Alignment': avg_score})

        df = pd.DataFrame(results)

        # Plot
        plt.figure(figsize=(10, 6))
        colors = ['#ff7f0e' if 'None' in m or 'none' in m.lower() else '#1f77b4'
                  for m in df['Model']]
        sns.barplot(data=df, x='Model', y='Semantic_Alignment', palette=colors)

        plt.ylabel('Semantic Alignment Score')
        plt.title('Factor Alignment with Semantic Map\n(Higher = More Coherent)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.out_dir / 'semantic_alignment.png', dpi=150)
        plt.show()

        # Save
        df.to_csv(self.out_dir / 'semantic_alignment.csv', index=False)

        return df

    def get_factor_labels(self, enrichment_results, q_threshold=0.05):
        """
        Assign biological labels to factors based on top enrichment.

        For each model and factor, finds the most significantly enriched gene set
        and assigns it as the factor's label.

        Parameters
        ----------
        enrichment_results : dict
            Output from benchmark_enrichment()
        q_threshold : float
            Only label factors with at least one significant enrichment

        Returns
        -------
        dict : {model_name: {factor: label}}
        """
        print(f"\n--- Factor Labels ---")

        sig_threshold = -np.log10(q_threshold)
        labels = {}

        for model_name, pivot in enrichment_results.items():
            labels[model_name] = {}

            for factor in pivot.columns:
                scores = pivot[factor].dropna()
                if len(scores) == 0:
                    continue

                best_set = scores.idxmax()
                best_score = scores.max()

                if best_score > sig_threshold:
                    # Clean up label (remove HALLMARK_ prefix, etc.)
                    label = best_set.replace('HALLMARK_', '').replace('_', ' ')
                    labels[model_name][factor] = label
                    print(f"  > {model_name} {factor}: {label} (q={10**(-best_score):.2e})")
                else:
                    labels[model_name][factor] = "Unlabeled"

        return labels

    # =========================================================================
    # PER-PROJECTION BIOLOGY (LLM + MSigDB spec/sens)
    # =========================================================================

    def _factor_top_symbols(self, model, n_top):
        """Return ``{factor_idx: [symbols]}`` for top-n genes per factor."""
        loadings = self._get_loadings(model)
        if loadings is None:
            return None
        out = {}
        for factor_col in loadings.columns:
            top_raw = loadings[factor_col].nlargest(n_top).index.tolist()
            if self.id_map:
                syms = [self.id_map.get(g, g) for g in top_raw]
            else:
                syms = list(top_raw)
            out[int(factor_col) if str(factor_col).isdigit() else factor_col] = syms
        return out

    @staticmethod
    def _best_program_match(genes_set, library):
        """argmax over programs of ``|p ∩ s|`` (ties broken by smaller |s|)."""
        best_name = None
        best_overlap = -1
        best_size = None
        for s_name, s_genes in library.items():
            ov = len(genes_set & s_genes)
            if ov > best_overlap or (ov == best_overlap and (best_size is None or len(s_genes) < best_size)):
                best_overlap = ov
                best_name = s_name
                best_size = len(s_genes)
        return best_name, best_overlap, best_size

    def benchmark_per_projection_biology(
        self,
        lib1_gmt=None,
        lib2_gmt=None,
        n_top=50,
        enable_llm=True,
        llm_cache_dir=None,
    ):
        """4 biological metrics per projection (factor) per model.

        Metrics
        -------
        - ``{persona}_score`` ∈ [0,100] for each LLM judge configured on this
          benchmark (default: ``claude_opus``, ``gemini_pro``)
        - ``lib1_spec`` = |p ∩ s*| / |p|, ``lib1_sens`` = |p ∩ s*| / |s*|
        - ``lib2_spec`` = |p ∩ s*| / |p|, ``lib2_sens`` = |p ∩ s*| / |s*|

        ``p`` = top-``n_top`` genes (by largest loading) for the factor, mapped
        to symbols. ``s*`` = MSigDB program with maximum |p ∩ s| in the library
        (ties broken by smaller |s|).
        """
        print("\n--- [Benchmark] Per-Projection Biology (LLM + MSigDB) ---")

        per_proj = self._collect_top_genes_per_factor(n_top)
        if not per_proj:
            print("  ! No models with extractable loadings; skipping.")
            return None

        # Partition: which models have a valid score cache for this level?
        # Cache hit = same gene-list fingerprint as last save.
        level_tag = "projection"
        # LLM gates caching: if grading is on, we need cached rows to include
        # LLM columns. If a cached entry was written without LLM scores and the
        # user now enables them, treat as miss for that model.
        hits, fresh, fingerprints = self._partition_by_cache(per_proj, level_tag)
        if enable_llm:
            expected_score_cols = {f"{p}_score" for p, _, _ in self._judges}
            for m in list(hits):
                cached_rows = hits[m]["rows"]
                cached_cols = {k for r in cached_rows for k in r}
                if cached_rows and not (expected_score_cols & cached_cols):
                    print(f"  > cache miss: {m} (needs LLM scores)")
                    fresh[m] = per_proj[m]
                    del hits[m]

        cached_rows = []
        cached_enrich = []
        for m, payload in hits.items():
            cached_rows.extend(payload["rows"])
            cached_enrich.extend(payload["enrichment"])

        # Compute fresh rows only for models that missed the cache.
        rows_fresh = self._init_per_projection_rows(fresh) if fresh else []

        if fresh:
            if enable_llm:
                self._score_per_projection_llm(fresh, rows_fresh, llm_cache_dir)
            else:
                print("  > LLM scoring disabled (enable_llm=False).")
            for label, gmt_path in [("lib1", lib1_gmt), ("lib2", lib2_gmt)]:
                if gmt_path is None:
                    print(f"  > Skipping {label} (no GMT path).")
                    continue
                self._score_per_projection_msigdb(fresh, rows_fresh, label, gmt_path)
        else:
            print(f"  > all {len(hits)} models cache-hit; skipping LLM + MSigDB compute")

        # HG enrichment for fresh models only
        libs = self._build_hg_libraries(lib1_gmt, lib2_gmt)
        enrich_fresh = pd.DataFrame()
        if libs and fresh:
            enrich_fresh = self._score_hg_enrichment_for_rows(
                fresh, rows_fresh, libs, id_col="Factor", level_tag=level_tag,
            )

        # Persist per-model cache for the freshly-computed models
        if fresh:
            enrich_fresh_records = enrich_fresh.to_dict("records") if not enrich_fresh.empty else []
            for m in fresh:
                rows_m = [r for r in rows_fresh if r["Model"] == m]
                enr_m = [r for r in enrich_fresh_records if r.get("Model") == m]
                self._save_score_cache(m, level_tag, fingerprints[m], rows_m, enr_m)
                print(f"  > cached scores: {m} ({level_tag})")

        # Merge cached + fresh
        rows = cached_rows + rows_fresh
        cached_enrich_df = pd.DataFrame(cached_enrich) if cached_enrich else pd.DataFrame()
        enrich_long = pd.concat([cached_enrich_df, enrich_fresh], ignore_index=True) if not enrich_fresh.empty or not cached_enrich_df.empty else pd.DataFrame()

        # Write per-library enrichment CSVs from the merged long-form
        if libs:
            self._write_enrichment_csvs(enrich_long, libs, level_tag)

        df = pd.DataFrame(rows)
        full_csv = self.out_dir / "per_projection_biology.csv"
        df.to_csv(full_csv, index=False)
        print(f"  > Wrote {full_csv} ({len(df)} rows)")

        summary = self._summarize_per_projection(df)
        summary_csv = self.out_dir / "per_projection_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"  > Wrote {summary_csv}")

        self._plot_per_projection(df, enrich_long)
        return df

    def _collect_top_genes_per_factor(self, n_top):
        per_proj = {}
        for name, model in self.models.items():
            top = self._factor_top_symbols(model, n_top)
            if top is not None:
                per_proj[name] = top
        return per_proj

    @staticmethod
    def _init_per_projection_rows(per_proj):
        rows = []
        for name, fg in per_proj.items():
            for factor in sorted(fg):
                rows.append({"Model": name, "Factor": factor})
        return rows

    @staticmethod
    def _default_judges():
        """Return ``[(persona_id, ScorerClass, ctor_kwargs), ...]``.

        Defaults:
        - Claude Opus via the ``claude -p`` CLI (uses your Claude Code subscription,
          no extra API charges).
        - Gemini 3.5 Flash via the Google API (free-tier friendly).

        Override by passing ``judges=`` to ``SemanticBenchmark(...)``.
        """
        try:
            from llm_scorers import ClaudeCLIScorer, GeminiAPIScorer
        except ImportError:
            from scripts.llm_scorers import ClaudeCLIScorer, GeminiAPIScorer
        return [
            ("claude_opus",   ClaudeCLIScorer, {"model": "opus"}),
            ("gemini_flash",  GeminiAPIScorer, {"model": "gemini-3.5-flash"}),
        ]

    _JUDGE_DISPLAY_NAMES = {
        "claude_opus":   "Claude Opus (CLI)",
        "gemini_flash":  "Gemini 3.5 Flash",
        "gemini_pro":    "Gemini 3.1 Pro",
        "claude_sonnet": "Claude Sonnet",
        "claude_haiku":  "Claude Haiku",
    }

    @classmethod
    def _judge_display(cls, persona_id):
        return cls._JUDGE_DISPLAY_NAMES.get(
            persona_id, persona_id.replace("_", " ").title()
        )

    def _score_per_projection_llm(self, per_proj, rows, llm_cache_dir):
        cache_dir = Path(llm_cache_dir) if llm_cache_dir else self.out_dir / "_llm_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        for persona, scorer_cls, ctor_kwargs in self._judges:
            scorer = scorer_cls(
                cache_path=cache_dir / f"{persona}_cache.pkl",
                cell_type_context=self.cell_type_context,
                **ctor_kwargs,
            )
            for model_name, fg in per_proj.items():
                tag = f"{persona}_{model_name}"
                print(f"  > LLM scoring [{tag}] ({len(fg)} factors)...")
                name_to_genes = {f"factor_{f}": genes for f, genes in fg.items()}
                try:
                    out = scorer.score_batch(name_to_genes)
                except Exception as e:
                    print(f"    ! {tag} failed: {e}; setting NaN")
                    out = {k: {"score": float("nan"), "program": None} for k in name_to_genes}
                for row in rows:
                    if row["Model"] != model_name:
                        continue
                    rec = out.get(f"factor_{row['Factor']}", {"score": float("nan"), "program": None})
                    if isinstance(rec, dict):
                        row[f"{persona}_score"] = rec.get("score", float("nan"))
                        row[f"{persona}_program"] = rec.get("program")
                    else:
                        row[f"{persona}_score"] = float(rec)
                        row[f"{persona}_program"] = None

    def _score_per_projection_msigdb(self, per_proj, rows, label, gmt_path):
        library = self._load_gmt(gmt_path)
        print(f"  > MSigDB {label}: {len(library)} programs from {gmt_path}")
        for row in rows:
            genes = per_proj[row["Model"]][row["Factor"]]
            p_set = set(genes)
            best_name, best_overlap, best_size = self._best_program_match(p_set, library)
            spec = best_overlap / len(p_set) if p_set else float("nan")
            sens = best_overlap / best_size if best_size else float("nan")
            row[f"{label}_spec"] = spec
            row[f"{label}_sens"] = sens
            row[f"{label}_best_program"] = best_name
            row[f"{label}_best_size"] = best_size
            row[f"{label}_overlap"] = best_overlap

    # ------------------------------------------------------------------
    # Per-model score cache — LLM + MSigDB + HG enrichment results are
    # keyed by a fingerprint of the per-(level, model) gene-list dict, so
    # reusing a trained model (force_train=False) reuses these scores too.
    # ------------------------------------------------------------------

    _SCORE_CACHE_DIR = "_score_cache"

    @staticmethod
    def _fingerprint(gene_lists):
        """Stable sha256 of ``{group_id: [symbols]}`` for cache keying."""
        import hashlib, json
        canon = {
            str(k): sorted(str(g) for g in v)
            for k, v in (gene_lists or {}).items()
        }
        blob = json.dumps(canon, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def _score_cache_path(self, model_name, level):
        return self.out_dir / self._SCORE_CACHE_DIR / f"{model_name}__{level}.json"

    def _load_score_cache(self, model_name, level, fingerprint):
        """Return cached ``{"rows": [...], "enrichment": [...]}`` if fingerprint
        matches; else ``None`` (no usable cache)."""
        import json
        path = self._score_cache_path(model_name, level)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"  ! cache read failed ({path.name}: {exc!r}); recomputing")
            return None
        if payload.get("fingerprint") != fingerprint:
            return None
        return {
            "rows": payload.get("rows", []),
            "enrichment": payload.get("enrichment", []),
        }

    def _save_score_cache(self, model_name, level, fingerprint, rows, enrichment):
        import json
        path = self._score_cache_path(model_name, level)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Drop non-JSON-serializable values (e.g., NaN floats become null safely).
        def _clean(v):
            if isinstance(v, float) and (v != v):  # NaN
                return None
            return v
        rows_safe = [{k: _clean(v) for k, v in r.items()} for r in rows]
        enr_safe = [{k: _clean(v) for k, v in r.items()} for r in enrichment]
        payload = {
            "fingerprint": fingerprint,
            "rows": rows_safe,
            "enrichment": enr_safe,
        }
        path.write_text(json.dumps(payload))

    def _partition_by_cache(self, per_group, level):
        """Return ``(hits, fresh, fingerprints)``.

        - ``hits``: ``{model: {"rows": [...], "enrichment": [...]}}`` for cache
          hits.
        - ``fresh``: ``{model: gene_lists}`` for models still needing compute.
        - ``fingerprints``: ``{model: hash}`` for every model (used at save).
        """
        hits, fresh, fingerprints = {}, {}, {}
        for model_name, gene_lists in per_group.items():
            fp = self._fingerprint(gene_lists)
            fingerprints[model_name] = fp
            cached = self._load_score_cache(model_name, level, fp)
            if cached is not None:
                hits[model_name] = cached
                print(f"  > cache hit: {model_name} ({level})")
            else:
                fresh[model_name] = gene_lists
        return hits, fresh, fingerprints

    def _build_hg_libraries(self, lib1_gmt, lib2_gmt):
        """Build the H / C2_immune / C7 library map from the two GMT paths.

        H + C2_immune come from splitting ``lib1_gmt`` by the ``HALLMARK_``
        prefix; C7 = full ``lib2_gmt``. Missing GMTs are dropped silently.
        """
        libs = {}
        if lib1_gmt:
            raw = self._load_gmt(lib1_gmt)
            h, c2 = _split_hallmark(raw)
            libs["H"] = h
            libs["C2_immune"] = c2
        if lib2_gmt:
            libs["C7"] = self._load_gmt(lib2_gmt)
        for label in libs:
            print(f"  > MSigDB {label}: {len(libs[label])} programs")
        return libs

    def _universe_symbols(self):
        if self.id_map:
            return set(self.id_map.get(g, g) for g in self.adata.var_names)
        return set(self.adata.var_names)

    @staticmethod
    def _hg_enrichment_rows(p_genes, library, universe):
        """For each gene set ``c`` in ``library``, compute fold-enrichment + HG p.

        ER = (|f ∩ c|/|c|) / (|f|/|G|) with f, c restricted to ``universe``.
        Returns a list of dicts (one per gene set; empty intersections skipped).
        """
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
            er = (overlap / n_c) / (n_f / G) if (n_c > 0 and n_f > 0) else float("nan")
            # one-sided hypergeometric: P(X >= overlap)
            pval = float(hypergeom.sf(overlap - 1, G, n_c, n_f))
            out.append({
                "gene_set": s_name,
                "overlap": overlap,
                "set_size": n_c,
                "factor_size": n_f,
                "G": G,
                "ER": er,
                "pvalue": pval,
            })
        return out

    def _score_hg_enrichment_for_rows(self, per_group, rows, libs, id_col, level_tag, q_thresh=0.05):
        """Add HG-enrichment columns to ``rows`` and return a long-form DataFrame.

        ``per_group`` maps ``model_name -> {row_id: [symbols]}`` (row_id is the
        factor id for projections, or the cluster id for clusters).
        ``id_col`` is the column name in ``rows`` carrying the row_id.
        ``level_tag`` is the CSV filename suffix (e.g. ``projection``,
        ``cluster_umap_leiden``).
        """
        universe = self._universe_symbols()
        long_rows = []

        # Tracks each (Model, library) BH-correction block; map row index ->
        # rec dict + p-value, finalize q-values when block is complete.
        for label, library in libs.items():
            # Collect (row_idx, pvals, recs) per Model so BH is per (Model, Lib)
            per_model_buf = {}
            for ridx, row in enumerate(rows):
                model_name = row["Model"]
                group_id = row[id_col]
                genes = per_group.get(model_name, {}).get(group_id)
                if genes is None:
                    continue
                recs = self._hg_enrichment_rows(genes, library, universe)
                per_model_buf.setdefault(model_name, []).append((ridx, group_id, recs))

            for model_name, entries in per_model_buf.items():
                # Flatten all p-values for this (model, library) to BH
                flat = []
                ranges = []
                for ridx, group_id, recs in entries:
                    start = len(flat)
                    flat.extend(r["pvalue"] for r in recs)
                    ranges.append((ridx, group_id, recs, start, len(flat)))
                if not flat:
                    for ridx, group_id, _recs in entries:
                        rows[ridx][f"{label}_n_sig"] = 0
                        rows[ridx][f"{label}_max_ER_sig"] = float("nan")
                    continue
                if multipletests is not None:
                    _, q_all, _, _ = multipletests(flat, method="fdr_bh")
                else:
                    # Fallback: no correction (matches existing pattern when
                    # statsmodels missing).
                    q_all = np.array(flat)

                for ridx, group_id, recs, start, end in ranges:
                    qs = q_all[start:end]
                    n_sig = 0
                    max_er_sig = float("nan")
                    for rec, q in zip(recs, qs):
                        rec_q = float(q)
                        sig = bool(rec_q < q_thresh)
                        if sig:
                            n_sig += 1
                            er = rec["ER"]
                            if er == er:  # not NaN
                                if max_er_sig != max_er_sig or er > max_er_sig:
                                    max_er_sig = er
                        long_rows.append({
                            "Model": model_name,
                            id_col: group_id,
                            "Library": label,
                            "gene_set": rec["gene_set"],
                            "overlap": rec["overlap"],
                            "set_size": rec["set_size"],
                            "factor_size": rec["factor_size"],
                            "G": rec["G"],
                            "ER": rec["ER"],
                            "pvalue": rec["pvalue"],
                            "qvalue": rec_q,
                            "significant": sig,
                        })
                    rows[ridx][f"{label}_n_sig"] = n_sig
                    rows[ridx][f"{label}_max_ER_sig"] = max_er_sig

        return pd.DataFrame(long_rows)

    def _write_enrichment_csvs(self, long_df, libs, level_tag):
        if long_df is None or long_df.empty:
            return
        for label in libs:
            sub = long_df[long_df["Library"] == label]
            if sub.empty:
                continue
            csv = self.out_dir / f"enrichment_{level_tag}_{label}.csv"
            sub.to_csv(csv, index=False)
            print(f"  > Wrote {csv} ({len(sub)} rows)")

    @staticmethod
    def _summarize_per_projection(df):
        cols = [c for c in df.columns if any(k in c for k in ("score", "spec", "sens"))]
        agg = df.groupby("Model")[cols].agg(["mean", "median"])
        agg.columns = [f"{c}_{s}" for c, s in agg.columns]
        agg = agg.reset_index()
        return agg

    def _plot_per_projection(self, df, enrich_long=None):
        models = df["Model"].unique().tolist()
        palette = sns.color_palette("tab10", len(models))
        color_map = dict(zip(models, palette))
        n_rows = max(len(df), 1)

        judge_cols = [
            (p, f"{p}_score", f"{p}_program", self._judge_display(p))
            for p, _, _ in self._judges
        ]
        has_llm = any(sc in df.columns for _, sc, _, _ in judge_cols)
        n_judges = max(len(judge_cols), 1)

        if has_llm:
            # 1) LLM scores (boxplots)
            fig, axes = plt.subplots(1, n_judges, figsize=(6.5 * n_judges, 5), squeeze=False)
            for ax_idx, (_, score_col, _, label) in enumerate(judge_cols):
                _safe_box(df, axes[0][ax_idx], score_col,
                          f"{label} score (per factor)", color_map)
            fig.suptitle("Per-Projection Biology — Stage 1/4: LLM scores")
            fig.tight_layout()
            out = self.out_dir / "01_per_projection_a_scores.png"
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")

            # 2) LLM interpretations (program labels)
            height = max(4, min(0.20 * n_rows + 1, 14))
            fig, axes = plt.subplots(1, n_judges, figsize=(6.5 * n_judges, height), squeeze=False)
            for ax_idx, (_, score_col, prog_col, label) in enumerate(judge_cols):
                ax = axes[0][ax_idx]
                if score_col in df.columns:
                    _render_text_panel(
                        ax,
                        _format_interpretation_lines(df, score_col, prog_col, "Factor", "Z_"),
                        f"{label} — what each projection is",
                    )
                else:
                    ax.axis("off")
                    ax.set_title(f"{label} — (not graded yet)")
            fig.suptitle("Per-Projection Biology — Stage 2/4: LLM interpretations")
            fig.tight_layout()
            out = self.out_dir / "02_per_projection_b_interpretations.png"
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")

        # 3) ER-distribution per library (H / C2_immune / C7)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        if enrich_long is None or enrich_long.empty:
            for ax, label in zip(axes, _HG_LIB_ORDER):
                ax.text(0.5, 0.5, "(no enrichment data)", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(_HG_LIB_TITLES.get(label, label))
        else:
            for ax, label in zip(axes, _HG_LIB_ORDER):
                _plot_enrichment_distribution(enrich_long, ax, label, color_map, models)
        fig.suptitle("Per-Projection Biology — Stage 3/4: HG enrichment (significant gene sets, q<0.05)")
        fig.tight_layout()
        out = self.out_dir / "03_per_projection_c_specsens.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")

        # 4) Top-10 by spec, with the matched MSigDB program names
        height = max(4, min(0.22 * n_rows + 1, 14))
        fig, axes = plt.subplots(1, 2, figsize=(13, height))
        _render_text_panel(
            axes[0],
            _format_top10_lines(df, "lib1", "Factor", "Z_", top_n=10),
            "Top-10 by lib1 spec — best matching C2/H program",
        )
        _render_text_panel(
            axes[1],
            _format_top10_lines(df, "lib2", "Factor", "Z_", top_n=10),
            "Top-10 by lib2 spec — best matching C7 program",
        )
        fig.suptitle("Per-Projection Biology — Stage 4/4: top spec/sens picks")
        fig.tight_layout()
        out = self.out_dir / "04_per_projection_d_top10.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")

    # =========================================================================
    # PER-CLUSTER BIOLOGY (LLM + MSigDB on gene clusters from 2 methods)
    # =========================================================================

    @staticmethod
    def _top_symbols_per_cluster(
        cluster_df, cluster_col="Cluster", n_top_per_cluster=50, min_cluster_size=5
    ):
        """For each cluster, return the top-``n`` symbols by ``max_loading``.

        Skips clusters with fewer than ``min_cluster_size`` genes. Cluster IDs
        are coerced to ``str`` so they sort cleanly in CSV output.
        """
        out = {}
        for cid, group in cluster_df.groupby(cluster_col):
            if len(group) < min_cluster_size:
                continue
            top = group.sort_values("max_loading", ascending=False).head(n_top_per_cluster)
            out[str(cid)] = top["Symbol"].tolist()
        return out

    def _leiden_to_cluster_df(self, adata_genes):
        """Normalize the Leiden AnnData into the same DataFrame shape as
        ``_get_hierarchical_clusters``: index = gene ID, columns =
        ``Cluster``, ``max_loading``, ``Symbol``.
        """
        if adata_genes is None:
            return None
        if "original_id" in adata_genes.obs.columns:
            ids = adata_genes.obs["original_id"].tolist()
        else:
            ids = adata_genes.obs_names.tolist()
        df = pd.DataFrame(
            {
                "Cluster": adata_genes.obs["leiden"].astype(str).tolist(),
                "max_loading": adata_genes.obs["max_loading"].values,
                "Symbol": adata_genes.obs_names.tolist(),
            },
            index=ids,
        )
        return df

    def benchmark_per_cluster_biology(
        self,
        lib1_gmt=None,
        lib2_gmt=None,
        n_top=500,
        n_top_per_cluster=50,
        enable_llm=True,
        llm_cache_dir=None,
        leiden_resolution=1.0,
        max_k=20,
        min_cluster_size=5,
    ):
        """4 biological metrics per gene cluster, for two clustering methods.

        Methods
        -------
        - ``umap_leiden``: Leiden clusters of top-``n_top`` genes in loading
          space (same as the existing gene-UMAP plot).
        - ``hierarchical``: agglomerative clusters with ``k`` chosen by
          silhouette (same as the existing module-heatmap plot).

        For each (method × model × cluster), uses the top-``n_top_per_cluster``
        genes within the cluster (by ``max_loading``) as ``p`` and computes
        LLM scores from every configured judge, lib1 spec/sens, lib2 spec/sens
        — same definitions as :meth:`benchmark_per_projection_biology`.
        """
        print("\n--- [Benchmark] Per-Cluster Biology (LLM + MSigDB) ---")

        # Step 1: collect cluster gene lists from both methods
        per_method = {}  # {method_name: {model_name: {cluster_id: [symbols]}}}
        for method_name in ("umap_leiden", "hierarchical"):
            per_method[method_name] = {}
            for model_name, model in self.models.items():
                if method_name == "umap_leiden":
                    adata_genes = self._get_gene_clusters(
                        model_name, model, n_top=n_top, resolution=leiden_resolution
                    )
                    cluster_df = self._leiden_to_cluster_df(adata_genes)
                else:
                    cluster_df = self._get_hierarchical_clusters(
                        model_name, model, n_top=n_top, max_k=max_k
                    )
                if cluster_df is None or cluster_df.empty:
                    print(f"  ! {method_name}/{model_name}: no clusters; skipping.")
                    continue
                cluster_to_genes = self._top_symbols_per_cluster(
                    cluster_df,
                    cluster_col="Cluster",
                    n_top_per_cluster=n_top_per_cluster,
                    min_cluster_size=min_cluster_size,
                )
                if cluster_to_genes:
                    per_method[method_name][model_name] = cluster_to_genes
                    print(
                        f"  > {method_name}/{model_name}: "
                        f"{len(cluster_to_genes)} clusters (size>={min_cluster_size})"
                    )

        if not any(per_method.values()):
            print("  ! No clusters across all methods; skipping per-cluster biology.")
            return None

        # Per-method: partition models by cache hit, compute only fresh, then
        # merge with cached. Each (method, model) gets its own cache entry.
        libs = self._build_hg_libraries(lib1_gmt, lib2_gmt)
        cache_dir = Path(llm_cache_dir) if llm_cache_dir else self.out_dir / "_llm_cache"
        if enable_llm:
            cache_dir.mkdir(parents=True, exist_ok=True)

        rows = []  # combined rows across methods
        enrich_long_by_method = {}

        for method_name, models_clusters in per_method.items():
            if not models_clusters:
                continue
            level_tag = f"cluster_{method_name}"
            hits, fresh, fingerprints = self._partition_by_cache(models_clusters, level_tag)
            # If user enabled LLM but cache has no LLM cols, miss those.
            if enable_llm:
                expected_score_cols = {f"{p}_score" for p, _, _ in self._judges}
                for m in list(hits):
                    cached_rows_m = hits[m]["rows"]
                    cached_cols = {k for r in cached_rows_m for k in r}
                    if cached_rows_m and not (expected_score_cols & cached_cols):
                        print(f"  > cache miss: {m} ({level_tag}, needs LLM scores)")
                        fresh[m] = models_clusters[m]
                        del hits[m]

            cached_rows = []
            cached_enrich = []
            for m, payload in hits.items():
                cached_rows.extend(payload["rows"])
                cached_enrich.extend(payload["enrichment"])

            # Build rows for fresh models only
            rows_fresh = []
            for model_name, ctg in fresh.items():
                for cid in sorted(ctg, key=lambda x: (len(x), x)):
                    rows_fresh.append({
                        "Method": method_name,
                        "Model": model_name,
                        "Cluster": cid,
                        "n_genes": len(ctg[cid]),
                    })

            # LLM scoring (one call per model × persona) — fresh only
            if fresh and enable_llm:
                for persona, scorer_cls, ctor_kwargs in self._judges:
                    scorer = scorer_cls(
                        cache_path=cache_dir / f"{persona}_cache.pkl",
                        cell_type_context=self.cell_type_context,
                        **ctor_kwargs,
                    )
                    for model_name, ctg in fresh.items():
                        tag = f"{persona}_{method_name}_{model_name}"
                        print(f"  > LLM scoring [{tag}] ({len(ctg)} clusters)...")
                        name_to_genes = {f"cluster_{cid}": genes for cid, genes in ctg.items()}
                        try:
                            out = scorer.score_batch(name_to_genes)
                        except Exception as e:
                            print(f"    ! {tag} failed: {e}; setting NaN")
                            out = {k: {"score": float("nan"), "program": None} for k in name_to_genes}
                        for row in rows_fresh:
                            if row["Model"] != model_name:
                                continue
                            rec = out.get(
                                f"cluster_{row['Cluster']}",
                                {"score": float("nan"), "program": None},
                            )
                            if isinstance(rec, dict):
                                row[f"{persona}_score"] = rec.get("score", float("nan"))
                                row[f"{persona}_program"] = rec.get("program")
                            else:
                                row[f"{persona}_score"] = float(rec)
                                row[f"{persona}_program"] = None
            elif not enable_llm:
                print(f"  > LLM scoring disabled (enable_llm=False) [{method_name}]")
            else:
                print(f"  > all {len(hits)} models cache-hit for {method_name}; skipping LLM + MSigDB compute")

            # MSigDB spec/sens (kept for top-10 panel) — fresh only
            if fresh:
                for label, gmt_path in [("lib1", lib1_gmt), ("lib2", lib2_gmt)]:
                    if gmt_path is None:
                        print(f"  > Skipping {label} (no GMT path).")
                        continue
                    library = self._load_gmt(gmt_path)
                    for row in rows_fresh:
                        genes = fresh[row["Model"]][row["Cluster"]]
                        p_set = set(genes)
                        best_name, best_overlap, best_size = self._best_program_match(p_set, library)
                        spec = best_overlap / len(p_set) if p_set else float("nan")
                        sens = best_overlap / best_size if best_size else float("nan")
                        row[f"{label}_spec"] = spec
                        row[f"{label}_sens"] = sens
                        row[f"{label}_best_program"] = best_name
                        row[f"{label}_best_size"] = best_size
                        row[f"{label}_overlap"] = best_overlap

            # HG enrichment over the 3-library split — fresh only
            enrich_fresh = pd.DataFrame()
            if libs and fresh:
                enrich_fresh = self._score_hg_enrichment_for_rows(
                    fresh, rows_fresh, libs, id_col="Cluster", level_tag=level_tag,
                )

            # Persist per-model cache for the freshly-computed models
            if fresh:
                enrich_fresh_records = enrich_fresh.to_dict("records") if not enrich_fresh.empty else []
                for m in fresh:
                    rows_m = [r for r in rows_fresh if r["Model"] == m]
                    enr_m = [r for r in enrich_fresh_records if r.get("Model") == m]
                    self._save_score_cache(m, level_tag, fingerprints[m], rows_m, enr_m)
                    print(f"  > cached scores: {m} ({level_tag})")

            # Merge cached + fresh for this method
            method_rows = cached_rows + rows_fresh
            rows.extend(method_rows)
            cached_enrich_df = pd.DataFrame(cached_enrich) if cached_enrich else pd.DataFrame()
            if not enrich_fresh.empty or not cached_enrich_df.empty:
                enrich_long_by_method[method_name] = pd.concat(
                    [cached_enrich_df, enrich_fresh], ignore_index=True
                )

        if not rows:
            print("  ! No rows produced; skipping per-cluster biology.")
            return None

        # Write per-library enrichment CSVs from the merged long-form
        if libs:
            for method_name, long_df in enrich_long_by_method.items():
                self._write_enrichment_csvs(long_df, libs, f"cluster_{method_name}")

        df = pd.DataFrame(rows)
        full_csv = self.out_dir / "per_cluster_biology.csv"
        df.to_csv(full_csv, index=False)
        print(f"  > Wrote {full_csv} ({len(df)} rows)")

        summary = self._summarize_per_cluster(df)
        summary_csv = self.out_dir / "per_cluster_summary.csv"
        summary.to_csv(summary_csv, index=False)
        print(f"  > Wrote {summary_csv}")

        for method_name in df["Method"].unique():
            sub = df[df["Method"] == method_name]
            self._plot_per_cluster(sub, method_name, enrich_long_by_method.get(method_name))
        return df

    @staticmethod
    def _summarize_per_cluster(df):
        cols = [c for c in df.columns if any(k in c for k in ("score", "spec", "sens"))]
        agg = df.groupby(["Method", "Model"])[cols].agg(["mean", "median"])
        agg.columns = [f"{c}_{s}" for c, s in agg.columns]
        agg = agg.reset_index()
        return agg

    # Numeric-prefix order used so the HTML report renders the per-cluster
    # stages right after the per-projection ones; per method we reserve a
    # 4-step block (a/b/c/d).
    _CLUSTER_PREFIX = {"umap_leiden": ("05", "06", "07", "08"),
                       "hierarchical": ("09", "10", "11", "12")}

    def _plot_per_cluster(self, df, method_name, enrich_long=None):
        if df.empty:
            return
        models = df["Model"].unique().tolist()
        palette = sns.color_palette("tab10", len(models))
        color_map = dict(zip(models, palette))
        n_rows = max(len(df), 1)
        prefixes = self._CLUSTER_PREFIX.get(method_name, ("99", "99", "99", "99"))

        judge_cols = [
            (p, f"{p}_score", f"{p}_program", self._judge_display(p))
            for p, _, _ in self._judges
        ]
        has_llm = any(sc in df.columns for _, sc, _, _ in judge_cols)
        n_judges = max(len(judge_cols), 1)

        if has_llm:
            # 1) LLM scores
            fig, axes = plt.subplots(1, n_judges, figsize=(6.5 * n_judges, 5), squeeze=False)
            for ax_idx, (_, score_col, _, label) in enumerate(judge_cols):
                _safe_box(df, axes[0][ax_idx], score_col,
                          f"{label} score (per cluster) — {method_name}", color_map)
            fig.suptitle(f"Per-Cluster Biology [{method_name}] — Stage 1/4: LLM scores")
            fig.tight_layout()
            out = self.out_dir / f"{prefixes[0]}_per_cluster_{method_name}_a_scores.png"
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")

            # 2) LLM interpretations
            height = max(4, min(0.18 * n_rows + 1, 16))
            fig, axes = plt.subplots(1, n_judges, figsize=(6.5 * n_judges, height), squeeze=False)
            for ax_idx, (_, score_col, prog_col, label) in enumerate(judge_cols):
                ax = axes[0][ax_idx]
                if score_col in df.columns:
                    _render_text_panel(
                        ax,
                        _format_interpretation_lines(df, score_col, prog_col, "Cluster", "C"),
                        f"{label} — what each {method_name} cluster is",
                    )
                else:
                    ax.axis("off")
                    ax.set_title(f"{label} — (not graded yet)")
            fig.suptitle(f"Per-Cluster Biology [{method_name}] — Stage 2/4: LLM interpretations")
            fig.tight_layout()
            out = self.out_dir / f"{prefixes[1]}_per_cluster_{method_name}_b_interpretations.png"
            fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")

        # 3) ER-distribution per library (H / C2_immune / C7)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
        if enrich_long is None or enrich_long.empty:
            for ax, label in zip(axes, _HG_LIB_ORDER):
                ax.text(0.5, 0.5, "(no enrichment data)", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(_HG_LIB_TITLES.get(label, label))
        else:
            for ax, label in zip(axes, _HG_LIB_ORDER):
                _plot_enrichment_distribution(enrich_long, ax, label, color_map, models)
        fig.suptitle(f"Per-Cluster Biology [{method_name}] — Stage 3/4: HG enrichment (significant gene sets, q<0.05)")
        fig.tight_layout()
        out = self.out_dir / f"{prefixes[2]}_per_cluster_{method_name}_c_specsens.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")

        # 4) Top-10 by spec, with MSigDB program names
        height = max(4, min(0.20 * n_rows + 1, 16))
        fig, axes = plt.subplots(1, 2, figsize=(13, height))
        _render_text_panel(
            axes[0],
            _format_top10_lines(df, "lib1", "Cluster", "C", top_n=10),
            f"Top-10 by lib1 spec — {method_name} clusters (C2/H)",
        )
        _render_text_panel(
            axes[1],
            _format_top10_lines(df, "lib2", "Cluster", "C", top_n=10),
            f"Top-10 by lib2 spec — {method_name} clusters (C7)",
        )
        fig.suptitle(f"Per-Cluster Biology [{method_name}] — Stage 4/4: top spec/sens picks")
        fig.tight_layout()
        out = self.out_dir / f"{prefixes[3]}_per_cluster_{method_name}_d_top10.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  > Wrote {out}")


def build_pathway_index(gmt_path, save_path=None):
    """
    Parses a GMT file and builds a reverse index (Gene -> Pathways).
    Returns a dictionary containing the index and metadata.
    """
    print(f"Building pathway index from: {gmt_path}")
    
    pathways = {}
    gene_to_pathways = {}
    
    with open(gmt_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                pname = parts[0]
                genes = [g for g in parts[2:] if g]
                
                # 1. Store forward map (Pathway -> Genes)
                pathways[pname] = set(genes)
                
                # 2. Store reverse map (Gene -> Pathways)
                for g in genes:
                    if g not in gene_to_pathways:
                        gene_to_pathways[g] = set()
                    gene_to_pathways[g].add(pname)
    
    index_data = {
        "pathways": pathways,
        "gene_to_pathways": gene_to_pathways,
        "M": len(pathways) # Total number of pathways
    }
    
    print(f"  > Indexed {len(pathways)} pathways.")
    print(f"  > Indexed {len(gene_to_pathways)} unique genes.")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(index_data, f)
        print(f"  > Saved index to {save_path}")

    return index_data


# ---------------------------------------------------------------------------
# SemanticSCVIVA (spatial) benchmark
# ---------------------------------------------------------------------------


class SemanticSpatialBenchmark(SemanticBenchmark):
    """Benchmark suite for :class:`scvi.external.scviva.SemanticSCVIVAModel`.

    Inherits every metric in :class:`SemanticBenchmark` (they all work via
    ``model.module.get_effective_loadings()`` and
    ``model.get_latent_representation()``, which :class:`SemanticSCVIVA`
    exposes with the same contract) and adds spatial/niche-aware metrics.

    Parameters
    ----------
    spatial_key
        ``adata.obsm`` key for spatial coordinates (e.g. ``"spatial"``).
    niche_composition_key
        ``adata.obsm`` key for per-cell neighborhood composition vectors.
    """

    def __init__(
        self,
        models_dict,
        adata,
        pathway_index,
        gene_mapping=None,
        out_dir="benchmark_results",
        spatial_key: str = "spatial",
        niche_composition_key: str = "neighborhood_composition",
    ):
        super().__init__(
            models_dict,
            adata,
            pathway_index,
            gene_mapping=gene_mapping,
            out_dir=out_dir,
        )
        self.spatial_key = spatial_key
        self.niche_composition_key = niche_composition_key

    def benchmark_niche_coherence(self, n_neighbors: int = 15):
        """Spatial smoothness of latent factors: mean inter-neighbor cosine similarity.

        For each model, compute K-nearest-neighbor cosine similarity of the
        latent representation in ``obsm``-spatial space. Higher = smoother.
        """
        from sklearn.neighbors import NearestNeighbors
        from sklearn.metrics.pairwise import cosine_similarity

        if self.spatial_key not in self.adata.obsm:
            print(f"  Skip niche coherence: obsm[{self.spatial_key!r}] missing.")
            return None

        coords = np.asarray(self.adata.obsm[self.spatial_key])
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        _, idx = nbrs.kneighbors(coords)
        neighbors = idx[:, 1:]  # drop self

        rows = []
        for name, model in self.models.items():
            z = np.asarray(model.get_latent_representation())
            sims = []
            for i, nbrs_i in enumerate(neighbors):
                sims.append(cosine_similarity(z[i : i + 1], z[nbrs_i]).mean())
            mean_sim = float(np.mean(sims))
            rows.append({"model": name, "niche_coherence": mean_sim})
            print(f"  {name}: niche_coherence = {mean_sim:.3f}")
        return pd.DataFrame(rows)

    def run_all(self, hallmark_gmt=None, gobp_gmt=None, semantic_map=None):
        """Core semantic metrics + spatial niche coherence."""
        super().run_all(
            hallmark_gmt=hallmark_gmt, gobp_gmt=gobp_gmt, semantic_map=semantic_map
        )
        print("\n" + "=" * 60)
        print("SPATIAL / NICHE METRICS")
        print("=" * 60)
        self.benchmark_niche_coherence()


# Backward-compatibility alias — prior entrypoints used ``ModelBenchmark``.
ModelBenchmark = SemanticBenchmark


