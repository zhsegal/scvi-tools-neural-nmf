"""Helpers for the skin-T TCR + inferCNV malignancy notebooks (14 and 15).

Moved out of the notebooks to keep them short. The notebooks orchestrate; this module
holds the mechanical blocks (TCR fold-in, dominant-clone rule, inferCNV prep/run,
GMM caller, MRVI smoothing, chromosome heatmaps).

Shared by both:
- 14_skin_T_tcr_cnv_malignancy  — shared HC/healthy diploid reference.
- 15_skin_cd4_cd8ref_cnv_malignancy — per-sample CD8 reference, CD4 query
  (`prepare_cd4_cd8ref_inputs`, `run_per_sample_cd8ref_infercnv`).
The TCR / GMM-caller / MRVI-smoothing / strategy-agreement / heatmap blocks are generic
across both; the heatmap fns take `shared_ref=None` when the reference is already in `acnv`.
"""
from __future__ import annotations

import gc
from pathlib import Path

import anndata as ad
import infercnvpy as cnv
import numpy as np
import pandas as pd
import scanpy as sc

# default reference benign-T (49-type) labels for the external integrated atlas
BENIGN_T = ["Tc", "Th", "Treg", "Tc17_Th17", "Tc_IL13_IL22"]


# ---------------------------------------------------------------- Step 1: load
def build_or_load_tcr_object(obj: Path, tcr_obj: Path, li_tcr: Path, H):
    """Load the TCR-complete T object, or build+cache it on first run.

    Build folds in Li2024/Haniffa TCR (parquet keyed by barcode after '|') then writes
    the ~12 GB cache. Returns adata with `cached_malignant` set.
    """
    if tcr_obj.exists():
        adata = sc.read_h5ad(tcr_obj)
        print("loaded cached TCR-complete object ->", tcr_obj.name, adata.shape)
    else:
        adata = sc.read_h5ad(obj)
        assert "raw_counts" in adata.layers, "expected raw_counts layer"
        bc_key = adata.obs_names.to_series().str.split("|", n=1).str[1]
        is_li = adata.obs["study"].astype(str).eq("li2024").to_numpy()
        li = pd.read_parquet(li_tcr)
        print("li2024 parquet:", li.shape, "| key overlap:",
              int(np.isin(bc_key[is_li], li.index).sum()), "/", int(is_li.sum()))
        li_bare = (li["tcr_clone_id"].where(li["has_tcr"].astype(bool), "").astype(str)
                   .str.split("::", n=1).str[-1])              # bare 'TRB:..' key
        adata.obs["_li_clone_key"] = ""
        adata.obs.loc[is_li, "_li_clone_key"] = bc_key[is_li].map(li_bare).fillna("").values
        has = adata.obs["has_tcr"].astype(bool).to_numpy().copy()
        has[is_li] = bc_key[is_li].map(li["has_tcr"].astype(bool)).fillna(False).values
        adata.obs["has_tcr"] = has
        adata.write_h5ad(tcr_obj)                              # HEAVY I/O (~12 GB)
        print("built + cached TCR-complete object ->", tcr_obj.name, adata.shape)

    adata.obs["cached_malignant"] = (adata.obs["cell_type"].astype(str) == "tumor_cell")
    print("\nstudy:\n", adata.obs["study"].value_counts())
    print("\nhas_tcr by study:\n",
          adata.obs.groupby("study", observed=True)["has_tcr"].agg(["size", "sum"]))
    print("cached tumor_cell:", int(adata.obs["cached_malignant"].sum()), "/", adata.n_obs)
    return adata


# ----------------------------------------------------- Step 2: TCR malignancy
def recompute_dominant_clone(adata, H, is_li, frac_thresh, ratio_thresh, expanded_min):
    """Unified TRB-primary clone key + per-donor dominant-clone malignancy.

    Sets obs: tcr_clone_id, tcr_clone_size, tcr_is_expanded, tcr_is_dominant_clone,
    tcr_is_malignant. Returns the per-donor dominance table.
    """
    tra = adata.obs["tra_cdr3"].astype(str).fillna("").values
    trb = adata.obs["trb_cdr3"].astype(str).fillna("").values
    key_cdr3 = np.array([H.clone_id_from_cdr3(a, b) for a, b in zip(tra, trb)])
    clone_key = np.where(is_li, adata.obs["_li_clone_key"].astype(str).values, key_cdr3)
    clone_key[~adata.obs["has_tcr"].to_numpy()] = ""          # only TCR+ cells carry a key
    adata.obs["tcr_clone_id"] = clone_key

    obs = adata.obs
    tcr = obs["has_tcr"].to_numpy() & (clone_key != "")
    df = pd.DataFrame({"donor": obs["donor"].astype(str).values,
                       "clone": clone_key}, index=obs.index)[tcr]

    clone_size = pd.Series(0, index=obs.index, dtype=int)
    is_dom = pd.Series(False, index=obs.index)
    dom_rows = []
    for d, sub in df.groupby("donor", sort=False):
        sizes = sub["clone"].value_counts()
        n = len(sub); top = sizes.index[0]; top_n = int(sizes.iloc[0])
        second_n = int(sizes.iloc[1]) if len(sizes) > 1 else 0
        dom_frac = top_n / max(1, n)
        ratio = top_n / second_n if second_n else np.inf
        is_dom_donor = (dom_frac >= frac_thresh) and (ratio >= ratio_thresh)
        clone_size.loc[sub.index] = sub["clone"].map(sizes).astype(int).values
        if is_dom_donor:
            is_dom.loc[sub.index[sub["clone"] == top]] = True
        dom_rows.append({"donor": d, "n_tcr": n, "top_clone": top, "top_n": top_n,
                         "dom_frac": round(dom_frac, 3),
                         "ratio": round(ratio, 2) if np.isfinite(ratio) else np.inf,
                         "is_dominant": is_dom_donor})

    adata.obs["tcr_clone_size"]        = clone_size.values
    adata.obs["tcr_is_expanded"]       = (clone_size >= expanded_min).values & tcr
    adata.obs["tcr_is_dominant_clone"] = is_dom.values
    adata.obs["tcr_is_malignant"]      = is_dom.values        # dominant clone == TCR malignant
    return pd.DataFrame(dom_rows).sort_values("donor")


def clone_summary_table(adata, frac_thresh, ratio_thresh):
    """Per-donor clone summary (largest clone, fold-change, malignant call)."""
    obs = adata.obs
    has = obs["has_tcr"].astype(str).isin(["True", "1"]).to_numpy()
    tcr = has & (obs["tcr_clone_id"].astype(str).to_numpy() != "")
    pc = pd.DataFrame({"donor": obs["donor"].astype(str).values,
                       "clone_id": obs["tcr_clone_id"].astype(str).values})[tcr]
    rows = []
    for sid, sub in pc.groupby("donor", sort=False):
        sizes = sub["clone_id"].value_counts()
        n = len(sub); top_n = int(sizes.iloc[0])
        second_n = int(sizes.iloc[1]) if len(sizes) > 1 else 0
        dom_frac = top_n / max(1, n); ratio = top_n / second_n if second_n else np.inf
        rows.append({
            "donor": sid, "n_tcr_cells": n, "n_clones": int(len(sizes)),
            "largest_clone": top_n, "second_clone": second_n,
            "fold_change": ratio, "dom_frac": dom_frac,
            "pass_frac (>=%.2f)" % frac_thresh: dom_frac >= frac_thresh,
            "pass_ratio (>=%.1f)" % ratio_thresh: ratio >= ratio_thresh,
            "malignant": (dom_frac >= frac_thresh) and (ratio >= ratio_thresh),
        })
    return pd.DataFrame(rows).sort_values("largest_clone", ascending=False).reset_index(drop=True)


# --------------------------------------------------- Step 3: inferCNV inputs
def _clean_ref(a, label):
    a = a.copy()
    a.X = a.layers["raw_counts"].copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    out = ad.AnnData(X=a.X.copy(),
                     obs=pd.DataFrame({"cnv_ref": label, "donor": f"{label.upper()}_REF",
                                       "cell_type": a.obs["cell_type"].astype(str).values},
                                      index=a.obs_names.astype(str)),
                     var=pd.DataFrame(index=a.var_names.astype(str)))
    out.var_names_make_unique()
    return out


def prepare_infercnv_inputs(adata, gtf: Path, integrated_h5: Path, seed: int,
                            n_hc_ref=5000, n_healthy_ref=3000, benign_t=BENIGN_T,
                            use_external=True):
    """Build the inferCNV query + shared diploid reference.

    Query = annotated T cells (cell_type_T != UNK) of disease-bearing donors. Reference =
    within-donor non-clonal T (`nonclonal`) + atlas HC T (`hc_atlas`, preferred) + external
    healthy/AD/pso benign-T (`healthy`, fallback). Adds genomic positions from `gtf`.
    Returns (acnv, shared_ref, cnv_donors, hc_donors).
    """
    ct      = adata.obs["cell_type"].astype(str)
    ctT     = adata.obs["cell_type_T"].astype(str)
    disease = adata.obs["disease"].astype(str)
    donor   = adata.obs["donor"].astype(str)

    hc_donors = sorted(donor[disease.eq("HC")].unique())
    qmask     = (~donor.isin(hc_donors)) & ctT.ne("UNK")
    cnv_donors = sorted(donor[qmask].unique())
    print("HC donors held out:", len(hc_donors), "| CNV (query) donors:", len(cnv_donors),
          "| query cells:", int(qmask.sum()))

    # ---- query: T cells of disease donors, log-normalised ----
    acnv = adata[qmask.to_numpy()].copy()
    acnv.X = acnv.layers["raw_counts"].copy()
    sc.pp.normalize_total(acnv, target_sum=1e4)
    sc.pp.log1p(acnv)
    acnv.obs["cnv_ref"] = "query"
    nonclonal = (acnv.obs["has_tcr"].to_numpy()
                 & ~acnv.obs["tcr_is_dominant_clone"].to_numpy()
                 & (acnv.obs["cell_type"].astype(str) != "tumor_cell").to_numpy())
    acnv.obs.loc[nonclonal, "cnv_ref"] = "nonclonal"

    # ---- atlas-internal HC T reference (same platform; preferred) ----
    rng = np.random.default_rng(seed)
    hc_idx = np.where((disease.eq("HC") & ctT.ne("UNK")).to_numpy())[0]
    if hc_idx.size > n_hc_ref:
        hc_idx = np.sort(rng.choice(hc_idx, n_hc_ref, replace=False))
    assert hc_idx.size > 0, "no HC T cells found for the hc_atlas reference"
    hc_ref = _clean_ref(adata[hc_idx], "hc_atlas")
    print(f"hc_atlas reference: {hc_ref.n_obs} cells")
    refs = [hc_ref]

    # ---- external healthy/AD/pso benign-T reference (fallback) ----
    if use_external and integrated_h5.exists():
        rf = sc.read_h5ad(integrated_h5, backed="r")
        rmask = (rf.obs["sample_type"].astype(str).isin(["healthy_skin", "AD", "psoriasis"])
                 & rf.obs["cell_type"].astype(str).isin(benign_t)).to_numpy()
        sel = np.where(rmask)[0]
        if sel.size > n_healthy_ref:
            sel = np.sort(rng.choice(sel, n_healthy_ref, replace=False))
        rsub = rf[sel].to_memory()
        healthy = ad.AnnData(
            X=rsub.layers["raw_counts"].copy(),
            obs=pd.DataFrame({"cnv_ref": "healthy", "donor": "HEALTHY_REF",
                              "cell_type": rsub.obs["cell_type"].astype(str).values},
                             index=rsub.obs_names.astype(str)),
            var=pd.DataFrame(index=pd.Index(rsub.var["genes"].astype(str).values)))
        rf.file.close()
        healthy.var_names_make_unique()
        sc.pp.normalize_total(healthy, target_sum=1e4)
        sc.pp.log1p(healthy)
        refs.append(healthy)
        print(f"external healthy reference: {healthy.n_obs} cells")

    # ---- common gene space + genomic positions; one shared reference ----
    common = acnv.var_names
    for r in refs:
        common = common.intersection(r.var_names)
    acnv = acnv[:, common].copy()
    cnv.io.genomic_position_from_gtf(gtf, adata=acnv, gtf_gene_id="gene_name")
    shared_ref = ad.concat([r[:, common] for r in refs], join="inner", index_unique=None)
    shared_ref.var = acnv.var.copy()                          # share genomic positions

    n_annot = int(acnv.var[["chromosome", "start", "end"]].notna().all(axis=1).sum())
    print(f"common genes: {len(common)} | with genomic position: {n_annot} / {acnv.n_vars}")
    assert n_annot > 8000, "too few genes annotated — check GTF gene_name / symbol intersection"
    print("query cells:", acnv.n_obs,
          "| within-donor nonclonal:", int((acnv.obs["cnv_ref"] == "nonclonal").sum()),
          "| shared ref:", shared_ref.n_obs, dict(shared_ref.obs["cnv_ref"].value_counts()))
    return acnv, shared_ref, cnv_donors, hc_donors


# ----------------------------------------------- Step 5: per-donor inferCNV
def run_per_donor_infercnv(acnv, shared_ref, cnv_donors, cache: Path, *, window=250,
                           topk_frac=0.10, leiden_res=2.0, n_jobs=8, chunk=2500, force=False):
    """Per-donor inferCNV vs the shared reference; focal per-cell score; cached to parquet.

    Fills acnv.obs: cnv_score (per cnv_leiden cluster), cnv_cell_score, cnv_focal_score,
    cnv_leiden. Reloads `cache` unless `force` or the cache is missing donors.
    """
    acnv.obs["cnv_score"]       = np.nan   # per cnv_leiden CLUSTER (heatmaps/diagnostics)
    acnv.obs["cnv_cell_score"]  = np.nan   # per CELL: genome-wide mean |X_cnv|
    acnv.obs["cnv_focal_score"] = np.nan   # per CELL: mean of top-K% |X_cnv| (drives the call)
    acnv.obs["cnv_leiden"]      = ""

    use_cache = (not force) and cache.exists()
    if use_cache:
        cc = pd.read_parquet(cache).set_index("obs_name")
        use_cache = ({"cnv_cell_score", "cnv_focal_score"}.issubset(cc.columns)
                     and set(cnv_donors).issubset(set(cc["donor"].astype(str).unique())))
    if use_cache:
        for col in ["cnv_score", "cnv_cell_score", "cnv_focal_score"]:
            acnv.obs[col] = cc[col].reindex(acnv.obs_names).to_numpy()
        acnv.obs["cnv_leiden"] = cc["cnv_leiden"].reindex(acnv.obs_names).fillna("").to_numpy()
        print(f"loaded cached inferCNV ({cache.name}) for {len(cnv_donors)} donors")
        return

    for d in cnv_donors:
        q = acnv[acnv.obs["donor"] == d]
        sub = ad.concat([q, shared_ref], join="inner", index_unique=None)
        sub.var = acnv.var.loc[sub.var_names].copy()          # restore genomic positions
        ref_cats = [c for c in ["nonclonal", "hc_atlas", "healthy"]
                    if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
        n_nc = int((q.obs["cnv_ref"] == "nonclonal").sum())
        print(f"[{d}] query={q.n_obs:>6}  nonclonal={n_nc:>5}  ref={ref_cats}")
        cnv.tl.infercnv(sub, reference_key="cnv_ref", reference_cat=ref_cats,
                        window_size=window, n_jobs=n_jobs, chunksize=chunk)
        cnv.tl.pca(sub)
        cnv.pp.neighbors(sub)
        cnv.tl.leiden(sub, resolution=leiden_res)             # finer CNV clusters
        cnv.tl.cnv_score(sub)                                 # per-cluster -> obs['cnv_score']
        Xc = sub.obsm["X_cnv"]
        Xc = np.abs(Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc))
        k  = max(1, int(round(topk_frac * Xc.shape[1])))
        sub.obs["cnv_cell_score"]  = Xc.mean(axis=1)
        sub.obs["cnv_focal_score"] = np.partition(Xc, Xc.shape[1] - k, axis=1)[:, -k:].mean(axis=1)
        del Xc
        keep = (sub.obs["donor"] == d).to_numpy()             # drop shared ref before stitching
        for col in ["cnv_score", "cnv_cell_score", "cnv_focal_score"]:
            acnv.obs.loc[sub.obs_names[keep], col] = sub.obs.loc[keep, col].to_numpy()
        acnv.obs.loc[sub.obs_names[keep], "cnv_leiden"] = (
            d + "_" + sub.obs.loc[keep, "cnv_leiden"].astype(str)).to_numpy()
        del sub, q
        gc.collect()
    out = acnv.obs[["donor", "cnv_score", "cnv_cell_score", "cnv_focal_score", "cnv_leiden"]].copy()
    out.index.name = "obs_name"
    out.reset_index().to_parquet(cache)
    print(f"computed inferCNV; cached -> {cache}")


# ------------------------------------------------- Step 6: malignancy callers
def diploid_mask(acnv):
    """Benign/diploid cells: within-donor non-clonal T, or any non-tumor non-dominant T."""
    return (acnv.obs["cnv_ref"].eq("nonclonal")
            | ((acnv.obs["cell_type"].astype(str) != "tumor_cell")
               & ~acnv.obs["tcr_is_dominant_clone"]))


def _best_f1_thr(score, y):                                   # F1-optimal cut over a 99-quantile grid
    ok = np.isfinite(score)
    if ok.sum() < 50 or not (0 < y[ok].sum() < ok.sum()):
        return np.nan, np.nan, int(ok.sum())
    s, yy = score[ok], y[ok].astype(bool)
    best_f1, best_t = 0.0, np.nan
    for t in np.unique(np.percentile(s, np.linspace(1, 99, 99))):
        yp = s >= t
        tp = int((yp & yy).sum()); fp = int((yp & ~yy).sum()); fn = int((~yp & yy).sum())
        pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn); f1 = 2 * pr * rc / max(1e-9, pr + rc)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1, int(ok.sum())


def call_per_donor(score_col, diploid, donors, obs, verbose=True, seed=0, thr_scale=1.0):
    """Per-donor 2-comp GMM crossover on log(score) + diploid-median floor.

    `thr_scale` < 1 lowers the final threshold (more cells malignant), > 1 raises it.
    Returns a bool ndarray aligned to `obs`. Falls back to ref-p90 / global-p90 when the
    GMM is degenerate. Prints per-donor diagnostics (incl. TCR-F1-optimal cut) if verbose.
    """
    from sklearn.mixture import GaussianMixture

    call   = pd.Series(False, index=obs.index)
    method = pd.Series("", index=obs.index, dtype=object)
    dip = diploid.to_numpy() if hasattr(diploid, "to_numpy") else np.asarray(diploid)
    for d in donors:
        m = (obs["donor"] == d).to_numpy()
        s = obs.loc[m, score_col].to_numpy(dtype=float)
        finite = np.isfinite(s) & (s > 0)
        ref_s = obs.loc[m & dip, score_col].to_numpy(dtype=float)
        ref_s = ref_s[np.isfinite(ref_s) & (ref_s > 0)]
        floor = np.nan
        if ref_s.size >= 20:
            ref_s = ref_s[ref_s <= np.percentile(ref_s, 90)]
            floor = float(np.median(ref_s))
        thr, meth = np.nan, "gmm_xover"
        if finite.sum() >= 50:
            x = np.log(s[finite]).reshape(-1, 1)
            gm = GaussianMixture(2, random_state=seed, n_init=3).fit(x)
            mu = gm.means_.ravel(); lo, hi = np.argsort(mu)
            n_hi = int((gm.predict(x) == hi).sum())
            if (np.exp(mu[hi] - mu[lo]) > 1.1) and n_hi >= 20:
                grid = np.linspace(mu[lo], mu[hi], 2001).reshape(-1, 1)
                post = gm.predict_proba(grid)[:, hi]
                thr = float(np.exp(grid[np.argmin(np.abs(post - 0.5)), 0]))
        if not np.isfinite(thr):
            thr = float(np.percentile(ref_s, 90)) if ref_s.size >= 20 else float(np.nanpercentile(s, 90))
            meth = "ref_p90_fallback" if ref_s.size >= 20 else "global_p90_fallback"
        if np.isfinite(floor):
            thr = max(thr, floor)
        thr = thr * thr_scale                                 # nudge the cut
        c = np.where(np.isfinite(s), s > thr, False)
        call.loc[obs.index[m]]   = c
        method.loc[obs.index[m]] = meth
        if verbose:
            tcr_m = m & obs["has_tcr"].to_numpy()
            yt = obs.loc[tcr_m, "tcr_is_malignant"].to_numpy().astype(bool)
            f1t, f1v, n_tcr = _best_f1_thr(obs.loc[tcr_m, score_col].to_numpy(dtype=float), yt)
            msg = f"  | TCR-F1opt thr={f1t:.4f} f1={f1v:.2f} n={n_tcr}" if np.isfinite(f1t) else ""
            print(f"[{d}] {meth:18s} thr={thr:.4f}  malignant={int(c.sum())}/{int(m.sum())}{msg}")
    return call.to_numpy(), method.to_numpy()


# ----------------------------------------- Step 6: MRVI-Leiden smoothing
def compute_mrvi_leiden(acnv, res, seed=0, key="mrvi_leiden", rep="X_mrvi_u"):
    """High-res Leiden on the MRVI latent (computed once, shared by the smoothed methods).

    HEAVY (neighbors + Leiden on ~400k cells) -> GPU kernel.
    """
    assert rep in acnv.obsm, f"{rep} missing on acnv (expected carried from adata)"
    sc.pp.neighbors(acnv, use_rep=rep, random_state=seed)
    sc.tl.leiden(acnv, resolution=res, random_state=seed, key_added=key,
                 flavor="igraph", n_iterations=2, directed=False)
    print(f"{acnv.obs[key].nunique()} MRVI clusters @res={res}")


def vote_cluster(acnv, call_col, frac, leiden_key="mrvi_leiden"):
    """Majority-vote a per-cell boolean call across MRVI Leiden clusters.

    A cluster is malignant if >= `frac` of its cells are True in `call_col`.
    Returns a bool ndarray aligned to acnv.obs.
    """
    cluster_frac = acnv.obs.groupby(leiden_key, observed=True)[call_col].mean()
    return acnv.obs[leiden_key].map(cluster_frac >= frac).astype(bool).to_numpy()


# ----------------------------------------------- Step 7: strategy comparison
def strategy_agreement(adata, strat_cols, tcr_col, avail_mask):
    """precision/recall/F1/Jaccard of each strategy vs the TCR call on available cells."""
    ref = adata.obs[tcr_col].to_numpy().astype(bool)
    rows = []
    for name, col in strat_cols.items():
        yp = adata.obs[col].to_numpy().astype(bool)[avail_mask]; yt = ref[avail_mask]
        tp = int((yp & yt).sum()); fp = int((yp & ~yt).sum()); fn = int((~yp & yt).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec); jac = tp / max(1, tp + fp + fn)
        rows.append(dict(strategy=name, n_called=int(yp.sum()), precision=round(prec, 3),
                         recall=round(rec, 3), f1=round(f1, 3), jaccard=round(jac, 3)))
    return pd.DataFrame(rows).set_index("strategy")


def tcr_cnv_quality(adata, strat_cols, tcr_col, avail_mask):
    """Pooled TCR<->CNV agreement per strategy on TCR+ cells.
    sensitivity = of TCR-malignant, fraction CNV-malignant;
    specificity = of TCR-non-malignant, fraction CNV-non-malignant."""
    ref = adata.obs[tcr_col].to_numpy().astype(bool)[avail_mask]
    rows = []
    for name, col in strat_cols.items():
        yp = adata.obs[col].to_numpy().astype(bool)[avail_mask]
        tp = int((yp & ref).sum());  fn = int((~yp & ref).sum())
        tn = int((~yp & ~ref).sum()); fp = int((yp & ~ref).sum())
        rows.append(dict(strategy=name,
                         n_tcr_malig=tp + fn, n_tcr_benign=tn + fp, tp=tp, tn=tn,
                         sensitivity=round(tp / max(1, tp + fn), 3),
                         specificity=round(tn / max(1, tn + fp), 3)))
    return pd.DataFrame(rows).set_index("strategy")


# ------------------------------------------------- Step 9: chromosome heatmaps
def _chr_separators(chr_pos):
    items = sorted(chr_pos.items(), key=lambda kv: kv[1])     # {chromosome: start_col}
    bounds = [v for _, v in items]
    labels = [k.replace("chr", "") for k, _ in items]
    return bounds, labels


def _heatmap_payload(sub, order_by):
    """Minimal arrays needed to draw a CNV heatmap, extracted from an inferCNV'd sub object.

    Caching this (rather than the AnnData) lets a re-run skip the per-donor inferCNV recompute.
    """
    q = sub[sub.obs["cnv_ref"] == "query"]
    Xc = q.obsm["X_cnv"]
    Xc = Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc)
    payload = {"obs_names": np.asarray(q.obs_names, dtype=object),
               "X_cnv": np.asarray(Xc, dtype=np.float32),
               "chr_pos": dict(sub.uns["cnv"]["chr_pos"])}
    if order_by == "recompute":
        payload["hm_leiden"] = q.obs["hm_leiden"].astype(str).to_numpy()
    return payload


def _save_payload(path: Path, payload):
    cp = payload["chr_pos"]
    arrs = {"obs_names": payload["obs_names"], "X_cnv": payload["X_cnv"],
            "chr_labels": np.asarray(list(cp.keys()), dtype=object),
            "chr_pos_val": np.asarray(list(cp.values()), dtype=np.int64)}
    if "hm_leiden" in payload:
        arrs["hm_leiden"] = payload["hm_leiden"]
    np.savez_compressed(path, **arrs)


def _load_payload(path: Path):
    d = np.load(path, allow_pickle=True)
    payload = {"obs_names": d["obs_names"], "X_cnv": d["X_cnv"],
               "chr_pos": {str(k): int(v) for k, v in zip(d["chr_labels"], d["chr_pos_val"])}}
    if "hm_leiden" in d.files:
        payload["hm_leiden"] = d["hm_leiden"]
    return payload


def _plot_cnv_heatmap(payload, donor, vlim, acnv, *, call_col, ann_cols, order_by, cmap, fig_dir,
                      fig_prefix="skin_T_cnv_heatmap", fmt="png", dpi=150):
    from matplotlib.colors import LogNorm, ListedColormap
    import matplotlib.pyplot as plt

    if fmt == "svg":
        plt.rcParams["svg.fonttype"] = "none"   # editable text; data layers rasterized below

    obs_names = pd.Index(payload["obs_names"])
    ann = acnv.obs.loc[obs_names, ann_cols]                   # query-only cols dropped by concat
    if order_by == "recompute":
        key = np.asarray(payload["hm_leiden"]).astype(str)    # this run's CNV clusters
        grp_label = "inferCNV leiden (this run)"
    else:
        key = ann["cnv_leiden"].astype(str).to_numpy()        # cached cluster = strat-3 unit
        grp_label = "cached cnv_leiden (strategy-3 unit)"
    codes = pd.factorize(key)[0]
    order = np.argsort(codes, kind="stable")
    ann = ann.iloc[order]
    Xc = np.asarray(payload["X_cnv"])[order]
    n  = Xc.shape[0]
    clu_bounds = np.where(np.diff(codes[order]) != 0)[0] + 1
    a_call = ann[call_col].to_numpy().astype(bool).astype(float)[:, None]
    a_tcr  = ann["tcr_is_malignant"].to_numpy().astype(bool).astype(float)[:, None]
    a_sz   = ann["tcr_clone_size"].to_numpy(dtype=float)
    a_sz   = np.where(a_sz > 0, a_sz, np.nan)[:, None]

    bounds, labels = _chr_separators(payload["chr_pos"])
    centers = [(bounds[i] + (bounds[i + 1] if i + 1 < len(bounds) else Xc.shape[1])) / 2
               for i in range(len(bounds))]
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 4, width_ratios=[40, 1, 1, 1.4], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    im0 = ax.imshow(Xc, aspect="auto", cmap=cmap, vmin=-vlim, vmax=vlim, interpolation="none",
                    rasterized=True)
    for b in bounds[1:]:
        ax.axvline(b, color="k", lw=0.4, alpha=0.5)
    for r in clu_bounds:
        ax.axhline(r - 0.5, color="k", lw=0.5, alpha=0.6)
    ax.set_xticks(centers); ax.set_xticklabels(labels, fontsize=5, rotation=90)
    ax.set_yticks([]); ax.set_ylabel(f"{n} query cells (grouped by {grp_label})")
    ax.set_title(f"inferCNV — {donor}: CNV clusters + {call_col} + TCR clone")
    ax.set_ylim(n - 0.5, -0.5)
    fig.colorbar(im0, ax=ax, fraction=0.015, pad=0.01, label="inferCNV")

    def _strip(gi, arr, title, cmap, norm=None, vmin=None, vmax=None, cbar=False):
        a = fig.add_subplot(gs[gi], sharey=ax)
        cmap = cmap.copy() if hasattr(cmap, "copy") else cmap
        if hasattr(cmap, "set_bad"):
            cmap.set_bad("white")
        im = a.imshow(arr, aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                      interpolation="none", rasterized=True)
        for r in clu_bounds:
            a.axhline(r - 0.5, color="k", lw=0.5, alpha=0.6)
        a.set_xticks([0]); a.set_xticklabels([title], fontsize=6, rotation=90); a.set_yticks([])
        if cbar:
            fig.colorbar(im, ax=a, fraction=0.6, pad=0.05)

    _strip(1, a_call, f"{call_col}", ListedColormap(["#e5e5e5", "#d62728"]), vmin=0, vmax=1)
    _strip(2, a_tcr, "TCR malig clone", ListedColormap(["#e5e5e5", "#111111"]), vmin=0, vmax=1)
    sz_norm = (LogNorm(vmin=max(1, np.nanmin(a_sz)), vmax=np.nanmax(a_sz))
               if np.isfinite(a_sz).any() else None)
    _strip(3, a_sz, "clone size", plt.cm.viridis, norm=sz_norm, cbar=True)

    fig.savefig(fig_dir / f"{fig_prefix}_{donor}.{fmt}", dpi=dpi, bbox_inches="tight", format=fmt)
    plt.show()


def run_cnv_heatmaps(acnv, shared_ref, *, call_col, fig_dir: Path, n_per_study=2,
                     window=150, vlim_scale=3.0, cmap="RdBu_r", order_by="cnv_leiden",
                     reference_cat=("nonclonal", "hc_atlas", "healthy"),
                     fig_prefix="skin_T_cnv_heatmap", n_jobs=8, chunk=2500,
                     fmt="png", dpi=150, cnv_cache_dir=None, force_cnv=False):
    """Re-run inferCNV (keeping X_cnv) for top-burden donors and draw annotated heatmaps.

    Picks the top `n_per_study` donors per study by TCR-malignant burden, then for each draws
    a chromosome heatmap with right-side strips: the chosen method's call (`call_col`),
    malignant-TCR-clone membership, and clone size. HEAVY (GPU kernel).

    `shared_ref` is concatenated to each per-donor query block (nb14). Pass `shared_ref=None`
    when the reference cells already live in `acnv` (nb15 same-sample CD8 ref). `reference_cat`
    lists the candidate `cnv_ref` categories to use as the inferCNV baseline (kept if >=20 cells).

    If `cnv_cache_dir` is given, each donor's heatmap inputs (X_cnv + obs_names + chr_pos) are
    cached to ``<cnv_cache_dir>/<fig_prefix>_<donor>.npz`` and reloaded on later runs, skipping
    the per-donor inferCNV recompute. The cache is keyed only by donor/`fig_prefix`; pass
    `force_cnv=True` to invalidate it after changing `window`/`order_by`/the cohort.
    """
    ann_cols = ["cnv_leiden", call_col, "tcr_is_malignant", "tcr_clone_size"]
    burden   = acnv.obs.groupby("donor", observed=True)["tcr_is_malignant"].sum()
    study_of = acnv.obs.groupby("donor", observed=True)["study"].first().astype(str)
    sel = pd.DataFrame({"study": study_of, "burden": burden})
    hm_donors = (sel.sort_values("burden", ascending=False)
                 .groupby("study", observed=True).head(n_per_study).index.tolist())
    print(f"heatmap donors ({n_per_study}/study, by TCR burden):", hm_donors,
          "| call_col =", call_col, "| ORDER_BY =", order_by)
    if cnv_cache_dir is not None:
        cnv_cache_dir = Path(cnv_cache_dir)
        cnv_cache_dir.mkdir(parents=True, exist_ok=True)
    for d in hm_donors:
        cache_f = (cnv_cache_dir / f"{fig_prefix}_{d}.npz") if cnv_cache_dir is not None else None
        if cache_f is not None and cache_f.exists() and not force_cnv:
            payload = _load_payload(cache_f)
            print(f"[{d}] loaded cached heatmap inputs ({cache_f.name})  X_cnv={payload['X_cnv'].shape}")
        else:
            q = acnv[acnv.obs["donor"] == d]
            if shared_ref is not None:
                sub = ad.concat([q, shared_ref], join="inner", index_unique=None)
                sub.var = acnv.var.loc[sub.var_names].copy()  # restore genomic positions
            else:
                sub = q.copy()                                # reference already in acnv
            ref_cats = [c for c in reference_cat
                        if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
            cnv.tl.infercnv(sub, reference_key="cnv_ref", reference_cat=ref_cats,
                            window_size=window, dynamic_threshold=None,
                            n_jobs=n_jobs, chunksize=chunk)
            if order_by == "recompute":
                cnv.tl.pca(sub); cnv.pp.neighbors(sub); cnv.tl.leiden(sub, key_added="hm_leiden")
            print(f"[{d}] n={sub.n_obs}  ref={ref_cats}")
            payload = _heatmap_payload(sub, order_by)
            if cache_f is not None:
                _save_payload(cache_f, payload)
                print(f"[{d}] cached heatmap inputs -> {cache_f.name}")
            del sub, q
            gc.collect()
        vlim = (float(np.nanpercentile(np.abs(payload["X_cnv"]), 99)) or 0.05) * vlim_scale
        st = acnv.obs.loc[acnv.obs["donor"] == d, "study"].iloc[0]
        print(f"[{st}/{d}] color vlim=±{vlim:.4f}")
        _plot_cnv_heatmap(payload, d, vlim, acnv, call_col=call_col, ann_cols=ann_cols,
                          order_by=order_by, cmap=cmap, fig_dir=fig_dir, fig_prefix=fig_prefix,
                          fmt=fmt, dpi=dpi)
        del payload
        gc.collect()


# ------------------------------------------- Step 10: per-cell arm-level CNV
# hg38 centromere positions (bp), used only to split each chromosome into p / q arms.
HG38_CENTROMERE = {
    "chr1": 123_400_000, "chr2": 93_900_000, "chr3": 90_900_000, "chr4": 50_000_000,
    "chr5": 48_800_000, "chr6": 59_800_000, "chr7": 60_100_000, "chr8": 45_200_000,
    "chr9": 43_000_000, "chr10": 39_800_000, "chr11": 53_400_000, "chr12": 35_500_000,
    "chr13": 17_700_000, "chr14": 17_200_000, "chr15": 19_000_000, "chr16": 36_800_000,
    "chr17": 25_100_000, "chr18": 18_500_000, "chr19": 26_200_000, "chr20": 28_100_000,
    "chr21": 12_000_000, "chr22": 15_000_000, "chrX": 61_000_000,
}


def _arm_labels(var, centromeres=HG38_CENTROMERE):
    """Map each gene (var row) to its chromosome arm, e.g. 'chr8q'. '' if unplaceable."""
    chrom = var["chromosome"].astype(str).to_numpy()
    start = pd.to_numeric(var["start"], errors="coerce").to_numpy()
    out = np.full(len(var), "", dtype=object)
    for i, (c, s) in enumerate(zip(chrom, start)):
        cen = centromeres.get(c)
        if cen is None or not np.isfinite(s):
            continue
        out[i] = f"{c}{'p' if s < cen else 'q'}"
    return out


def compute_arm_cnv_per_cell(acnv, cnv_donors, cache: Path, *, shared_ref=None,
                             reference_cat=("cd8_ref",), window=250, step=10,
                             n_jobs=8, chunk=100, min_genes_arm=15, force=False):
    """Per-sample inferCNV keeping per-gene CNV; aggregate to chromosome arms.

    For each sample runs `cnv.tl.infercnv(... calculate_gene_values=True)` so the per-gene CNV
    layer (`gene_values_cnv`, aligned to var_names, NaN where a gene isn't in any window) is
    available, then averages genes within each chromosome arm -> a compact `query_cells x ~39 arm`
    matrix. Only `cnv_ref == "query"` cells are kept.

    Reference: by default the same-sample CD8 cells already in `acnv` (nb15). Pass `shared_ref`
    (the strat_3 diploid reference) to concat it onto each per-donor query block instead, with
    `reference_cat` listing the candidate `cnv_ref` baseline categories (kept if >=20 cells).

    Cached to `cache` (parquet); reload unless `force` or the cache is missing donors. HEAVY
    (GPU kernel, per-sample inferCNV). Returns the arm DataFrame (index = obs_name, cols = arms + 'donor').
    """
    if (not force) and cache.exists():
        arm = pd.read_parquet(cache)
        have = set(arm["donor"].astype(str).unique()) if "donor" in arm else set()
        if set(map(str, cnv_donors)).issubset(have):
            print(f"loaded cached arm-CNV ({cache.name}) for {len(cnv_donors)} samples", arm.shape)
            return arm.set_index("obs_name") if "obs_name" in arm else arm

    arm_label_full = pd.Series(_arm_labels(acnv.var), index=acnv.var_names)
    keep_arms = sorted({a for a in arm_label_full.unique() if a})  # stable column order
    parts = []
    for d in cnv_donors:
        q = acnv[acnv.obs["donor"] == d]
        if shared_ref is not None:
            sub = ad.concat([q, shared_ref], join="inner", index_unique=None)
            sub.var = acnv.var.loc[sub.var_names].copy()       # restore genomic positions
        else:
            sub = q.copy()                                     # reference already in acnv
        ref_cats = [c for c in reference_cat
                    if int((sub.obs["cnv_ref"] == c).sum()) >= 20]
        cnv.tl.infercnv(sub, reference_key="cnv_ref", reference_cat=ref_cats,
                        window_size=window, step=step, n_jobs=n_jobs, chunksize=chunk,
                        calculate_gene_values=True)
        G = sub.layers["gene_values_cnv"]                      # cells x genes (NaN-filled)
        G = G.toarray() if hasattr(G, "toarray") else np.asarray(G)
        lab = arm_label_full.reindex(sub.var_names).to_numpy()
        qmask = (sub.obs["cnv_ref"] == "query").to_numpy()
        rows = {}
        for a in keep_arms:
            cols = np.where(lab == a)[0]
            if cols.size < min_genes_arm:
                continue
            with np.errstate(invalid="ignore"):
                rows[a] = np.nanmean(G[np.ix_(qmask, cols)], axis=1)
        df = pd.DataFrame(rows, index=sub.obs_names[qmask])
        df["donor"] = d
        parts.append(df)
        print(f"[{d}] query={int(qmask.sum()):>6}  arms={df.shape[1] - 1}")
        del sub, G
        gc.collect()
    arm = pd.concat(parts, axis=0)
    arm.index.name = "obs_name"
    arm.reset_index().to_parquet(cache)
    print(f"computed arm-CNV; cached -> {cache}  {arm.shape}")
    return arm


# ============================ nb15: CD4 query / same-sample CD8 reference ============================
def prepare_cd4_cd8ref_inputs(adata, gtf: Path, *, kept_lineages, min_cd8_ref=20, std_chr=None):
    """Build the CD4 inferCNV query + same-sample CD8 reference (nb15).

    Query = CD4 T cells; reference = CD8 T cells from the SAME sample. Keeps samples whose
    `lineage` is in `kept_lineages` and that have >= `min_cd8_ref` CD8 cells and >= 1 CD4 cell.
    Log-normalises, sets `cnv_ref` (cd8_ref / query) and `donor = sample_id`, adds genomic
    positions from `gtf`, and restricts to `std_chr` (drops the unpositioned 'chrnan' block +
    chrM that would otherwise dominate the heatmap and dilute cnv_cell_score). Returns
    (acnv, cnv_donors).
    """
    if std_chr is None:
        std_chr = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    lcol = "cell_type_T2" if "cell_type_T2" in adata.obs.columns else "cell_type_T"
    lin5 = adata.obs[lcol].astype(str).str.split("_").str[0]   # CD4 / CD8 / NK (or UNK)
    adata.obs["lin5"] = lin5.values
    kept_sample = adata.obs["lineage"].astype(str).isin(kept_lineages)
    sel = kept_sample.to_numpy() & lin5.isin(["CD4", "CD8"]).to_numpy()
    print("kept samples (lineage CD4/NA/unresolved):", adata.obs.loc[kept_sample, "sample_id"].nunique(),
          "/", adata.obs["sample_id"].nunique(),
          "| excluded (CD8/gamma_delta):", adata.obs.loc[~kept_sample, "sample_id"].nunique())

    # ---- query (CD4) + reference (CD8), log-normalised ----
    acnv = adata[sel].copy()
    acnv.X = acnv.layers["raw_counts"].copy()
    sc.pp.normalize_total(acnv, target_sum=1e4)
    sc.pp.log1p(acnv)
    acnv.obs["cnv_ref"] = np.where(acnv.obs["lin5"].to_numpy() == "CD8", "cd8_ref", "query")
    acnv.obs["donor"]   = acnv.obs["sample_id"].astype(str).values     # call_per_donor groups on 'donor'

    # ---- keep only samples with enough CD8 reference + at least one CD4 query ----
    n_cd8 = acnv.obs[acnv.obs["cnv_ref"] == "cd8_ref"].groupby("donor", observed=True).size()
    n_cd4 = acnv.obs[acnv.obs["cnv_ref"] == "query"].groupby("donor", observed=True).size()
    all_d = acnv.obs["donor"].unique()
    n_cd8 = n_cd8.reindex(all_d, fill_value=0); n_cd4 = n_cd4.reindex(all_d, fill_value=0)
    cnv_donors = sorted(d for d in all_d if n_cd8[d] >= min_cd8_ref and n_cd4[d] >= 1)
    dropped = sorted(set(map(str, all_d)) - set(cnv_donors))
    print(f"usable samples (CD8 ref >= {min_cd8_ref}): {len(cnv_donors)} | dropped (too few CD8): {len(dropped)}")
    if dropped:
        print("  dropped:", dropped)
    acnv = acnv[acnv.obs["donor"].isin(cnv_donors)].copy()

    # ---- genomic positions; drop unpositioned 'chrnan' / chrM (a huge near-zero window block) ----
    cnv.io.genomic_position_from_gtf(gtf, adata=acnv, gtf_gene_id="gene_name")
    acnv = acnv[:, acnv.var["chromosome"].astype(str).isin(std_chr)].copy()
    print(f"query (CD4) cells: {int((acnv.obs['cnv_ref']=='query').sum())} | "
          f"reference (CD8) cells: {int((acnv.obs['cnv_ref']=='cd8_ref').sum())} | "
          f"genes on chr1-22/X/Y: {acnv.n_vars}")
    assert acnv.n_vars > 8000, "too few positioned genes — check GTF gene_name / symbol match + 'chr' prefix"
    return acnv, cnv_donors


def run_per_sample_cd8ref_infercnv(acnv, cnv_donors, cache: Path, *, window=250, leiden_res=2.0,
                                   n_jobs=8, chunk=2500, std_chr=None, force=False):
    """Per-sample inferCNV with same-sample CD8 reference; cluster + score the CD4 query (nb15).

    Fills acnv.obs: cnv_cell_score (per CD4 cell: genome-wide mean |X_cnv|), cnv_leiden
    (sample-local CNV Leiden cluster), cnv_score (per-cnv_leiden group-mean of cnv_cell_score
    == infercnvpy cnv_score). Cached to `cache` (parquet, cols donor/cnv_ref/cnv_cell_score/
    cnv_leiden); reloads unless `force` or the cache is missing donors. HEAVY (GPU kernel).
    """
    if std_chr is None:
        std_chr = [f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]]
    acnv.obs["cnv_cell_score"] = np.nan   # per CD4 cell: genome-wide mean |X_cnv|
    acnv.obs["cnv_leiden"]     = ""       # per CD4 cell: sample-local cnv Leiden cluster

    use_cache = (not force) and cache.exists()
    if use_cache:
        cc = pd.read_parquet(cache).set_index("obs_name")
        use_cache = ({"cnv_cell_score", "cnv_leiden"}.issubset(cc.columns)
                     and set(cnv_donors).issubset(set(cc["donor"].astype(str).unique())))
    if use_cache:
        acnv.obs["cnv_cell_score"] = cc["cnv_cell_score"].reindex(acnv.obs_names).to_numpy()
        acnv.obs["cnv_leiden"]     = cc["cnv_leiden"].reindex(acnv.obs_names).fillna("").to_numpy()
        print(f"loaded cached inferCNV ({cache.name}) for {len(cnv_donors)} samples @res={leiden_res:g}")
    else:
        for d in cnv_donors:
            sub = acnv[acnv.obs["donor"] == d].copy()                          # CD4 query + same-sample CD8 ref
            sub = sub[:, sub.var["chromosome"].astype(str).isin(std_chr)].copy()   # drop unpositioned chrnan / chrM
            print(f"[{d}] CD4 query={int((sub.obs['cnv_ref']=='query').sum()):>6}  "
                  f"CD8 ref={int((sub.obs['cnv_ref']=='cd8_ref').sum()):>5}")
            cnv.tl.infercnv(sub, reference_key="cnv_ref", reference_cat=["cd8_ref"],
                            window_size=window, n_jobs=n_jobs, chunksize=chunk)
            subq = sub[sub.obs["cnv_ref"] == "query"].copy()                   # cluster + score the CD4 query only
            cnv.tl.pca(subq)
            cnv.pp.neighbors(subq)
            cnv.tl.leiden(subq, resolution=leiden_res)                         # -> subq.obs['cnv_leiden']
            Xc = subq.obsm["X_cnv"]
            Xc = np.abs(Xc.toarray() if hasattr(Xc, "toarray") else np.asarray(Xc))
            subq.obs["cnv_cell_score"] = Xc.mean(axis=1)
            del Xc
            acnv.obs.loc[subq.obs_names, "cnv_cell_score"] = subq.obs["cnv_cell_score"].to_numpy()
            acnv.obs.loc[subq.obs_names, "cnv_leiden"] = (
                d + "_" + subq.obs["cnv_leiden"].astype(str)).to_numpy()
            del sub, subq
            gc.collect()
        out = acnv.obs[["donor", "cnv_ref", "cnv_cell_score", "cnv_leiden"]].copy()
        out.index.name = "obs_name"
        out.reset_index().to_parquet(cache)
        print(f"computed inferCNV; cached -> {cache}")

    # per-cnv-cluster score = group-mean of per-cell cnv_cell_score (== infercnvpy cnv_score)
    _q = (acnv.obs["cnv_ref"] == "query").to_numpy()
    acnv.obs["cnv_score"] = np.nan
    acnv.obs.loc[_q, "cnv_score"] = (acnv.obs.loc[_q]
                                     .groupby("cnv_leiden", observed=True)["cnv_cell_score"].transform("mean"))
