"""Generate the per-population modalities notebooks from the Haniffa-CD8 template.

Each output is byte-identical to benchmark_modalities_haniffa_cd8.ipynb EXCEPT:
  * the Cell-1 config header (RUN_NAME / DATASET_STEM / LIB2_NAME / BATCH_KEY /
    BIO_VARS / BATCH_VARS), and
  * the title markdown cell.
All outputs are cleared so the committed notebooks are clean; the user executes them
on the GPU kernel to populate results.

Run:  python notebooks/modalities/_gen_notebooks.py
"""
from __future__ import annotations
import copy
import json
from pathlib import Path

NB_DIR = Path("/home/projects/nyosef/zvise/scvi-tools-neural-nmf/notebooks")
TEMPLATE = NB_DIR / "benchmark_modalities_haniffa_cd8.ipynb"
MOD_DIR = NB_DIR / "modalities"
CONFIG_CELL_ID = "d5a67f2b"
TITLE_CELL_ID = "f733c50c"

HANIFFA_BIO = ["cell_type", "Status", "Status_on_day_collection_summary", "Sex"]
HANIFFA_BATCH = ["Site", "donor_id"]

# run_name -> config. (CD8 reference already exists as the template; not regenerated.)
RUNS = {
    "haniffa_cd4":    dict(stem="haniffa_cd4_clean",  lib2="lib2_cd4",      batch="Site",
                          bio=HANIFFA_BIO, batch_vars=HANIFFA_BATCH,
                          title="Haniffa CD4 T", note=""),
    "haniffa_mono":   dict(stem="haniffa_mono_clean", lib2="lib2_monocyte", batch="Site",
                          bio=HANIFFA_BIO, batch_vars=HANIFFA_BATCH,
                          title="Haniffa classical monocytes", note=""),
    "haniffa_nk":     dict(stem="haniffa_nk_clean",   lib2="lib2_nk",       batch="Site",
                          bio=HANIFFA_BIO, batch_vars=HANIFFA_BATCH,
                          title="Haniffa NK cells", note=""),
    "haniffa_b":      dict(stem="haniffa_b_clean",    lib2="lib2_bcell",    batch="Site",
                          bio=HANIFFA_BIO, batch_vars=HANIFFA_BATCH,
                          title="Haniffa B cells", note=""),
    "cellxgene_mono": dict(stem="monocytes_clean",    lib2="lib2_monocyte", batch="donor_id",
                          bio=["cell_type"], batch_vars=["donor_id"],
                          title="CELLxGENE classical monocytes",
                          note=("**Note:** this dataset has a single `cell_type` and `disease` "
                                "value, so the *biological* side of the metadata modality (mod5) "
                                "is degenerate (bio η²≈0 for every model) — the other four "
                                "modalities carry the comparison here.")),
    "tonsil_b":       dict(stem="tonsil_b_clean",     lib2="lib2_bcell",    batch="donor_id",
                          bio=["cell_type"], batch_vars=["donor_id", "assay"],
                          title="Tonsil B cells", note=""),
    "zheng_cd8":      dict(stem="zheng_cd8_clean",    lib2="lib2_cd8",      batch="donor_id",
                          bio=["cell_type", "disease"], batch_vars=["donor_id"],
                          title="Zheng pan-cancer CD8 T", note=""),
    # Perez et al. 2022 SLE lupus — disease (SLE/normal) + ancestry design factor.
    "perez_sle_mono": dict(stem="perez_sle_mono_clean", lib2="lib2_monocyte", batch="donor_id",
                          bio=["cell_type", "disease", "self_reported_ethnicity", "sex"],
                          batch_vars=["donor_id", "assay"],
                          title="Perez SLE classical monocytes",
                          note="**Note:** `disease` = SLE vs normal; `self_reported_ethnicity` = ancestry."),
    "perez_sle_cd4":  dict(stem="perez_sle_cd4_clean",  lib2="lib2_cd4",      batch="donor_id",
                          bio=["cell_type", "disease", "self_reported_ethnicity", "sex"],
                          batch_vars=["donor_id", "assay"],
                          title="Perez SLE CD4 T",
                          note="**Note:** `disease` = SLE vs normal; `self_reported_ethnicity` = ancestry."),
    # COMBAT blood atlas — `Source` disease group (healthy / COVID mild·severe·critical / sepsis / flu).
    "combat_mono":    dict(stem="combat_mono_clean",    lib2="lib2_monocyte", batch="donor_id",
                          bio=["cell_type", "Source"], batch_vars=["donor_id", "Pool_ID"],
                          title="COMBAT classical monocytes",
                          note="**Note:** `Source` = disease group (healthy / COVID mild·severe·critical / sepsis / influenza)."),
    "combat_cd8":     dict(stem="combat_cd8_clean",     lib2="lib2_cd8",      batch="donor_id",
                          bio=["cell_type", "Source"], batch_vars=["donor_id", "Pool_ID"],
                          title="COMBAT CD8 T",
                          note="**Note:** `Source` = disease group (healthy / COVID / sepsis / influenza)."),
}

# Cell-1 source with %%sentinels%% for the per-notebook config. Everything below the
# CONFIG block is identical to the template (verbatim).
CONFIG_CELL = '''# ============================================================
# Config — paths, hyperparameters (mirror the reference notebook), modality knobs.
# ============================================================
import hashlib, json
from pathlib import Path


def _find_nb_dir() -> Path:
    for p in [Path.cwd(), *Path.cwd().parents]:
        for cand in (
            p / "notebooks" / "benchmark_helpers.py",
            p / "benchmark_helpers.py",
            p / "scvi-tools-neural-nmf" / "notebooks" / "benchmark_helpers.py",
        ):
            if cand.exists():
                return cand.parent.resolve()
    raise FileNotFoundError(
        f"benchmark_helpers.py not found from {Path.cwd()}. "
        "Launch jupyter under the scvi-tools-neural-nmf tree, or set NB_DIR manually."
    )


def _slug(kwargs, max_epochs, warmup, kl_warmup, hvg, n=10):
    blob = json.dumps({"kwargs": dict(sorted(kwargs.items())), "max_epochs": max_epochs,
                       "warmup_epochs": warmup, "n_epochs_kl_warmup": kl_warmup,
                       "hvg_top_n": hvg}, default=str, sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:n]


NB_DIR = _find_nb_dir()

# ====== PER-NOTEBOOK CONFIG (only this block differs across the modalities notebooks) ======
RUN_NAME     = "%%RUN_NAME%%"        # -> benchmark_results/<RUN_NAME>_modalities, .model_cache_<RUN_NAME>
DATASET_STEM = "%%DATASET_STEM%%"    # <stem>.h5ad + <stem>_{geneformer,genept_3large}.pt in NB_DIR
LIB2_NAME    = "%%LIB2_NAME%%"       # NB_DIR/<LIB2_NAME>.gmt  (lib1_immune is universal)
BATCH_KEY    = %%BATCH_KEY%%          # batch / condition key for the VAEs + EXPIMAP
BIO_VARS     = %%BIO_VARS%%           # mod5 biological obs columns (higher assoc = better)
BATCH_VARS   = %%BATCH_VARS%%         # mod5 technical/batch obs columns (lower assoc = better)
# ===========================================================================================

ADATA_PATH = NB_DIR / f"{DATASET_STEM}.h5ad"
SEMANTIC_CACHE_GENEFORMER = NB_DIR / f"{DATASET_STEM}_geneformer.pt"
SEMANTIC_CACHE_GENEPT = NB_DIR / f"{DATASET_STEM}_genept_3large.pt"
GENEPT_PICKLE_PATH = NB_DIR / "GenePT_gene_protein_embedding_model_3_text.pickle"
LIB1_GMT = NB_DIR / "lib1_immune.gmt"
LIB2_GMT = NB_DIR / f"{LIB2_NAME}.gmt"
OUT_DIR = NB_DIR / "benchmark_results" / f"{RUN_NAME}_modalities"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_DIR = NB_DIR / f".model_cache_{RUN_NAME}"          # per-run model cache (fresh train)
STABILITY_DIR = MODEL_CACHE_DIR / "stability"                  # subsample retrains

# ---- Preprocessing / shared ----
HVG_TOP_N = 2500
HVG_FLAVOR = "seurat_v3"
SUBSAMPLE_N = 40000
N_LATENT = 10
GENE_MAPPING = ("feature_id", "feature_name")
MODEL_NAMES = ["semantic_geom", "semantic_genept", "ldvae_nn", "schpf_k10", "cnmf_k10", "expimap_k10"]

# ---- SemanticSCVI (Geneformer / GenePT) ----
SEM_GEOM_MAX_EPOCHS = SEM_GENEPT_MAX_EPOCHS = 200
SEM_GEOM_WARMUP = SEM_GENEPT_WARMUP = 20
SEM_GEOM_KL_WARMUP = SEM_GENEPT_KL_WARMUP = 100
_SEM_KWARGS = dict(loss_mode="geometric", coherence_weight=2000.0, n_gene_sample=1024,
                   n_latent=N_LATENT, n_layers=2, n_hidden=128, dropout_rate=0.1,
                   gene_likelihood="nb", weights_positive=True, use_batch_norm=False)
SEM_GEOM_KWARGS = dict(_SEM_KWARGS)
SEM_GENEPT_KWARGS = dict(_SEM_KWARGS)

# ---- LDVAE+ ----
LDVAE_MAX_EPOCHS = 250
LDVAE_KWARGS = dict(n_hidden=128, n_latent=N_LATENT, n_layers=1, dropout_rate=0.1,
                    gene_likelihood="nb", weights_positive=True, use_batch_norm=False)

# ---- scHPF ----
N_FACTORS = N_LATENT

# ---- cNMF (consensus NMF, Kotliar et al. 2019) ----
CNMF_N_ITER = 20                  # NMF replicates pooled into the consensus
CNMF_DENSITY_THRESHOLD = 0.5      # consensus outlier-spectra filter
CNMF_NUM_HIGHVAR_GENES = None     # None -> force cNMF onto ALL pipeline genes (genes_file)
CNMF_BETA_LOSS = "frobenius"
CNMF_INIT = "random"
CNMF_LOADINGS = "score"           # "score" (z-scored, canonical) or "tpm" (non-negative)

# ---- EXPIMAP ----
EXPIMAP_MASK_TERMS = [
    "HALLMARK_INTERFERON_GAMMA_RESPONSE", "HALLMARK_IL2_STAT5_SIGNALING",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB", "HALLMARK_INFLAMMATORY_RESPONSE",
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION", "HALLMARK_GLYCOLYSIS",
    "HALLMARK_MTORC1_SIGNALING", "HALLMARK_HYPOXIA",
    "HALLMARK_E2F_TARGETS", "HALLMARK_G2M_CHECKPOINT",
]
EXPIMAP_SOURCE_GMT = LIB1_GMT
EXPIMAP_MASK_GMT = NB_DIR / f"{ADATA_PATH.stem}_expimap_hallmark.gmt"
EXPIMAP_N_EPOCHS = 400
EXPIMAP_ALPHA = 0.7
EXPIMAP_ALPHA_KL = 0.5
EXPIMAP_ALPHA_EPOCH_ANNEAL = 130
EXPIMAP_KWARGS = dict(condition_key=BATCH_KEY, recon_loss="nb", hidden_layer_sizes=(256, 256))

# ---- Param-hash cache dirs (so a re-run LOADS instead of retraining) ----
SEM_GEOM_CACHE = MODEL_CACHE_DIR / "semantic_scvi" / _slug(
    {**SEM_GEOM_KWARGS, "batch_key": BATCH_KEY}, SEM_GEOM_MAX_EPOCHS, SEM_GEOM_WARMUP, SEM_GEOM_KL_WARMUP, HVG_TOP_N)
SEM_GENEPT_CACHE = MODEL_CACHE_DIR / "semantic_scvi_genept" / _slug(
    {**SEM_GENEPT_KWARGS, "batch_key": BATCH_KEY}, SEM_GENEPT_MAX_EPOCHS, SEM_GENEPT_WARMUP, SEM_GENEPT_KL_WARMUP, HVG_TOP_N)
LDVAE_CACHE = MODEL_CACHE_DIR / f"ldvae_nn_batch_{BATCH_KEY or 'none'}"
EXPIMAP_CACHE = MODEL_CACHE_DIR / "expimap_k10" / _slug(
    {**EXPIMAP_KWARGS, "terms": tuple(EXPIMAP_MASK_TERMS), "alpha": EXPIMAP_ALPHA,
     "alpha_kl": EXPIMAP_ALPHA_KL, "alpha_epoch_anneal": EXPIMAP_ALPHA_EPOCH_ANNEAL},
    EXPIMAP_N_EPOCHS, 0, EXPIMAP_ALPHA_EPOCH_ANNEAL, HVG_TOP_N)

# ---- cfg dict consumed by benchmark_modalities.train_all_models (stability retrains) ----
cfg = dict(
    N_LATENT=N_LATENT, BATCH_KEY=BATCH_KEY, N_FACTORS=N_FACTORS,
    CNMF_N_ITER=CNMF_N_ITER, CNMF_DENSITY_THRESHOLD=CNMF_DENSITY_THRESHOLD,
    CNMF_NUM_HIGHVAR_GENES=CNMF_NUM_HIGHVAR_GENES, CNMF_BETA_LOSS=CNMF_BETA_LOSS,
    CNMF_INIT=CNMF_INIT, CNMF_LOADINGS=CNMF_LOADINGS,
    SEM_GEOM_MAX_EPOCHS=SEM_GEOM_MAX_EPOCHS, SEM_GEOM_WARMUP=SEM_GEOM_WARMUP, SEM_GEOM_KL_WARMUP=SEM_GEOM_KL_WARMUP, SEM_GEOM_KWARGS=SEM_GEOM_KWARGS,
    SEM_GENEPT_MAX_EPOCHS=SEM_GENEPT_MAX_EPOCHS, SEM_GENEPT_WARMUP=SEM_GENEPT_WARMUP, SEM_GENEPT_KL_WARMUP=SEM_GENEPT_KL_WARMUP, SEM_GENEPT_KWARGS=SEM_GENEPT_KWARGS,
    LDVAE_MAX_EPOCHS=LDVAE_MAX_EPOCHS, LDVAE_KWARGS=LDVAE_KWARGS,
    EXPIMAP_MASK_GMT=str(EXPIMAP_MASK_GMT), EXPIMAP_SOURCE_GMT=str(EXPIMAP_SOURCE_GMT), EXPIMAP_MASK_TERMS=EXPIMAP_MASK_TERMS,
    EXPIMAP_N_EPOCHS=EXPIMAP_N_EPOCHS, EXPIMAP_ALPHA=EXPIMAP_ALPHA, EXPIMAP_ALPHA_KL=EXPIMAP_ALPHA_KL,
    EXPIMAP_ALPHA_EPOCH_ANNEAL=EXPIMAP_ALPHA_EPOCH_ANNEAL, EXPIMAP_KWARGS=EXPIMAP_KWARGS,
)

# ---- Force-retrain switches (full models) ----
FORCE_TRAIN = dict(geom=False, genept=False, ldvae=False, schpf=False, cnmf=False, expimap=False)

# ---- Modality knobs ----
SPARSITY_MASS_FRAC = 0.9     # mod1: #genes to reach this fraction of loading mass
SPARSITY_EPS = 1e-9          # mod1: |loading|>eps counts as "used" (structural-zero robust)
COLLIN_THRESHOLD = 0.5       # mod2: |r| above this counts as a redundant program pair
N_TOP = 30                   # mod4: top genes per factor for enrichment
Q_THRESH = 0.05              # mod4: BH-adjusted p threshold
ER_THRESH = 2.0              # mod4: fold-enrichment ("large effect") threshold
BEST_BY = "pval"             # mod4: best program per factor by "pval" (min p) or "ER" (max fold)
FRACTIONS = [0.75, 0.8, 0.9]  # mod3: cell subsample fractions
SEEDS = [0, 1]               # mod3: seeds per fraction

# mod4 held-out eval: drop EXPIMAP's mask programs from H/C2 (applied to ALL models) so its
# enrichment isn't circular — its factors ARE those HALLMARK terms. C7 is mask-disjoint.
if EXPIMAP_MASK_GMT.exists():
    MOD4_HELD_OUT_TERMS = [ln.split("\\t")[0] for ln in EXPIMAP_MASK_GMT.read_text().splitlines() if ln.strip()]
else:
    MOD4_HELD_OUT_TERMS = list(EXPIMAP_MASK_TERMS)

print(f"NB_DIR = {NB_DIR}")
print(f"RUN_NAME = {RUN_NAME} | ADATA = {ADATA_PATH.name} | LIB2 = {LIB2_GMT.name} | BATCH_KEY = {BATCH_KEY}")
print(f"OUT_DIR = {OUT_DIR}")
print(f"mod4 held-out terms ({len(MOD4_HELD_OUT_TERMS)}): {MOD4_HELD_OUT_TERMS}")
'''

TITLE_CELL = '''# Five-modality factor-model benchmark — %%TITLE%%

Runs the six-model factor benchmark (SemanticSCVI-Geneformer/GenePT, LDVAE+, scHPF,
cNMF, EXPIMAP) on `%%DATASET_STEM%%.h5ad` and compares them along five interpretable axes,
with all logic in `benchmark_modalities.py`:

1. **Gene sparsity** — few important genes per factor (good) vs all genes loaded.
2. **Program independence** — pairwise factor collinearity (some allowed, not too much).
3. **Stability** — retrain on fewer cells; the recovered representation should recur
   (rotation-invariant CCA on shared cells; legacy per-axis |corr| kept for reference).
4. **MSigDB best-program-per-factor** — fraction of factors whose best-fitting program is
   BH-significant **and** large-effect.
5. **Factor↔metadata** — factors should track biology, not batch.

Same pipeline as `benchmark_modalities_haniffa_cd8.ipynb`; **only the Cell-1 config**
(data paths, gene-set library, batch/metadata columns, cache dirs) differs. Models are
trained fresh into `.model_cache_%%RUN_NAME%%` on first run, then loaded.%%NOTE%%

Output: self-contained `benchmark_results/%%RUN_NAME%%_modalities/report_modalities.html`.
'''


def _clear_outputs(nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    nb.get("metadata", {}).pop("widgets", None)


def main():
    template = json.loads(TEMPLATE.read_text())
    cfg_idx = next(i for i, c in enumerate(template["cells"]) if c.get("id") == CONFIG_CELL_ID)
    title_idx = next(i for i, c in enumerate(template["cells"]) if c.get("id") == TITLE_CELL_ID)

    for run_name, cfg in RUNS.items():
        nb = copy.deepcopy(template)
        cfg_src = (CONFIG_CELL
                   .replace("%%RUN_NAME%%", run_name)
                   .replace("%%DATASET_STEM%%", cfg["stem"])
                   .replace("%%LIB2_NAME%%", cfg["lib2"])
                   .replace("%%BATCH_KEY%%", repr(cfg["batch"]))
                   .replace("%%BIO_VARS%%", repr(cfg["bio"]))
                   .replace("%%BATCH_VARS%%", repr(cfg["batch_vars"])))
        note = f"\n\n{cfg['note']}" if cfg["note"] else ""
        title_src = (TITLE_CELL
                     .replace("%%TITLE%%", cfg["title"])
                     .replace("%%DATASET_STEM%%", cfg["stem"])
                     .replace("%%RUN_NAME%%", run_name)
                     .replace("%%NOTE%%", note))
        nb["cells"][cfg_idx]["source"] = cfg_src.splitlines(keepends=True)
        nb["cells"][title_idx]["source"] = title_src.splitlines(keepends=True)
        _clear_outputs(nb)

        out_dir = MOD_DIR / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"benchmark_modalities_{run_name}.ipynb"
        out_path.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"wrote {out_path.relative_to(NB_DIR)}", flush=True)


if __name__ == "__main__":
    main()
