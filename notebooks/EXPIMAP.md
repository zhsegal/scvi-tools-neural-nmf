# expiMap (scArches) — quick reference for Claude Code

**Purpose.** Use expiMap as a benchmark baseline against `SemanticSCVI`/`SemanticSCVIVAModel`. expiMap = interpretable conditional VAE with linear masked decoder where each latent dim is one gene program (GP). Paper: Lotfollahi et al., *Nat Cell Biol* 2023 (`s41556-022-01072-x`).

Read this file in full before writing any expiMap code. Skip the upstream docs — they redirect a lot and waste tokens. Everything below was verified against installed source `scArches==0.6.1` (PyPI, 2024-02-29 — current latest).

---

## 1. Install (CPU or GPU)

```bash
pip install scarches            # pulls torch, scvi-tools>=0.12.1, scanpy>=1.6, anndata>=0.7.4
```

For the project env: `mamba activate neural_nmf_env && pip install scarches`. No extras flag needed; expiMap ships in the base package. Python 3.9–3.11 supported.

Sanity check:

```python
import scarches as sca
print(sca.__version__)        # '0.6.1'
sca.models.EXPIMAP            # class exists
sca.utils.add_annotations     # mask builder exists
```

---

## 2. The four objects you'll touch

| Object | Where | What it does |
|---|---|---|
| `sca.utils.add_annotations` | `scarches/utils/annotations.py` | Reads GMT-style files, fills `adata.varm['I']` (binary `n_vars × n_terms`) and `adata.uns['terms']` (term names). |
| `sca.models.EXPIMAP` | `scarches/models/expimap/expimap_model.py` | High-level model. `__init__`, `train`, `get_latent`, `latent_enrich`, `term_genes`, `nonzero_terms`, `latent_directions`, `save`, `load`, `load_query_data`. |
| `sca.models.expimap.expiMap` | same dir, `expimap.py` | The underlying `nn.Module`. Don't touch unless extending. |
| `expiMapTrainer` | `scarches/trainers/expimap/regularized.py` | Constructed by `EXPIMAP.train`; takes the regularization hyperparams (alpha, alpha_l1, gamma_ext, beta). |

---

## 3. Golden-path script (reference training)

```python
import scanpy as sc
import scarches as sca

# --- 1. Data prep ---
adata = sc.read_h5ad("reference.h5ad")          # raw counts in .X
adata.layers["counts"] = adata.X.copy()         # keep raw — recon_loss='nb' needs counts
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="study", subset=True)
adata.X = adata.layers["counts"]                # restore raw counts in .X for training

# --- 2. Build the GP mask ---
# Reactome GMT from MSigDB (or PanglaoDB). min_genes=12 is the paper's filter.
sca.utils.add_annotations(
    adata,
    files="reactome.gmt",        # tab-delimited: term_name  gene1  gene2 ...
    min_genes=12,
    max_genes=None,
    varm_key="I",                # mask lands in adata.varm['I']
    uns_key="terms",             # term names in adata.uns['terms']
    clean=True,                  # strips 'REACTOME_' prefix, truncates to 30 chars
    genes_use_upper=True,
)
# Optional but recommended: drop genes that aren't in any GP, then re-HVG.

# --- 3. Construct & train ---
model = sca.models.EXPIMAP(
    adata,
    condition_key="study",       # batch column in adata.obs
    hidden_layer_sizes=[256, 256],
    recon_loss="nb",             # use raw counts; 'mse' if you have log-norm
    mask_key="I",                # picks up adata.varm['I']
    soft_mask=False,             # True = allow off-list genes via L1 (alpha_l1)
)

model.train(
    n_epochs=400,
    alpha_epoch_anneal=130,      # KL annealing (default; set None to disable)
    alpha_kl=0.5,                # KL weight (default 0.35; paper uses 0.5)
    alpha=0.7,                   # group-lasso weight (paper rec for 300–500 GPs)
    # alpha_l1=0.5,              # only if soft_mask=True
    # gamma_ext=0.5,             # only if n_ext>0 (de novo GPs)
    # beta=1.0,                  # HSIC weight (only if use_hsic=True)
)

model.save("models/expimap_ref", overwrite=True, save_anndata=False)
```

**Why each kwarg:** `alpha` drives entire latent columns to zero (GP pruning). `alpha_l1` lets soft-mask GPs absorb new genes. `gamma_ext` makes de novo (query-time) GPs sparse. `beta` enforces HSIC independence between de novo and reference GPs. Anneal anything you want with `*_epoch_anneal`.

**Defaults that bite you (verified in source):**
- If you don't pass `alpha_kl`, it gets set to `0.35` (was `1.0` pre-0.5.x) — model prints a warning. Set explicitly to `0.5` to match paper.
- If you don't pass `alpha_epoch_anneal`, it gets set to `min(130, n_epochs)` automatically.
- `recon_loss='nb'` requires raw counts in `adata.X`. Putting log-norm there silently produces garbage.

---

## 4. Reading results

```python
# All cells, latent matrix; only_active=True drops GPs killed by group lasso
Z = model.get_latent(mean=True, only_active=False)     # shape: (n_cells, n_GPs)
active = model.nonzero_terms()                          # ndarray of surviving GP indices

# Sign-correct latent so positive = upregulated; stored in adata.uns['directions']
model.latent_directions(method="sum", key_added="directions")

# Differential GP test (replaces "DE then GSEA")
model.latent_enrich(
    groups="cell_type",         # column in adata.obs OR a {group: [obs_names]} dict
    comparison="rest",          # or list of group names
    n_sample=5000,
    use_directions=True,
    directions_key="directions",
    exact=True,                 # closed-form Gaussian Bayes factor (faster, paper default)
    key_added="bf_scores",
)
# Results in adata.uns['bf_scores'][group_name] -> dict(p_h0, p_h1, bf)
# Paper threshold: |bf| >= 2.3 = "enriched" GP.

# Genes driving a specific GP (sorted by |decoder weight|, in_mask flag)
df = model.term_genes("INTERFERON_ALPHA_BETA_SIGNALING")  # str OR int index
# df: columns ['genes', 'weights', 'in_mask']
```

`adata.uns['terms']` is the canonical list of GP names — the i-th entry corresponds to the i-th column of the mask, the i-th latent dim, and the i-th column of `model.model.decoder.L0.expr_L.weight`. Use it to index everything.

---

## 5. Query → reference mapping (architecture surgery)

```python
query = sc.read_h5ad("query.h5ad")
# query.var_names MUST match reference order (load_query_data validates this)

q_model = sca.models.EXPIMAP.load_query_data(
    adata=query,
    reference_model="models/expimap_ref",   # path or in-memory model
    freeze=True,                             # freezes everything except batch & ext weights
    freeze_expression=True,                  # freeze first-layer non-batch weights
    unfreeze_ext=True,                       # train new GP weights
    new_n_ext=10,                            # de novo (unconstrained) GPs to add
    new_n_ext_m=0,                           # constrained new GPs (need new_ext_mask)
    # new_ext_mask=<np.ndarray>,             # required if new_n_ext_m > 0
)

q_model.train(
    n_epochs=200,                            # paper rec: smaller queries need MORE epochs
    early_stopping_kwargs={"early_stopping_metric": "val_loss", "patience": 20},
    alpha_kl=0.5,
    alpha=0.7,
    gamma_ext=0.5,                           # L1 on the 10 new GPs
    beta=1.0,                                # HSIC: keep new GPs independent of reference
    print_stats=True,                        # monitors soft-mask de-activation share
)

# Add names for the new GPs into adata.uns['terms']
q_model.update_terms(terms="terms")          # appends 'unconstrained_0' ... 'unconstrained_9'
```

Then run `get_latent` / `latent_enrich` on `q_model` exactly as in §4.

**Hyperparam rules of thumb (from the paper, §"Choice of hyperparameters"):**
- `alpha_kl` default 0.5: latent collapses to a blob → lower; bad integration → raise.
- `alpha`: 0.7 for 300–500 GPs, scale up for more.
- Query mapping: 200 epochs + early stopping; *small* queries need more epochs (counterintuitive).
- New unconstrained GPs: start with ≥10, use `gamma_ext` for sparsity.
- Soft mask: at end of training, `print_stats=True` should report "Share of de-activated inactive genes" > 0.9. If lower, raise `alpha_l1` and retrain.

---

## 6. Save / load

```python
model.save("dir", overwrite=True, save_anndata=False)
loaded = sca.models.EXPIMAP.load("dir", adata=adata)   # adata required if not saved
```

`load_query_data` accepts either a path string or a live `EXPIMAP` instance as `reference_model`.

---

## 7. EXPIMAP `__init__` — full signature (verified from source)

```python
EXPIMAP(
    adata,
    condition_key=None,                  # str column in adata.obs for batches
    conditions=None,                     # explicit list (else inferred from adata)
    hidden_layer_sizes=[256, 256],
    dr_rate=0.05,                        # dropout
    recon_loss='nb',                     # 'nb' or 'mse'
    use_l_encoder=False,                 # library-size encoder (only if decoder='softmax')
    use_bn=False, use_ln=True,
    mask=None, mask_key='I',             # one of these MUST resolve
    decoder_last_layer=None,             # 'softmax'|'softplus'|'exp'|'relu'|None
    soft_mask=False,
    n_ext=0,                             # de novo extension GPs (set in load_query_data)
    n_ext_m=0,                           # constrained extension GPs
    use_hsic=False, hsic_one_vs_all=False,
    ext_mask=None, soft_ext_mask=False,
)
```

Note: `latent_dim` is **not** a free parameter — it equals `len(mask) = adata.varm['I'].shape[1] = n_GPs`.

## 8. EXPIMAP.train — full signature

```python
model.train(
    n_epochs=400, lr=1e-3, eps=0.01,
    alpha=None,                          # group lasso (None = off)
    omega=None,                          # per-group weight tensor for alpha
    # passed via **kwargs to expiMapTrainer:
    alpha_kl=0.35,                       # KL weight; default printed as warning
    alpha_epoch_anneal=130,              # KL anneal epochs (auto-set)
    alpha_l1=None,                       # soft-mask L1 (None = off)
    alpha_l1_epoch_anneal=None, alpha_l1_anneal_each=5,
    gamma_ext=None,                      # L1 for new unconstrained GPs
    gamma_epoch_anneal=None, gamma_anneal_each=5,
    beta=1.0,                            # HSIC weight (only if use_hsic=True)
    print_stats=False,
    # plus standard scvi-trainer kwargs: batch_size, weight_decay,
    # early_stopping_kwargs, train_frac, n_workers, monitor_only, ...
)
```

---

## 9. Common errors / gotchas

- **`ValueError: Please provide mask.`** → you didn't run `add_annotations` or passed wrong `mask_key`.
- **All-NaN losses** → `recon_loss='nb'` with non-integer values in `.X`. Restore raw counts.
- **Latent collapses to one cluster** → lower `alpha_kl`. Try 0.1.
- **No GPs survive (`nonzero_terms()` returns empty)** → `alpha` too aggressive. Halve it.
- **Query gene order mismatch** → `_validate_var_names` reorders `query.var` to match the reference; if a reference gene is missing in query, the call errors. Pre-align with `query = query[:, ref.var_names].copy()` (after intersecting).
- **`load_query_data` complains about `ext_mask`** → if you pass `new_n_ext_m > 0` you must also pass `new_ext_mask` of shape `(new_n_ext_m, n_genes)`.

---

## 10. Wiring as a SemanticBenchmark baseline

Add a thin adapter in `scripts/benchmarking.py` that:
1. Calls `add_annotations` with the same Reactome/PanglaoDB GMT used as your semantic map.
2. Trains `EXPIMAP` on the same train split as `SemanticSCVI`.
3. Exposes a `.get_latent_representation()`-shaped method by wrapping `model.get_latent(mean=True)`.
4. For pathway-level metrics: use `latent_enrich(groups='cell_type')` → `adata.uns['bf_scores']`. Compare `|bf| >= 2.3` enriched terms to ground-truth perturbation pathways.
5. For integration metrics (cLISI/NMI/ARI/ASW/kBET/iLISI): feed `Z = model.get_latent(only_active=True, mean=True)` into the existing scIB pipeline you already use.

The reference implementation of the paper's full benchmarks lives at `github.com/theislab/expiMap_reproducibility` — clone it locally if you want the exact GMT preprocessing, HVG counts, and integration splits per dataset.

---

## 11. Source-of-truth files

When in doubt, read these (all in the installed package, no need to fetch from GitHub):
- `scarches/models/expimap/expimap_model.py` — `EXPIMAP` class (the high-level API, ~500 lines)
- `scarches/models/expimap/expimap.py` — `expiMap` nn.Module
- `scarches/trainers/expimap/regularized.py` — trainer + proximal operator implementations
- `scarches/utils/annotations.py` — `add_annotations` (only ~50 lines)
- `scarches/models/base/_base.py` — `BaseMixin.save/load`, `SurgeryMixin.load_query_data`

Don't fetch tutorials over the network on a first pass — the API in this doc is exhaustive for benchmarking work.
