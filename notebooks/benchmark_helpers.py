"""Helpers for ``four_way_benchmark.ipynb``.

Provides the Geneformer V2 semantic-map builder, training entry-points for
SemanticSCVI / LDVAE+ / scHPF / cNMF (all kwargs forwarded — pass training
parameters from the notebook), an ``_ScviAdapter`` to side-step adata-UUID
collisions between scvi models, and ``build_report`` to render a single
self-contained HTML from the artifacts ``SemanticBenchmark`` writes.

The heavy benchmarking machinery (``SemanticBenchmark``, ``ClaudeCLIScorer``,
``train_schpf_model``/``train_nmf_model``) lives in sibling modules
(``benchmarking.py``, ``llm_scorers.py``, ``train_schpf.py``) so the notebook
imports them directly when needed.
"""
from __future__ import annotations

import base64
import os
import pickle
import time
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from scvi.model import LinearSCVI
from scvi.model._semantic_scvi import SemanticSCVI


# ---------------------------------------------------------------------------
# Geneformer V2 semantic-map builder
# ---------------------------------------------------------------------------

GENEFORMER_REPO_ID = "ctheodoris/Geneformer"
GENEFORMER_TOKEN_DICT_PATH = "geneformer/token_dictionary_gc104M.pkl"


def build_geneformer_semantic_map(
    adata,
    *,
    var_id_key: str | None = None,
    model_id: str = GENEFORMER_REPO_ID,
    fill_missing: str = "zero",
    device: str | torch.device = "cpu",
    return_mask: bool = False,
):
    """Build a Geneformer V2 (gc104M) semantic map for an AnnData.

    Each row is the Geneformer V2 input token-embedding for the corresponding
    gene in ``adata.var``. Static lookup against
    ``model.bert.embeddings.word_embeddings.weight`` (shape 20275 x 1152) — no
    forward pass, no aggregation, no fine-tuning. First call downloads
    Geneformer to the HF cache (~415 MB); subsequent calls are instant.
    """
    if fill_missing not in ("zero", "raise"):
        raise ValueError(f"fill_missing must be 'zero' or 'raise', got {fill_missing!r}")

    from huggingface_hub import snapshot_download
    from transformers import BertForMaskedLM

    snap_dir = snapshot_download(
        repo_id=model_id,
        allow_patterns=["config.json", "model.safetensors", GENEFORMER_TOKEN_DICT_PATH],
    )

    tdict_path = os.path.join(snap_dir, GENEFORMER_TOKEN_DICT_PATH)
    with open(tdict_path, "rb") as f:
        token_dict: dict[str, int] = pickle.load(f)

    model = BertForMaskedLM.from_pretrained(snap_dir).eval()
    weights = model.bert.embeddings.word_embeddings.weight.detach().to(device)

    gene_ids = (
        adata.var[var_id_key].astype(str).tolist()
        if var_id_key is not None
        else adata.var_names.astype(str).tolist()
    )

    n_genes = len(gene_ids)
    semantic_map = torch.zeros(n_genes, weights.shape[1], dtype=weights.dtype, device=device)
    mask = torch.zeros(n_genes, dtype=torch.bool, device=device)
    missing: list[tuple[int, str]] = []
    for i, gene in enumerate(gene_ids):
        tid = token_dict.get(gene)
        if tid is None:
            missing.append((i, gene))
            continue
        semantic_map[i] = weights[tid]
        mask[i] = True

    if missing and fill_missing == "raise":
        sample = ", ".join(g for _, g in missing[:5])
        raise KeyError(
            f"{len(missing)}/{n_genes} genes missing from Geneformer token dict "
            f"(first 5: {sample})"
        )

    return (semantic_map, mask) if return_mask else semantic_map


def get_or_build_geneformer_map(adata, cache_path, **kwargs):
    """Load cached semantic map if it exists, else build and cache."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Loading cached Geneformer map from {cache_path}")
        return torch.load(cache_path, weights_only=False)
    print(f"Building Geneformer map (will cache to {cache_path})")
    semantic_map = build_geneformer_semantic_map(adata, **kwargs)
    torch.save(semantic_map, cache_path)
    return semantic_map


# ---------------------------------------------------------------------------
# GenePT (text-embedding-3-large) semantic-map builder
# ---------------------------------------------------------------------------

GENEPT_ZENODO_URL = "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip"
GENEPT_PICKLE_NAME = "GenePT_gene_protein_embedding_model_3_text.pickle"


def get_or_build_genept_map(
    adata,
    cache_path,
    pickle_path,
    *,
    symbol_key: str = "feature_name",
    fill_missing: str = "zero",
    zenodo_url: str = GENEPT_ZENODO_URL,
    pickle_basename: str = GENEPT_PICKLE_NAME,
):
    """Build/load a GenePT (text-embedding-3-large) semantic map for an AnnData.

    Rows align 1:1 to adata.var; lookup uses uppercased HGNC symbols (GenePT
    pickle uses UPPER-CASE HGNC keys). Missing genes are zero-filled (or raise
    if fill_missing='raise'). Cached as a torch tensor on first call.
    First-time setup downloads the Zenodo v2 bundle (~574 MB) and extracts the
    text-embedding-3-large pickle. Subsequent calls just load the cached tensor.
    """
    if fill_missing not in ("zero", "raise"):
        raise ValueError(f"fill_missing must be 'zero' or 'raise', got {fill_missing!r}")

    cache_path = Path(cache_path)
    pickle_path = Path(pickle_path)

    if cache_path.exists():
        print(f"Loading cached GenePT map from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    if not pickle_path.exists():
        zip_path = pickle_path.with_name("GenePT_emebdding_v2.zip")
        if not zip_path.exists():
            print(f"Downloading GenePT v2 bundle (~574 MB) from {zenodo_url}")
            urllib.request.urlretrieve(zenodo_url, zip_path)
            print(f"  saved to {zip_path}")
        print(f"Extracting {pickle_basename} from {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            # Some Zenodo bundles carry a stray trailing dot in member names —
            # strip trailing dots when matching by basename.
            target = pickle_basename.rstrip(".")
            member = next(
                (m for m in zf.namelist() if Path(m).name.rstrip(".") == target),
                None,
            )
            if member is None:
                raise FileNotFoundError(
                    f"{pickle_basename} not in zip; members: {zf.namelist()[:10]}"
                )
            with zf.open(member) as src, open(pickle_path, "wb") as dst:
                dst.write(src.read())
        print(f"  extracted to {pickle_path}")

    print(f"Loading GenePT pickle from {pickle_path}")
    with open(pickle_path, "rb") as f:
        gene2vec = pickle.load(f)

    symbols = adata.var[symbol_key].astype(str).str.upper().tolist()
    d = int(np.asarray(next(iter(gene2vec.values()))).shape[0])  # 3072 for text-embedding-3-large
    sm = torch.zeros(len(symbols), d, dtype=torch.float32)
    missing = []
    for i, g in enumerate(symbols):
        v = gene2vec.get(g)
        if v is None:
            missing.append(g)
            continue
        sm[i] = torch.as_tensor(np.asarray(v, dtype=np.float32))

    if missing and fill_missing == "raise":
        raise KeyError(
            f"{len(missing)}/{len(symbols)} symbols missing from GenePT "
            f"(first 5: {missing[:5]})"
        )
    print(
        f"GenePT: {len(symbols) - len(missing)}/{len(symbols)} genes mapped, "
        f"{len(missing)} zero-filled (d={d})"
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sm, cache_path)
    print(f"  cached aligned map to {cache_path}")
    return sm


# ---------------------------------------------------------------------------
# Trainers — all training/architecture knobs forwarded from the notebook.
# Each trainer caches under cache_dir; pass force_train=True to retrain.
# ---------------------------------------------------------------------------


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def train_or_load_semantic_scvi(
    adata,
    semantic_map,
    *,
    cache_dir: Path,
    force_train: bool,
    max_epochs: int,
    warmup_epochs: int,
    n_epochs_kl_warmup: int = 40,
    labels_key: str = "cell_type",
    batch_key: str | None = None,
    **kwargs,
):
    """Train (or reload from cache) ``SemanticSCVI``.

    Cache layout: scvi-tools' native ``model.save(<dir>)`` artifacts under
    ``cache_dir``. ``force_train=True`` retrains and overwrites the cache.
    """
    cache_dir = Path(cache_dir)
    SemanticSCVI.setup_anndata(
        adata, layer=None, labels_key=labels_key, batch_key=batch_key,
    )

    if cache_dir.exists() and not force_train:
        try:
            model = SemanticSCVI.load(str(cache_dir), adata=adata)
            _log(f"  loaded SemanticSCVI from {cache_dir}")
            return model
        except Exception as exc:
            _log(f"  load failed ({exc!r}); retraining")

    _log(
        f"Training SemanticSCVI (max_epochs={max_epochs}, warmup_epochs={warmup_epochs}, "
        f"n_epochs_kl_warmup={n_epochs_kl_warmup})..."
    )
    model = SemanticSCVI(adata, semantic_map=semantic_map, **kwargs)
    t0 = time.time()
    model.train(
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        n_epochs_kl_warmup=n_epochs_kl_warmup,
    )
    _log(f"  SemanticSCVI done in {time.time() - t0:.1f}s")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cache_dir), overwrite=True)
    _log(f"  saved to {cache_dir}")
    return model


def train_or_load_nonneg_ldvae(
    adata,
    *,
    cache_dir: Path,
    force_train: bool,
    max_epochs: int,
    batch_key: str | None = None,
    **kwargs,
):
    """Train (or reload from cache) ``LinearSCVI`` with ``weights_positive=True``."""
    cache_dir = Path(cache_dir)
    LinearSCVI.setup_anndata(adata, layer=None, batch_key=batch_key)
    kwargs.setdefault("weights_positive", True)

    if cache_dir.exists() and not force_train:
        try:
            model = LinearSCVI.load(str(cache_dir), adata=adata)
            _log(f"  loaded LDVAE+ from {cache_dir}")
            return model
        except Exception as exc:
            _log(f"  load failed ({exc!r}); retraining")

    _log(f"Training LDVAE+ (max_epochs={max_epochs})...")
    model = LinearSCVI(adata, **kwargs)
    t0 = time.time()
    model.train(max_epochs=max_epochs)
    _log(f"  LDVAE+ done in {time.time() - t0:.1f}s")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cache_dir), overwrite=True)
    _log(f"  saved to {cache_dir}")
    return model


def train_or_load_pickle(name, train_fn, cache_path: Path, force_train: bool):
    """Pickle-based cache for non-scvi factor models (scHPF / NMF wrappers)."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force_train:
        with cache_path.open("rb") as fh:
            obj = pickle.load(fh)
        _log(f"  loaded {name} from {cache_path}")
        return obj
    obj = train_fn()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(obj, fh)
    _log(f"  saved {name} to {cache_path}")
    return obj


# ---------------------------------------------------------------------------
# scvi adapter — binds each model to its own registered adata so the
# SemanticBenchmark methods can safely share a single adata reference.
# ---------------------------------------------------------------------------


class _ScviAdapter:
    def __init__(self, model, adata):
        self._model = model
        self._adata = adata

    @property
    def module(self):
        return self._model.module

    def get_latent_representation(self, adata=None):
        return self._model.get_latent_representation(self._adata)

    def get_loadings(self):
        return self._model.get_loadings()


# ---------------------------------------------------------------------------
# NMF wrapper — exposes the same contract as the scvi/scHPF models.
# ---------------------------------------------------------------------------


class NMFWrapper:
    def __init__(self, model, W, H, feature_names):
        self.model = model
        self.W = W
        self.H = H
        self.feature_names = feature_names

    def get_latent_representation(self, adata=None):
        return self.W

    def get_loadings(self):
        return pd.DataFrame(
            self.H.T,
            index=self.feature_names,
            columns=[f"Factor_{i}" for i in range(self.H.shape[0])],
        )


# ---------------------------------------------------------------------------
# EXPIMAP (scArches) integration: GMT builder, trainer/cache, adapter.
# ---------------------------------------------------------------------------


def build_expimap_mask_gmt(
    adata,
    source_gmt: Path,
    term_names,
    out_gmt: Path,
    *,
    symbol_col: str = "feature_name",
    min_genes: int = 12,
):
    """Filter ``source_gmt`` to ``term_names`` and translate HUGO symbols → ``adata.var_names``.

    Skips terms whose translated gene list falls below ``min_genes`` (paper default).
    Writes a small GMT next to the adata; idempotent — re-call only when missing.
    """
    source_gmt = Path(source_gmt)
    out_gmt = Path(out_gmt)
    sym_to_id = dict(
        zip(adata.var[symbol_col].astype(str), adata.var_names.astype(str))
    )
    keep = set(term_names)
    written, dropped = [], []
    with source_gmt.open() as fin, out_gmt.open("w") as fout:
        for line in fin:
            parts = line.rstrip("\n").split("\t")
            term = parts[0]
            if term not in keep:
                continue
            ids = [sym_to_id[s] for s in parts[2:] if s in sym_to_id]
            if len(ids) < min_genes:
                dropped.append((term, len(ids)))
                continue
            fout.write("\t".join([term, "NA", *ids]) + "\n")
            written.append((term, len(ids)))
    _log(f"  built {out_gmt.name}: {len(written)} terms (dropped {len(dropped)})")
    for term, n in written:
        print(f"    {term}: {n} genes")
    for term, n in dropped:
        print(f"    [skip] {term}: only {n} genes (< {min_genes})")
    return out_gmt


def train_or_load_expimap(
    adata,
    *,
    gmt_path: Path,
    cache_dir: Path,
    force_train: bool,
    n_epochs: int,
    condition_key: str | None = None,
    alpha_kl: float = 0.5,
    alpha: float = 0.7,
    alpha_epoch_anneal: int = 130,
    hidden_layer_sizes=(256, 256),
    recon_loss: str = "nb",
    **train_kwargs,
):
    """Train (or reload from cache) ``scarches.models.EXPIMAP``.

    Cache layout: scArches' native ``model.save(<dir>)`` directory. Re-running
    with the same params hits the cache; ``force_train=True`` retrains.
    """
    import scarches as sca

    cache_dir = Path(cache_dir)
    sca.utils.add_annotations(
        adata,
        files=str(gmt_path),
        min_genes=12,
        varm_key="I",
        uns_key="terms",
        clean=True,
        genes_use_upper=False,
    )
    n_terms = len(adata.uns["terms"])
    _log(f"  EXPIMAP mask: {n_terms} terms × {adata.varm['I'].shape[0]} genes")

    if cache_dir.exists() and not force_train:
        try:
            # map_location: scArches' default (None) deserializes onto the saved
            # device, so a GPU-trained cache fails to load on a CPU-only kernel
            # and silently retrains. Pin to the available device.
            map_location = "cuda" if torch.cuda.is_available() else "cpu"
            model = sca.models.EXPIMAP.load(
                str(cache_dir), adata=adata, map_location=map_location
            )
            _log(f"  loaded EXPIMAP from {cache_dir}")
            return model
        except Exception as exc:
            _log(f"  load failed ({exc!r}); retraining")

    _log(f"Training EXPIMAP (n_epochs={n_epochs}, alpha={alpha}, alpha_kl={alpha_kl})...")
    model = sca.models.EXPIMAP(
        adata,
        condition_key=condition_key,
        hidden_layer_sizes=list(hidden_layer_sizes),
        recon_loss=recon_loss,
        mask_key="I",
        soft_mask=False,
    )
    # Stratified sampling crashes when condition_key=None (AnnotatedDataset has no .conditions attr).
    # Disable it for the unconditional case to match the other models in the benchmark.
    if condition_key is None:
        train_kwargs.setdefault("use_stratified_sampling", False)
    # Force single-process data loading: scArches' default multi-worker DataLoader
    # stalls on CPU-only kernels.
    train_kwargs.setdefault("n_workers", 0)
    t0 = time.time()
    model.train(
        n_epochs=n_epochs,
        alpha_epoch_anneal=alpha_epoch_anneal,
        alpha_kl=alpha_kl,
        alpha=alpha,
        **train_kwargs,
    )
    _log(f"  EXPIMAP done in {time.time() - t0:.1f}s")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(cache_dir), overwrite=True, save_anndata=False)
    _log(f"  saved to {cache_dir}")
    return model


class _ExpimapAdapter:
    """Exposes EXPIMAP via the standard ``get_latent_representation`` / ``get_loadings`` contract."""

    def __init__(self, model, adata):
        self._model = model
        self._adata = adata
        self._terms = list(adata.uns["terms"])

    def get_latent_representation(self, adata=None):
        # scArches' get_latent defaults x=self.adata.X (often sparse) and indexes
        # it with a torch tensor, which newer scipy rejects ("'torch.dtype' has
        # no attribute 'kind'"). Pass a dense ndarray + matching condition vector
        # to bypass the sparse fancy-indexing path entirely.
        X = self._adata.X
        x = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        c = None
        cond_key = getattr(self._model, "condition_key_", None)
        if cond_key is not None and cond_key in self._adata.obs.columns:
            c = self._adata.obs[cond_key].to_numpy()
        return np.asarray(
            self._model.get_latent(x=x, c=c, mean=True, only_active=False)
        )

    def get_loadings(self):
        # decoder.L0.expr_L.weight: nn.Linear weight (out=genes, in=GPs).
        # column i = gene loadings for GP i → matches the order of adata.uns['terms'].
        w = (
            self._model.model.decoder.L0.expr_L.weight.detach().cpu().numpy()
        )
        return pd.DataFrame(w, index=self._adata.var_names, columns=self._terms)

    @property
    def history(self):
        return getattr(self._model, "history", None) or getattr(
            self._model, "history_", None
        )


# ---------------------------------------------------------------------------
# Training-curve plotting — reads scvi ``model.history`` (dict[str, DataFrame])
# and draws train/val curves for the available metrics.
# ---------------------------------------------------------------------------


def _series(history, key):
    """Return ``(epochs, values)`` numpy arrays for ``history[key]`` or None."""
    if not history or key not in history:
        return None
    df = history[key]
    if df is None or len(df) == 0:
        return None
    col = df.columns[0] if hasattr(df, "columns") else None
    if col is None:
        return None
    s = df[col].dropna()
    if s.empty:
        return None
    return np.asarray(s.index), np.asarray(s.values)


def _line(ax, history, key, label, color=None, linestyle="-"):
    pair = _series(history, key)
    if pair is None:
        return False
    epochs, values = pair
    ax.plot(epochs, values, label=label, color=color, linestyle=linestyle)
    return True


def plot_training_curves(model, name, out_path, *, semantic: bool):
    """Save a multi-panel figure of training curves for ``model``.

    Reads ``model.history`` (dict of ``{metric_name: DataFrame}``). If
    ``semantic=True`` also renders coherence-loss and semantic-scale panels.
    Skips silently if the model has no usable history.
    """
    history = getattr(model, "history", None) or getattr(model, "history_", None)
    if not history:
        print(f"  ! {name}: no history attribute; skipping training-curve plot")
        return None

    if semantic:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        ax_total, ax_recon, ax_coh, ax_scale = axes.ravel()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        ax_total, ax_recon, ax_kl = axes
        ax_coh = ax_scale = None

    drew_total = False
    drew_total |= _line(ax_total, history, "train_loss_epoch", "train", color="C0")
    drew_total |= _line(ax_total, history, "validation_loss", "validation", color="C1")
    ax_total.set_title("Total loss")
    ax_total.set_xlabel("epoch")
    ax_total.set_ylabel("loss")
    if drew_total:
        ax_total.legend(fontsize=8)
    else:
        ax_total.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax_total.transAxes)

    drew_recon = False
    drew_recon |= _line(ax_recon, history, "reconstruction_loss_train", "train recon", color="C0")
    drew_recon |= _line(ax_recon, history, "reconstruction_loss_validation", "val recon", color="C1")
    ax_recon.set_title("Reconstruction loss")
    ax_recon.set_xlabel("epoch")
    if drew_recon:
        ax_recon.legend(fontsize=8)
    else:
        ax_recon.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax_recon.transAxes)

    if semantic:
        # KL panel shares the recon axis as a twin? Keep it simple: drop KL into
        # the coherence-loss row instead so semantic gets its own dedicated panels.
        drew_coh = False
        drew_coh |= _line(ax_coh, history, "coherence_loss_train", "train", color="C0")
        drew_coh |= _line(ax_coh, history, "coherence_loss_validation", "validation", color="C1")
        ax_coh.set_title("Coherence (semantic) loss — weighted")
        ax_coh.set_xlabel("epoch")
        if drew_coh:
            ax_coh.legend(fontsize=8)
        else:
            ax_coh.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax_coh.transAxes)

        drew_scale = False
        drew_scale |= _line(ax_scale, history, "semantic_scale_train", "train scale", color="C2")
        drew_scale |= _line(ax_scale, history, "semantic_scale_validation", "val scale", color="C3", linestyle="--")
        # KL goes on a twin axis of the semantic_scale panel — both are bounded/small.
        kl_drawn = False
        kl_drawn |= _line(ax_scale, history, "kl_local_train", "kl_local train", color="C4")
        kl_drawn |= _line(ax_scale, history, "kl_local_validation", "kl_local val", color="C5", linestyle="--")
        ax_scale.set_title("Semantic warmup scale + KL local")
        ax_scale.set_xlabel("epoch")
        if drew_scale or kl_drawn:
            ax_scale.legend(fontsize=7, loc="best")
        else:
            ax_scale.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax_scale.transAxes)
    else:
        drew_kl = False
        drew_kl |= _line(ax_kl, history, "kl_local_train", "train", color="C0")
        drew_kl |= _line(ax_kl, history, "kl_local_validation", "validation", color="C1")
        ax_kl.set_title("KL local")
        ax_kl.set_xlabel("epoch")
        if drew_kl:
            ax_kl.legend(fontsize=8)
        else:
            ax_kl.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax_kl.transAxes)

    fig.suptitle(f"{name} — training dynamics", fontsize=11)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"  > Wrote {out_path}")
    return fig


# ---------------------------------------------------------------------------
# HTML report — embeds every PNG the SemanticBenchmark writes plus the
# summary CSVs into one self-contained file.
# ---------------------------------------------------------------------------


def _embed(path: Path, alt: str) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("ascii")
    return f'<div class="figure"><img src="data:image/png;base64,{b64}" alt="{alt}"></div>'


def _summary_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p><em>(no data)</em></p>"

    def fmt(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.3f}"
        return str(v)

    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = "".join(f"<td>{fmt(r[c])}</td>" for c in df.columns)
        rows.append(f"<tr>{cells}</tr>")
    return (
        '<table class="summary"><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def build_report(out_dir: Path, model_names: list[str], adata_shape, notes: str = "") -> Path:
    """Render ``out_dir/report.html`` from the artifacts ``SemanticBenchmark`` wrote."""
    out_dir = Path(out_dir)
    sections = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections.append(
        f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<title>Four-way factor benchmark</title>
<style>
 body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-width: 1300px; margin: 0 auto; padding: 24px; background: #fafafa; color: #222; }}
 h1 {{ border-bottom: 3px solid #16213e; padding-bottom: 8px; }}
 h2 {{ border-bottom: 1px solid #ddd; padding-bottom: 4px; margin-top: 36px; }}
 h3 {{ color: #0f3460; margin-top: 24px; }}
 .meta {{ color: #666; font-size: 14px; margin-bottom: 16px; }}
 .models {{ background: #e8f4f8; padding: 12px 16px; border-radius: 6px;
            margin-bottom: 24px; font-size: 14px; }}
 .models code {{ background: #d0e8f0; padding: 2px 8px; border-radius: 3px;
                 margin: 2px; display: inline-block; font-family: monospace; }}
 table.summary {{ border-collapse: collapse; font-size: 13px; width: 100%;
                  margin: 16px 0; }}
 table.summary th, table.summary td {{ padding: 6px 10px;
   border-bottom: 1px solid #eee; text-align: right; }}
 table.summary th {{ background: #f5f5f7; font-weight: 600; }}
 table.summary td:first-child, table.summary th:first-child {{
   text-align: left; font-family: monospace; }}
 table.summary tr:nth-child(even) {{ background: #fafafa; }}
 .figure {{ margin: 16px 0; text-align: center; }}
 .figure img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.08); }}
 .note {{ background: #fffaeb; border-left: 4px solid #fbbf24;
          padding: 8px 14px; margin: 12px 0; font-size: 13px; }}
 details {{ margin: 12px 0; padding: 8px 14px; background: #f9f9fb;
            border-left: 3px solid #4a4e69; border-radius: 3px; }}
 details summary {{ cursor: pointer; font-weight: 600; font-family: monospace; }}
</style></head><body>
<h1>Four-way factor benchmark</h1>
<div class="meta">Generated: {timestamp}</div>
<div class="models">
  <strong>Models compared:</strong>
  {' '.join(f'<code>{n}</code>' for n in model_names)}<br>
  <strong>Data:</strong> {adata_shape[0]} cells &times; {adata_shape[1]} genes
</div>"""
    )
    if notes:
        sections.append(f'<div class="note">{notes}</div>')

    # Training dynamics
    train_figs = [
        ("training_curves_semantic.png", "SemanticSCVI (Geneformer prior) — total / recon / coherence / semantic_scale + KL"),
        ("training_curves_semantic_genept.png", "SemanticSCVI (GenePT prior) — total / recon / coherence / semantic_scale + KL"),
        ("training_curves_ldvae.png", "LDVAE+ — total / recon / KL"),
        ("training_curves_expimap.png", "EXPIMAP — total / recon / KL"),
    ]
    train_snippets = [(_embed(out_dir / f, f), c) for f, c in train_figs]
    if any(snip for snip, _ in train_snippets):
        sections.append("<h2>Training dynamics</h2>")
        for snip, caption in train_snippets:
            if snip:
                sections.append(f"<h3>{caption}</h3>{snip}")

    # Summary tables
    proj_summary_csv = out_dir / "per_projection_summary.csv"
    cluster_summary_csv = out_dir / "per_cluster_summary.csv"
    if proj_summary_csv.exists():
        df = pd.read_csv(proj_summary_csv)
        sections.append("<h2>Per-projection summary (mean across factors)</h2>")
        sections.append(_summary_table(df))
    if cluster_summary_csv.exists():
        df = pd.read_csv(cluster_summary_csv)
        sections.append("<h2>Per-cluster summary (mean across clusters)</h2>")
        sections.append(_summary_table(df))

    # Per-projection figures
    sections.append("<h2>Per-projection biology — LLM grading + sens/spec</h2>")
    for fname, caption in [
        ("01_per_projection_a_scores.png", "Stage 1/4 — LLM judge scores per factor"),
        ("02_per_projection_b_interpretations.png", "Stage 2/4 — what each projection is (LLM)"),
        ("03_per_projection_c_specsens.png", "Stage 3/4 — HG enrichment (ER, q<0.05) across H / C2-immune / C7"),
        ("04_per_projection_d_top10.png", "Stage 4/4 — top-10 by spec, with matched program"),
    ]:
        snippet = _embed(out_dir / fname, fname)
        if snippet:
            sections.append(f"<h3>{caption}</h3>{snippet}")

    # Per-cluster figures
    sections.append("<h2>Per-cluster biology — gene modules</h2>")
    sections.append("<h3>UMAP/Leiden modules</h3>")
    for fname, caption in [
        ("05_per_cluster_umap_leiden_a_scores.png", "LLM scores"),
        ("06_per_cluster_umap_leiden_b_interpretations.png", "LLM interpretation"),
        ("07_per_cluster_umap_leiden_c_specsens.png", "HG enrichment (ER, q<0.05) across H / C2-immune / C7"),
        ("08_per_cluster_umap_leiden_d_top10.png", "top-10 by spec"),
    ]:
        snippet = _embed(out_dir / fname, fname)
        if snippet:
            sections.append(f"<h4>{caption}</h4>{snippet}")
    sections.append("<h3>Hierarchical modules</h3>")
    for fname, caption in [
        ("09_per_cluster_hierarchical_a_scores.png", "LLM scores"),
        ("10_per_cluster_hierarchical_b_interpretations.png", "LLM interpretation"),
        ("11_per_cluster_hierarchical_c_specsens.png", "HG enrichment (ER, q<0.05) across H / C2-immune / C7"),
        ("12_per_cluster_hierarchical_d_top10.png", "top-10 by spec"),
    ]:
        snippet = _embed(out_dir / fname, fname)
        if snippet:
            sections.append(f"<h4>{caption}</h4>{snippet}")

    # Matrix / program quality
    sections.append("<h2>Matrix &amp; program quality</h2>")
    for fname, caption in [
        ("1_depth_correlation_summary.png", "Depth correlation (lower = better)"),
        ("3_orthogonality_summary.png", "Latent factor orthogonality"),
        ("6_omega_coherence.png", "Per-program omega"),
        ("7_jaccard_similarity.png", "Per-program Jaccard"),
        ("10_cluster_omega_barplot.png", "Per-cluster omega"),
        ("semantic_alignment.png", "Semantic alignment score"),
    ]:
        snippet = _embed(out_dir / fname, fname)
        if snippet:
            sections.append(f"<h3>{caption}</h3>{snippet}")

    # Per-model
    sections.append("<h2>Per-model gene-space artifacts</h2>")
    for name in model_names:
        sub = []
        for fname, caption in [
            (f"5_latent_profile_{name}.png", "latent profile (top-500 genes)"),
            (f"8_gene_umap_{name}.png", "gene UMAP / Leiden modules"),
            (f"module_heatmap_{name}.png", "hierarchical module heatmap"),
        ]:
            snippet = _embed(out_dir / fname, fname)
            if snippet:
                sub.append(f"<h4>{caption}</h4>{snippet}")
        if sub:
            sections.append(
                f'<details open><summary>{name}</summary>{"".join(sub)}</details>'
            )

    # Artifacts list
    sections.append("<h2>Artifacts on disk</h2><ul>")
    for f in (
        out_dir / "per_projection_biology.csv",
        out_dir / "per_projection_summary.csv",
        out_dir / "per_cluster_biology.csv",
        out_dir / "per_cluster_summary.csv",
        out_dir / "program_recovery.csv",
        out_dir / "semantic_alignment.csv",
    ):
        if f.exists():
            sections.append(f"<li><code>{f}</code></li>")
    sections.append("</ul>")
    sections.append("</body></html>")

    out_html = out_dir / "report.html"
    out_html.write_text("".join(sections))
    return out_html
