from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd
import torch

from scvi import REGISTRY_KEYS
from scvi.external.scviva._model import SCVIVA

from ._module import semantic_nicheVAE

if TYPE_CHECKING:
    from typing import Literal

    from anndata import AnnData

logger = logging.getLogger(__name__)


class SemanticSCVIVA(SCVIVA):
    """Spatial semantic NMF model: SemanticSCVI x scVIVA.

    A two-stage spatial extension of the semantic model. The expression arm is a positive
    *linear* NMF decoder constrained by a gene-embedding semantic prior (as in
    :class:`~scvi.model.SemanticSCVI`), fed a *structured* latent of size
    ``n_latent * (1 + n_labels)`` (shared block + one soft/hard-gated block per cell type).
    The niche-activation and Dirichlet-composition arms of
    :class:`~scvi.external.SCVIVA` are retained.

    Typical workflow
    ----------------
    >>> # Stage 1: semantic model provides the embedding seed (replaces scVI/SCANVI)
    >>> stage1 = scvi.model.SemanticSCVI(adata_s1, semantic_map=semantic_map)
    >>> stage1.train(max_epochs=200, warmup_epochs=20)
    >>> adata.obsm["X_scVI"] = stage1.get_latent_representation(adata_s1)
    >>> # Stage 2: build niche targets, warm-start, train jointly
    >>> SemanticSCVIVA.preprocessing_anndata(adata, sample_key="slide", labels_key="cell_type")
    >>> SemanticSCVIVA.setup_anndata(adata, labels_key="cell_type", sample_key="slide")
    >>> model = SemanticSCVIVA.from_semantic_scvi(stage1, adata, semantic_map=semantic_map)
    >>> model.train(max_epochs=200, warmup_epochs=20)

    Parameters
    ----------
    adata
        AnnData registered via :meth:`setup_anndata` (and prepared via
        :meth:`preprocessing_anndata`).
    semantic_map
        ``(n_genes, d)`` gene-embedding matrix for the semantic prior.
    coherence_weight, loss_mode, n_gene_sample
        Semantic-loss parameters (see :class:`~scvi.model.SemanticSCVI`).
    gate_mode
        ``"soft"`` (default), ``"hard"``, or ``"soft_label_blend"``.
    decorrelation_loss_weight, weights_positive, gate_temperature, gate_blend_label,
    tie_gate_to_classifier
        See :class:`~scvi.external.semantic_scviva._module.semantic_nicheVAE`.
    **kwargs
        Forwarded to :class:`~scvi.external.semantic_scviva._module.semantic_nicheVAE`.

    See Also
    --------
    :class:`~scvi.external.SCVIVA`
    :class:`~scvi.model.SemanticSCVI`
    """

    _module_cls = semantic_nicheVAE

    def __init__(
        self,
        adata: AnnData | None = None,
        semantic_map: torch.Tensor | None = None,
        *,
        coherence_weight: float = 10.0,
        loss_mode: Literal["centroid", "geometric"] = "geometric",
        n_gene_sample: int = 1024,
        gate_mode: Literal["hard", "soft", "soft_label_blend"] = "soft",
        gate_temperature: float = 1.0,
        gate_blend_label: float = 0.5,
        tie_gate_to_classifier: bool = True,
        decorrelation_loss_weight: float = 120.0,
        weights_positive: bool = True,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **kwargs,
    ):
        if semantic_map is None:
            raise ValueError("`semantic_map` is required for SemanticSCVIVA.")

        super().__init__(
            adata,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            # forwarded to semantic_nicheVAE via SCVIVA.__init__(**kwargs)
            semantic_map=semantic_map,
            coherence_weight=coherence_weight,
            loss_mode=loss_mode,
            n_gene_sample=n_gene_sample,
            gate_mode=gate_mode,
            gate_temperature=gate_temperature,
            gate_blend_label=gate_blend_label,
            tie_gate_to_classifier=tie_gate_to_classifier,
            decorrelation_loss_weight=decorrelation_loss_weight,
            weights_positive=weights_positive,
            **kwargs,
        )

        self._model_summary_string = (
            f"SemanticSCVIVA (gate_mode={gate_mode}, loss_mode={loss_mode}, "
            f"coherence_weight={coherence_weight}, n_latent={n_latent}, "
            f"n_labels={self.n_labels})"
        )
        self.init_params_ = self._get_init_params(locals())

    # ------------------------------------------------------------------ #
    # two-stage helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def preprocess_with_semantic_scvi(
        cls,
        adata: AnnData,
        semantic_scvi_model,
        semantic_scvi_adata: AnnData | None = None,
        expression_embedding_key: str = "X_scVI",
        **pp_kwargs,
    ) -> None:
        """Write Stage-1 SemanticSCVI latents into ``adata`` and build the niche targets.

        ``semantic_scvi_adata`` (defaults to ``adata``) must be in the same cell order as
        ``adata`` -- e.g. an ``adata.copy()`` used for the UUID-isolated Stage-1 run.
        """
        src = semantic_scvi_adata if semantic_scvi_adata is not None else adata
        latent = semantic_scvi_model.get_latent_representation(src)
        if latent.shape[0] != adata.n_obs:
            raise ValueError(
                "Stage-1 latent has a different number of cells than `adata`; "
                "pass the matching `semantic_scvi_adata`."
            )
        adata.obsm[expression_embedding_key] = latent
        cls.preprocessing_anndata(
            adata, expression_embedding_key=expression_embedding_key, **pp_kwargs
        )

    @classmethod
    def from_semantic_scvi(
        cls,
        semantic_scvi_model,
        adata: AnnData,
        *,
        semantic_map: torch.Tensor | None = None,
        n_latent: int | None = None,
        init_type_blocks: Literal["shared", "random"] = "shared",
        warm_start_encoder: bool = False,
        **model_kwargs,
    ) -> SemanticSCVIVA:
        """Build a SemanticSCVIVA and warm-start its decoder from a trained SemanticSCVI.

        ``adata`` must already be preprocessed (:meth:`preprocess_with_semantic_scvi` or
        :meth:`preprocessing_anndata`) and registered (:meth:`setup_anndata`).
        """
        src_module = semantic_scvi_model.module
        if semantic_map is None:
            semantic_map = src_module.semantic_map
        if n_latent is None:
            n_latent = semantic_scvi_model.n_latent

        model = cls(adata, semantic_map=semantic_map, n_latent=n_latent, **model_kwargs)
        _warm_start_from_semantic_scvi(
            model.module,
            src_module,
            n_latent=n_latent,
            T=model.n_labels,
            init_type_blocks=init_type_blocks,
            warm_start_encoder=warm_start_encoder,
        )
        return model

    # ------------------------------------------------------------------ #
    # training / inspection
    # ------------------------------------------------------------------ #
    def train(self, max_epochs=None, warmup_epochs: int = 0, n_epochs_kl_warmup: int = 40, **kwargs):
        """Train with the semantic-loss warmup (off for ``warmup_epochs``) + KL warmup."""
        from scvi.model._semantic_scvi import SemanticWarmupCallback

        callbacks = kwargs.get("callbacks", []) or []
        if warmup_epochs > 0:
            logger.info(
                f"Semantic warmup: coherence loss OFF for {warmup_epochs} epochs ({self.module.loss_mode})."
            )
            callbacks.insert(0, SemanticWarmupCallback(warmup_epochs))
            kwargs["callbacks"] = callbacks

        plan_kwargs = kwargs.get("plan_kwargs") or {}
        plan_kwargs.setdefault("n_epochs_kl_warmup", n_epochs_kl_warmup)
        kwargs["plan_kwargs"] = plan_kwargs

        super().train(max_epochs=max_epochs, **kwargs)

    def get_loadings(self) -> pd.DataFrame:
        """Gene-by-factor loadings; columns are ``shared_*`` then ``<cell_type>_*`` blocks."""
        n_latent = self.module.n_latent_block
        T = self.module.n_type_blocks
        cols = [f"shared_{i}" for i in range(n_latent)]
        ct_names = list(
            self.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY).categorical_mapping
        )
        for t in range(T):
            name = ct_names[t] if t < len(ct_names) else f"ct{t}"
            cols += [f"{name}_{i}" for i in range(n_latent)]
        return pd.DataFrame(self.module.get_loadings(), index=self.adata.var_names, columns=cols)


# ---------------------------------------------------------------------- #
# warm-start utilities
# ---------------------------------------------------------------------- #
@torch.no_grad()
def _expand_dual_linear(tgt, src, n_latent: int, T: int, init_type_blocks: str) -> None:
    """Copy a ``(n_genes, n_latent)`` positive layer into the shared (and type) blocks."""
    src_w = src.X_layer.weight.data
    tgt_w = tgt.X_layer.weight.data
    tgt_w[:, :n_latent].copy_(src_w)
    if init_type_blocks == "shared":
        for t in range(T):
            tgt_w[:, n_latent * (1 + t) : n_latent * (2 + t)].copy_(src_w)

    if tgt.X_layer.bias is not None and src.X_layer.bias is not None:
        tgt.X_layer.bias.data.copy_(src.X_layer.bias.data)

    if tuple(tgt.covariate_layer.weight.shape) == tuple(src.covariate_layer.weight.shape):
        tgt.covariate_layer.weight.data.copy_(src.covariate_layer.weight.data)
        if tgt.covariate_layer.bias is not None and src.covariate_layer.bias is not None:
            tgt.covariate_layer.bias.data.copy_(src.covariate_layer.bias.data)


@torch.no_grad()
def _warm_start_encoder(target, source) -> None:
    """Best-effort encoder transfer (architectures differ: BN/LN, split vs fused heads)."""
    try:
        result = target.z_encoder.encoder.load_state_dict(
            source.z_encoder.encoder.state_dict(), strict=False
        )
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                f"Encoder FCLayers warm-start partial: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        src_m, src_v = source.z_encoder.mean_encoder, source.z_encoder.var_encoder
        de = target.z_encoder.dist_encoder
        n_out = src_m.weight.shape[0]
        de.weight.data[:n_out].copy_(src_m.weight.data)
        de.weight.data[n_out:].copy_(src_v.weight.data)
        de.bias.data[:n_out].copy_(src_m.bias.data)
        de.bias.data[n_out:].copy_(src_v.bias.data)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Encoder warm-start failed; leaving encoder freshly initialized: {e}")


@torch.no_grad()
def _warm_start_from_semantic_scvi(
    target,
    source,
    n_latent: int,
    T: int,
    init_type_blocks: str = "shared",
    warm_start_encoder: bool = False,
) -> None:
    """Warm-start ``target`` (semantic_nicheVAE) from ``source`` (SemanticLDVAE)."""
    _expand_dual_linear(
        target.decoder.factor_regressor.fc_layers[0][0],
        source.decoder.factor_regressor.fc_layers[0][0],
        n_latent,
        T,
        init_type_blocks,
    )
    _expand_dual_linear(
        target.decoder.px_dropout_decoder.fc_layers[0][0],
        source.decoder.px_dropout_decoder.fc_layers[0][0],
        n_latent,
        T,
        init_type_blocks,
    )

    if target.use_decoder_batch_norm and getattr(source, "use_batch_norm", False):
        try:
            target.decoder.factor_regressor.fc_layers[0][1].load_state_dict(
                source.decoder.factor_regressor.fc_layers[0][1].state_dict()
            )
            target.decoder.px_dropout_decoder.fc_layers[0][1].load_state_dict(
                source.decoder.px_dropout_decoder.fc_layers[0][1].state_dict()
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Skipped decoder BatchNorm warm-start: {e}")

    if hasattr(source, "px_r") and hasattr(target, "px_r"):
        if tuple(source.px_r.shape) == tuple(target.px_r.shape):
            target.px_r.data.copy_(source.px_r.data)

    if warm_start_encoder:
        _warm_start_encoder(target, source)
