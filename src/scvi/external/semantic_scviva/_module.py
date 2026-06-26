from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from scvi import REGISTRY_KEYS
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import auto_move_data
from scvi.nn import LinearDecoderSCVI

from ..scviva._constants import SCVIVA_MODULE_KEYS
from ..scviva._module import nicheVAE
from ._components import GateHead

if TYPE_CHECKING:
    from typing import Literal

    from torch.distributions import Distribution

logger = logging.getLogger(__name__)

GATE_KEY = "gate"


class semantic_nicheVAE(nicheVAE):
    """nicheVAE with a structured, gated, linear semantic-NMF expression arm.

    Extends :class:`~scvi.external.scviva._module.nicheVAE` by:

    1. Replacing the (non-linear) expression decoder with a positive linear decoder
       (NMF-like, ``softplus`` weights) constrained by a gene-embedding *semantic* prior
       (centroid/geometric) and a decorrelation regularizer -- ported from
       :class:`~scvi.module.SemanticLDVAE`.
    2. Feeding that decoder a *structured* latent of size ``n_latent * (1 + n_labels)``:
       a shared block plus one block per cell type, each type-block scaled by a soft/hard
       *gate*. The niche-activation and composition decoders keep the un-expanded
       ``n_latent`` latent, so the encoder, KL, and niche-target dimensions are unchanged.

    Parameters beyond :class:`~scvi.external.scviva._module.nicheVAE`
    ----------------------------------------------------------------
    semantic_map
        ``(n_genes, d)`` gene-embedding matrix used by the semantic prior.
    coherence_weight, loss_mode, n_gene_sample, use_importance_sampling, use_huber_loss,
    huber_delta
        Semantic-loss parameters (see :class:`~scvi.module.SemanticLDVAE`).
    decorrelation_loss_weight, min_cor
        Decorrelation regularizer parameters (see :class:`~scvi.module.LDVAE`).
    weights_positive, decoder_bias, use_decoder_batch_norm
        Linear decoder parameters.
    gate_mode
        ``"hard"`` (each cell uses shared + its own type-block via the label one-hot),
        ``"soft"`` (learned ``softmax`` over type-blocks), or ``"soft_label_blend"``
        (soft blended with the label one-hot for labeled cells).
    gate_temperature, gate_blend_label, gate_entropy_weight, tie_gate_to_classifier
        Gating parameters.
    """

    def __init__(
        self,
        n_input: int,
        n_output_niche: int,
        semantic_map: torch.Tensor,
        n_labels: int = 0,
        n_latent: int = 10,
        # --- semantic NMF expression arm ---
        coherence_weight: float = 10.0,
        loss_mode: Literal["centroid", "geometric"] = "geometric",
        n_gene_sample: int = 1024,
        use_importance_sampling: bool = True,
        use_huber_loss: bool = False,
        huber_delta: float = 0.1,
        decorrelation_loss_weight: float = 120.0,
        min_cor: float = 0.0,
        weights_positive: bool = True,
        decoder_bias: bool = False,
        use_decoder_batch_norm: bool = True,
        # --- structured latent / gating ---
        gate_mode: Literal["hard", "soft", "soft_label_blend"] = "soft",
        gate_temperature: float = 1.0,
        gate_blend_label: float = 0.5,
        gate_entropy_weight: float = 0.0,
        tie_gate_to_classifier: bool = True,
        **kwargs,
    ):
        kwargs.setdefault("gene_likelihood", "nb")
        super().__init__(
            n_input=n_input,
            n_output_niche=n_output_niche,
            n_labels=n_labels,
            n_latent=n_latent,
            **kwargs,
        )

        if self.batch_representation != "one-hot":
            raise NotImplementedError(
                "semantic_nicheVAE only supports batch_representation='one-hot' "
                "for the linear semantic decoder."
            )

        # structured latent bookkeeping
        self.n_latent_block = n_latent
        self.n_type_blocks = n_labels
        self.n_latent_expanded = n_latent * (1 + n_labels)

        # gating config
        self.gate_mode = gate_mode
        self.gate_temperature = gate_temperature
        self.gate_blend_label = gate_blend_label
        self.gate_entropy_weight = gate_entropy_weight
        self.tie_gate_to_classifier = tie_gate_to_classifier

        # decoder config
        self.weights_positive = weights_positive
        self.use_decoder_batch_norm = use_decoder_batch_norm
        self.decorrelation_loss_weight = decorrelation_loss_weight
        self.min_cor = min_cor

        # replace the expression decoder with a positive linear (NMF) decoder over z_eff
        self.decoder = LinearDecoderSCVI(
            self.n_latent_expanded,
            n_input,
            n_cat_list=[self.n_batch],
            use_batch_norm=use_decoder_batch_norm,
            use_layer_norm=False,
            bias=decoder_bias,
            weights_positive=weights_positive,
        )

        # gate head (only needed for soft modes when not tying to the classifier)
        if (not tie_gate_to_classifier) or (self.classifier is None):
            self.gate_head = GateHead(n_latent, n_labels)
        else:
            self.gate_head = None

        # semantic-prior buffers (mirror SemanticLDVAE)
        self.register_buffer("semantic_map", semantic_map)
        self.coherence_weight = coherence_weight
        self.loss_mode = loss_mode
        self.n_gene_sample = n_gene_sample
        self.use_importance_sampling = use_importance_sampling
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        # updated by SemanticWarmupCallback (semantic loss off during warmup)
        self.register_buffer("semantic_loss_scale", torch.tensor(1.0))
        gene_norms = torch.norm(semantic_map, p=2, dim=1)
        probs = gene_norms + 1e-6
        self.register_buffer("sampling_probs", probs / probs.sum())

    # ------------------------------------------------------------------ #
    # structured latent + gating
    # ------------------------------------------------------------------ #
    def _broadcast_gate(self, g: torch.Tensor, lead: torch.Size) -> torch.Tensor:
        """Broadcast a ``(n_obs, T)`` gate to leading dims ``lead`` (handles n_samples)."""
        if tuple(g.shape[:-1]) == tuple(lead):
            return g
        while g.dim() - 1 < len(lead):
            g = g.unsqueeze(0)
        return g.expand(*lead, g.shape[-1])

    def _compute_gate(self, z: torch.Tensor, y: torch.Tensor | None) -> torch.Tensor:
        """Per-cell weights over the ``T`` type-blocks; shape ``(..., T)``."""
        lead = z.shape[:-1]
        T = self.n_type_blocks

        if self.gate_mode == "hard":
            if y is None:
                raise ValueError("gate_mode='hard' requires labels `y`.")
            yy = y.reshape(-1).long().clamp(min=0, max=T - 1)
            g = F.one_hot(yy, T).float()
            return self._broadcast_gate(g, lead)

        # soft / soft_label_blend
        z2 = z.reshape(-1, z.shape[-1])
        if self.tie_gate_to_classifier and self.classifier is not None:
            logits = self.classifier(z2)
        else:
            logits = self.gate_head(z2)
        g = F.softmax(logits / max(self.gate_temperature, 1e-4), dim=-1)
        g = g.reshape(*lead, T)

        if self.gate_mode == "soft_label_blend" and y is not None:
            yy = y.reshape(-1).long()
            labeled = (yy >= 0) & (yy < T)
            one_hot_y = torch.zeros(yy.shape[0], T, device=z.device)
            if labeled.any():
                one_hot_y[labeled] = F.one_hot(yy[labeled], T).float()
            one_hot_y = self._broadcast_gate(one_hot_y, lead)
            labeled_b = self._broadcast_gate(labeled.float().unsqueeze(-1), lead).bool()
            blended = (1.0 - self.gate_blend_label) * g + self.gate_blend_label * one_hot_y
            g = torch.where(labeled_b, blended, g)
        return g

    def _assemble_z_eff(self, z: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """z_eff = concat(z, g_1 * z, ..., g_T * z); shape ``(..., n_latent * (1 + T))``."""
        blocks = g.unsqueeze(-1) * z.unsqueeze(-2)  # (..., T, n_latent)
        blocks = blocks.reshape(*z.shape[:-1], self.n_type_blocks * self.n_latent_block)
        return torch.cat([z, blocks], dim=-1)

    # ------------------------------------------------------------------ #
    # semantic / decorrelation helpers (ported from SemanticLDVAE / LDVAE)
    # ------------------------------------------------------------------ #
    def _get_effective_loadings(self) -> torch.Tensor:
        """``W_eff = softplus(W) * (gamma / sigma)``; shape ``(n_genes, n_latent * (1 + T))``."""
        layer = self.decoder.factor_regressor.fc_layers[0][0]
        if self.weights_positive:
            w_raw = F.softplus(layer.X_layer.weight)
        else:
            w_raw = F.softplus(layer.weight)
        if self.use_decoder_batch_norm:
            bn = self.decoder.factor_regressor.fc_layers[0][1]
            scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).unsqueeze(1)
            return w_raw * scale
        return w_raw

    @torch.inference_mode()
    def get_loadings(self):
        """Gene-by-factor effective loadings as a NumPy array."""
        return self._get_effective_loadings().detach().cpu().numpy()

    def _decorrelation_loss(self, W: torch.Tensor) -> torch.Tensor:
        W_normalized = F.normalize(W, p=2, dim=1)
        gram = W_normalized @ W_normalized.T
        off_diag = gram - torch.eye(gram.shape[0], device=W.device)
        return torch.clamp((off_diag**2).mean(), min=self.min_cor)

    def _semantic_loss(self, W: torch.Tensor) -> torch.Tensor:
        if self.loss_mode == "centroid":
            W_prob = W / (W.sum(dim=0, keepdim=True) + 1e-6)
            centroids = torch.matmul(W_prob.T, self.semantic_map)
            distances = (self.semantic_map.unsqueeze(1) - centroids.unsqueeze(0)).pow(2).sum(dim=2)
            return (W_prob * distances).sum()

        # geometric (isometry between factor- and gene-embedding similarity)
        n_genes = self.semantic_map.shape[0]
        curr = min(self.n_gene_sample, n_genes)
        if self.use_importance_sampling:
            indices = torch.multinomial(self.sampling_probs, num_samples=curr, replacement=False)
        else:
            indices = torch.randperm(n_genes, device=self.device)[:curr]
        W_norm = F.normalize(W[indices], p=2, dim=1)
        S_norm = F.normalize(self.semantic_map[indices], p=2, dim=1)
        sim_W = torch.mm(W_norm, W_norm.t())
        sim_S = torch.mm(S_norm, S_norm.t())
        if self.use_huber_loss:
            return F.huber_loss(sim_W, sim_S, delta=self.huber_delta)
        return F.mse_loss(sim_W, sim_S)

    # ------------------------------------------------------------------ #
    # generative / loss overrides
    # ------------------------------------------------------------------ #
    @auto_move_data
    def generative(
        self,
        z: torch.Tensor,
        library: torch.Tensor,
        batch_index: torch.Tensor,
        cont_covs: torch.Tensor | None = None,
        cat_covs: torch.Tensor | None = None,
        size_factor: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        transform_batch: torch.Tensor | None = None,
    ) -> dict[str, Distribution | None]:
        """Run the generative process (structured/gated linear expression arm)."""
        from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
        from torch.nn.functional import linear, one_hot

        from scvi.distributions import (
            NegativeBinomial,
            Poisson,
            ZeroInflatedNegativeBinomial,
        )

        # ----- niche/composition decoder input uses the un-expanded latent z -----
        if cont_covs is None:
            niche_input = z
        elif z.dim() != cont_covs.dim():
            niche_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            niche_input = torch.cat([z, cont_covs], dim=-1)

        categorical_input = torch.split(cat_covs, 1, dim=1) if cat_covs is not None else ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        # ----- expression decoder input is the structured, gated latent z_eff -----
        gate = self._compute_gate(z, y)
        z_eff = self._assemble_z_eff(z, gate)
        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            z_eff,
            size_factor,
            batch_index,
            *categorical_input,
            y,
        )

        if self.dispersion == "gene-label":
            px_r = linear(one_hot(y, self.n_labels), self.px_r)
        elif self.dispersion == "gene-batch":
            px_r = linear(one_hot(batch_index, self.n_batch), self.px_r)
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate, theta=px_r, zi_logits=px_dropout, scale=px_scale
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        # library prior
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())

        # latent prior
        if self.prior_mixture is True:
            u_prior_logits = self.prior_logits
            u_prior_means = self.prior_means
            u_prior_scales = torch.exp(self.prior_log_scales) + 1e-4

            if self.semisupervised:
                logits_input = (
                    torch.stack(
                        [
                            torch.nn.functional.one_hot(y_i, self.n_labels)
                            if y_i < self.n_labels
                            else torch.zeros(self.n_labels)
                            for y_i in y.ravel()
                        ]
                    )
                    .to(z.device)
                    .float()
                )
                u_prior_logits = u_prior_logits + 10 * logits_input
                u_prior_means = u_prior_means.expand(y.shape[0], -1, -1)
                u_prior_scales = u_prior_scales.expand(y.shape[0], -1, -1)
            cats = Categorical(logits=u_prior_logits)
            normal_dists = Independent(
                Normal(u_prior_means, u_prior_scales), reinterpreted_batch_ndims=1
            )
            pz = MixtureSameFamily(cats, normal_dists)
        else:
            pz = Normal(torch.zeros_like(z), torch.ones_like(z))

        niche_composition = self.composition_decoder(
            niche_input, batch_index, *categorical_input
        )
        niche_mean, niche_variance = self.niche_decoder(
            niche_input, batch_index, *categorical_input
        )

        if self.niche_likelihood == "poisson":
            niche_expression = torch.distributions.Poisson(niche_variance)
        else:
            niche_expression = Normal(niche_mean, niche_variance)

        return {
            MODULE_KEYS.PX_KEY: px,
            MODULE_KEYS.PL_KEY: pl,
            MODULE_KEYS.PZ_KEY: pz,
            SCVIVA_MODULE_KEYS.NICHE_MEAN: niche_mean,
            SCVIVA_MODULE_KEYS.NICHE_VARIANCE: niche_variance,
            SCVIVA_MODULE_KEYS.P_NICHE_EXPRESSION: niche_expression,
            SCVIVA_MODULE_KEYS.P_NICHE_COMPOSITION: niche_composition,
            GATE_KEY: gate,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, **kwargs):
        """nicheVAE loss + decorrelation + semantic prior (+ optional gate entropy)."""
        loss_output = super().loss(tensors, inference_outputs, generative_outputs, **kwargs)
        extra = 0.0 * loss_output.loss

        # decorrelation over the expanded positive loadings
        if self.decorrelation_loss_weight != 0 and self.weights_positive:
            w = F.softplus(self.decoder.factor_regressor.fc_layers[0][0].X_layer.weight)
            decor = self._decorrelation_loss(w)
            extra = extra + self.decorrelation_loss_weight * decor
            loss_output.extra_metrics["decorrelation_loss"] = decor

        # semantic prior, gated by semantic_loss_scale (warmup)
        W = self._get_effective_loadings()
        weighted_sem = self.semantic_loss_scale * self.coherence_weight * self._semantic_loss(W)
        extra = extra + weighted_sem
        loss_output.extra_metrics["coherence_loss"] = weighted_sem
        loss_output.extra_metrics["semantic_scale"] = self.semantic_loss_scale

        # optional: encourage the batch-marginal gate to use all cell-type blocks
        # (resists collapse to a single shared type) without forcing per-cell uniformity
        if self.gate_entropy_weight != 0 and GATE_KEY in generative_outputs:
            g = generative_outputs[GATE_KEY]
            marginal = g.reshape(-1, g.shape[-1]).mean(dim=0)
            marginal_entropy = -(marginal * (marginal + 1e-8).log()).sum()
            extra = extra - self.gate_entropy_weight * marginal_entropy
            loss_output.extra_metrics["gate_marginal_entropy"] = marginal_entropy

        object.__setattr__(loss_output, "loss", loss_output.loss + extra)
        return loss_output
