from __future__ import annotations

import torch
from torch import nn

# re-export the scVIVA components so the subpackage is self-describing
from ..scviva._components import DirichletDecoder, Encoder, NicheDecoder

__all__ = ["DirichletDecoder", "Encoder", "NicheDecoder", "GateHead"]


class GateHead(nn.Module):
    """Linear head mapping the latent ``z`` to per-cell-type gate logits.

    Used to softly weight the cell-type-specific latent blocks when the gate is not
    tied to the model's classifier.
    """

    def __init__(self, n_latent: int, n_labels: int):
        super().__init__()
        self.fc = nn.Linear(n_latent, n_labels)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)
