from __future__ import annotations
import torch
import logging
from lightning.pytorch.callbacks import Callback
from scvi.model import LinearSCVI
from scvi.module._vae import SemanticLDVAE

logger = logging.getLogger(__name__)

class SemanticWarmupCallback(Callback):
    def __init__(self, warmup_epochs=10):
        super().__init__()
        self.warmup_epochs = warmup_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        current_epoch = trainer.current_epoch
        if current_epoch < self.warmup_epochs:
            if pl_module.module.semantic_loss_scale.item() != 0.0:
                pl_module.module.semantic_loss_scale.fill_(0.0)
        else:
            if pl_module.module.semantic_loss_scale.item() != 1.0:
                pl_module.module.semantic_loss_scale.fill_(1.0)

class SemanticSCVI(LinearSCVI):
    _module_cls = SemanticLDVAE 

    def __init__(
        self,
        adata,
        semantic_map: torch.Tensor,
        coherence_weight: float = 10.0,
        loss_mode: str = 'centroid', # or 'geometric'
        n_gene_sample: int = 1024,   
        **kwargs
    ):
        super().__init__(
            adata, 
            semantic_map=semantic_map, 
            coherence_weight=coherence_weight, 
            loss_mode=loss_mode,
            n_gene_sample=n_gene_sample,
            **kwargs
        )
        
        self._model_summary_string = (
            f"SemanticSCVI Model (Mode: {loss_mode})\n"
            f"Weight: {coherence_weight}, n_latent: {self.n_latent}"
        )

    def train(self, max_epochs=None, warmup_epochs=0, **kwargs):
        train_callbacks = kwargs.get("callbacks", []) or []
        if warmup_epochs > 0:
            print(f"🕒 Warmup Enabled: Semantic Loss ({self.module.loss_mode}) OFF for {warmup_epochs} epochs.")
            train_callbacks.insert(0, SemanticWarmupCallback(warmup_epochs))
            kwargs["callbacks"] = train_callbacks
        
        super(SemanticSCVI, self).train(max_epochs=max_epochs, **kwargs)