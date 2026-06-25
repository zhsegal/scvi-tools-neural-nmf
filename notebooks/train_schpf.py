import schpf
import numpy as np
import pandas as pd
from scipy.sparse import issparse, csr_matrix, coo_matrix

# --- 1. The Wrapper ---
class SCHPFWrapper:
    def __init__(self, model, cell_scores, gene_scores, feature_names):
        self.model = model
        self.cell_scores = cell_scores
        self.gene_scores = gene_scores
        self.feature_names = feature_names

    def get_latent_representation(self, adata=None):
        return self.cell_scores

    def get_loadings(self):
        return pd.DataFrame(
            self.gene_scores, 
            index=self.feature_names,
            columns=[f"Factor_{i}" for i in range(self.gene_scores.shape[1])]
        )

# --- 2. The Corrected Training Function ---
def train_schpf_model(adata, n_factors=10):
    print(f"--- Training scHPF (Factors={n_factors}) ---")
    
    # 1. Preprocess: scHPF needs COO Format + Integers
    # We convert to COO immediately to satisfy the '.col' requirement
    if issparse(adata.X):
        X_train = adata.X.tocoo()
    else:
        import scipy.sparse as sp
        X_train = sp.coo_matrix(adata.X)
    
    # Round to integers (scHPF requirement)
    # COO matrices generally don't support in-place data modification as easily as CSR,
    # so we cast the .data array directly.
    X_train.data = np.rint(X_train.data).astype(np.int32)
    
    # 2. Train Model
    model = schpf.scHPF(nfactors=n_factors, verbose=False)
    
    # scHPF often internally converts to CSR, but for the input check that failed, 
    # passing COO is the safest bet given the error.
    model.fit(X_train)
    
    print("scHPF training complete.")
    
    # 3. Return Wrapper
    return SCHPFWrapper(
        model=model,
        cell_scores=model.cell_score(),
        gene_scores=model.gene_score(),
        feature_names=adata.var_names
    )
    

from sklearn.decomposition import NMF
import pandas as pd
import numpy as np
from scipy.sparse import issparse

from sklearn.decomposition import NMF
import pandas as pd
import numpy as np
from scipy.sparse import issparse

# --- 1. The Wrapper (Must be defined first) ---
class NMFWrapper:
    def __init__(self, model, W, H, feature_names):
        self.model = model
        self.W = W
        self.H = H
        self.feature_names = feature_names

    def get_latent_representation(self, adata=None):
        return self.W

    def get_loadings(self):
        # Transpose H from (factors, genes) to (genes, factors)
        return pd.DataFrame(
            self.H.T, 
            index=self.feature_names,
            columns=[f"Factor_{i}" for i in range(self.H.shape[0])]
        )

def train_nmf_model(adata, n_factors=10):
    print(f"--- Training NMF (Factors={n_factors}) ---")
    
    # Handle Data (Negatives and Sparsity)
    if issparse(adata.X):
        X_train = adata.X
        if (X_train.data < 0).any():
             print("Warning: Sparse data contains negatives. Clipping to 0.")
             X_train.data = np.maximum(X_train.data, 0)
    else:
        X_train = adata.X
        if (X_train < 0).any():
            print("Warning: Dense data contains negatives. Clipping to 0.")
            X_train = np.maximum(X_train, 0)

    # Train
    model = NMF(n_components=n_factors, init='nndsvd', random_state=42, max_iter=500)
    W = model.fit_transform(X_train)
    H = model.components_
    
    return model, W, H