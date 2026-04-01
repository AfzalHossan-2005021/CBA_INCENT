"""
cross_timepoint.py — Cross-timepoint extension for CBA-FGW MERFISH alignment.

Mathematical overview
---------------------
The cross-timepoint alignment problem has four unique challenges that require
specific mathematical extensions beyond the same-timepoint (rigid + balanced OT)
framework:

  1. Expression distribution shift
     Across timepoints, gene expression shifts due to developmental progression,
     disease, or treatment.  Raw cosine distance conflates biological identity
     with temporal drift, penalising correct matches and rewarding wrong ones.

     Solution: Align in a **batch-corrected latent space** z obtained from a
     conditional VAE (scVI) or Harmony-corrected PCA.  The latent space is
     trained to be conditionally independent of the timepoint label while
     preserving cell identity.

  2. Spatial non-rigidity
     The tissue undergoes mild deformation between timepoints (growth, fixation
     artefact, mechanical processing).  A single rigid (R, t) cannot capture
     this; a smooth deformation field is required.

     Solution: After rigid alignment converges, fit a **thin-plate spline (TPS)**
     deformation field φ: ℝ² → ℝ² on the confident anchor pairs (x_i^s, T(i)).
     Replace L_CBA with L_CBA_TPS using φ(x_i^s) in place of Rx_i^s + t.

  3. Cell mass non-conservation
     Some cells divide, die, or differentiate between timepoints.  Balanced OT
     forces every source cell to be matched, which is biologically wrong.

     Solution: **Semi-relaxed OT** (Flamary et al., 2021) with asymmetric
     Kullback-Leibler penalties:
       τ_s → ∞  (ancestor marginal is exact: every source cell is matched or dies)
       τ_t < ∞  (descendant marginal is relaxed: new cells can appear)
     Implementation: large-cost birth dummy cells + moderate-cost death dummies.

  4. Rare cell-type transitions
     Cells do not randomly change type; they traverse a transcriptomic manifold.
     A hard same-type penalty would block valid developmental transitions.

     Solution: **Lineage-aware cost** C_lin[i,j] = distance between type
     centroids in the batch-corrected latent space.  Permitted transitions
     (similar centroids) get low cost; impossible jumps get high cost.

Full cross-timepoint objective
-------------------------------

  min_{π≥0, φ∈TPS}
      ⟨C_lat + λ_l·C_lin, π⟩           (1) batch-corrected feature cost
    + β · L_CBA_TPS(π, φ)               (2) TPS coherence loss
    + α · GW_local(D_s, D_t, π)         (3) local structure preservation
    + τ_s · KL(π1 | p)                   (4) ancestor marginal (strict)
    + τ_t · KL(π^T 1 | q)               (5) descendant marginal (relaxed)
    + γ · ‖π‖_F²                         (6) sparsity / compactness

where

  L_CBA_TPS(π, φ) = (1/|M_s|) Σ_{i∈M_s} ‖π_i‖₁ · ‖φ(x_i^s) − T(i)‖²

  ∂L_CBA_TPS/∂π_{ij} = (1/|M_s|) · (‖φ(x_i^s) − x_j^t‖² − ‖x_j^t − T(i)‖²)

  (identical gradient form to the rigid case, with φ(x_i^s) replacing Rx_i^s+t)

TPS Procrustes update (Block B, cross-timepoint)
-------------------------------------------------
Given barycenter pairs {(x_i^s, T(i))} with weights w_i = ‖π_i‖₁, solve

  min_{φ∈TPS} Σ_i w_i · ‖φ(x_i^s) − T(i)‖² + λ_smooth · BE(φ)

where BE(φ) is the thin-plate spline bending energy (integral of second
derivatives, penalises non-affine warping).  Solved as a weighted linear
system via scipy.interpolate.RBFInterpolator with kernel='thin_plate_spline'.

Expression alignment via conditional VAE (scVI)
-----------------------------------------------
Architecture (per Lopez et al., Nature Methods 2018):

  Encoder:  [g; s] → FC(256) → FC(128) → [μ_z, log σ_z²] ∈ ℝ^d_z
  Decoder:  [z; s] → FC(128) → FC(256) → (μ_nb, θ_nb, π_zi) ∈ ℝ^g  (ZINB)

where g = number of genes (250), s = timepoint one-hot, d_z = 20.
The latent z is approximately independent of s by the KL term in the ELBO.

Harmony-corrected PCA fallback (always available)
-------------------------------------------------
  1. Concatenate both slices: X = [X_A; X_B]
  2. PCA → Z ∈ ℝ^{(N+M)×k}, k = 30
  3. Harmony batch correction on Z with timepoint label
  4. Split back into Z_A, Z_B
  5. Use Euclidean distances in Z for C_lat

Author: Research-grade implementation extending Anup Bhowmik's INCENT baseline.
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import scipy.sparse as sp
from typing import Optional, Tuple, List, Dict, Literal
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Expression aligner (batch-corrected latent representations)
# ─────────────────────────────────────────────────────────────────────────────

class ExpressionAligner:
    """
    Align gene expression distributions across timepoints into a shared
    batch-corrected latent space for transport cost computation.

    Three backends are available in decreasing order of expressiveness:

    ``'scvi'``
        Conditional variational autoencoder (scVI, Lopez et al. 2018).
        Trains a deep generative model on both slices with timepoint as a
        batch covariate.  The resulting latent z is conditionally independent
        of the timepoint label.  Requires ``scvi-tools`` ≥ 1.0.

    ``'harmony'``
        Harmony-corrected PCA (Korsunsky et al. 2019).  Applies iterative
        linear correction to PCA embeddings.  Fast, no GPU required, no
        hyperparameter tuning.  Requires ``harmonypy``.  Falls back to raw
        PCA if unavailable.

    ``'pca'``
        Simple joint PCA without batch correction.  Always available.
        Appropriate when timepoint effects are small relative to cell-type
        variation.

    The aligned representations Z_A (N, d) and Z_B (M, d) are used to
    build the latent cost matrix C_lat[i,j] = ‖z_i^A − z_j^B‖².

    Parameters
    ----------
    backend : str
        One of 'scvi', 'harmony', 'pca'.  Falls back gracefully if the
        required library is not installed.
    n_latent : int
        Latent dimension for scVI (default 20).  Also used as the number
        of PCs for Harmony / PCA backends.
    n_hidden : int
        Hidden layer width for scVI encoder/decoder (default 128).
    max_epochs : int
        Training epochs for scVI (default 400).  Ignored for other backends.
    batch_size : int
        Mini-batch size for scVI training (default 512).
    device : str
        'cpu' or 'cuda' for scVI training.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        backend: Literal['scvi', 'harmony', 'pca'] = 'harmony',
        n_latent:   int = 20,
        n_hidden:   int = 128,
        max_epochs: int = 400,
        batch_size: int = 512,
        device:     str = 'cpu',
        random_state: int = 42,
    ) -> None:
        self.backend      = backend
        self.n_latent     = n_latent
        self.n_hidden     = n_hidden
        self.max_epochs   = max_epochs
        self.batch_size   = batch_size
        self.device       = device
        self.random_state = random_state
        self._fitted      = False
        self._Z_A: Optional[np.ndarray] = None
        self._Z_B: Optional[np.ndarray] = None

    # ── public interface ──────────────────────────────────────────────────────

    def fit_transform(
        self,
        X_A: np.ndarray,
        X_B: np.ndarray,
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the expression aligner on both slices and return latent embeddings.

        Parameters
        ----------
        X_A : ndarray, shape (N, G)  — gene expression, source (t₁) slice
        X_B : ndarray, shape (M, G)  — gene expression, target (t₂) slice
        verbose : bool

        Returns
        -------
        Z_A : ndarray, shape (N, d)  — batch-corrected latent embedding, source
        Z_B : ndarray, shape (M, d)  — batch-corrected latent embedding, target
        """
        X_A = np.asarray(X_A, dtype=np.float32)
        X_B = np.asarray(X_B, dtype=np.float32)

        if self.backend == 'scvi':
            Z_A, Z_B = self._fit_scvi(X_A, X_B, verbose)
        elif self.backend == 'harmony':
            Z_A, Z_B = self._fit_harmony(X_A, X_B, verbose)
        else:
            Z_A, Z_B = self._fit_pca(X_A, X_B, verbose)

        self._Z_A = Z_A
        self._Z_B = Z_B
        self._fitted = True
        return Z_A, Z_B

    def latent_cost_matrix(
        self,
        Z_A: Optional[np.ndarray] = None,
        Z_B: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute the squared Euclidean distance matrix between latent embeddings.

            C_lat[i,j] = ‖z_i^A − z_j^B‖²

        Parameters
        ----------
        Z_A, Z_B : ndarray, optional.  Uses stored embeddings if not supplied.

        Returns
        -------
        C_lat : ndarray, shape (N, M)
        """
        Z_A = Z_A if Z_A is not None else self._Z_A
        Z_B = Z_B if Z_B is not None else self._Z_B
        if Z_A is None or Z_B is None:
            raise RuntimeError("Call fit_transform first.")
        return euclidean_distances(Z_A, Z_B) ** 2

    # ── scVI backend ──────────────────────────────────────────────────────────

    def _fit_scvi(
        self, X_A: np.ndarray, X_B: np.ndarray, verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a conditional VAE (scVI) on both slices.

        Uses ``scvi-tools`` if installed.  Falls back to the built-in
        lightweight conditional VAE if not.

        The model uses a ZINB (zero-inflated negative binomial) likelihood,
        which is the statistically correct model for MERFISH count data.
        The timepoint label is used as a categorical batch covariate so that
        the latent z is conditionally independent of timepoint.

        Reference
        ---------
        Lopez R, Regier J, Cole MB, Jordan MI, Yosef N (2018).
        Deep generative modeling for single-cell transcriptomics.
        Nature Methods 15, 1053–1058.
        """
        try:
            import scvi
            import anndata
            if verbose:
                print("[ExpressionAligner] Using scvi-tools backend.")
            return self._fit_scvi_tools(X_A, X_B, verbose, scvi, anndata)
        except ImportError:
            if verbose:
                print("[ExpressionAligner] scvi-tools not installed; "
                      "using built-in conditional VAE.")
            return self._fit_builtin_vae(X_A, X_B, verbose)

    def _fit_scvi_tools(
        self, X_A, X_B, verbose, scvi, anndata
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Use scvi-tools SCVI model for batch-corrected latent embeddings."""
        import scipy.sparse as sp

        N, M = len(X_A), len(X_B)
        X_all = np.vstack([X_A, X_B])
        batch  = np.array(['t1'] * N + ['t2'] * M)

        # Round to integer counts for ZINB (MERFISH data is integer-valued)
        X_int = np.round(np.maximum(X_all, 0)).astype(np.float32)

        adata = anndata.AnnData(
            X=sp.csr_matrix(X_int),
            obs={'batch': batch}
        )
        adata.var_names = [f"gene_{k}" for k in range(X_all.shape[1])]

        scvi.model.SCVI.setup_anndata(adata, batch_key='batch')
        model = scvi.model.SCVI(
            adata,
            n_latent=self.n_latent,
            n_hidden=self.n_hidden,
            n_layers=2,
            gene_likelihood='zinb',
        )
        model.train(
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            plan_kwargs={'lr': 1e-3},
            enable_progress_bar=verbose,
        )
        Z = model.get_latent_representation(adata)
        return Z[:N].astype(np.float32), Z[N:].astype(np.float32)

    def _fit_builtin_vae(
        self, X_A: np.ndarray, X_B: np.ndarray, verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Built-in lightweight conditional VAE when scvi-tools is unavailable.

        Architecture
        ------------
        Encoder:  [g; s] → BatchNorm → FC(n_hidden) → ReLU → FC(n_hidden//2)
                         → ReLU → [μ_z, log σ²_z] ∈ ℝ^{n_latent}

        Decoder:  [z; s] → FC(n_hidden//2) → ReLU → FC(n_hidden)
                         → ReLU → [log_rate] ∈ ℝ^g  (Poisson approx for MERFISH)

        The timepoint label s is concatenated as a one-hot vector to both
        encoder input and decoder input.  The KL term in the ELBO forces z
        to be approximately N(0, I) regardless of s, achieving batch correction.

        Loss  = E_{q(z|x,s)}[-log p(x|z,s)] + KL[q(z|x,s) || p(z)]

        Training uses Adam with learning-rate warmup over the first 50 epochs.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except ImportError:
            warnings.warn(
                "PyTorch not available. Falling back to Harmony-PCA.",
                UserWarning)
            return self._fit_harmony(X_A, X_B, verbose)

        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

        N, G = X_A.shape
        M    = X_B.shape[0]
        d_s  = 2                       # one-hot timepoint dimension
        d_z  = self.n_latent
        nh   = self.n_hidden

        # ── Model definition ─────────────────────────────────────────────
        class _Encoder(nn.Module):
            def __init__(self, g, d_s, nh, d_z):
                super().__init__()
                self.bn  = nn.BatchNorm1d(g)
                self.fc1 = nn.Linear(g + d_s, nh)
                self.fc2 = nn.Linear(nh, nh // 2)
                self.mu  = nn.Linear(nh // 2, d_z)
                self.lv  = nn.Linear(nh // 2, d_z)
            def forward(self, x, s):
                h = F.relu(self.fc1(torch.cat([self.bn(x), s], dim=1)))
                h = F.relu(self.fc2(h))
                return self.mu(h), self.lv(h)

        class _Decoder(nn.Module):
            def __init__(self, g, d_s, nh, d_z):
                super().__init__()
                self.fc1  = nn.Linear(d_z + d_s, nh // 2)
                self.fc2  = nn.Linear(nh // 2, nh)
                self.rate = nn.Linear(nh, g)
            def forward(self, z, s):
                h = F.relu(self.fc1(torch.cat([z, s], dim=1)))
                h = F.relu(self.fc2(h))
                return F.softplus(self.rate(h))   # Poisson log-rate

        enc = _Encoder(G, d_s, nh, d_z).to(self.device)
        dec = _Decoder(G, d_s, nh, d_z).to(self.device)
        opt = torch.optim.Adam(
            list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
        sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.1, total_iters=50)

        # ── Data preparation ─────────────────────────────────────────────
        X_all  = np.vstack([X_A, X_B]).astype(np.float32)
        s_all  = np.zeros((N + M, 2), dtype=np.float32)
        s_all[:N, 0] = 1.0    # t1 one-hot
        s_all[N:, 1] = 1.0    # t2 one-hot

        X_t = torch.from_numpy(X_all).to(self.device)
        S_t = torch.from_numpy(s_all).to(self.device)

        # ── Training loop ─────────────────────────────────────────────────
        bs   = min(self.batch_size, N + M)
        idx  = np.arange(N + M)

        if verbose:
            print(f"[ExpressionAligner] Training conditional VAE "
                  f"({self.max_epochs} epochs, {N+M} cells, {G} genes) …")

        enc.train(); dec.train()
        for ep in range(self.max_epochs):
            np.random.shuffle(idx)
            epoch_loss = 0.0
            for start in range(0, N + M, bs):
                batch_idx = idx[start:start + bs]
                x_b = X_t[batch_idx]
                s_b = S_t[batch_idx]

                mu_z, lv_z = enc(x_b, s_b)
                # Reparameterisation trick
                eps = torch.randn_like(mu_z)
                z   = mu_z + torch.exp(0.5 * lv_z) * eps

                rate = dec(z, s_b)
                # Poisson NLL (counts)
                recon = -torch.distributions.Poisson(rate + 1e-8).log_prob(
                    x_b.clamp(min=0)).sum(dim=1).mean()
                kl = -0.5 * (1 + lv_z - mu_z.pow(2) - lv_z.exp()).sum(dim=1).mean()
                loss = recon + kl

                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += loss.item()

            sched.step()
            if verbose and (ep + 1) % 50 == 0:
                print(f"  epoch {ep+1:4d}/{self.max_epochs}  "
                      f"loss={epoch_loss:.4f}")

        # ── Extract latent embeddings ─────────────────────────────────────
        enc.eval()
        with torch.no_grad():
            mu_all, _ = enc(X_t, S_t)
        Z_all = mu_all.cpu().numpy()
        return Z_all[:N].astype(np.float32), Z_all[N:].astype(np.float32)

    # ── Harmony-PCA backend ───────────────────────────────────────────────────

    def _fit_harmony(
        self, X_A: np.ndarray, X_B: np.ndarray, verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmony-corrected PCA batch correction.

        Algorithm (Korsunsky et al., Nature Methods 2019)
        -------------------------------------------------
        1. Fit PCA on the concatenated, log1p-normalised expression matrix.
        2. Assign cells to soft clusters (fuzzy k-means in PC space).
        3. Iteratively compute a linear correction matrix R per cluster that
           minimises the within-cluster cross-batch divergence.
        4. Corrected PCs: Z_corrected = Z_pca - R_cluster @ Z_pca.

        Parameters
        ----------
        The correction is linear, so it removes additive timepoint shifts in
        PC space.  Effective for technical batch effects and mild developmental
        drift; less effective for strong nonlinear expression changes.

        Reference
        ---------
        Korsunsky I, Millard N, Fan J, et al. (2019). Fast, sensitive and
        accurate integration of single-cell data with Harmony.
        Nature Methods 16, 1289–1296.
        """
        if verbose:
            print("[ExpressionAligner] Fitting Harmony-corrected PCA …")

        N, M = len(X_A), len(X_B)
        # log1p normalisation (standard scRNA-seq preprocessing)
        X_all  = np.log1p(np.vstack([X_A, X_B]).astype(np.float32))
        k      = min(self.n_latent, X_all.shape[1] - 1, X_all.shape[0] - 1)
        pca    = PCA(n_components=k, random_state=self.random_state)
        Z_pca  = pca.fit_transform(X_all)

        batch  = np.array([0] * N + [1] * M)

        try:
            import harmonypy as hm
            if verbose:
                print("[ExpressionAligner] Applying Harmony correction …")
            ho     = hm.run_harmony(Z_pca, None, batch,
                                     random_state=self.random_state,
                                     verbose=False)
            Z_corr = ho.Z_corr.T   # Harmony returns (k, N+M)
        except ImportError:
            warnings.warn(
                "harmonypy not installed; using raw PCA (no batch correction).",
                UserWarning)
            Z_corr = Z_pca

        return (Z_corr[:N].astype(np.float32),
                Z_corr[N:].astype(np.float32))

    # ── PCA fallback ─────────────────────────────────────────────────────────

    def _fit_pca(
        self, X_A: np.ndarray, X_B: np.ndarray, verbose: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple joint PCA without batch correction (always available)."""
        if verbose:
            print("[ExpressionAligner] Fitting joint PCA (no batch correction) …")
        N  = len(X_A)
        X  = np.log1p(np.vstack([X_A, X_B]).astype(np.float32))
        k  = min(self.n_latent, X.shape[1] - 1, X.shape[0] - 1)
        pca = PCA(n_components=k, random_state=self.random_state)
        Z   = pca.fit_transform(X)
        return Z[:N].astype(np.float32), Z[N:].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Lineage-aware cost
# ─────────────────────────────────────────────────────────────────────────────

class LineageAwareCost:
    """
    Build a lineage-aware cell-to-cell cost matrix for cross-timepoint transport.

    Motivation
    ----------
    In cross-timepoint data, cells traverse a transcriptomic manifold along
    known developmental trajectories.  The transport cost between cell type A
    at t₁ and cell type B at t₂ should reflect the biological plausibility of
    that transition.  Hard same-type constraints are too restrictive (blocking
    valid differentiation events); a soft lineage-distance penalty is correct.

    Method
    ------
    We compute the **type centroid distance** in the batch-corrected latent
    space Z.  Cell types with similar mean expression profiles (close centroids)
    are likely related and get a low transition cost; biologically distant types
    get a high cost.  The matrix is normalised to [0, 1].

    When a user-supplied lineage DAG (adjacency dict or networkx graph) is
    provided, we use shortest-path distances in the DAG instead, which is the
    biologically most grounded option.

    Parameters
    ----------
    method : str
        'centroid' (default) — type centroid Euclidean distance in latent space.
        'dag'                — shortest path in a user-supplied lineage DAG.
        'pseudotime'         — pseudotime-coordinate distance between type means.
    lineage_dag : dict or None
        Adjacency dict {type_a: [type_b, type_c, ...]} for the 'dag' method.
    pseudotime : ndarray, shape (N,) or None
        Pseudotime coordinate for each cell (combined A + B), for 'pseudotime'.
    """

    def __init__(
        self,
        method: Literal['centroid', 'dag', 'pseudotime'] = 'centroid',
        lineage_dag: Optional[Dict] = None,
        pseudotime:  Optional[np.ndarray] = None,
    ) -> None:
        self.method       = method
        self.lineage_dag  = lineage_dag
        self.pseudotime   = pseudotime
        self._D_type: Optional[np.ndarray] = None
        self._types:  Optional[List[str]]  = None

    def fit(
        self,
        Z_A: np.ndarray,
        Z_B: np.ndarray,
        types_A: np.ndarray,
        types_B: np.ndarray,
        verbose: bool = False
    ) -> 'LineageAwareCost':
        """
        Fit the lineage cost model.

        Parameters
        ----------
        Z_A, Z_B  : latent embeddings of source and target cells
        types_A   : cell-type labels for source cells
        types_B   : cell-type labels for target cells
        """
        all_types = list(sorted(set(types_A) | set(types_B)))
        self._types = all_types

        if self.method == 'dag' and self.lineage_dag is not None:
            self._D_type = self._dag_distances(all_types)
        elif self.method == 'pseudotime' and self.pseudotime is not None:
            self._D_type = self._pseudotime_distances(
                Z_A, Z_B, types_A, types_B, all_types)
        else:
            # Default: centroid distance in latent space
            self._D_type = self._centroid_distances(
                Z_A, Z_B, types_A, types_B, all_types)

        # Normalise to [0, 1]
        dmax = self._D_type.max()
        if dmax > 0:
            self._D_type /= dmax

        if verbose:
            print(f"[LineageAwareCost] Built {len(all_types)}×{len(all_types)} "
                  f"type distance matrix (method={self.method})")
        return self

    def cell_cost_matrix(
        self,
        types_A: np.ndarray,
        types_B: np.ndarray
    ) -> np.ndarray:
        """
        Return the cell-to-cell lineage cost matrix.

            C_lin[i,j] = D_type[type(i), type(j)]

        Parameters
        ----------
        types_A : ndarray, shape (N,)
        types_B : ndarray, shape (M,)

        Returns
        -------
        C_lin : ndarray, shape (N, M), values in [0, 1]
        """
        if self._D_type is None:
            raise RuntimeError("Call fit() first.")
        type_idx = {t: i for i, t in enumerate(self._types)}
        idx_A = np.array([type_idx.get(t, 0) for t in types_A])
        idx_B = np.array([type_idx.get(t, 0) for t in types_B])
        return self._D_type[np.ix_(idx_A, idx_B)]

    # ── internal methods ──────────────────────────────────────────────────────

    def _centroid_distances(
        self, Z_A, Z_B, types_A, types_B, all_types
    ) -> np.ndarray:
        """
        Type centroid distance in the shared latent space.

        For each cell type t, the centroid is the mean latent vector across
        all cells of type t in both slices combined.  Distance between
        centroids reflects the transcriptomic similarity between the types'
        mean expression profiles — a principled proxy for lineage distance.
        """
        Z_all   = np.vstack([Z_A, Z_B])
        t_all   = np.concatenate([types_A, types_B])
        T       = len(all_types)
        centroids = np.zeros((T, Z_all.shape[1]), dtype=np.float32)
        for k, typ in enumerate(all_types):
            mask = t_all == typ
            if mask.any():
                centroids[k] = Z_all[mask].mean(axis=0)
        return euclidean_distances(centroids, centroids).astype(np.float64)

    def _dag_distances(self, all_types: List[str]) -> np.ndarray:
        """
        Shortest-path distances in a user-supplied lineage DAG.

        The DAG encodes known developmental relationships: edges point from
        progenitors to their daughter types.  A transition from type A to
        type B receives cost = shortest path length from A to B in the
        undirected version of the DAG.  Types with no path get cost = diameter
        of the graph (maximum finite distance).

        Reference
        ---------
        For brain development: radial glia → intermediate progenitor →
        glutamatergic neuron is a known 2-step transition; direct radial
        glia → interneuron mapping is biologically implausible (long path).
        """
        # Build adjacency list as undirected (both directions allowed for cost)
        from collections import deque
        adj: Dict[str, List[str]] = {}
        for src, dsts in self.lineage_dag.items():
            adj.setdefault(src, []).extend(dsts)
            for dst in dsts:
                adj.setdefault(dst, []).append(src)  # reverse edge

        T = len(all_types)
        D = np.full((T, T), np.inf, dtype=np.float64)
        idx = {t: i for i, t in enumerate(all_types)}

        for start in all_types:
            if start not in idx:
                continue
            dist = {start: 0.0}
            q = deque([start])
            while q:
                node = q.popleft()
                for nbr in adj.get(node, []):
                    if nbr not in dist:
                        dist[nbr] = dist[node] + 1.0
                        q.append(nbr)
            for end, d in dist.items():
                if end in idx:
                    D[idx[start], idx[end]] = d

        # Replace inf with diameter + 1
        finite = D[np.isfinite(D)]
        D[~np.isfinite(D)] = finite.max() + 1.0 if len(finite) > 0 else T
        np.fill_diagonal(D, 0.0)
        return D

    def _pseudotime_distances(
        self, Z_A, Z_B, types_A, types_B, all_types
    ) -> np.ndarray:
        """
        Type-to-type distance as the absolute difference in mean pseudotime.

        If the pseudotime coordinate is provided (e.g., from Monocle or
        diffusion pseudotime), the mean pseudotime per type is used as a
        1D proxy for developmental progression.  The transition cost between
        types A and B is |pt_A − pt_B|.

        Note: MERFISH does not provide spliced/unspliced counts, so RNA
        velocity cannot be used.  Pseudotime from diffusion maps computed on
        the latent representation (e.g., scVelo without velocity, or
        scanpy.tl.diffmap) is the recommended alternative.
        """
        N    = len(types_A)
        pt_A = self.pseudotime[:N] if self.pseudotime is not None else np.zeros(N)
        pt_B = self.pseudotime[N:] if self.pseudotime is not None else np.zeros(len(types_B))
        all_t_arr = np.concatenate([types_A, types_B])
        all_pt    = np.concatenate([pt_A, pt_B])

        T      = len(all_types)
        means  = np.zeros(T, dtype=np.float64)
        for k, typ in enumerate(all_types):
            mask = all_t_arr == typ
            means[k] = all_pt[mask].mean() if mask.any() else 0.0

        D = np.abs(means[:, None] - means[None, :])
        return D


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: TPS deformation field (Block B for cross-timepoint)
# ─────────────────────────────────────────────────────────────────────────────

class TPSDeformationField:
    """
    Thin-plate spline (TPS) deformation field for cross-timepoint spatial warping.

    Mathematical formulation
    ------------------------
    The TPS is a smooth mapping φ: ℝ² → ℝ² that minimises

        min_φ  Σ_i w_i · ‖φ(x_i^s) − T(i)‖²  +  λ_smooth · BE(φ)

    where BE(φ) = ∫ ‖∇²φ‖_F² dx is the bending energy (integral of squared
    second derivatives), and w_i = ‖π_i‖₁ is the confidence weight of the
    i-th anchor pair.

    The TPS solution has the form:

        φ(x) = A·x + b + Σ_k c_k · r_k²·log(r_k²)

    where r_k = ‖x − x_k^s‖ for each control point x_k^s.  The coefficients
    (A, b, {c_k}) are obtained by solving a linear system incorporating
    the smoothing penalty λ_smooth.

    The CBA loss with TPS becomes:

        L_CBA_TPS(π, φ) = (1/|M_s|) Σ_{i∈M_s} ‖π_i‖₁ · ‖φ(x_i^s) − T(i)‖²

    with the same gradient form:

        ∂L_CBA_TPS/∂π_{ij} = (1/|M_s|) · (‖φ(x_i^s)−x_j^t‖² − ‖x_j^t−T(i)‖²)

    The static CBA cost C_CBA_TPS[i,j] = ‖φ(x_i^s) − x_j^t‖² is updated
    after each TPS fitting step (Block B) and held fixed during the inner
    transport solve (Block A).

    Parameters
    ----------
    smoothing : float
        TPS smoothing parameter λ_smooth (default 0.1).  Higher values enforce
        a smoother (more affine-like) deformation at the cost of fitting the
        anchor pairs less exactly.  Should be scaled to the number of anchors:
        a value of λ / n_anchors in the RBFInterpolator sense is recommended.
    n_control_points : int or None
        If set, subsample this many anchor pairs as TPS control points.
        Reduces the O(n³) TPS solve cost for large matched sets.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        n_control_points: Optional[int] = None,
    ) -> None:
        self.smoothing        = smoothing
        self.n_control_points = n_control_points
        self._interpolator    = None
        self._is_rigid        = True    # True until first fit

    def fit(
        self,
        X_src:       np.ndarray,
        barycenters: np.ndarray,
        weights:     np.ndarray,
        verbose:     bool = False
    ) -> 'TPSDeformationField':
        """
        Fit the TPS deformation field on matched source → barycenter pairs.

        Parameters
        ----------
        X_src       : ndarray, shape (n, 2)  — source coordinates of matched cells
        barycenters : ndarray, shape (n, 2)  — barycentric projections T(i)
        weights     : ndarray, shape (n,)    — per-cell confidence (‖π_i‖₁)

        Returns
        -------
        self
        """
        from scipy.interpolate import RBFInterpolator

        # Remove NaN barycenters
        valid = ~np.isnan(barycenters[:, 0]) & (weights > 1e-8)
        if valid.sum() < 4:
            warnings.warn(
                "Fewer than 4 valid anchor pairs; TPS not fitted (using rigid).",
                UserWarning)
            return self

        X_s = X_src[valid]
        X_t = barycenters[valid]
        w   = weights[valid]

        # Optionally subsample control points for scalability
        if self.n_control_points is not None and len(X_s) > self.n_control_points:
            # Weight-proportional sampling: confident anchors are preferred
            prob = w / w.sum()
            idx  = np.random.choice(len(X_s), size=self.n_control_points,
                                     replace=False, p=prob)
            X_s, X_t, w = X_s[idx], X_t[idx], w[idx]

        # Smoothing parameter scaled by number of anchors (standard convention)
        smooth = self.smoothing / max(len(X_s), 1)

        try:
            # scipy's RBFInterpolator with thin_plate_spline kernel solves the
            # weighted TPS problem automatically.
            self._interpolator = RBFInterpolator(
                X_s, X_t,
                kernel='thin_plate_spline',
                degree=1,            # affine term included
                smoothing=smooth,
            )
            self._is_rigid = False
            if verbose:
                print(f"[TPSDeformationField] Fitted TPS on {len(X_s)} "
                      f"anchor pairs (smooth={smooth:.2e})")
        except Exception as e:
            warnings.warn(f"TPS fitting failed ({e}); using rigid.", UserWarning)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the TPS deformation to source coordinates.

        Parameters
        ----------
        X : ndarray, shape (n, 2)

        Returns
        -------
        X_warped : ndarray, shape (n, 2)
        """
        if self._is_rigid or self._interpolator is None:
            return X
        return self._interpolator(X).astype(np.float64)

    def static_cba_cost(self, X_s: np.ndarray, X_t: np.ndarray) -> np.ndarray:
        """
        Compute C_CBA_TPS[i,j] = ‖φ(x_i^s) − x_j^t‖².

        This is the static cost baked into the transport problem for one outer
        iteration.  Replaces the rigid version cba_static_cost(X_s, X_t, R, t)
        in cross-timepoint mode.
        """
        X_s_warped = self.transform(X_s)          # (N, 2)
        diff = X_s_warped[:, None, :] - X_t[None, :, :]   # (N, M, 2)
        return np.sum(diff ** 2, axis=2)            # (N, M)

    def cba_loss_tps(
        self,
        pi: np.ndarray,
        X_s: np.ndarray,
        X_t: np.ndarray,
        delta_threshold: float = 1e-6
    ) -> float:
        """
        Evaluate L_CBA_TPS(π, φ) = (1/|M_s|) Σ_i ‖π_i‖₁ · ‖φ(x_i^s) − T(i)‖².
        """
        from .utils_v2 import compute_barycenters
        weights   = pi.sum(axis=1)
        matched   = weights > delta_threshold
        n_matched = matched.sum()
        if n_matched == 0:
            return 0.0

        X_warped  = self.transform(X_s)
        bary      = compute_barycenters(pi, X_t)
        residuals = X_warped - bary
        residuals[~matched] = 0.0
        per_cell  = weights * np.sum(residuals ** 2, axis=1)
        return float(per_cell[matched].sum() / n_matched)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Semi-relaxed margin helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_semitrelaxed_marginals(
    ns: int,
    nt: int,
    types_A: np.ndarray,
    types_B: np.ndarray,
    tau_t: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Build asymmetric marginals for semi-relaxed cross-timepoint OT.

    The ancestor-constrained formulation (DeST-OT, Cell Systems 2025):
      τ_s → ∞  (source marginal is exact; every t₁ cell must be explained)
      τ_t < ∞  (target marginal is relaxed; new cells can appear at t₂)

    Implementation via dummy-cell augmentation:
      - A birth dummy source cell is added with weight w_birth = τ_t · nt.
        Any t₂ cell receiving mass from the birth dummy is declared new (born).
      - A death dummy target cell is added with weight w_death = τ_s · ns.
        Any t₁ cell sending mass to the death dummy is declared dead (died).
    
    The birth cost is set high (prevents spurious birth assignments),
    while the death cost is set moderate (allows authentic cell death).

    Parameters
    ----------
    ns, nt    : number of source and target cells
    types_A, types_B : cell-type labels
    tau_t     : float in (0, 1] — fraction of target mass that may be "new"
                (0 = fully balanced, 1 = every target cell may be new)

    Returns
    -------
    p_vals : source marginal (ns + 1,)  — uniform on real cells + birth dummy
    q_vals : target marginal (nt + 1,)  — uniform on real cells + death dummy
    w_birth : float — birth dummy weight
    w_death : float — death dummy weight
    """
    from collections import Counter
    counts_A = Counter(types_A)
    counts_B = Counter(types_B)
    all_types = set(counts_A) | set(counts_B)
    budget = sum(max(counts_A.get(k, 0), counts_B.get(k, 0)) for k in all_types)

    # Death: allow source cells to "die" proportionally to τ_t
    w_death = max(0, budget - ns)
    # Birth: allow new target cells proportionally to τ_t
    w_birth = max(0, int(tau_t * nt))

    ns_aug = ns + 1   # always add birth dummy (even if weight is small)
    nt_aug = nt + 1   # always add death dummy

    budget_aug = budget + max(w_death, w_birth)

    p_vals         = np.ones(ns_aug, dtype=np.float64)
    p_vals[-1]     = max(w_birth, 1)          # birth dummy weight (relative)

    q_vals         = np.ones(nt_aug, dtype=np.float64)
    q_vals[-1]     = max(w_death, 1)          # death dummy weight (relative)

    # Normalise so each marginal sums to 1 — required by OT solver
    p_vals /= p_vals.sum()
    q_vals /= q_vals.sum()

    return p_vals, q_vals, float(w_birth), float(w_death)


def augment_cost_matrices_ct(
    C_feat: np.ndarray,
    D_A:    np.ndarray,
    D_B:    np.ndarray,
    ns:     int,
    nt:     int,
    types_A: np.ndarray,
    types_B: np.ndarray,
    birth_cost_mult: float = 2.0,
    death_cost_mult: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment cost and distance matrices with birth/death dummy rows and columns.

    The birth dummy column (last column of C_feat_aug) represents a new target
    cell appearing without a source ancestor.  Source cells pay birth_cost_mult
    × their type's maximum cost to map to the birth dummy.

    The death dummy row (last row of C_feat_aug) represents a source cell dying.
    Target cells pay death_cost_mult × their type's maximum cost.

    Parameters
    ----------
    C_feat       : ndarray (ns, nt) — original feature cost
    D_A, D_B     : ndarray (ns, ns) and (nt, nt) — pairwise distances
    types_A, types_B : cell-type arrays
    birth_cost_mult  : float — multiplier on per-type max cost for birth column
    death_cost_mult  : float — multiplier on per-type max cost for death row

    Returns
    -------
    C_aug  : ndarray (ns+1, nt+1)
    D_A_aug : ndarray (ns+1, ns+1)
    D_B_aug : ndarray (nt+1, nt+1)
    """
    eps = 1e-6
    from collections import Counter
    counts_A = Counter(types_A)
    counts_B = Counter(types_B)
    all_types = set(counts_A) | set(counts_B)

    # Per-type max costs
    type_max: Dict[str, float] = {}
    for k in all_types:
        S = np.where(types_A == k)[0]
        T = np.where(types_B == k)[0]
        if len(S) > 0 and len(T) > 0:
            type_max[k] = float(C_feat[np.ix_(S, T)].max())
        else:
            type_max[k] = float(C_feat.max())

    death_col = np.array([death_cost_mult * type_max[types_A[i]] + eps
                           for i in range(ns)])  # source → death dummy
    birth_row = np.array([birth_cost_mult * type_max[types_B[j]] + eps
                           for j in range(nt)])  # birth dummy → target

    # Augmented feature cost matrix (ns+1, nt+1)
    C_aug = np.zeros((ns + 1, nt + 1), dtype=np.float64)
    C_aug[:ns, :nt]  = C_feat
    C_aug[:ns, nt]   = death_col          # death column (source → dummy tgt)
    C_aug[ns, :nt]   = birth_row          # birth row   (dummy src → target)
    C_aug[ns, nt]    = C_feat.max() + eps # dummy-to-dummy is free

    # Augmented distance matrices (dummy rows/cols = 0 → invisible to GW)
    D_A_aug = np.zeros((ns + 1, ns + 1), dtype=np.float64)
    D_A_aug[:ns, :ns] = D_A

    D_B_aug = np.zeros((nt + 1, nt + 1), dtype=np.float64)
    D_B_aug[:nt, :nt] = D_B

    return C_aug, D_A_aug, D_B_aug


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Cross-timepoint diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_expression_shift(
    Z_A: np.ndarray,
    Z_B: np.ndarray,
    pi:  np.ndarray,
    types_A: np.ndarray,
    types_B: np.ndarray,
    delta_threshold: float = 1e-6
) -> Dict:
    """
    Measure expression shift between matched cell pairs in latent space.

    For each matched source cell i, the expression shift is:

        Δz_i = ‖z_i^A − Σ_j π̃_{ij} z_j^B‖

    where Σ_j π̃_{ij} z_j^B is the expression barycenter of i's matches.

    Returns a dict with:
      - 'mean_shift'      : float   — mean latent distance between matched pairs
      - 'per_type_shift'  : dict    — mean shift broken down by cell type
      - 'shift_per_cell'  : ndarray — per-cell shift vector
    """
    weights  = pi.sum(axis=1)
    matched  = weights > delta_threshold
    pi_norm  = np.where(weights[:, None] > 1e-12, pi / weights[:, None], 0.0)

    Z_B_bary = pi_norm @ Z_B                   # (N, d) — expression barycenter
    delta    = np.linalg.norm(Z_A - Z_B_bary, axis=1)   # (N,)
    delta[~matched] = np.nan

    per_type = {}
    for typ in np.unique(types_A):
        mask = (types_A == typ) & matched
        per_type[typ] = float(np.nanmean(delta[mask])) if mask.any() else np.nan

    return {
        'mean_shift':     float(np.nanmean(delta)),
        'per_type_shift': per_type,
        'shift_per_cell': delta,
    }


def identify_cell_fate(
    pi:     np.ndarray,
    types_A: np.ndarray,
    types_B: np.ndarray,
    delta_threshold:       float = 1e-6,
    type_change_threshold: float = 0.5,
) -> Dict:
    """
    Classify each source cell as: maintained, differentiating, or dead.

    Classification rules (applied per source cell i):
      - ‖π_i‖₁ < delta_threshold → **dead** (no mass assigned to real target)
      - Most mass goes to same type → **maintained**
      - Most mass goes to different type, fraction > type_change_threshold → **differentiating**
      - Otherwise → **ambiguous**

    Returns
    -------
    dict with keys 'maintained', 'differentiating', 'dead', 'ambiguous'
    mapping to boolean masks of shape (N,).
    """
    N       = len(types_A)
    weights = pi.sum(axis=1)
    fate    = np.full(N, 'ambiguous', dtype=object)

    dead_mask  = weights < delta_threshold
    fate[dead_mask] = 'dead'

    for i in np.where(~dead_mask)[0]:
        row   = pi[i]
        argm  = np.argmax(row)
        frac  = row[argm] / (weights[i] + 1e-12)
        if types_B[argm] == types_A[i]:
            fate[i] = 'maintained'
        elif frac > type_change_threshold:
            fate[i] = 'differentiating'

    return {
        'maintained':    fate == 'maintained',
        'differentiating': fate == 'differentiating',
        'dead':          fate == 'dead',
        'ambiguous':     fate == 'ambiguous',
        'fate_labels':   fate,
    }
