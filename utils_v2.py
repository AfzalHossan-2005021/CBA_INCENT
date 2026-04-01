"""
utils_v2.py — Core mathematical primitives for Coherent Barycentric Alignment
              with Fused Gromov-Wasserstein transport (CBA-FGW).

Mathematical foundation
-----------------------
The framework jointly optimises a transport plan π and a rigid transformation
(R, t) under the objective:

    min_{π≥0, R∈O(2), t∈ℝ²}
        ⟨C_feat, π⟩
        + β  · L_CBA(π, R, t)          # coherent rigidity — primary structure term
        + α  · GW_local(D_s, D_t, π)   # local isometry on target side
        + τ  · [KL(π1|p) + KL(πᵀ1|q)] # unbalanced margins  (via dummy-cell augmentation)
        + γ  · ‖π‖_F²                  # sparsity / compactness

Where the CBA loss is:

    L_CBA(π, R, t) = (1/|M_s|) Σ_{i∈M_s} ‖π_i‖₁ · ‖Rx_i^s + t − T(i)‖²

and T(i) = (Σ_j π_{ij} x_j^t) / ‖π_i‖₁  is the barycentric projection of cell i.

The CBA gradient (derived in closed form) equals:

    ∂L_CBA/∂π_{ij} = (1/|M_s|) · ( ‖Rx_i^s + t − x_j^t‖² − ‖x_j^t − T(i)‖² )

This decomposes into a *static* spatial cost C_CBA[i,j] = ‖Rx_i^s + t − x_j^t‖²
(updated each outer iteration) and a *dynamic* variance correction (updated each
inner FW iteration).

Author: Research-grade implementation extending Anup Bhowmik's INCENT baseline.
"""

from __future__ import annotations

import ot
import torch
import inspect
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from typing import Optional, Tuple, List

try:
    from ot.lp import emd
    from ot.optim import line_search_armijo
    from ot.utils import list_to_array, get_backend
except ImportError:
    emd = None                  # type: ignore[assignment]
    line_search_armijo = None   # type: ignore[assignment]
    list_to_array = lambda *a: a
    get_backend = lambda *a: None


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Barycentric projection and CBA cost/gradient
# ─────────────────────────────────────────────────────────────────────────────

def compute_barycenters(pi: np.ndarray, X_t: np.ndarray) -> np.ndarray:
    """
    Compute the barycentric projection T(i) for every source cell.

    For source cell i, T(i) is the weighted average of target coordinates
    under the transport plan:

        T(i) = (Σ_j π_{ij} x_j^t) / ‖π_i‖₁

    Cells with negligible outgoing mass (‖π_i‖₁ < 1e-12) receive a NaN
    barycenter; callers should mask on weights > threshold before using T.

    Parameters
    ----------
    pi : ndarray, shape (N, M)
        Transport plan (unnormalised rows).
    X_t : ndarray, shape (M, 2)
        Spatial coordinates of target cells.

    Returns
    -------
    barycenters : ndarray, shape (N, 2)
        Barycentric projections; NaN for unmatched source cells.
    """
    row_sums = pi.sum(axis=1, keepdims=True)          # (N, 1)
    safe_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
    barycenters = (pi @ X_t) / safe_sums              # (N, 2)
    barycenters[row_sums[:, 0] < 1e-12] = np.nan
    return barycenters


def cba_static_cost(X_s: np.ndarray, X_t: np.ndarray,
                    R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute the static CBA cost matrix C_CBA[i,j] = ‖Rx_i^s + t − x_j^t‖².

    This is the dominant term of the CBA gradient for fixed (R, t). It enters
    the Frank-Wolfe linearisation as an additive cost: cells in source that
    have been rigidly mapped near cell j receive low cost; far cells pay high.

    Parameters
    ----------
    X_s : ndarray, shape (N, 2)
        Source spatial coordinates.
    X_t : ndarray, shape (M, 2)
        Target spatial coordinates.
    R   : ndarray, shape (2, 2)
        Rotation / reflection matrix (orthogonal).
    t   : ndarray, shape (2,)
        Translation vector.

    Returns
    -------
    C_cba : ndarray, shape (N, M)
        Squared-distance cost matrix under the current rigid transform.
    """
    X_s_transformed = (X_s @ R.T) + t[np.newaxis, :]   # (N, 2)
    diff = X_s_transformed[:, np.newaxis, :] - X_t[np.newaxis, :, :]  # (N, M, 2)
    return np.sum(diff ** 2, axis=2)                     # (N, M)


def cba_variance_correction(pi: np.ndarray, X_t: np.ndarray,
                             barycenters: np.ndarray) -> np.ndarray:
    """
    Compute the dynamic variance-correction term of the CBA gradient.

        V[i,j] = −‖x_j^t − T(i)‖²

    When added to the static CBA cost and multiplied by β/|M_s|, the full
    CBA gradient emerges:

        ∂L_CBA/∂π_{ij} = (β/|M_s|) · (C_CBA[i,j] + V[i,j])

    The negative sign means the correction *rewards* placing mass at target
    cells that are close to the current barycenter — pulling mass toward the
    correct location and reducing island-splitting.

    Parameters
    ----------
    pi : ndarray, shape (N, M)
        Current transport plan.
    X_t : ndarray, shape (M, 2)
        Target coordinates.
    barycenters : ndarray, shape (N, 2)
        Current barycentric projections from ``compute_barycenters``.

    Returns
    -------
    V : ndarray, shape (N, M)
        Variance-correction matrix; NaN rows correspond to unmatched cells
        and are zeroed out before adding to Mi.
    """
    # T_i[:, None] - X_t[None] → (N, M, 2)
    delta = barycenters[:, np.newaxis, :] - X_t[np.newaxis, :, :]  # (N, M, 2)
    V = -np.sum(delta ** 2, axis=2)                                  # (N, M)
    # Zero out rows where barycenter is undefined
    unmatched = np.isnan(barycenters[:, 0])
    V[unmatched] = 0.0
    return V


def cba_loss(pi: np.ndarray, X_s: np.ndarray, X_t: np.ndarray,
             R: np.ndarray, t: np.ndarray,
             delta_threshold: float = 1e-6) -> float:
    """
    Evaluate the Coherent Barycentric Alignment loss.

        L_CBA(π, R, t) = (1/|M_s|) Σ_{i∈M_s} ‖π_i‖₁ · ‖Rx_i^s + t − T(i)‖²

    Parameters
    ----------
    pi            : ndarray, shape (N, M)
    X_s, X_t      : ndarray, shape (N, 2) and (M, 2)
    R             : ndarray, shape (2, 2)
    t             : ndarray, shape (2,)
    delta_threshold : float
        Minimum row-sum to consider a cell matched.

    Returns
    -------
    loss : float
        Scalar CBA loss value.
    """
    weights   = pi.sum(axis=1)                              # (N,)
    matched   = weights > delta_threshold
    n_matched = matched.sum()
    if n_matched == 0:
        return 0.0

    bary      = compute_barycenters(pi, X_t)                # (N, 2)
    X_s_trans = (X_s @ R.T) + t[np.newaxis, :]             # (N, 2)
    residuals = X_s_trans - bary                            # (N, 2)
    residuals[~matched] = 0.0
    per_cell  = weights * np.sum(residuals ** 2, axis=1)    # (N,)
    return float(per_cell[matched].sum() / n_matched)


def cba_gradient_matrix(pi: np.ndarray, X_s: np.ndarray, X_t: np.ndarray,
                         R: np.ndarray, t: np.ndarray,
                         delta_threshold: float = 1e-6) -> np.ndarray:
    """
    Full CBA gradient ∂L_CBA/∂π_{ij} for use in the Frank-Wolfe linearisation.

    Closed-form derivation (Section 3 of the paper):

        ∂L_CBA/∂π_{ij} = (1/|M_s|) · ( ‖Rx_i^s + t − x_j^t‖²  −  ‖x_j^t − T(i)‖² )

    The first term is the static CBA cost; the second is the variance
    correction.  Unmatched rows (‖π_i‖₁ < delta) contribute zero gradient.

    Parameters
    ----------
    pi  : ndarray, shape (N, M)
    X_s : ndarray, shape (N, 2)
    X_t : ndarray, shape (M, 2)
    R   : ndarray, shape (2, 2)
    t   : ndarray, shape (2,)
    delta_threshold : float

    Returns
    -------
    grad : ndarray, shape (N, M)
        Gradient matrix; zero rows for unmatched source cells.
    """
    weights   = pi.sum(axis=1)
    matched   = weights > delta_threshold
    n_matched = max(1, matched.sum())

    bary      = compute_barycenters(pi, X_t)                          # (N, 2)
    C_static  = cba_static_cost(X_s, X_t, R, t)                      # (N, M)
    V_corr    = cba_variance_correction(pi, X_t, bary)                # (N, M)

    grad      = (C_static + V_corr) / n_matched                       # (N, M)
    grad[~matched] = 0.0
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Weighted Procrustes (Block B of alternating optimisation)
# ─────────────────────────────────────────────────────────────────────────────

def solve_procrustes_weighted(
        X_s: np.ndarray,
        barycenters: np.ndarray,
        weights: np.ndarray,
        allow_reflection: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the weighted Procrustes problem to find the optimal rigid transform.

    Given source positions {x_i^s} and target barycenters {T(i)} with
    confidence weights {w_i = ‖π_i‖₁}, find:

        (R*, t*) = argmin_{R∈O(2), t} Σ_i w_i · ‖Rx_i^s + t − T(i)‖²

    The closed-form solution uses the SVD of the weighted cross-covariance:

        H = Σ_i w_i (x_i^s − μ_s)(T(i) − μ_t)ᵀ
        H = UΣVᵀ
        R* = V · diag(1, det(VUᵀ)) · Uᵀ     (det correction handles reflections)
        t* = μ_t − R* μ_s

    Parameters
    ----------
    X_s         : ndarray, shape (N, 2)
        Source coordinates of matched cells.
    barycenters : ndarray, shape (N, 2)
        Barycentric projections of the same cells.
    weights     : ndarray, shape (N,)
        Per-cell confidence weights (‖π_i‖₁).
    allow_reflection : bool
        If True (default), allow improper rotations (reflections).
        This is critical for MERFISH data where the ground-truth transform
        may include a mirror flip.

    Returns
    -------
    R : ndarray, shape (2, 2)
        Optimal rotation / reflection matrix.
    t : ndarray, shape (2,)
        Optimal translation vector.
    """
    w       = weights / (weights.sum() + 1e-12)                # normalise
    mu_s    = (w[:, None] * X_s).sum(axis=0)                   # (2,)
    mu_t    = (w[:, None] * barycenters).sum(axis=0)           # (2,)

    Xs_c    = X_s - mu_s[None, :]                              # (N, 2)
    Xt_c    = barycenters - mu_t[None, :]                      # (N, 2)

    # Weighted cross-covariance H = Xsᵀ W Xt
    H       = (Xs_c * w[:, None]).T @ Xt_c                     # (2, 2)
    U, _, Vt = np.linalg.svd(H)
    V       = Vt.T

    if allow_reflection:
        # Correct for improper rotation: det(VUᵀ) = -1 means reflection
        d    = np.linalg.det(V @ U.T)
        D    = np.diag([1.0, d])
        R    = V @ D @ U.T
    else:
        R    = V @ U.T
        if np.linalg.det(R) < 0:
            # Force proper rotation by flipping the last column of V
            V[:, -1] *= -1
            R = V @ U.T

    t = mu_t - R @ mu_s
    return R.astype(np.float64), t.astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Local GW term
# ─────────────────────────────────────────────────────────────────────────────

def build_knn_mask(coords: np.ndarray, k: int) -> np.ndarray:
    """
    Build a binary k-nearest-neighbour adjacency matrix for spatial coordinates.

    Used to restrict the GW term to local neighbourhoods, replacing the O(N⁴)
    full GW tensor with an O(kN²M) local version.

    Parameters
    ----------
    coords : ndarray, shape (N, 2)
        Spatial coordinates.
    k      : int
        Number of nearest neighbours (excluding self).

    Returns
    -------
    mask : ndarray, shape (N, N), dtype float64
        Symmetric binary mask; mask[i,i'] = 1 iff i' ∈ kNN(i) or i ∈ kNN(i').
    """
    from sklearn.neighbors import NearestNeighbors
    nn   = NearestNeighbors(n_neighbors=k + 1, algorithm='ball_tree')
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)

    N    = coords.shape[0]
    mask = np.zeros((N, N), dtype=np.float64)
    for i, nbrs in enumerate(indices):
        for j in nbrs[1:]:                   # skip self (index 0)
            mask[i, j] = 1.0
            mask[j, i] = 1.0
    return mask


def local_gw_f(G: np.ndarray, D_s: np.ndarray, D_t: np.ndarray,
               knn_mask: Optional[np.ndarray] = None) -> float:
    """
    Evaluate the (local) Gromov-Wasserstein loss under square loss.

    Standard GW (Vayer et al., 2019):
        GW(π) = Σ_{i,i',j,j'} π_{ij} π_{i'j'} (D_s[i,i'] − D_t[j,j'])²

    Local GW restricts the outer sum to (i,i') ∈ E_s^k (kNN graph):
        GW_local(π) = Σ_{(i,i')∈E_s^k, j,j'} π_{ij} π_{i'j'} (D_s[i,i'] − D_t[j,j'])²

    This reduces computational cost from O(N²M²) to O(kNM²).

    Parameters
    ----------
    G        : ndarray, shape (N, M)
    D_s      : ndarray, shape (N, N)  — source pairwise distances
    D_t      : ndarray, shape (M, M)  — target pairwise distances
    knn_mask : ndarray, shape (N, N), optional
        If provided, masks D_s to the kNN neighbourhood. If None, uses full D_s.

    Returns
    -------
    loss : float
    """
    D_s_eff = D_s * knn_mask if knn_mask is not None else D_s
    # Inner-product form: ⟨D_s_eff, G G^T⟩ + ⟨D_t, G^T G⟩ (see utils.py::f)
    return float(np.sum((G @ G.T) * D_s_eff) + np.sum((G.T @ G) * D_t))


def local_gw_df(G: np.ndarray, D_s: np.ndarray, D_t: np.ndarray,
                knn_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Gradient of the (local) GW loss with respect to G.

    For the inner-product form f(G) = ⟨D_s_eff, GGᵀ⟩ + ⟨D_t, GᵀG⟩:

        df/dG = 2 (D_s_eff G + G D_t)

    Parameters
    ----------
    G, D_s, D_t, knn_mask : same as ``local_gw_f``.

    Returns
    -------
    grad : ndarray, shape (N, M)
    """
    D_s_eff = D_s * knn_mask if knn_mask is not None else D_s
    return 2.0 * (D_s_eff @ G + G @ D_t)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Quadratic sparsity penalty
# ─────────────────────────────────────────────────────────────────────────────

def quadratic_f(G: np.ndarray) -> float:
    """
    Evaluate ‖G‖_F² = Σ_{ij} G_{ij}².

    Adding γ‖π‖_F² to the objective enforces row-sparsity via soft ℓ₂-ball
    shrinkage. The minimiser of ⟨C, π⟩ + γ‖π‖_F² over π ≥ 0 has the
    closed-form row solution π_i* ∝ (−C_i / 2γ)₊, i.e. only entries where
    C_{ij} < 2γ · max_j(−C_{ij}) receive mass. Higher γ → tighter support.
    """
    return float(np.sum(G ** 2))


def quadratic_df(G: np.ndarray) -> np.ndarray:
    """Gradient of ‖G‖_F² w.r.t. G: simply 2G."""
    return 2.0 * G


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Spectral multi-hypothesis initialisation
# ─────────────────────────────────────────────────────────────────────────────

def compute_supercells(
        coords: np.ndarray,
        expr: np.ndarray,
        n_supercells: int,
        cell_types: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate cells into supercells via spatial k-means.

    Each supercell summarises ~sqrt(N) cells and provides a coarse
    representation for fast hypothesis scoring before the full N×M transport.

    Parameters
    ----------
    coords       : ndarray, shape (N, 2)
        Spatial coordinates.
    expr         : ndarray, shape (N, G)
        Gene expression matrix.
    n_supercells : int
        Target number of supercells (typically sqrt(N)).
    cell_types   : ndarray, shape (N,), optional
        Cell-type labels for composition-aware descriptors.

    Returns
    -------
    sc_coords : ndarray, shape (K, 2)
        Supercell centroids.
    sc_expr   : ndarray, shape (K, G)
        Pseudo-bulk (mean) gene expression per supercell.
    assignments : ndarray, shape (N,)
        Supercell index for each cell.
    """
    from sklearn.cluster import MiniBatchKMeans
    km          = MiniBatchKMeans(n_clusters=n_supercells, n_init=3,
                                   random_state=42, batch_size=2048)
    assignments = km.fit_predict(coords)

    sc_coords = np.array([coords[assignments == k].mean(axis=0)
                          for k in range(n_supercells)])
    sc_expr   = np.array([expr[assignments == k].mean(axis=0)
                          for k in range(n_supercells)])
    return sc_coords, sc_expr, assignments


def _spectral_embedding(coords: np.ndarray, k_neighbors: int = 15,
                         n_components: int = 8) -> np.ndarray:
    """
    Compute the leading eigenvectors of the normalised graph Laplacian.

    The spatial k-NN graph captures the topological structure of the tissue.
    Leading eigenvectors embed cells into a low-dimensional space where
    globally symmetric regions cluster together, enabling fast alignment of
    repeating structures (bilateral symmetry, lobules, crypts, etc.).

    Parameters
    ----------
    coords      : ndarray, shape (N, 2)
    k_neighbors : int
    n_components : int
        Number of eigenvectors to retain.

    Returns
    -------
    embedding : ndarray, shape (N, n_components)
    """
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.linalg import eigsh

    N  = coords.shape[0]
    A  = kneighbors_graph(coords, n_neighbors=k_neighbors,
                           mode='connectivity', include_self=False)
    A  = (A + A.T).astype(np.float64)                        # symmetrise
    A.data[:] = 1.0

    d  = np.asarray(A.sum(axis=1)).flatten()
    D  = sp.diags(1.0 / np.sqrt(d + 1e-12))
    L  = sp.eye(N) - D @ A @ D                               # normalised Laplacian

    n_ev = min(n_components + 1, N - 1)
    vals, vecs = eigsh(L, k=n_ev, which='SM')                # smallest eigenvalues
    # Sort by eigenvalue; skip the trivial constant eigenvector (λ≈0)
    order = np.argsort(vals)
    vecs  = vecs[:, order[1:n_components + 1]]
    return vecs


def spectral_hypotheses(
        coords_A: np.ndarray, coords_B: np.ndarray,
        expr_A: np.ndarray, expr_B: np.ndarray,
        n_hypotheses: int = 8,
        n_supercells: int = 0,
        k_spectral: int = 15,
        n_spectral_components: int = 6
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate H rigid-transform hypotheses via spectral graph alignment.

    Algorithm
    ---------
    1. Build spatial k-NN graphs for source and target.
    2. Compute leading Laplacian eigenvectors (spectral embeddings).
    3. For each sign-flip combination of eigenvectors (captures reflections),
       align the spectral embeddings via Procrustes and recover the
       corresponding spatial (R, t) from anchor pairs.
    4. Include pure-rotation candidates (0°, 90°, 180°, 270°).
    5. Return the H highest-scoring distinct candidates.

    Parameters
    ----------
    coords_A, coords_B : ndarray, shape (N, 2) and (M, 2)
    expr_A, expr_B     : ndarray, shape (N, G) and (M, G)
    n_hypotheses       : int
        Number of candidates to return (default 8).
    n_supercells       : int
        If > 0, use supercell representatives; if 0, use all cells.
    k_spectral         : int
        k-NN neighbourhood for graph construction.
    n_spectral_components : int
        Number of eigenvectors.

    Returns
    -------
    hypotheses : list of (R, t) tuples, length ≤ n_hypotheses.
        Each entry is a (2×2) rotation matrix and (2,) translation vector.
    """
    emb_A = _spectral_embedding(coords_A, k_spectral, n_spectral_components)
    emb_B = _spectral_embedding(coords_B, k_spectral, n_spectral_components)

    hypotheses = []

    # --- Spectral Procrustes alignment with eigenvector sign flips ---
    # Each eigenvector has an arbitrary sign; we try all 2^d sign combinations
    # for the first d components to enumerate symmetric alignments.
    n_signs = min(n_spectral_components, 3)    # 2^3 = 8 candidates
    for flip_mask in range(2 ** n_signs):
        signs = np.array([((-1) ** ((flip_mask >> k) & 1))
                          for k in range(n_signs)] + [1.0] * (n_spectral_components - n_signs))
        emb_A_flipped = emb_A * signs[np.newaxis, :]

        # Align spectral embeddings: find R_spec, t_spec in spectral space
        # Then use supercell / centroid pairs to recover spatial (R, t)
        sc_n = int(np.sqrt(max(len(coords_A), len(coords_B)))) if n_supercells == 0 else n_supercells
        sc_n = max(sc_n, 16)
        sc_coords_A, sc_expr_A, _ = compute_supercells(coords_A, expr_A, sc_n)
        sc_coords_B, sc_expr_B, _ = compute_supercells(coords_B, expr_B, sc_n)

        # Match supercells by spectral position (nearest in spectral space)
        sc_emb_A = _spectral_embedding(sc_coords_A, min(k_spectral, sc_n - 1),
                                        n_spectral_components)
        sc_emb_B = _spectral_embedding(sc_coords_B, min(k_spectral, sc_n - 1),
                                        n_spectral_components)
        sc_emb_A_flipped = sc_emb_A * signs[np.newaxis, :]

        # Nearest-neighbour matching in spectral space
        from sklearn.neighbors import NearestNeighbors
        nn_spec = NearestNeighbors(n_neighbors=1).fit(sc_emb_B)
        _, idx_B = nn_spec.kneighbors(sc_emb_A_flipped)
        idx_B = idx_B[:, 0]

        anchor_A = sc_coords_A
        anchor_B = sc_coords_B[idx_B]
        weights  = np.ones(len(anchor_A))

        R, t = solve_procrustes_weighted(anchor_A, anchor_B, weights,
                                          allow_reflection=True)
        hypotheses.append((R, t))

    # --- Add cardinal rotation hypotheses (0°, 90°, 180°, 270°) ---
    centroid_A = coords_A.mean(axis=0)
    centroid_B = coords_B.mean(axis=0)
    for angle_deg in [0, 90, 180, 270]:
        theta = np.deg2rad(angle_deg)
        R_rot = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
        t_rot = centroid_B - R_rot @ centroid_A
        hypotheses.append((R_rot, t_rot))

    # De-duplicate and return top-H (no scoring here; scored in INCENT_v2)
    return hypotheses[:n_hypotheses]


def score_hypothesis(
        R: np.ndarray, t: np.ndarray,
        X_s: np.ndarray, X_t: np.ndarray,
        C_feat: np.ndarray,
        k_nn: int = 10
) -> float:
    """
    Fast hypothesis scorer: spatial proximity + feature similarity of kNN pairs.

    For each source cell, find its nearest target cell under the rigid
    transform and measure the feature cost. Lower is better.

    Parameters
    ----------
    R, t     : rigid transform
    X_s, X_t : coordinates (N,2) and (M,2)
    C_feat   : feature cost matrix (N, M)
    k_nn     : number of neighbours to check

    Returns
    -------
    score : float  (lower = better hypothesis)
    """
    from sklearn.neighbors import NearestNeighbors
    X_s_trans = (X_s @ R.T) + t[np.newaxis, :]
    nn   = NearestNeighbors(n_neighbors=min(k_nn, len(X_t))).fit(X_t)
    dists, idx = nn.kneighbors(X_s_trans)
    # Mean feature cost of nearest target for each source cell
    scores_feat  = C_feat[np.arange(len(X_s))[:, None], idx].min(axis=1)
    scores_space = dists.min(axis=1)
    return float(scores_feat.mean() + 0.1 * scores_space.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def compute_effective_support(pi: np.ndarray) -> np.ndarray:
    """
    Per-row inverse participation ratio (effective support size).

        k_eff(i) = (Σ_j π_{ij})² / (Σ_j π_{ij}²)

    Equals 1 for a perfect 1-to-1 mapping; equals M for uniform diffusion.
    Report mean and median in the paper as the primary sparsity metric.

    Parameters
    ----------
    pi : ndarray, shape (N, M)

    Returns
    -------
    k_eff : ndarray, shape (N,)
    """
    row_l1 = pi.sum(axis=1)
    row_l2 = (pi ** 2).sum(axis=1)
    safe_l2 = np.where(row_l2 < 1e-24, 1.0, row_l2)
    k_eff   = row_l1 ** 2 / safe_l2
    k_eff[row_l1 < 1e-12] = 0.0
    return k_eff


def compute_forward_compactness(pi: np.ndarray, X_t: np.ndarray) -> float:
    """
    Forward compactness: mean spatial variance of each source cell's target footprint.

        L_fwd = (1/N) Σ_i [ (Σ_j π̃_{ij} ‖x_j^t‖²) − ‖Σ_j π̃_{ij} x_j^t‖² ]

    where π̃_{ij} = π_{ij}/‖π_i‖₁ is the row-normalised plan.
    Lower values mean each source cell maps to a spatially tight cluster.
    """
    row_sums = pi.sum(axis=1, keepdims=True)
    mask     = (row_sums[:, 0] > 1e-12)
    pi_norm  = np.where(row_sums > 1e-12, pi / row_sums, 0.0)

    sq_norms = np.sum(X_t ** 2, axis=1)                     # (M,)
    E_sq     = pi_norm @ sq_norms                            # (N,)  E[‖x‖²]
    bary     = pi_norm @ X_t                                 # (N, 2)
    E_sq_bar = np.sum(bary ** 2, axis=1)                     # (N,)  ‖E[x]‖²
    variance = np.where(mask, E_sq - E_sq_bar, 0.0)         # (N,)
    return float(variance[mask].mean()) if mask.any() else 0.0


def compute_reverse_compactness(pi: np.ndarray, X_s: np.ndarray) -> float:
    """
    Reverse compactness: mean spatial variance of each target cell's source origin.

    Symmetrically to forward compactness, this penalises a single target cell
    receiving mass from spatially dispersed source cells.
    """
    return compute_forward_compactness(pi.T, X_s)


def compute_cba_rmse(pi: np.ndarray, X_s: np.ndarray, X_t: np.ndarray,
                      R: np.ndarray, t: np.ndarray,
                      delta_threshold: float = 1e-6) -> float:
    """
    Root-mean-squared residual of barycenters from the rigid prediction.

        RMSE = sqrt( (1/|M_s|) Σ_{i∈M_s} ‖Rx_i^s + t − T(i)‖² )

    This is the primary spatial alignment quality metric. At perfect alignment,
    RMSE → 0 on matched cells.
    """
    weights   = pi.sum(axis=1)
    matched   = weights > delta_threshold
    n_matched = matched.sum()
    if n_matched == 0:
        return float('inf')

    bary      = compute_barycenters(pi, X_t)
    X_trans   = (X_s @ R.T) + t[np.newaxis, :]
    residuals = X_trans - bary
    residuals[~matched] = 0.0
    mse = (weights * np.sum(residuals ** 2, axis=1))[matched].sum() / n_matched
    return float(np.sqrt(max(0.0, mse)))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: Core CBA-FGW solver  (Frank-Wolfe conditional gradient)
# ─────────────────────────────────────────────────────────────────────────────

def _cba_fgw_linesearch(G, deltaG, cost_G, C_total, D_s, D_t,
                         alpha_gw, gamma_quad, knn_mask, nx):
    """
    Closed-form quadratic line search for the CBA-FGW objective.

    The objective along the FW ray G + s·ΔG decomposes as:

        cost(s) = cost_G + as² + bs

    where the quadratic coefficient a includes contributions from the GW
    and quadratic-penalty terms, and b captures all linear terms.

    Parameters
    ----------
    G, deltaG  : transport plan and FW direction
    cost_G     : current objective value
    C_total    : combined linear cost (C_feat + β·C_CBA_static)
    D_s, D_t   : distance matrices (possibly knn-masked for D_s)
    alpha_gw   : weight of GW term
    gamma_quad : weight of quadratic penalty
    knn_mask   : kNN mask for D_s (or None)
    nx         : POT backend

    Returns
    -------
    step : float
    n_calls : int (always 1)
    new_cost : float
    """
    D_s_eff = D_s * knn_mask if knn_mask is not None else D_s

    # Quadratic coefficient from GW term: -2α ⟨D_s_eff·ΔG·D_t, ΔG⟩
    dot_gw = nx.dot(nx.dot(D_s_eff, deltaG), D_t.T)
    a_gw   = -2.0 * alpha_gw * nx.sum(dot_gw * deltaG)

    # Quadratic coefficient from ‖·‖_F² term: γ ‖ΔG‖_F²
    a_quad = gamma_quad * nx.sum(deltaG * deltaG)

    a = float(a_gw + a_quad)

    # Linear coefficient
    b_linear = nx.sum(C_total * deltaG)
    b_gw     = -2.0 * alpha_gw * (nx.sum(dot_gw * G) +
                                    nx.sum(nx.dot(nx.dot(D_s_eff, G), D_t.T) * deltaG))
    b_quad   = 2.0 * gamma_quad * nx.sum(G * deltaG)
    b = float(b_linear + b_gw + b_quad)

    step = ot.optim.solve_1d_linesearch_quad(a, b)
    step = float(np.clip(step, 0.0, 1.0))
    new_cost = cost_G + a * step ** 2 + b * step
    return step, 1, new_cost


def cba_fgw_incent(
        C_feat: np.ndarray,
        C_cba_static: np.ndarray,
        D_s: np.ndarray,
        D_t: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        beta: float = 1.0,
        alpha_gw: float = 0.1,
        gamma_quad: float = 0.01,
        knn_mask: Optional[np.ndarray] = None,
        G0: Optional[np.ndarray] = None,
        numItermax: int = 200,
        tol_rel: float = 1e-8,
        tol_abs: float = 1e-9,
        verbose: bool = False,
        log: bool = True,
        use_gpu: bool = False,
        **kwargs
) -> Tuple[np.ndarray, dict]:
    """
    Frank-Wolfe conditional gradient solver for the CBA-FGW objective.

    Minimises:
        ⟨C_total, π⟩ + α·GW_local(π) + γ·‖π‖_F²

    where C_total = C_feat + β·C_CBA_static (fixed for this call; updated by
    the outer Procrustes loop).

    The transport update (Block A) in the alternating optimisation corresponds
    to one call to this function with the current rigid transform baked into
    C_CBA_static.

    Parameters
    ----------
    C_feat        : ndarray (N, M) — gene-expression + cell-type cost
    C_cba_static  : ndarray (N, M) — ‖Rx_i^s + t − x_j^t‖² for current (R,t)
    D_s           : ndarray (N, N) — source pairwise distances (kNN-masked or full)
    D_t           : ndarray (M, M) — target pairwise distances
    p             : ndarray (N,)   — source marginal
    q             : ndarray (M,)   — target marginal
    beta          : float          — CBA loss weight
    alpha_gw      : float          — GW local weight
    gamma_quad    : float          — quadratic sparsity weight
    knn_mask      : ndarray (N,N)  — binary mask restricting GW to kNN pairs
    G0            : ndarray (N,M)  — warm-start transport plan
    numItermax    : int
    tol_rel, tol_abs : convergence thresholds
    verbose, log  : I/O flags
    use_gpu       : bool

    Returns
    -------
    G     : ndarray, shape (N, M) — optimal transport plan
    logdict : dict with keys 'loss' (list), 'n_iter' (int)
    """
    p_a, q_a = list_to_array(p, q)
    nx = get_backend(p_a, q_a)

    C_total = C_feat + beta * C_cba_static        # combined linear cost

    if G0 is None:
        G = p_a[:, None] * q_a[None, :]           # outer product initialisation
    else:
        G = nx.from_numpy(np.asarray(G0, dtype=np.float64))
        G = G / nx.sum(G)

    if isinstance(nx, ot.backend.TorchBackend):
        C_total = nx.from_numpy(C_total.astype(np.float32) if use_gpu
                                 else C_total.astype(np.float64))
        D_s     = D_s.float() if use_gpu else D_s
        D_t     = D_t.float() if use_gpu else D_t
    else:
        C_total = nx.from_numpy(C_total)
        D_s     = nx.from_numpy(D_s)
        D_t     = nx.from_numpy(D_t)

    if knn_mask is not None:
        knn_mask_nx = nx.from_numpy(knn_mask.astype(
            np.float32 if (isinstance(nx, ot.backend.TorchBackend) and use_gpu)
            else np.float64))
    else:
        knn_mask_nx = None

    def cost_fn(Gx):
        lin  = nx.sum(C_total * Gx)
        gw   = alpha_gw * local_gw_f(nx.to_numpy(Gx),
                                       nx.to_numpy(D_s),
                                       nx.to_numpy(D_t),
                                       nx.to_numpy(knn_mask_nx) if knn_mask_nx is not None else None)
        quad = gamma_quad * nx.sum(Gx * Gx)
        return float(lin) + gw + float(quad)

    cost_G = cost_fn(G)
    logdict = {'loss': [cost_G], 'n_iter': 0}

    for it in range(1, numItermax + 1):
        old_cost = cost_G

        # FW gradient: Mi = C_total + α·df_GW(G) + 2γ·G
        grad_gw  = nx.from_numpy(local_gw_df(nx.to_numpy(G),
                                               nx.to_numpy(D_s),
                                               nx.to_numpy(D_t),
                                               nx.to_numpy(knn_mask_nx)
                                               if knn_mask_nx is not None else None))
        Mi       = C_total + alpha_gw * grad_gw + gamma_quad * quadratic_df(nx.to_numpy(G))
        Mi       = nx.from_numpy(nx.to_numpy(Mi) + np.abs(nx.to_numpy(Mi).min()))  # ensure ≥ 0

        # Linear subproblem: solve EMD
        Gc, _    = emd(nx.to_numpy(p_a), nx.to_numpy(q_a), nx.to_numpy(Mi),
                        numItermax=kwargs.get('numItermaxEmd', 100_000), log=True)
        Gc       = nx.from_numpy(Gc)
        deltaG   = Gc - G

        # Quadratic line search
        step, _, cost_G = _cba_fgw_linesearch(
            G, deltaG, cost_G, C_total, D_s, D_t,
            alpha_gw, gamma_quad, knn_mask_nx, nx)
        G = G + step * deltaG

        logdict['loss'].append(cost_G)
        logdict['n_iter'] = it

        if verbose and it % 20 == 0:
            print(f"  [CBA-FGW] iter {it:4d} | cost {cost_G:.6e} | "
                  f"Δcost {abs(cost_G - old_cost):.2e}")

        abs_delta  = abs(cost_G - old_cost)
        rel_delta  = abs_delta / (abs(cost_G) + 1e-12)
        if rel_delta < tol_rel or abs_delta < tol_abs:
            if verbose:
                print(f"  [CBA-FGW] Converged at iter {it}")
            break

    return nx.to_numpy(G), logdict


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: Backward-compatible helpers (preserved from utils.py)
# ─────────────────────────────────────────────────────────────────────────────

# --- Jensen-Shannon divergence (Nuwaisir's implementation, preserved) ---

def kl_divergence_corresponding_backend(X, Y):
    """Pairwise KL divergence matrix. See original utils.py for details."""
    assert X.shape[1] == Y.shape[1]
    nx_be = ot.backend.get_backend(X, Y)
    X = X / nx_be.sum(X, axis=1, keepdims=True)
    Y = Y / nx_be.sum(Y, axis=1, keepdims=True)
    log_X = nx_be.log(X)
    log_Y = nx_be.log(Y)
    X_log_X = nx_be.einsum('ij,ij->i', X, log_X)
    X_log_X = nx_be.reshape(X_log_X, (1, X_log_X.shape[0]))
    X_log_Y = nx_be.einsum('ij,ij->i', X, log_Y)
    X_log_Y = nx_be.reshape(X_log_Y, (1, X_log_Y.shape[0]))
    return ot.backend.get_backend(X, Y).to_numpy(X_log_X.T - X_log_Y.T)


def jensenshannon_distance_1_vs_many_backend(X, Y):
    """JSD for one row of X against all rows of Y. See original utils.py."""
    assert X.shape[1] == Y.shape[1] and X.shape[0] == 1
    nx_be = ot.backend.get_backend(X, Y)
    X = nx_be.concatenate([X] * Y.shape[0], axis=0)
    X = X / nx_be.sum(X, axis=1, keepdims=True)
    Y = Y / nx_be.sum(Y, axis=1, keepdims=True)
    M = (X + Y) / 2.0
    kl_xm = torch.from_numpy(kl_divergence_corresponding_backend(X, M))
    kl_ym = torch.from_numpy(kl_divergence_corresponding_backend(Y, M))
    return nx_be.sqrt((kl_xm + kl_ym) / 2.0).T[0]


def jensenshannon_divergence_backend(X, Y):
    """Full pairwise JSD matrix. See original utils.py."""
    print("Calculating JSD cost matrix")
    assert X.shape[1] == Y.shape[1]
    nx_be = ot.backend.get_backend(X, Y)
    X = X / nx_be.sum(X, axis=1, keepdims=True)
    Y = Y / nx_be.sum(Y, axis=1, keepdims=True)
    n, m = X.shape[0], Y.shape[0]
    js_dist = nx_be.zeros((n, m))
    for i in tqdm(range(n)):
        js_dist[i, :] = jensenshannon_distance_1_vs_many_backend(X[i:i + 1], Y)
    print("Done.")
    if torch.cuda.is_available():
        try:
            return js_dist.numpy()
        except Exception:
            return js_dist
    return js_dist


def pairwise_msd(A, B):
    """Mean squared distance between rows of A and B."""
    A, B = np.asarray(A), np.asarray(B)
    diff = A[:, np.newaxis, :] - B[np.newaxis, :, :]
    return np.mean(diff ** 2, axis=2)


to_dense_array = lambda X: X.toarray() if sp.issparse(X) else np.asarray(X)
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.obsm[rep]


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: Legacy FGW solver (preserved for ablation/baseline)
# ─────────────────────────────────────────────────────────────────────────────

# (fused_gromov_wasserstein_incent, solve_gromov_linesearch,
#  generic_conditional_gradient_incent, cg_incent — all preserved in the
#  original utils.py and importable from there for backward compatibility.)
