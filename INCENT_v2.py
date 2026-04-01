"""
INCENT_v2.py — Coherent Barycentric Alignment (CBA-FGW) for MERFISH slices.

Overview
--------
This module implements the research-grade alignment method proposed in:

    "Symmetry-Aware Coherent Partial Registration for MERFISH Slices"

Two alignment modes are supported:

same_timepoint
    The gold-standard case: slices come from the same developmental stage.
    Gene expression and cell identity are unchanged; only spatial placement
    differs.  Alignment uses raw cosine expression cost and a rigid transform
    (R, t) fitted by weighted Procrustes in Block B of the alternating loop.

    Objective:
        min_{π≥0, R∈O(2), t}
            ⟨C_feat, π⟩  +  β·L_CBA(π,R,t)  +  α·GW_local(π)
          + τ·[KL(π1|p)+KL(πᵀ1|q)]  +  γ·‖π‖_F²

cross_timepoint
    Slices come from different developmental stages: expression distributions
    shift, cell mass is not conserved, and the tissue may be mildly deformed.

    Four mathematical extensions over same_timepoint:

    (1) Batch-corrected expression cost
        Raw cosine distance fails because developmental drift penalises correct
        matches.  We instead compute cost in a batch-corrected latent space z
        obtained from a conditional VAE (scVI) or Harmony-corrected PCA.
            C_feat_CT = (1-λ_l)·C_lat  +  λ_l·C_lin
        where C_lat[i,j] = ‖z_i^A − z_j^B‖² and C_lin is the lineage-aware
        cost derived from type centroid distances in the latent space.

    (2) TPS deformation field in Block B
        After rigid alignment converges, Block B fits a thin-plate spline (TPS)
        φ: ℝ² → ℝ² on confident anchor pairs (x_i^s, T(i)).  The TPS replaces
        the rigid map in subsequent outer iterations, capturing mild tissue
        deformation.  L_CBA_TPS uses φ(x_i^s) in place of Rx_i^s + t.

    (3) Semi-relaxed margins (ancestor-constrained OT)
        τ_s → ∞: every source cell must be matched or declared dead.
        τ_t < ∞: new target cells can appear without a source ancestor.
        Implemented via asymmetric birth/death dummy augmentation.

    (4) Cell fate classification
        Post-alignment, each source cell is classified as: maintained,
        differentiating, or dead, based on the transport plan and type labels.

    Cross-timepoint objective:
        min_{π≥0, φ∈TPS}
            ⟨C_feat_CT, π⟩  +  β·L_CBA_TPS(π,φ)  +  α·GW_local(π)
          + τ_s·KL(π1|p)  +  τ_t·KL(πᵀ1|q)  +  γ·‖π‖_F²

Public API
----------
coherent_pairwise_align(sliceA, sliceB, mode='same_timepoint', ...) -> AlignmentResult
    Main research-grade entry point for both modes.

pairwise_align(...)
    Backward-compatible legacy wrapper (calls original INCENT baseline).

Author: Research-grade implementation extending Anup Bhowmik's INCENT baseline.
"""

from __future__ import annotations

import os
import ot
import time
import torch
import datetime
import warnings
import numpy as np
import pandas as pd

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Union, Literal
from tqdm import tqdm
from anndata import AnnData
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

from .utils_v2 import (
    compute_barycenters,
    cba_static_cost,
    cba_gradient_matrix,
    cba_loss,
    solve_procrustes_weighted,
    build_knn_mask,
    local_gw_f,
    local_gw_df,
    quadratic_f,
    quadratic_df,
    spectral_hypotheses,
    score_hypothesis,
    compute_effective_support,
    compute_forward_compactness,
    compute_reverse_compactness,
    compute_cba_rmse,
    cba_fgw_incent,
    to_dense_array,
    extract_data_matrix,
    jensenshannon_divergence_backend,
    pairwise_msd,
)

from .cross_timepoint import (
    ExpressionAligner,
    LineageAwareCost,
    TPSDeformationField,
    build_semitrelaxed_marginals,
    augment_cost_matrices_ct,
    compute_expression_shift,
    identify_cell_fate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data class: AlignmentResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignmentResult:
    """
    Container for all outputs of ``coherent_pairwise_align``.

    Attributes
    ----------
    pi_fwd : ndarray (N, M)
        Forward transport plan (source → target).  Row i of pi_fwd encodes
        how much mass source cell i sends to each target cell.

    pi_rev : ndarray (M, N)
        Reverse transport plan (target → source).  Obtained by solving the
        same CBA-FGW objective with source and target swapped, using the
        final (R, t) from the forward pass as initialisation.

    R : ndarray (2, 2)
        Optimal rigid rotation/reflection matrix.

    t : ndarray (2,)
        Optimal translation vector.

    src_warped : ndarray (N, 2)
        Source coordinates after applying (R, t):  X_s @ R.T + t.

    forward_barycenters : ndarray (N, 2)
        Barycentric projection T(i) of each source cell into target space.
        NaN for unmatched cells.

    reverse_barycenters : ndarray (M, 2)
        Barycentric projection T(j) of each target cell into source space.

    matched_src : ndarray (N,) bool
        Mask of matched source cells (‖π_i‖₁ > delta_threshold).

    matched_tgt : ndarray (M,) bool
        Mask of matched target cells (‖π_j‖₁ > delta_threshold, col-sum).

    raw_unmatched_src : float
        Fraction of source mass that went to the dummy (unmatched).

    raw_unmatched_tgt : float
        Fraction of target mass that went to the dummy (unmatched).

    metrics : dict
        Diagnostic metrics computed post-alignment:
        - 'cba_rmse'           : RMSE of barycenters from rigid prediction
        - 'fwd_compactness'    : mean forward spatial variance
        - 'rev_compactness'    : mean reverse spatial variance
        - 'mean_k_eff'         : mean effective support size per source row
        - 'median_k_eff'       : median effective support size per source row
        - 'n_matched_src'      : number of matched source cells
        - 'n_matched_tgt'      : number of matched target cells
        - 'hypothesis_scores'  : scores for all H initial hypotheses
        - 'selected_hypothesis': index of the chosen hypothesis

    selected_hypothesis : int
        Index in the hypothesis list that was selected.

    hypothesis_scores : List[float]
        Scores for all enumerated rigid hypotheses (lower = better).

    loss_history : List[float]
        Objective values across inner FW iterations of the final outer step.

    runtime_seconds : float
        Total wall-clock time for the alignment.
    """
    pi_fwd:              np.ndarray = field(repr=False)
    pi_rev:              np.ndarray = field(repr=False)
    R:                   np.ndarray = field(repr=False)
    t:                   np.ndarray = field(repr=False)
    src_warped:          np.ndarray = field(repr=False)
    forward_barycenters: np.ndarray = field(repr=False)
    reverse_barycenters: np.ndarray = field(repr=False)
    matched_src:         np.ndarray = field(repr=False)
    matched_tgt:         np.ndarray = field(repr=False)
    raw_unmatched_src:   float      = 0.0
    raw_unmatched_tgt:   float      = 0.0
    metrics:             dict       = field(default_factory=dict)
    selected_hypothesis: int        = 0
    hypothesis_scores:   list       = field(default_factory=list)
    loss_history:        list       = field(default_factory=list)
    runtime_seconds:     float      = 0.0

    # ── Cross-timepoint specific fields (None in same_timepoint mode) ─────────
    deformation_field:   Optional[object]     = None   # TPSDeformationField
    src_warped_tps:      Optional[np.ndarray] = field(default=None, repr=False)
    expression_shift:    Optional[dict]       = None   # from compute_expression_shift
    cell_fate:           Optional[dict]       = None   # from identify_cell_fate
    latent_A:            Optional[np.ndarray] = field(default=None, repr=False)
    latent_B:            Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        """Return a human-readable summary of the alignment result."""
        m = self.metrics
        lines = [
            "── AlignmentResult ──────────────────────────────────",
            f"  Mode                      : {m.get('mode', 'same_timepoint')}",
            f"  CBA RMSE (post-transform) : {m.get('cba_rmse', float('nan')):.4f}",
            f"  Forward compactness       : {m.get('fwd_compactness', float('nan')):.4f}",
            f"  Reverse compactness       : {m.get('rev_compactness', float('nan')):.4f}",
            f"  Mean k_eff (sparsity)     : {m.get('mean_k_eff', float('nan')):.2f}",
            f"  Matched src / tgt         : {m.get('n_matched_src', '?')} / "
            f"{m.get('n_matched_tgt', '?')}",
            f"  Unmatched src / tgt frac  : {self.raw_unmatched_src:.4f} / "
            f"{self.raw_unmatched_tgt:.4f}",
            f"  Selected hypothesis       : {self.selected_hypothesis} "
            f"(score {self.hypothesis_scores[self.selected_hypothesis]:.4f})",
        ]
        if m.get('mode') == 'cross_timepoint':
            lines += [
                f"  Expression aligner        : {m.get('expression_backend', '?')}",
                f"  Mean latent shift         : {m.get('mean_expression_shift', float('nan')):.4f}",
                f"  TPS anchors used          : {m.get('n_tps_anchors', '?')}",
                f"  Cell fate — maintained    : {m.get('n_maintained', '?')}",
                f"  Cell fate — differentiating: {m.get('n_differentiating', '?')}",
                f"  Cell fate — dead          : {m.get('n_dead', '?')}",
            ]
        lines += [
            f"  Runtime                   : {self.runtime_seconds:.1f}s",
            "─────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_cost_ct(
        sliceA: AnnData, sliceB: AnnData,
        filePath: str,
        sliceA_name: str, sliceB_name: str,
        use_rep: Optional[str],
        lambda_lineage: float,
        expr_backend: str,
        expr_aligner_kwargs: dict,
        overwrite: bool,
        verbose: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Construct the cross-timepoint feature cost matrix.

        C_feat_CT[i,j] = (1 − λ_l) · C_lat[i,j]  +  λ_l · C_lin[i,j]

    where:
      C_lat[i,j] = ‖z_i^A − z_j^B‖²  in the batch-corrected latent space
      C_lin[i,j] = lineage distance between type(i) and type(j) (type centroid
                   Euclidean distance in the same latent space, normalised to [0,1])

    Parameters
    ----------
    sliceA, sliceB    : AnnData
    lambda_lineage    : float — weight of lineage cost (0 = pure latent, 1 = pure lineage)
    expr_backend      : str   — 'scvi' | 'harmony' | 'pca'
    expr_aligner_kwargs : dict — passed to ExpressionAligner.__init__
    overwrite         : bool
    verbose           : bool

    Returns
    -------
    C_feat_ct : ndarray (N, M)
    Z_A       : ndarray (N, d) — latent embeddings for source
    Z_B       : ndarray (M, d) — latent embeddings for target
    backend_used : str         — actual backend used (may differ from requested)
    """
    X_A = to_dense_array(extract_data_matrix(sliceA, use_rep)).astype(np.float32)
    X_B = to_dense_array(extract_data_matrix(sliceB, use_rep)).astype(np.float32)

    cache_lat_A = os.path.join(filePath, f"latent_{sliceA_name}_{expr_backend}.npy")
    cache_lat_B = os.path.join(filePath, f"latent_{sliceB_name}_{expr_backend}.npy")

    aligner = ExpressionAligner(backend=expr_backend, **expr_aligner_kwargs)

    if (os.path.exists(cache_lat_A) and os.path.exists(cache_lat_B) and not overwrite):
        if verbose:
            print(f"[CT] Loading cached latent embeddings ({expr_backend}) …")
        Z_A = np.load(cache_lat_A)
        Z_B = np.load(cache_lat_B)
        backend_used = expr_backend
    else:
        if verbose:
            print(f"[CT] Computing batch-corrected latent embeddings ({expr_backend}) …")
        Z_A, Z_B = aligner.fit_transform(X_A, X_B, verbose=verbose)
        backend_used = aligner.backend   # may have fallen back
        np.save(cache_lat_A, Z_A)
        np.save(cache_lat_B, Z_B)

    # Latent cost: squared Euclidean distance in corrected space
    C_lat = euclidean_distances(Z_A, Z_B).astype(np.float64) ** 2
    # Normalise to [0, 1] for weighting
    lat_max = C_lat.max()
    if lat_max > 0:
        C_lat /= lat_max

    # Lineage-aware cost: type centroid distances
    types_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    types_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    lin_cost = LineageAwareCost(method='centroid')
    lin_cost.fit(Z_A, Z_B, types_A, types_B, verbose=verbose)
    C_lin = lin_cost.cell_cost_matrix(types_A, types_B)   # already in [0, 1]

    C_feat_ct = (1.0 - lambda_lineage) * C_lat + lambda_lineage * C_lin
    return C_feat_ct, Z_A, Z_B, backend_used


def _run_ct_alternating_loop(
        C_feat_aug:    np.ndarray,
        D_A_aug:       np.ndarray,
        D_B_aug:       np.ndarray,
        D_A_local_aug: np.ndarray,
        p_vals:        np.ndarray,
        q_vals:        np.ndarray,
        X_s:           np.ndarray,
        X_t:           np.ndarray,
        ns:            int,
        nt:            int,
        ns_aug:        int,
        nt_aug:        int,
        # rigid initialisation
        R_init:        np.ndarray,
        t_init:        np.ndarray,
        # solver weights
        beta_cba:      float,
        alpha_gw:      float,
        gamma_quad:    float,
        # TPS settings
        tps_smoothing:         float,
        tps_n_control:         Optional[int],
        tps_start_outer:       int,
        # loop settings
        max_outer_iters:  int,
        inner_iters:      int,
        delta_threshold:  float,
        tol_procrustes:   float,
        allow_reflection: bool,
        verbose:          bool,
        log_path:         str,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TPSDeformationField, list]:
    """
    Cross-timepoint alternating optimisation loop.

    Phases
    ------
    Phase 1 (outer iters 0 … tps_start_outer − 1):
        Block A — CBA-FGW transport with rigid CBA static cost C_CBA_rigid[i,j]
        Block B — Weighted Procrustes → update (R, t)

    Phase 2 (outer iters tps_start_outer … max_outer_iters − 1):
        Block A — CBA-FGW transport with TPS CBA static cost C_CBA_TPS[i,j]
        Block B — Weighted TPS fit → update deformation field φ

    This phased approach ensures the rigid alignment converges before the TPS
    introduces non-affine degrees of freedom, avoiding premature overfitting
    of the deformation field to a wrong rigid placement.

    Returns
    -------
    pi            : ndarray (ns, nt)   — final transport plan
    R             : ndarray (2, 2)     — final rigid rotation (from Phase 1)
    t             : ndarray (2,)       — final rigid translation (from Phase 1)
    tps_field     : TPSDeformationField — fitted TPS (may be rigid-only if Phase 2
                                          did not start)
    loss_history  : list[float]
    """
    R, t      = R_init.copy(), t_init.copy()
    tps_field = TPSDeformationField(smoothing=tps_smoothing,
                                     n_control_points=tps_n_control)
    # Keep two plan variables:
    #   pi_warmstart — augmented (ns_aug, nt_aug), used as G0 for next FW call
    #   pi_real      — stripped (ns, nt), used for Procrustes, barycenters, output
    pi_warmstart = None
    pi_real      = None
    death_mass_out = 0.0
    birth_mass_out = 0.0
    loss_history: list = []

    for outer in range(max_outer_iters):
        in_tps_phase = (outer >= tps_start_outer) and not tps_field._is_rigid

        if verbose:
            phase_str = "TPS" if in_tps_phase else "rigid"
            print(f"\n[CT-loop] Outer {outer+1}/{max_outer_iters} "
                  f"({phase_str}) — Block A (transport update) …")

        # ── Compute CBA static cost (rigid or TPS) ─────────────────────────
        C_static_full = np.zeros((ns_aug, nt_aug), dtype=np.float64)
        if in_tps_phase:
            C_static_full[:ns, :nt] = tps_field.static_cba_cost(X_s, X_t)
        else:
            C_static_full[:ns, :nt] = cba_static_cost(X_s, X_t, R, t)

        # ── Block A: transport solve (G0 = augmented warm-start) ──────────
        pi_aug, inner_log = cba_fgw_incent(
            C_feat       = C_feat_aug,
            C_cba_static = C_static_full,
            D_s          = D_A_local_aug,
            D_t          = D_B_aug,
            p            = p_vals,
            q            = q_vals,
            beta         = beta_cba,
            alpha_gw     = alpha_gw,
            gamma_quad   = gamma_quad,
            G0           = pi_warmstart,   # augmented (ns_aug, nt_aug) or None
            numItermax   = inner_iters,
            tol_rel      = 1e-7,
            tol_abs      = 1e-9,
            verbose      = verbose,
            log          = True,
            numItermaxEmd = kwargs.get('numItermaxEmd', 500_000),
        )
        loss_history.extend(inner_log['loss'])

        # Keep augmented plan for next warm-start (correct shape)
        pi_warmstart = pi_aug

        # Birth/death mass before stripping
        death_mass_out = float(pi_aug[:ns, -1].sum()) if nt_aug > nt else 0.0
        birth_mass_out = float(pi_aug[-1, :nt].sum()) if ns_aug > ns else 0.0

        # Strip dummy rows/cols → real (ns, nt) plan
        pi_stripped = pi_aug[:ns, :nt].copy()
        pi_sum = pi_stripped.sum()
        if pi_sum > 0:
            pi_stripped /= pi_sum
        pi_real = pi_stripped

        # ── Compute barycenters ────────────────────────────────────────────
        weights     = pi_real.sum(axis=1)
        matched     = weights > delta_threshold
        barycenters = compute_barycenters(pi_real, X_t)   # (N, 2)
        n_matched   = matched.sum()

        if verbose:
            print(f"[CT-loop]   Matched cells: {n_matched}")

        if n_matched < 4:
            warnings.warn("Fewer than 4 matched cells; stopping alternating loop.")
            break

        # ── Block B: update transform (Procrustes or TPS) ─────────────────
        if outer < tps_start_outer:
            # Phase 1: weighted Procrustes
            R_new, t_new = solve_procrustes_weighted(
                X_s[matched], barycenters[matched], weights[matched],
                allow_reflection=allow_reflection)
            delta_R = np.linalg.norm(R_new - R, 'fro')
            delta_t = np.linalg.norm(t_new - t)
            R, t = R_new, t_new

            if verbose:
                rmse_r = compute_cba_rmse(pi_real, X_s, X_t, R, t, delta_threshold)
                print(f"[CT-loop]   Procrustes ΔR={delta_R:.2e} Δt={delta_t:.2e} "
                      f"CBA-RMSE={rmse_r:.4f}")

            with open(log_path, 'a') as lf:
                lf.write(f"CT outer {outer+1} [rigid]: n_matched={n_matched}, "
                         f"ΔR={delta_R:.2e}, Δt={delta_t:.2e}\n")

            # Transition to TPS phase when rigid has converged
            if delta_R < tol_procrustes and delta_t < tol_procrustes:
                if verbose:
                    print(f"[CT-loop] Rigid converged; switching to TPS phase.")
                tps_field.fit(X_s, barycenters, weights, verbose=verbose)
                with open(log_path, 'a') as lf:
                    lf.write(f"  TPS phase started at outer {outer+1}\n")

        else:
            # Phase 2: TPS update
            tps_field.fit(X_s, barycenters, weights, verbose=verbose)
            tps_rmse = tps_field.cba_loss_tps(pi_real, X_s, X_t, delta_threshold)
            if verbose:
                print(f"[CT-loop]   TPS CBA-RMSE={tps_rmse:.4f}")
            with open(log_path, 'a') as lf:
                lf.write(f"CT outer {outer+1} [TPS]: n_matched={n_matched}, "
                         f"TPS-RMSE={tps_rmse:.4f}\n")

    # Return stripped real plan; birth/death masses tracked separately
    return pi_real, R, t, tps_field, loss_history, death_mass_out, birth_mass_out


def _build_feature_cost(
        sliceA: AnnData, sliceB: AnnData,
        filePath: str,
        sliceA_name: str, sliceB_name: str,
        use_rep: Optional[str],
        beta_type: float,
        overwrite: bool
) -> np.ndarray:
    """
    Construct the combined feature cost matrix C_feat.

    C_feat[i,j] = (1 − β_type) · cosine_dist(expr_i, expr_j)
                + β_type       · [cell_type_i ≠ cell_type_j]

    The cell-type mismatch term acts as a near-hard constraint: it costs
    β_type extra to map across cell types, which strongly discourages
    biologically impossible matches while allowing occasional cross-type
    transport at higher total cost.

    Parameters
    ----------
    sliceA, sliceB : AnnData
    filePath       : str — cache directory
    sliceA_name, sliceB_name : str — identifiers for caching
    use_rep        : str or None — obsm key for expression; None uses .X
    beta_type      : float — weight of cell-type mismatch penalty
    overwrite      : bool

    Returns
    -------
    C_feat : ndarray, shape (N, M)
    """
    A_X = to_dense_array(extract_data_matrix(sliceA, use_rep)).astype(np.float32)
    B_X = to_dense_array(extract_data_matrix(sliceB, use_rep)).astype(np.float32)
    A_X += 0.01
    B_X += 0.01

    cache_path = os.path.join(filePath, f"cosine_dist_{sliceA_name}_{sliceB_name}.npy")
    if os.path.exists(cache_path) and not overwrite:
        cosine_dist = np.load(cache_path)
    else:
        cosine_dist = cosine_distances(A_X, B_X).astype(np.float64)
        np.save(cache_path, cosine_dist)

    lab_A = np.asarray(sliceA.obs['cell_type_annot'].values)
    lab_B = np.asarray(sliceB.obs['cell_type_annot'].values)
    mismatch = (lab_A[:, None] != lab_B[None, :]).astype(np.float64)

    return (1.0 - beta_type) * cosine_dist + beta_type * mismatch


def _build_neighborhood_cost(
        sliceA: AnnData, sliceB: AnnData,
        filePath: str,
        sliceA_name: str, sliceB_name: str,
        radius: float,
        overwrite: bool,
        method: str = 'jsd'
) -> np.ndarray:
    """
    Build the neighbourhood composition divergence cost matrix.

    Each cell i is described by the distribution of cell types within a
    spatial disc of radius `radius`. The JSD between these distributions is
    used as a secondary feature cost capturing the local tissue microenvironment.

    Parameters
    ----------
    sliceA, sliceB : AnnData
    radius         : float — neighbourhood radius in spatial units
    method         : str — 'jsd' | 'cosine' | 'msd'

    Returns
    -------
    M_nbhd : ndarray, shape (N, M)
    """
    def _compute_nbhd(sl, name):
        cache = os.path.join(filePath, f"nbhd_{name}.npy")
        if os.path.exists(cache) and not overwrite:
            nd = np.load(cache)
        else:
            nd = _neighborhood_distribution(sl, radius)
            nd += 0.01
            np.save(cache, nd)
        return nd

    nd_A = _compute_nbhd(sliceA, sliceA_name)
    nd_B = _compute_nbhd(sliceB, sliceB_name)

    if method == 'jsd':
        cache_jsd = os.path.join(filePath, f"jsd_{sliceA_name}_{sliceB_name}.npy")
        if os.path.exists(cache_jsd) and not overwrite:
            M = np.load(cache_jsd)
        else:
            M = np.array(jensenshannon_divergence_backend(
                torch.from_numpy(nd_A), torch.from_numpy(nd_B)))
            if hasattr(M, 'numpy'):
                M = M.numpy()
            M = np.asarray(M, dtype=np.float64)
            np.save(cache_jsd, M)
    elif method == 'cosine':
        num = nd_A @ nd_B.T
        den = (np.linalg.norm(nd_A, axis=1)[:, None] *
               np.linalg.norm(nd_B, axis=1)[None, :])
        M = 1.0 - num / (den + 1e-12)
    elif method == 'msd':
        M = pairwise_msd(nd_A, nd_B).astype(np.float64)
    else:
        raise ValueError(f"Unknown neighbourhood method: {method!r}")
    return M


def _neighborhood_distribution(curr_slice: AnnData, radius: float) -> np.ndarray:
    """
    Per-cell neighbourhood cell-type composition vector (probability distribution).

    Uses a BallTree for O(N log N) computation. Returns a (N, C) matrix where
    C is the number of unique cell types; each row sums to 1.
    """
    from sklearn.neighbors import BallTree
    cell_types = np.asarray(curr_slice.obs['cell_type_annot'].astype(str))
    unique_ct  = np.unique(cell_types)
    ct_idx     = {c: i for i, c in enumerate(unique_ct)}
    coords     = curr_slice.obsm['spatial']
    N          = curr_slice.shape[0]

    dist_mat   = np.zeros((N, len(unique_ct)), dtype=float)
    tree       = BallTree(coords)
    nbrs_list  = tree.query_radius(coords, r=radius)

    for i, nbrs in enumerate(nbrs_list):
        for j in nbrs:
            dist_mat[i][ct_idx[cell_types[j]]] += 1

    row_sums = dist_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return dist_mat / row_sums


def _augment_dummy_cells(
        ns: int, nt: int,
        lab_A: np.ndarray, lab_B: np.ndarray,
        C_feat: np.ndarray,
        C_nbhd: np.ndarray,
        D_A: np.ndarray,
        D_B: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, float, float, bool, bool, int]:
    """
    Add dummy source/target cells to model partially overlapping slices.

    Mirrors the logic in the original INCENT.py exactly for backward
    compatibility. Returns augmented matrices plus birth/death budgets.

    Returns
    -------
    C_feat_aug, C_nbhd_aug, D_A_aug, D_B_aug : augmented cost/distance matrices
    p_vals, q_vals : marginal distributions (normalised)
    w_dummy_src, w_dummy_tgt : dummy weights
    has_dummy_src, has_dummy_tgt : flags
    budget : int
    """
    from collections import Counter
    counts_A = Counter(lab_A)
    counts_B = Counter(lab_B)
    all_types = set(counts_A) | set(counts_B)
    budget   = sum(max(counts_A.get(k, 0), counts_B.get(k, 0)) for k in all_types)
    w_src    = budget - ns    # needed extra src weight (birth)
    w_tgt    = budget - nt    # needed extra tgt weight (death)
    has_src  = w_src > 0
    has_tgt  = w_tgt > 0

    ns_aug = ns + (1 if has_src else 0)
    nt_aug = nt + (1 if has_tgt else 0)
    eps    = 1e-6

    # --- Augment distance matrices ---
    def _aug_dist(D, n_aug):
        D_aug = np.zeros((n_aug, n_aug), dtype=np.float64)
        D_aug[:D.shape[0], :D.shape[1]] = D
        return D_aug

    D_A_aug = _aug_dist(D_A, ns_aug)
    D_B_aug = _aug_dist(D_B, nt_aug)

    # --- Per-type max costs for informative dummy penalties ---
    type_max_feat = {}
    type_max_nbhd = {}
    for k in all_types:
        S = np.where(lab_A == k)[0]
        T = np.where(lab_B == k)[0]
        if len(S) > 0 and len(T) > 0:
            type_max_feat[k] = float(C_feat[np.ix_(S, T)].max())
            type_max_nbhd[k] = float(C_nbhd[np.ix_(S, T)].max())
        else:
            type_max_feat[k] = float(C_feat.max())
            type_max_nbhd[k] = float(C_nbhd.max())

    death_feat = np.array([type_max_feat[lab_A[i]] for i in range(ns)])
    birth_feat = np.array([type_max_feat[lab_B[j]] for j in range(nt)])
    death_nbhd = np.array([type_max_nbhd[lab_A[i]] for i in range(ns)])
    birth_nbhd = np.array([type_max_nbhd[lab_B[j]] for j in range(nt)])

    # --- Augment cost matrices ---
    def _aug_cost(M, n_aug_s, n_aug_t, death_col, birth_row):
        M_aug = np.zeros((n_aug_s, n_aug_t), dtype=np.float64)
        M_aug[:ns, :nt] = M
        if has_tgt:
            M_aug[:ns, nt] = death_col + eps
        if has_src:
            M_aug[ns, :nt] = birth_row + eps
        if has_src and has_tgt:
            M_aug[ns, nt] = M.max() + eps
        return M_aug

    C_feat_aug = _aug_cost(C_feat, ns_aug, nt_aug, death_feat, birth_feat)
    C_nbhd_aug = _aug_cost(C_nbhd, ns_aug, nt_aug, death_nbhd, birth_nbhd)

    # --- Marginals ---
    p_vals = np.full(ns_aug, 1.0 / budget, dtype=np.float64)
    q_vals = np.full(nt_aug, 1.0 / budget, dtype=np.float64)
    if has_src:
        p_vals[-1] = float(w_src) / budget
    if has_tgt:
        q_vals[-1] = float(w_tgt) / budget

    return (C_feat_aug, C_nbhd_aug, D_A_aug, D_B_aug,
            p_vals, q_vals, w_src, w_tgt, has_src, has_tgt, budget)


def _strip_dummy(pi_aug: np.ndarray, ns: int, nt: int,
                  has_src: bool, has_tgt: bool
                  ) -> Tuple[np.ndarray, float, float]:
    """
    Remove dummy rows/columns from the augmented plan and report birth/death mass.

    Returns
    -------
    pi   : stripped plan (ns, nt)
    death_mass : fraction of source mass that went to dummy target
    birth_mass : fraction of target mass that came from dummy source
    """
    death_mass = float(pi_aug[:ns, -1].sum()) if has_tgt else 0.0
    birth_mass = float(pi_aug[-1, :nt].sum()) if has_src else 0.0

    if has_src and has_tgt:
        pi = pi_aug[:ns, :nt]
    elif has_src:
        pi = pi_aug[:ns, :]
    elif has_tgt:
        pi = pi_aug[:, :nt]
    else:
        pi = pi_aug

    pi_sum = pi.sum()
    if pi_sum > 0:
        pi = pi / pi_sum

    return pi, death_mass, birth_mass


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point: coherent_pairwise_align
# ─────────────────────────────────────────────────────────────────────────────

def coherent_pairwise_align(
        sliceA: AnnData,
        sliceB: AnnData,
        filePath: str,
        *,
        # Structural weights
        beta_cba: float         = 1.0,
        alpha_gw: float         = 0.1,
        gamma_quad: float       = 0.01,
        # Feature weights
        beta_type: float        = 0.3,
        gamma_nbhd: float       = 0.5,
        # Spatial neighbourhood
        radius: float           = 100.0,
        knn_k: int              = 10,
        # Hypothesis generation
        n_hypotheses: int       = 8,
        n_spectral: int         = 6,
        # Alternating optimisation
        max_outer_iters: int    = 6,
        inner_iters: int        = 150,
        delta_threshold: float  = 1e-6,
        tol_procrustes: float   = 1e-4,
        # Mode
        mode: str               = 'same_timepoint',
        allow_reflection: bool  = True,
        # ── Cross-timepoint parameters ───────────────────────────────────────
        # Expression alignment
        expr_backend: str       = 'harmony',    # 'scvi' | 'harmony' | 'pca'
        lambda_lineage: float   = 0.3,          # weight of lineage-aware cost
        n_latent: int           = 20,           # latent dim (scVI / PCA)
        scvi_epochs: int        = 400,          # scVI training epochs
        # TPS deformation
        tps_smoothing: float    = 0.1,          # TPS bending energy weight
        tps_n_control: Optional[int] = None,    # control points (None = all matched)
        tps_start_outer: int    = 3,            # outer iter to begin TPS phase
        # Semi-relaxed margins
        tau_t: float            = 0.3,          # fraction of target that may be new
        # Cell fate thresholds
        type_change_threshold: float = 0.5,     # min fraction for "differentiating"
        # I/O
        use_rep: Optional[str]  = None,
        sliceA_name: str        = 'sliceA',
        sliceB_name: str        = 'sliceB',
        overwrite: bool         = False,
        neighborhood_dissimilarity: str = 'jsd',
        dummy_cell: bool        = True,
        verbose: bool           = False,
        return_result: bool     = True,
        **kwargs
) -> Union[AlignmentResult, np.ndarray]:
    """
    Coherent Barycentric Alignment with Fused Gromov-Wasserstein (CBA-FGW).

    Jointly optimises a transport plan π and rigid transform (R, t) to align
    two MERFISH slices with partial overlap, arbitrary rotation/reflection,
    and optional cross-timepoint expression drift.

    The matched regions of source and target will satisfy simultaneously:
      1. Same gene expression and cell-type composition (feature match)
      2. Same cell-neighbourhood microenvironments (NCP match)
      3. Same internal spatial geometry — when you plot the mapped region,
         it looks like the region it maps to (CBA + GW_local)
      4. Compact 1-to-few transport (quadratic sparsity)
      5. No forced mapping for cells at removed borders (unbalanced margins)

    Alternating optimisation
    ------------------------
    Block A  (fix R, t)  — solve CBA-FGW transport via Frank-Wolfe:
        C_total = C_feat + γ_nbhd·C_nbhd + β_cba · ‖Rx_s + t − x_t‖²
        min_{π} ⟨C_total, π⟩ + α·GW_local(π) + γ·‖π‖_F²

    Block B  (fix π)  — weighted Procrustes on barycenter pairs:
        (R*, t*) = argmin_{R,t} Σ_i ‖π_i‖₁ · ‖Rx_i^s + t − T(i)‖²
        Closed-form SVD solution; handles reflections via det correction.

    Parameters
    ----------
    sliceA, sliceB  : AnnData
        MERFISH slices with .obs['cell_type_annot'], .obsm['spatial'], and
        shared gene expression in .X (or use_rep).
    filePath        : str
        Directory for caching precomputed cost matrices.
    beta_cba        : float (default 1.0)
        Weight of the CBA structural loss.  This is the primary knob for
        enforcing that the mapped region looks like the source region.
    alpha_gw        : float (default 0.1)
        Weight of the local GW term (target-side structure preservation).
        Set α ≈ β_cba / 5 as a sensible default.
    gamma_quad      : float (default 0.01)
        Weight of the ‖π‖_F² sparsity penalty.  Higher → fewer targets per
        source → higher confidence but less flexibility for partial overlaps.
    beta_type       : float (default 0.3)
        Fraction of feature cost attributed to cell-type mismatch penalty.
    gamma_nbhd      : float (default 0.5)
        Weight of neighbourhood divergence cost in C_feat.
    radius          : float (default 100.0)
        Radius (same units as .obsm['spatial']) for neighbourhood computation.
    knn_k           : int (default 10)
        k for local GW kNN graph.
    n_hypotheses    : int (default 8)
        Number of rigid-transform hypotheses from spectral initialisation.
    n_spectral      : int (default 6)
        Number of graph Laplacian eigenvectors for hypothesis generation.
    max_outer_iters : int (default 6)
        Maximum alternating-optimisation iterations (each = one full transport
        solve + one Procrustes step).
    inner_iters     : int (default 150)
        Maximum Frank-Wolfe iterations inside each transport update.
    delta_threshold : float (default 1e-6)
        Row-sum threshold below which a source cell is declared unmatched.
    tol_procrustes  : float (default 1e-4)
        Convergence tolerance for the rigid transform (Frobenius norm change).
    mode            : str (default 'same_timepoint')
        'same_timepoint' — rigid + reflection, raw expression cost.
        'cross_timepoint' — TPS deformation, batch-corrected latent cost,
                            semi-relaxed margins, cell fate classification.
    allow_reflection : bool (default True)
        Whether to include reflections in the Procrustes search. Critical for
        MERFISH data where the correct alignment may include a mirror flip.
    expr_backend    : str (default 'harmony')
        Expression alignment backend for cross-timepoint mode.
        'scvi'    — Conditional VAE (scVI). Requires scvi-tools ≥ 1.0.
                    Best for strong developmental shifts.
        'harmony' — Harmony-corrected PCA. Fast, no GPU required. Good
                    default for moderate temporal changes.
        'pca'     — Joint PCA without batch correction. Appropriate when
                    expression shift is small.
    lambda_lineage  : float (default 0.3)
        Weight of the lineage-aware cost in C_feat_CT.  At 0, pure latent
        distance is used; at 1, only type centroid distance.  Values in
        0.2–0.4 are recommended.
    n_latent        : int (default 20)
        Latent dimension for scVI or number of PCs for Harmony/PCA.
    scvi_epochs     : int (default 400)
        Training epochs for scVI. Ignored for other backends.
    tps_smoothing   : float (default 0.1)
        TPS bending energy smoothing.  Higher values → smoother (more affine)
        deformation. Scaled internally by 1/n_anchors.
    tps_n_control   : int or None (default None)
        Maximum TPS control points.  None uses all matched cells.  Set to
        ~500 for large datasets to reduce the O(n³) TPS solve.
    tps_start_outer : int (default 3)
        Outer iteration at which to switch from Procrustes to TPS (Block B).
        The first tps_start_outer iterations use rigid alignment to find a
        good placement before the TPS introduces non-affine freedom.
    tau_t           : float in (0, 1] (default 0.3)
        Fraction of target mass that may be "new" (born) cells.  Higher values
        relax the target marginal more, allowing more cell birth.
    type_change_threshold : float (default 0.5)
        Minimum fraction of outgoing mass to a single target type for a cell
        to be classified as "differentiating" (vs "ambiguous").
    use_rep         : str or None
        obsm key for expression representation; None uses .X.
    sliceA_name, sliceB_name : str
        String identifiers used for caching.
    overwrite       : bool
        If True, recompute all cached matrices.
    neighborhood_dissimilarity : str
        'jsd' | 'cosine' | 'msd' — neighbourhood cost measure.
    dummy_cell      : bool (default True)
        Use dummy-cell augmentation for unbalanced / partial-overlap transport.
        For cross_timepoint mode, semi-relaxed dummy cells are used instead of
        the balanced dummy logic; this flag is therefore ignored in CT mode.
    verbose         : bool
    return_result   : bool
        If True (default), return AlignmentResult; else return pi_fwd array only.

    Returns
    -------
    result : AlignmentResult (if return_result=True)
    pi     : ndarray (N, M) (if return_result=False)
    """
    t_start = time.time()
    os.makedirs(filePath, exist_ok=True)
    log_path = os.path.join(filePath, f"cba_fgw_{sliceA_name}_{sliceB_name}.log")

    with open(log_path, 'w') as logf:
        logf.write(f"coherent_pairwise_align — CBA-FGW v2\n")
        logf.write(f"Started: {datetime.datetime.now()}\n")
        logf.write(f"mode={mode}, sliceA={sliceA_name}, sliceB={sliceB_name}\n")
        logf.write(f"beta_cba={beta_cba}, alpha_gw={alpha_gw}, "
                   f"gamma_quad={gamma_quad}, beta_type={beta_type}\n")
        if mode == 'cross_timepoint':
            logf.write(f"expr_backend={expr_backend}, lambda_lineage={lambda_lineage}, "
                       f"tau_t={tau_t}, tps_smoothing={tps_smoothing}\n")

    # ── 1. Validate inputs ──────────────────────────────────────────────────
    for sl in [sliceA, sliceB]:
        if not len(sl):
            raise ValueError(f"Empty AnnData: {sl}")
        if 'spatial' not in sl.obsm:
            raise ValueError("Both slices must have .obsm['spatial'].")
        if 'cell_type_annot' not in sl.obs.columns:
            raise ValueError("Both slices must have .obs['cell_type_annot'].")

    # ── 2. Shared genes ─────────────────────────────────────────────────────
    shared_genes = sliceA.var_names.intersection(sliceB.var_names)
    if len(shared_genes) == 0:
        raise ValueError("No shared genes.")
    sliceA = sliceA[:, shared_genes]
    sliceB = sliceB[:, shared_genes]

    ns, nt = sliceA.shape[0], sliceB.shape[0]
    X_s    = sliceA.obsm['spatial'].astype(np.float64)
    X_t    = sliceB.obsm['spatial'].astype(np.float64)
    lab_A  = np.asarray(sliceA.obs['cell_type_annot'].values)
    lab_B  = np.asarray(sliceB.obs['cell_type_annot'].values)

    if verbose:
        print(f"[CBA-FGW] mode={mode}  sliceA: {ns} cells, sliceB: {nt} cells, "
              f"{len(shared_genes)} shared genes")

    # ── 3. Distance matrices for GW ─────────────────────────────────────────
    D_A = ot.dist(X_s, X_s, metric='euclidean').astype(np.float64)
    D_B = ot.dist(X_t, X_t, metric='euclidean').astype(np.float64)
    D_A /= (D_A.max() + 1e-12)
    D_B /= (D_B.max() + 1e-12)

    # ── 4. kNN mask for local GW ─────────────────────────────────────────────
    knn_mask_A  = build_knn_mask(X_s, k=knn_k)
    D_A_local   = D_A * knn_mask_A

    # ── 5. Feature cost ──────────────────────────────────────────────────────
    # In cross-timepoint mode, the expression cost uses batch-corrected latent
    # representations; the same-timepoint path uses raw cosine distance.
    Z_A = Z_B = None
    expr_backend_used = 'raw'

    if mode == 'cross_timepoint':
        if verbose:
            print(f"[CBA-FGW] Cross-timepoint mode: building latent feature cost "
                  f"(backend={expr_backend}) …")
        C_feat_base, Z_A, Z_B, expr_backend_used = _build_feature_cost_ct(
            sliceA, sliceB, filePath, sliceA_name, sliceB_name,
            use_rep, lambda_lineage, expr_backend,
            {'n_latent': n_latent, 'max_epochs': scvi_epochs},
            overwrite, verbose)
    else:
        C_feat_base = _build_feature_cost(
            sliceA, sliceB, filePath, sliceA_name, sliceB_name,
            use_rep, beta_type, overwrite)

    # ── 6. Neighbourhood cost ────────────────────────────────────────────────
    C_nbhd = _build_neighborhood_cost(
        sliceA, sliceB, filePath, sliceA_name, sliceB_name,
        radius, overwrite, method=neighborhood_dissimilarity)
    C_feat = C_feat_base + gamma_nbhd * C_nbhd

    # ── 7. Dummy-cell / semi-relaxed augmentation ────────────────────────────
    if mode == 'cross_timepoint':
        # Semi-relaxed: asymmetric birth/death augmentation
        # τ_s → ∞  (source marginal is exact)
        # τ_t < ∞  (target marginal relaxed by tau_t)
        if verbose:
            print(f"[CBA-FGW] Semi-relaxed margins (tau_t={tau_t}) …")
        p_vals, q_vals, w_birth, w_death = build_semitrelaxed_marginals(
            ns, nt, lab_A, lab_B, tau_t=tau_t)
        C_feat_aug, D_A_aug, D_B_aug = augment_cost_matrices_ct(
            C_feat, D_A, D_B, ns, nt, lab_A, lab_B,
            birth_cost_mult=2.0, death_cost_mult=1.0)
        # kNN-masked distance (dummy row/col stays zero)
        ns_aug, nt_aug = ns + 1, nt + 1
        D_A_local_aug = np.zeros((ns_aug, ns_aug), dtype=np.float64)
        D_A_local_aug[:ns, :ns] = D_A_local
        has_dummy_src = True
        has_dummy_tgt = True
        if verbose:
            print(f"[CBA-FGW]   w_birth={w_birth:.1f}, w_death={w_death:.1f}")
        with open(log_path, 'a') as logf:
            logf.write(f"[CT] w_birth={w_birth:.1f}, w_death={w_death:.1f}\n")

    elif dummy_cell:
        # Same-timepoint balanced dummy augmentation (original INCENT logic)
        (C_feat_aug, C_nbhd_aug,
         D_A_aug, D_B_aug,
         p_vals, q_vals,
         w_src, w_tgt,
         has_dummy_src, has_dummy_tgt,
         budget) = _augment_dummy_cells(
             ns, nt, lab_A, lab_B, C_feat, C_nbhd, D_A, D_B)
        ns_aug = ns + (1 if has_dummy_src else 0)
        nt_aug = nt + (1 if has_dummy_tgt else 0)
        D_A_local_aug = np.zeros((ns_aug, ns_aug), dtype=np.float64)
        D_A_local_aug[:ns, :ns] = D_A_local
        if verbose:
            print(f"[CBA-FGW] Same-timepoint dummy cells augmented.")
    else:
        C_feat_aug = C_feat
        D_A_aug = D_A; D_B_aug = D_B
        D_A_local_aug = D_A_local
        p_vals = np.ones(ns, dtype=np.float64) / ns
        q_vals = np.ones(nt, dtype=np.float64) / nt
        has_dummy_src = has_dummy_tgt = False
        ns_aug = ns; nt_aug = nt

    # ── 8. Spectral multi-hypothesis initialisation ──────────────────────────
    if verbose:
        print(f"[CBA-FGW] Generating {n_hypotheses} rigid hypotheses …")
    expr_A = to_dense_array(extract_data_matrix(sliceA, use_rep)).astype(np.float64)
    expr_B = to_dense_array(extract_data_matrix(sliceB, use_rep)).astype(np.float64)

    # For cross-timepoint, use latent representations for spectral hypotheses
    # if available (better spectral alignment under expression shift)
    if mode == 'cross_timepoint' and Z_A is not None:
        hyp_expr_A = Z_A.astype(np.float64)
        hyp_expr_B = Z_B.astype(np.float64)
    else:
        hyp_expr_A, hyp_expr_B = expr_A, expr_B

    hypotheses = spectral_hypotheses(
        X_s, X_t, hyp_expr_A, hyp_expr_B,
        n_hypotheses=n_hypotheses,
        n_spectral_components=n_spectral)

    h_scores = [score_hypothesis(R, t, X_s, X_t, C_feat)
                for R, t in hypotheses]
    best_idx = int(np.argmin(h_scores))
    R, t     = hypotheses[best_idx]

    if verbose:
        print(f"[CBA-FGW] Best hypothesis: {best_idx} (score {h_scores[best_idx]:.4f})")
    with open(log_path, 'a') as logf:
        for hi, (sc, _) in enumerate(zip(h_scores, hypotheses)):
            logf.write(f"  Hypothesis {hi}: score={sc:.4f}\n")
        logf.write(f"  Selected: {best_idx}\n")

    # ── 9. Alternating optimisation ──────────────────────────────────────────
    tps_field    = None
    loss_history = []
    death_mass   = 0.0
    birth_mass   = 0.0

    if mode == 'cross_timepoint':
        # Full cross-timepoint loop: rigid phase → TPS phase
        pi, R, t, tps_field, loss_history, death_mass, birth_mass = _run_ct_alternating_loop(
            C_feat_aug    = C_feat_aug,
            D_A_aug       = D_A_aug,
            D_B_aug       = D_B_aug,
            D_A_local_aug = D_A_local_aug,
            p_vals        = p_vals,
            q_vals        = q_vals,
            X_s           = X_s,
            X_t           = X_t,
            ns            = ns,
            nt            = nt,
            ns_aug        = ns_aug,
            nt_aug        = nt_aug,
            R_init        = R,
            t_init        = t,
            beta_cba      = beta_cba,
            alpha_gw      = alpha_gw,
            gamma_quad    = gamma_quad,
            tps_smoothing = tps_smoothing,
            tps_n_control = tps_n_control,
            tps_start_outer = tps_start_outer,
            max_outer_iters = max_outer_iters,
            inner_iters     = inner_iters,
            delta_threshold = delta_threshold,
            tol_procrustes  = tol_procrustes,
            allow_reflection = allow_reflection,
            verbose          = verbose,
            log_path         = log_path,
            **kwargs,
        )
        # death_mass and birth_mass are now returned directly from _run_ct_alternating_loop

    else:
        # Same-timepoint: rigid alternating loop
        # Keep two variables:
        #   pi_warmstart — augmented (ns_aug, nt_aug), used as G0 for next FW call
        #   pi_real      — stripped (ns, nt), used for Procrustes and output
        pi_warmstart = None
        pi_real      = None
        for outer in range(max_outer_iters):
            if verbose:
                print(f"\n[CBA-FGW] Outer iter {outer+1}/{max_outer_iters} — "
                      f"Block A (transport update) …")

            C_cba_static_full = np.zeros((ns_aug, nt_aug), dtype=np.float64)
            C_cba_static_full[:ns, :nt] = cba_static_cost(X_s, X_t, R, t)

            pi_aug, inner_log = cba_fgw_incent(
                C_feat       = C_feat_aug,
                C_cba_static = C_cba_static_full,
                D_s          = D_A_local_aug,
                D_t          = D_B_aug,
                p            = p_vals,
                q            = q_vals,
                beta         = beta_cba,
                alpha_gw     = alpha_gw,
                gamma_quad   = gamma_quad,
                G0           = pi_warmstart,   # augmented warm-start (correct shape)
                numItermax   = inner_iters,
                tol_rel      = 1e-7,
                tol_abs      = 1e-9,
                verbose      = verbose,
                log          = True,
                numItermaxEmd = kwargs.get('numItermaxEmd', 500_000),
            )
            loss_history.extend(inner_log['loss'])

            # Save augmented plan for next warm-start BEFORE stripping
            pi_warmstart = pi_aug

            # Strip dummy rows/cols for Procrustes and output
            pi_real, death_mass, birth_mass = _strip_dummy(
                pi_aug, ns, nt, has_dummy_src, has_dummy_tgt)

            weights     = pi_real.sum(axis=1)
            matched     = weights > delta_threshold
            barycenters = compute_barycenters(pi_real, X_t)
            n_matched   = matched.sum()

            if n_matched < 3:
                break

            if verbose:
                rmse = compute_cba_rmse(pi_real, X_s, X_t, R, t, delta_threshold)
                print(f"[CBA-FGW]   Matched: {n_matched}, CBA RMSE: {rmse:.4f}")

            R_new, t_new = solve_procrustes_weighted(
                X_s[matched], barycenters[matched], weights[matched],
                allow_reflection=allow_reflection)
            delta_R = np.linalg.norm(R_new - R, 'fro')
            delta_t = np.linalg.norm(t_new - t)
            R, t = R_new, t_new

            if verbose:
                print(f"[CBA-FGW]   ΔR={delta_R:.2e}, Δt={delta_t:.2e}")
            with open(log_path, 'a') as logf:
                logf.write(f"Outer {outer+1}: n_matched={n_matched}, "
                           f"ΔR={delta_R:.2e}, Δt={delta_t:.2e}\n")

            if delta_R < tol_procrustes and delta_t < tol_procrustes:
                if verbose:
                    print(f"[CBA-FGW] Converged at outer iter {outer+1}.")
                break

        # Use stripped plan as final output
        pi = pi_real if pi_real is not None else np.zeros((ns, nt), dtype=np.float64)

    # ── 10. Reverse transport plan ───────────────────────────────────────────
    if verbose:
        print("\n[CBA-FGW] Computing reverse transport plan …")

    # Reverse CBA static cost
    if mode == 'cross_timepoint' and tps_field is not None and not tps_field._is_rigid:
        C_cba_rev = tps_field.static_cba_cost(X_t, X_s)    # (nt, ns)
    else:
        C_cba_rev = cba_static_cost(X_t, X_s, R.T, -R.T @ t)   # (nt, ns)

    # Build properly augmented reverse costs.
    # IMPORTANT: np.pad with default zeros gives dummy rows/cols a cost of 0,
    # which causes the solver to trivially route all mass to the dummy.
    # Instead, dummy entries must carry a high penalty (2× max real cost).
    use_dummy = dummy_cell or mode == 'cross_timepoint'
    cost_max  = max(float(C_feat.max()), 1e-6) * 2.0 + 1e-6

    if use_dummy:
        C_feat_rev_aug = np.full((nt_aug, ns_aug), cost_max, dtype=np.float64)
        C_feat_rev_aug[:nt, :ns] = C_feat.T
        if has_dummy_src and has_dummy_tgt:
            C_feat_rev_aug[-1, -1] = cost_max   # dummy ↔ dummy is free (relative)

        C_cba_rev_aug = np.zeros((nt_aug, ns_aug), dtype=np.float64)
        C_cba_rev_aug[:nt, :ns] = C_cba_rev
        # Dummy row/col stays 0 in the CBA cost (invisible to structure term)

        p_rev = q_vals   # target becomes source in reverse
        q_rev = p_vals   # source becomes target in reverse
    else:
        C_feat_rev_aug = C_feat.T
        C_cba_rev_aug  = C_cba_rev
        p_rev = np.ones(nt, dtype=np.float64) / nt
        q_rev = np.ones(ns, dtype=np.float64) / ns

    pi_rev_aug, _ = cba_fgw_incent(
        C_feat       = C_feat_rev_aug,
        C_cba_static = C_cba_rev_aug,
        D_s          = D_B_aug    if use_dummy else D_B,
        D_t          = D_A_local_aug if use_dummy else D_A_local,
        p            = p_rev,
        q            = q_rev,
        beta         = beta_cba,
        alpha_gw     = alpha_gw,
        gamma_quad   = gamma_quad,
        numItermax   = inner_iters,
        verbose      = False,
        log          = True,
        numItermaxEmd = kwargs.get('numItermaxEmd', 500_000),
    )
    if use_dummy:
        pi_rev, _, _ = _strip_dummy(pi_rev_aug, nt, ns, has_dummy_tgt, has_dummy_src)
    else:
        pi_rev = pi_rev_aug

    # ── 11. Diagnostics ─────────────────────────────────────────────────────
    # For TPS mode, warped source uses the TPS field
    if mode == 'cross_timepoint' and tps_field is not None and not tps_field._is_rigid:
        src_warped = tps_field.transform(X_s)
    else:
        src_warped = (X_s @ R.T) + t[np.newaxis, :]

    bary_final  = compute_barycenters(pi, X_t)
    bary_rev    = compute_barycenters(pi_rev, X_s)
    w_fwd       = pi.sum(axis=1)
    matched_src = w_fwd > delta_threshold
    matched_tgt = pi.sum(axis=0) > delta_threshold

    k_eff     = compute_effective_support(pi)
    fwd_comp  = compute_forward_compactness(pi, X_t)
    rev_comp  = compute_reverse_compactness(pi, X_s)
    cba_rmse  = compute_cba_rmse(pi, X_s, X_t, R, t, delta_threshold)

    metrics: Dict = {
        'mode':              mode,
        'cba_rmse':          cba_rmse,
        'fwd_compactness':   fwd_comp,
        'rev_compactness':   rev_comp,
        'mean_k_eff':        float(k_eff[matched_src].mean()) if matched_src.any() else 0.0,
        'median_k_eff':      float(np.median(k_eff[matched_src])) if matched_src.any() else 0.0,
        'n_matched_src':     int(matched_src.sum()),
        'n_matched_tgt':     int(matched_tgt.sum()),
        'hypothesis_scores': h_scores,
        'selected_hypothesis': best_idx,
    }

    # Cross-timepoint specific metrics and cell fate
    expr_shift_info = None
    cell_fate_info  = None
    if mode == 'cross_timepoint':
        # Expression shift in latent space
        if Z_A is not None and Z_B is not None:
            expr_shift_info = compute_expression_shift(
                Z_A, Z_B, pi, lab_A, lab_B, delta_threshold)
            metrics['mean_expression_shift'] = expr_shift_info['mean_shift']
        else:
            metrics['mean_expression_shift'] = float('nan')

        # Cell fate classification
        cell_fate_info = identify_cell_fate(
            pi, lab_A, lab_B, delta_threshold, type_change_threshold)
        metrics['n_maintained']     = int(cell_fate_info['maintained'].sum())
        metrics['n_differentiating'] = int(cell_fate_info['differentiating'].sum())
        metrics['n_dead']            = int(cell_fate_info['dead'].sum())
        metrics['expression_backend'] = expr_backend_used

        # TPS anchor count
        metrics['n_tps_anchors'] = (
            int(matched_src.sum()) if (tps_field and not tps_field._is_rigid)
            else 0)

    runtime = time.time() - t_start
    with open(log_path, 'a') as logf:
        logf.write(f"\n── Final metrics ──\n")
        for k, v in metrics.items():
            logf.write(f"  {k}: {v}\n")
        logf.write(f"Runtime: {runtime:.2f}s\n")

    result = AlignmentResult(
        pi_fwd              = pi,
        pi_rev              = pi_rev,
        R                   = R,
        t                   = t,
        src_warped          = src_warped,
        forward_barycenters = bary_final,
        reverse_barycenters = bary_rev,
        matched_src         = matched_src,
        matched_tgt         = matched_tgt,
        raw_unmatched_src   = death_mass,
        raw_unmatched_tgt   = birth_mass,
        metrics             = metrics,
        selected_hypothesis = best_idx,
        hypothesis_scores   = h_scores,
        loss_history        = loss_history,
        runtime_seconds     = runtime,
        # CT-specific
        deformation_field   = tps_field,
        src_warped_tps      = src_warped if mode == 'cross_timepoint' else None,
        expression_shift    = expr_shift_info,
        cell_fate           = cell_fate_info,
        latent_A            = Z_A,
        latent_B            = Z_B,
    )

    if verbose:
        print(result.summary())

    if return_result:
        return result
    return pi

