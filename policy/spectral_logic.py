"""Exact Walsh-spectral threshold head for the game's Boolean logic (after Gorgi Pavlov,
arXiv:2601.13953 / arXiv:2605.01637).

The prereq head answers a Boolean question: "are role r's prerequisites all present?" =
AND over the prereq building-presence bits. A dense MLP learns a FUZZY approximation of that AND
that degrades under quantization. Instead we represent it EXACTLY in the Boolean-Fourier (Walsh)
basis: f(x) = sum_S f_hat(S) * chi_S(x), with f_hat read straight off the truth table by a WHT.
The coefficients are sparse and dyadic, so the head is exact at fp32 AND stays exact at low bit-
widths where the MLP head falls apart. (A prereq with one input is a 'dictator'; with several, an
'AND' -- the scaling classes of the Banach-Butterfly paper.)

    python policy/spectral_logic.py    # exact prereq logic; robust where the MLP degrades
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "yr_env"))
import game_model as gm                                   # noqa: E402
from wht_quant import fwht                                # noqa: E402

# the buildable roles (match offline_bc.BUILD_ROLES) and the building-presence bits they depend on
BUILD_ROLES = [gm.POWER, gm.ECONOMY, gm.PROD_INF, gm.PROD_VEH, gm.TECH_RADAR, gm.DEF_GROUND, gm.DEF_AA]
PREREQ_BITS = [gm.CONSTRUCTION, gm.POWER, gm.ECONOMY, gm.PROD_INF, gm.PROD_VEH, gm.TECH_RADAR]
BIT_IX = {r: i for i, r in enumerate(PREREQ_BITS)}
K = len(PREREQ_BITS)                                      # input bits
N = 1 << K                                                # truth-table size


def _truth_table(role):
    """tt[x] = 1 iff every prereq building of `role` is present in the bit-pattern x (0..2^K-1)."""
    need = [BIT_IX[p] for p in gm.PREREQ.get(role, []) if p in BIT_IX]
    tt = np.zeros(N, np.float32)
    for x in range(N):
        tt[x] = 1.0 if all((x >> b) & 1 for b in need) else 0.0
    return tt


def build_spectrum():
    """Exact Walsh spectrum of each buildable role's prereq-AND: coeffs (n_roles, N)."""
    coeffs = np.stack([fwht(_truth_table(r)) / N for r in BUILD_ROLES])   # f_hat(S) = WHT(tt)/2^K
    return coeffs.astype(np.float32)


def _parity_features(bits):
    """bits: (B, K) in {0,1} -> (B, N) Walsh characters chi_S(x) = (-1)^<S,x>, S = 0..N-1."""
    B = bits.shape[0]
    S = np.arange(N)[None, :, None]                       # (1, N, 1) subset masks
    Sbits = ((S >> np.arange(K)) & 1).astype(np.float32)  # (1, N, K)
    inner = (bits[:, None, :] * Sbits).sum(-1)            # (B, N) = <S, x>
    return np.where(inner % 2 == 0, 1.0, -1.0).astype(np.float32)


def predict(coeffs, bits):
    """Exact prereq head: f(x) = sum_S f_hat(S) chi_S(x) -> {0,1} per role. bits:(B,K) -> (B,n_roles)."""
    phi = _parity_features(np.asarray(bits, np.float32))  # (B, N)
    vals = phi @ coeffs.T                                 # (B, n_roles) == the Boolean values, exactly
    return (vals > 0.5).astype(np.float32)


def bits_from_buildings(own_buildings):
    """Presence bit-vector over PREREQ_BITS from a role->count dict."""
    return np.asarray([1.0 if own_buildings.get(r, 0) > 0 else 0.0 for r in PREREQ_BITS], np.float32)


def quantize_coeffs(coeffs, bits):
    """N-bit symmetric per-role quantization of the spectral coefficients (round-trip)."""
    qmax = 2 ** (bits - 1) - 1
    scale = np.maximum(np.abs(coeffs).max(1, keepdims=True), 1e-8) / qmax
    return np.clip(np.round(coeffs / scale), -qmax - 1, qmax) * scale


if __name__ == "__main__":
    coeffs = build_spectrum()
    nnz = (np.abs(coeffs) > 1e-6).sum(1)
    print("exact prereq spectrum (sparse, dyadic):")
    for r, c, k in zip(BUILD_ROLES, coeffs, nnz):
        print(f"  {r:11s} prereqs {str(gm.PREREQ.get(r, [])):28s} -> {int(k)} nonzero Walsh coeffs")

    # exhaustive check: the spectral head equals the true prereq-AND on ALL 2^K inputs, fp32 and low-bit
    allx = np.stack([[(x >> b) & 1 for b in range(K)] for x in range(N)]).astype(np.float32)
    truth = np.stack([_truth_table(r) for r in BUILD_ROLES]).T            # (N, n_roles)
    print("\nprereq-head accuracy over ALL 2^K building states (exact target):")
    print(f"  {'bits':>6} | {'spectral (Walsh)':>16} | {'note':<30}")
    for bits in ("fp32", 8, 4, 3, 2):
        c = coeffs if bits == "fp32" else quantize_coeffs(coeffs, bits)
        acc = float((predict(c, allx) == truth).mean())
        note = "exact" if acc == 1.0 else "approx"
        print(f"  {str(bits):>6} | {acc:>16.3f} | {note:<30}")
    print("\nThe Walsh head is exact and stays exact at low bit-widths -- the MLP head (see compare_brains) does not.")
