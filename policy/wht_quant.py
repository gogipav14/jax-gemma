"""WHT-rotated low-bit weight quantization (after Gorgi Pavlov, arXiv:2605.25203).

The thesis: a fixed, parameter-free Walsh-Hadamard rotation spreads each weight matrix's outliers
across all coordinates ("incoherence processing") so the matrix quantizes to very few bits with far
less error than naive per-channel rounding. For a linear y = x W:

    y = x W = (x H)(H W)            because H is orthogonal and symmetric (H H = I)

so we rotate the input by H (a free FWHT) and store the quantized H W. We measure FIDELITY by
round-tripping the weight: rotate -> quantize -> dequantize -> rotate back. Because H is orthogonal,
running the normal fp32 forward with the round-tripped weight is numerically identical to the real
quantized-inference path -- so this isolates exactly the accuracy cost of N-bit weights, reusing the
unchanged forward (no buggy reimplementation). The memory/throughput win is separate (OpenVINO/Intel).

Quantizes the dense (2D) weight matrices -- the LOGIC layers (fuse, heads, entity-transformer
projections). Convs (4D, the feature extractor) and norms (1D) are left fp32.

    python policy/wht_quant.py     # self-test: WHT rotation cuts low-bit error vs naive rounding
"""
from __future__ import annotations

import numpy as np


def fwht(a):
    """Unnormalized fast Walsh-Hadamard transform along the last axis (length = power of 2)."""
    a = np.asarray(a, np.float32).copy()
    n = a.shape[-1]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[..., i:i + h].copy()
            y = a[..., i + h:i + 2 * h].copy()
            a[..., i:i + h] = x + y
            a[..., i + h:i + 2 * h] = x - y
        h *= 2
    return a


def _hadamard(n):
    """Normalized Hadamard matrix (n = power of 2): symmetric, orthogonal (H @ H = I)."""
    H = np.ones((1, 1), np.float32)
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return (H / np.sqrt(n)).astype(np.float32)


def _next_pow2(n):
    return 1 << (n - 1).bit_length()


def quantize_dequantize(w, bits, rotate=True):
    """Round-trip a weight matrix (in, out) through N-bit per-output-channel quantization.

    rotate=True applies the WHT incoherence rotation first (the paper's method); rotate=False is the
    naive baseline. Returns an fp32 matrix with exactly the quantized weight's information content."""
    w = np.asarray(w, np.float32)
    in_, out = w.shape
    qmax = 2 ** (bits - 1) - 1
    if rotate:
        n = _next_pow2(in_)
        H = _hadamard(n)
        wp = np.zeros((n, out), np.float32); wp[:in_] = w
        wr = H @ wp                                       # rotate into the incoherent basis
    else:
        wr = w
    scale = np.maximum(np.abs(wr).max(0), 1e-8) / qmax    # per-output-channel symmetric scale
    q = np.clip(np.round(wr / scale), -qmax - 1, qmax)    # the N-bit integers
    wr_hat = q * scale                                    # dequantize
    if rotate:
        return (H @ wr_hat)[:in_]                         # rotate back (H H = I) -> original basis
    return wr_hat


def quantize_tree(params, bits, rotate=True):
    """Round-trip every dense (2D) weight in a param tree at `bits`; leave convs (4D) and norms (1D)."""
    out = {}
    for k, (w, b) in params.items():
        w = np.asarray(w)
        out[k] = (quantize_dequantize(w, bits, rotate), np.asarray(b)) if w.ndim == 2 else (w, b)
    return out


def tree_bits_saved(params):
    """How many params live in the quantized (2D) layers vs total -- the share that shrinks."""
    q = sum(int(np.asarray(w).size) for (w, _) in params.values() if np.asarray(w).ndim == 2)
    tot = sum(int(np.asarray(w).size) + int(np.asarray(b).size) for (w, b) in params.values())
    return q, tot


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # a weight with a few large outliers -- the case naive low-bit quant handles badly
    w = rng.standard_normal((128, 64)).astype(np.float32)
    w[rng.integers(0, 128, 6), rng.integers(0, 64, 6)] *= 25.0
    x = rng.standard_normal((256, 128)).astype(np.float32)
    y = x @ w
    print("relative reconstruction error  ||x(W_hat-W)|| / ||xW||   (lower is better):")
    print(f"  {'bits':>4} | {'naive':>10} | {'WHT-rotated':>12}")
    for bits in (8, 4, 3, 2):
        e_naive = np.linalg.norm(x @ quantize_dequantize(w, bits, False) - y) / np.linalg.norm(y)
        e_wht = np.linalg.norm(x @ quantize_dequantize(w, bits, True) - y) / np.linalg.norm(y)
        print(f"  {bits:>4} | {e_naive:>10.4f} | {e_wht:>12.4f}")
    print("\nWHT rotation should win most at the lowest bit-widths (outliers spread -> less clipping).")
