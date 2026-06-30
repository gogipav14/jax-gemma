"""v2 brain: the spectral heads wired in as the ACTUAL heads (not an A/B). The law/heuristic/judgment
split, made architecture:

  LAWS (frozen, exact):  prereq  = a fixed Walsh-spectral AND over the building-presence bits
                         power   = the exact threshold (surplus >= 0), straight from the obs
  HEURISTIC (plastic):   counter = a linear in the Walsh (parity) basis, INITIALIZED to the teacher's
                         exact spectrum but TRAINABLE -- a prior RL can bend, not a frozen oracle
  JUDGMENT (plastic):    policy + value -- the NN, CONDITIONED on the exact world-model (it consults
                         the laws/heuristic as inputs) but never frozen; this is where insight lives

The laws are exact and quantization-proof; the policy spends its capacity on strategy, not on
re-deriving rules. Reuses the v1 trunk (CNN + entity transformer + scalar MLP) from net.py.

    python policy/net_spectral.py     # forward-pass self-test (shapes + prereq exactness)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

import net                                                # reuse the v1 trunk + helpers
import spectral_logic as sl

PREREQ_COEFFS = jnp.asarray(sl.build_spectrum())          # (n_build=7, 2^Kp) FROZEN exact spectrum
KP, KC = sl.K, sl.KC                                      # prereq bits (6), counter bits (5)
N_BUILD = len(sl.BUILD_ROLES)                             # 7
N_COUNTER = sl.N_CLASS                                    # 4
EXTRA = N_BUILD + 1 + N_COUNTER                           # prereq(7) + power(1) + counter(4) = 12


def _parity(bits, k):
    """bits:(B,k) in {0,1} -> (B, 2^k) Walsh characters chi_S(x) = (-1)^<S,x>."""
    n = 1 << k
    S = jnp.arange(n)[None, :, None]
    Sb = ((S >> jnp.arange(k)) & 1).astype(jnp.float32)   # (1, 2^k, k)
    inner = (bits[:, None, :] * Sb).sum(-1)               # (B, 2^k)
    return jnp.where(inner % 2 == 0, 1.0, -1.0)


def init_params(key, hidden=128):
    p = net.init_params(key, hidden=hidden)               # full v1 tree (trunk + old heads)
    k = random.split(key, 6)
    for dead in ("bld", "cnt", "pwr"):                    # replaced by spectral law/heuristic heads
        p.pop(dead)
    p["pi"] = net._lin(k[0], hidden + EXTRA, p["pi"][0].shape[1])   # policy CONSULTS the world-model
    p["v"] = net._lin(k[1], hidden + EXTRA, 1)
    p["thr"] = net._lin(k[2], hidden, 1)                  # threat-at-base aux stays off the trunk
    p["ev"] = net._lin(k[3], hidden, 1)
    p["cnt_coeffs"] = jnp.asarray(sl.build_counter_spectrum())     # (4, 2^Kc) TRAINABLE, init=exact
    return p


def prereq_exact(build_bits):
    """The LAW: exact, frozen Walsh-spectral prereq AND. (B, KP) -> (B, N_BUILD) in {0,1}."""
    return (_parity(build_bits, KP) @ PREREQ_COEFFS.T > 0.5).astype(jnp.float32)


def heads(p, grid, scalar, entities, ent_mask, build_bits, enemy_bits, power_bit):
    """Full read. build_bits:(B,KP) presence over PREREQ_BITS; enemy_bits:(B,KC) over COUNTER_ROLES;
    power_bit:(B,1) = (surplus>=0). Returns pi, v, and the aux outputs (bld/cnt/pwr/thr/ev)."""
    f = net._trunk(p, grid, scalar, entities, ent_mask)
    bld = prereq_exact(build_bits)                        # exact law
    pwr = power_bit                                       # exact law (passthrough from obs)
    cnt = _parity(enemy_bits, KC) @ p["cnt_coeffs"].T     # spectral-init trainable heuristic
    ctx = jnp.concatenate([f, bld, pwr, cnt], -1)         # the policy consults the world-model
    return {"pi": net._ap(p["pi"], ctx), "v": net._ap(p["v"], ctx)[..., 0],
            "bld": bld, "cnt": cnt, "pwr": pwr[..., 0],
            "thr": net._ap(p["thr"], f)[..., 0], "ev": net._ap(p["ev"], f)[..., 0]}


def forward(p, grid, scalar, entities, ent_mask, build_bits, enemy_bits, power_bit):
    h = heads(p, grid, scalar, entities, ent_mask, build_bits, enemy_bits, power_bit)
    return h["pi"], h["v"]


def decide(p, grid, scalar, entities, ent_mask, build_bits, enemy_bits, power_bit, key=None, greedy=True):
    g = grid[None] if grid.ndim == 3 else grid
    s = scalar[None] if scalar.ndim == 1 else scalar
    e = entities[None] if entities.ndim == 2 else entities
    em = ent_mask[None] if ent_mask.ndim == 1 else ent_mask
    bb = build_bits[None] if build_bits.ndim == 1 else build_bits
    eb = enemy_bits[None] if enemy_bits.ndim == 1 else enemy_bits
    pw = np.reshape(power_bit, (1, 1))
    logits, value = forward(p, jnp.asarray(g), jnp.asarray(s), jnp.asarray(e), jnp.asarray(em),
                            jnp.asarray(bb), jnp.asarray(eb), jnp.asarray(pw))
    a = int(jnp.argmax(logits[0])) if greedy else int(random.categorical(key, logits[0]))
    return a, float(value[0])


if __name__ == "__main__":
    p = init_params(random.PRNGKey(0))
    n = sum(int(np.prod(np.shape(v))) for leaf in jax.tree.leaves(p) for v in [leaf])
    print(f"params: {n:,}   EXTRA world-model inputs to policy: {EXTRA}")
    B = 4
    g = jnp.asarray(np.random.rand(B, 7, 64, 64), jnp.float32)
    s = jnp.asarray(np.random.rand(B, 21), jnp.float32)
    e = jnp.asarray(np.random.rand(B, net.MAX_ENT, net.ENT_FEAT), jnp.float32)
    m = jnp.asarray((np.random.rand(B, net.MAX_ENT) > 0.5).astype(np.float32))
    bb = jnp.asarray((np.random.rand(B, KP) > 0.5).astype(np.float32))
    eb = jnp.asarray((np.random.rand(B, KC) > 0.5).astype(np.float32))
    pw = jnp.asarray(np.random.rand(B, 1).astype(np.float32))
    h = heads(p, g, s, e, m, bb, eb, pw)
    print("heads:", {k: tuple(v.shape) for k, v in h.items()})
    # the prereq LAW is exact vs the numpy reference on all building states
    allx = np.stack([[(x >> b) & 1 for b in range(KP)] for x in range(1 << KP)]).astype(np.float32)
    jax_pr = np.asarray(prereq_exact(jnp.asarray(allx)))
    np_pr = sl.predict(sl.build_spectrum(), allx)
    print(f"prereq law exact vs reference over all 2^{KP} states: {bool((jax_pr == np_pr).all())}")
    print("v2 spectral-wired brain forward OK on CPU")
