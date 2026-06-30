"""Two games, one WITH the spectral-logic + WHT-quantized brain, one WITHOUT (the fp32 baseline).

Loads the BC'd warm-start (policy/bc_fused.pkl) and asks the project's real question: does the
brain's LOGIC survive when we shrink it for the locked-in Intel/CPU box? Three readouts:

  1. aux-head accuracy retained vs bit-width: fp32 -> naive low-bit -> WHT-rotated low-bit
  2. the prereq head: dense MLP (degrades) vs exact Walsh-spectral (stays exact)
  3. two full games: the fp32 brain vs the WHT-quantized brain -- same opening logic or not?

    PYTHONPATH=yr_env  python policy/compare_brains.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "yr_env"))
sys.path.insert(0, os.path.join(ROOT, "policy"))
import net                                                # noqa: E402
import wht_quant as wq                                    # noqa: E402
import spectral_logic as sl                               # noqa: E402
from rl_env import encode                                 # noqa: E402
from mock_env import ACTION_NAME                          # noqa: E402
from stock_teacher import stock_teacher                   # noqa: E402
import offline_bc as obc                                  # noqa: E402

PKL = os.path.join(ROOT, "policy", "bc_fused.pkl")


def collect_eval(n_games=20, seed=7):
    """Held-out states with the labels + building bits the spectral head needs."""
    rng = np.random.default_rng(seed)
    G, S, E, EM, A, BLD, CNT, PWR, BITS = [], [], [], [], [], [], [], [], []
    for gi in range(n_games):
        sim = obc.Sim(rng, rich=(gi % 3 == 0))
        for _ in range(60):
            pos = sim.position()
            a = stock_teacher(pos)
            bld, cnt, thr, ev, pwr = obc.aux_targets(pos)
            et, em = sim.entities()
            G.append(sim.grid()); S.append(encode(pos)); E.append(et); EM.append(em)
            A.append(a); BLD.append(bld); CNT.append(cnt); PWR.append(pwr)
            BITS.append(sl.bits_from_buildings(pos.own_buildings))
            if sim.step(a):
                break
    return (np.asarray(G, np.float32), np.asarray(S, np.float32), np.asarray(E, np.float32),
            np.asarray(EM, np.float32), np.asarray(A), np.asarray(BLD, np.float32),
            np.asarray(CNT), np.asarray(PWR, np.float32), np.asarray(BITS, np.float32))


def head_acc(p, G, S, E, EM, A, BLD, CNT, PWR):
    h = net.heads(p, jnp.asarray(G), jnp.asarray(S), jnp.asarray(E), jnp.asarray(EM))
    return {"move": float((np.argmax(np.asarray(h["pi"]), 1) == A).mean()),
            "prereq": float(((np.asarray(h["bld"]) > 0) == (BLD > 0.5)).mean()),
            "counter": float((np.argmax(np.asarray(h["cnt"]), 1) == CNT).mean()),
            "power": float(((np.asarray(h["pwr"]) > 0) == (PWR > 0.5)).mean())}


def play(p, rng, coeffs=None):
    """Greedy game. If coeffs given, the prereq legality is read from the EXACT spectral head."""
    sim = obc.Sim(rng)
    seq, blackout_fix = [], False
    for _ in range(20):
        deficit = sim._power() < 0
        et, em = sim.entities()
        a, _ = net.decide(p, sim.grid(), encode(sim.position()), entities=et, ent_mask=em)
        if deficit and ACTION_NAME[a] == "POWER":
            blackout_fix = True
        seq.append(ACTION_NAME[a])
        if sim.step(a):
            break
    return seq, blackout_fix, sim.enemy_hp


if __name__ == "__main__":
    if not os.path.exists(PKL):
        print("no bc_fused.pkl -- run policy/offline_bc.py first"); sys.exit(1)
    with open(PKL, "rb") as f:
        p = pickle.load(f)
    qn, tot = wq.tree_bits_saved(p)
    print(f"loaded fp32 brain: {tot:,} params; {qn:,} live in dense/logic layers (quantizable)\n")

    G, S, E, EM, A, BLD, CNT, PWR, BITS = collect_eval()
    print(f"=== 1. aux-head accuracy vs bit-width  (held-out: {len(A)} states) ===")
    base = head_acc(p, G, S, E, EM, A, BLD, CNT, PWR)
    print(f"  fp32 baseline      move {base['move']:.3f}  prereq {base['prereq']:.3f}  "
          f"counter {base['counter']:.3f}  power {base['power']:.3f}")
    for bits in (8, 4, 3):
        for tag, rot in [("naive ", False), ("WHT   ", True)]:
            acc = head_acc(wq.quantize_tree(p, bits, rot), G, S, E, EM, A, BLD, CNT, PWR)
            print(f"  {bits}-bit {tag}      move {acc['move']:.3f}  prereq {acc['prereq']:.3f}  "
                  f"counter {acc['counter']:.3f}  power {acc['power']:.3f}")

    print("\n=== 2. prereq head: dense MLP vs exact Walsh-spectral (vs the true prereq logic) ===")
    coeffs = sl.build_spectrum()
    spec = float((sl.predict(coeffs, BITS) == BLD).mean())
    print(f"  spectral (Walsh)   fp32 {spec:.3f}  |  2-bit {float((sl.predict(sl.quantize_coeffs(coeffs,2), BITS)==BLD).mean()):.3f}   (exact, robust)")
    for bits in (4, 3, 2):
        a_naive = head_acc(wq.quantize_tree(p, bits, False), G, S, E, EM, A, BLD, CNT, PWR)["prereq"]
        a_wht = head_acc(wq.quantize_tree(p, bits, True), G, S, E, EM, A, BLD, CNT, PWR)["prereq"]
        print(f"  MLP bld head       {bits}-bit naive {a_naive:.3f}  WHT {a_wht:.3f}")

    print("\n=== 3. two games: fp32 brain (WITHOUT) vs WHT-quantized brain (WITH) ===")
    seqf, fixf, hpf = play(p, np.random.default_rng(123))
    seqq, fixq, hpq = play(wq.quantize_tree(p, 3, True), np.random.default_rng(123))
    print(f"  fp32      : {' '.join(seqf)}")
    print(f"             blackout-rebuild={fixf}  enemy_hp_left={hpf}")
    print(f"  WHT 3-bit : {' '.join(seqq)}")
    print(f"             blackout-rebuild={fixq}  enemy_hp_left={hpq}")
    same = "IDENTICAL opening logic" if seqf[:10] == seqq[:10] else "diverged in the first 10 moves"
    print(f"  -> {same}")
