"""v2 offline BC: train the spectral-wired brain (net_spectral) on the stock-AI teacher.

The laws (prereq, power) are EXACT by construction -- no loss needed, they're correct for free. BC
trains only the JUDGMENT (policy + value's use of the world-model) and shapes the trainable counter
prior. So the brain starts with the rulebook frozen and exact, and learns the teacher's moves while
CONSULTING that exact world-model -- the v2 of the spectral fork.

    PYTHONPATH=yr_env  python policy/offline_bc_v2.py
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "yr_env"))
sys.path.insert(0, os.path.join(ROOT, "policy"))
import net_spectral as ns                                 # noqa: E402
import spectral_logic as sl                               # noqa: E402
import offline_bc as obc                                  # noqa: E402  (reuse the Sim + aux_targets)
from rl_env import encode                                 # noqa: E402
from mock_env import ACTION_NAME                          # noqa: E402
from stock_teacher import stock_teacher                   # noqa: E402

OUT = os.path.join(ROOT, "policy", "bc_fused_v2.pkl")


def collect(n_games=120, seed=0):
    rng = np.random.default_rng(seed)
    G, S, E, EM, A, BB, EB, PW, CNT, THR, EV = [], [], [], [], [], [], [], [], [], [], []
    for gi in range(n_games):
        sim = obc.Sim(rng, rich=(gi % 3 == 0))
        for _ in range(60):
            pos = sim.position()
            a = stock_teacher(pos)
            _, cnt, thr, ev, _ = obc.aux_targets(pos)
            et, em = sim.entities()
            G.append(sim.grid()); S.append(encode(pos)); E.append(et); EM.append(em); A.append(a)
            BB.append(sl.bits_from_buildings(pos.own_buildings))
            EB.append(sl.bits_from_threats(pos))
            PW.append([1.0 if pos.power_surplus >= 0 else 0.0])
            CNT.append(cnt); THR.append(thr); EV.append(ev)
            if sim.step(a):
                break
    arr = lambda x, t: np.asarray(x, t)
    return (arr(G, np.float32), arr(S, np.float32), arr(E, np.float32), arr(EM, np.float32),
            arr(A, np.int32), arr(BB, np.float32), arr(EB, np.float32), arr(PW, np.float32),
            arr(CNT, np.int32), arr(THR, np.float32), arr(EV, np.float32))


def bc(data, steps=600, lr=2e-3, batch=512):
    G, S, E, EM, A, BB, EB, PW, CNT, THR, EV = data
    p = ns.init_params(random.PRNGKey(0))
    opt = optax.adam(lr); st = opt.init(p)

    def loss(pp, g, s, e, em, bb, eb, pw, a, cnt, thr, ev):
        h = ns.heads(pp, g, s, e, em, bb, eb, pw)
        n = a.shape[0]
        pi = -jnp.mean(jax.nn.log_softmax(h["pi"])[jnp.arange(n), a])              # judgment: the move
        cl = -jnp.mean(jax.nn.log_softmax(h["cnt"])[jnp.arange(n), cnt])           # shape the counter prior
        tl = jnp.mean(optax.sigmoid_binary_cross_entropy(h["thr"], thr))           # threat aux
        el = jnp.mean((h["ev"] - ev) ** 2)                                         # eval aux
        return pi + 0.5 * cl + 0.3 * tl + 0.1 * el, (pi, cl)
    gfn = jax.jit(jax.value_and_grad(loss, has_aux=True))
    rng = np.random.default_rng(1)
    n = len(A)
    for i in range(steps):
        idx = rng.integers(0, n, batch)
        bg = [jnp.asarray(x[idx]) for x in (G, S, E, EM, BB, EB, PW)]
        (l, parts), gr = gfn(p, *bg, jnp.asarray(A[idx]), jnp.asarray(CNT[idx]),
                             jnp.asarray(THR[idx]), jnp.asarray(EV[idx]))
        u, st = opt.update(gr, st, p); p = optax.apply_updates(p, u)
        if i % 100 == 0 or i == steps - 1:
            h = ns.heads(p, *bg)
            pacc = float((jnp.argmax(h["pi"], -1) == jnp.asarray(A[idx])).mean())
            cacc = float((jnp.argmax(h["cnt"], -1) == jnp.asarray(CNT[idx])).mean())
            print(f"  step {i:4d} | move {float(parts[0]):.3f}(acc {pacc:.2f})  counter {float(parts[1]):.3f}(acc {cacc:.2f})")
    return p


if __name__ == "__main__":
    print("=== v2 offline BC: spectral-wired brain (laws frozen-exact; learn judgment + counter) ===")
    data = collect()
    A = data[4]
    dist = {ACTION_NAME[i]: int((A == i).sum()) for i in sorted(set(A.tolist()))}
    print(f"  {len(A)} demos | actions: {dist}")
    p = bc(data)
    with open(OUT, "wb") as f:
        pickle.dump(jax.tree.map(lambda x: np.asarray(x), p), f)
    print(f"saved -> {OUT}")

    # sanity: the prereq LAW is exact (by construction); the brain plays a coherent opening
    G, S, E, EM, A, BB, EB, PW, CNT, THR, EV = data
    h = ns.heads(p, jnp.asarray(G[:512]), jnp.asarray(S[:512]), jnp.asarray(E[:512]), jnp.asarray(EM[:512]),
                 jnp.asarray(BB[:512]), jnp.asarray(EB[:512]), jnp.asarray(PW[:512]))
    true_bld = np.stack([sl.predict(sl.build_spectrum(), BB[:512])])[0]
    print(f"  prereq law exactness: {float((np.asarray(h['bld']) == true_bld).mean()):.3f}   "
          f"counter acc: {float((np.argmax(np.asarray(h['cnt']),1) == CNT[:512]).mean()):.3f}")
    for label, rich in [("scarce", False), ("rich", True)]:
        sim = obc.Sim(np.random.default_rng(99), rich=rich)
        seq = []
        for _ in range(16):
            pos = sim.position()
            et, em = sim.entities()
            bb = sl.bits_from_buildings(pos.own_buildings); eb = sl.bits_from_threats(pos)
            pw = 1.0 if pos.power_surplus >= 0 else 0.0
            a, _ = ns.decide(p, sim.grid(), encode(pos), et, em, bb, eb, pw)
            seq.append(ACTION_NAME[a])
            if sim.step(a):
                break
        print(f"  v2 opening [{label:6s}]: {' '.join(seq)}")
