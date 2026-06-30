"""Train the FUSED brain's BC warm-start OFFLINE on the distilled stock-AI teacher — fast, no game.

A lightweight sim plays the stock-AI teacher (build -> mass -> counter -> attack) over many games,
producing realistic (grid, scalar, action) triples: the grid is a synthetic 64x64x7 vision (own
base cluster, enemy across the map, ore, fog) so the CNN learns to read spatial layout; the scalar
is encode(Position); the action is the teacher's macro. We BC the fused net on thousands of these,
giving a SOLID warm-start the slow live RL (and the GPU box) build on — instead of one live game.

    python policy/offline_bc.py            # generate demos, BC the fused net, save policy/bc_fused.pkl
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
import game_model as gm                                   # noqa: E402
import net                                                # noqa: E402
from rl_env import encode                                 # noqa: E402
from stock_teacher import stock_teacher                   # noqa: E402

OUT = os.path.join(ROOT, "policy", "bc_fused.pkl")
ANCHOR = (44, 44)        # own base in 64-grid coords
ENEMY = (16, 16)         # enemy base across the map


class Sim:
    """Minimal offline YR for BC data: the teacher's progression + a synthetic spatial grid."""

    def __init__(self, rng):
        self.rng = rng
        self.terrain = (rng.random((64, 64)) * 255).astype(np.uint8)     # fixed per game
        self.height = (rng.random((64, 64)) * 60).astype(np.uint8)
        self.ore = np.zeros((64, 64), np.uint8)
        for _ in range(6):
            oy, ox = rng.integers(8, 56), rng.integers(8, 56)
            self.ore[oy:oy + 3, ox:ox + 3] = 200
        self.reset()

    def reset(self):
        self.b = {gm.POWER: 0, gm.ECONOMY: 0, gm.PROD_INF: 0, gm.PROD_VEH: 0,
                  gm.TECH_RADAR: 0, gm.DEF_GROUND: 0, gm.DEF_AA: 0}
        self.u = {gm.MAIN_BATTLE: 0, gm.ANTI_ARMOR: 0, gm.ANTI_AIR: 0}
        self.credits = 5000
        self.tick = 0
        self.deployed = False
        self.enemy_hp = 8
        self.threats = []                      # list of (role, count, at_base)
        self.attacking = 0                     # ticks our army has been pushing the enemy

    def position(self):
        p = gm.Position(prefix="NA", anchor=ANCHOR if self.deployed else None)   # no ConYard until deploy
        p.own_buildings = {k: v for k, v in self.b.items() if v > 0}
        if self.deployed:
            p.own_buildings[gm.CONSTRUCTION] = 1
        p.own_units = {k: v for k, v in self.u.items() if v > 0}
        army_n = self.u[gm.MAIN_BATTLE] + self.u[gm.ANTI_ARMOR]
        p.army = [{"unique_id": i, "x": ANCHOR[0], "y": ANCHOR[1]} for i in range(army_n)]
        p.tech_tier = 2 if self.b[gm.TECH_RADAR] and self.b.get(gm.PROD_VEH) else (1 if self.b[gm.TECH_RADAR] else 0)
        p.threats = [gm.Threat(r, c, c * 1.0, gm.counter_for(r), ab) for (r, c, ab) in self.threats]
        p.enemy_belief = {r: {"count": c, "age": 0, "positions": [ENEMY]} for (r, c, ab) in self.threats}
        p.enemy_seen = {r: c for (r, c, ab) in self.threats}
        return p

    def grid(self):
        g = np.zeros((7, 64, 64), np.float32)
        g[0] = self.terrain / 255.0
        g[6] = self.height / 255.0
        g[1] = self.ore / 255.0
        ay, ax = ANCHOR[1], ANCHOR[0]
        g[2, ay - 8:ay + 8, ax - 8:ax + 8] = 1.0           # visible around own base
        if self.deployed:
            nb = sum(self.b.values())
            g[5, ay - 2:ay + 2, ax - 2:ax + 2] = 1.0       # own buildings cluster
            g[5, ay, ax] = 1.0
            r = 1 + min(3, nb // 2)
            g[5, ay - r:ay + r, ax - r:ax + r] = 1.0
        army_n = self.u[gm.MAIN_BATTLE] + self.u[gm.ANTI_ARMOR]
        if army_n:
            # army sits at base, or strung toward the enemy when attacking
            t = min(1.0, self.attacking / 8.0)
            uy = int(ay + (ENEMY[1] - ay) * t); ux = int(ax + (ENEMY[0] - ax) * t)
            g[3, max(0, uy - 2):uy + 2, max(0, ux - 2):ux + 2] = 1.0
            g[2, max(0, uy - 4):uy + 4, max(0, ux - 4):ux + 4] = np.maximum(g[2, max(0, uy - 4):uy + 4, max(0, ux - 4):ux + 4], 1.0)
        for (r, c, ab) in self.threats:                     # visible enemy units
            ey, ex = (ay, ax) if ab else (ENEMY[1], ENEMY[0])
            g[4, max(0, ey - 1):ey + 2, max(0, ex - 1):ex + 2] = 1.0
        if self.attacking >= 6 and self.enemy_hp > 0:       # scouted/at the enemy base
            g[4, ENEMY[1] - 2:ENEMY[1] + 2, ENEMY[0] - 2:ENEMY[0] + 2] = 1.0
        return g

    def step(self, a):
        self.tick += 1
        self.credits += self.b[gm.ECONOMY] * 500 + 300
        if self.tick > 12 and self.rng.random() < 0.25:     # enemy applies pressure
            roles = [gm.MAIN_BATTLE, gm.ARTILLERY, gm.SUPERUNIT]
            self.threats = [(roles[self.rng.integers(3)], int(self.rng.integers(1, 4)), bool(self.rng.random() < 0.6))]
        elif self.rng.random() < 0.3:
            self.threats = []
        role_b = {2: gm.POWER, 3: gm.ECONOMY, 4: gm.PROD_INF, 5: gm.PROD_VEH, 6: gm.TECH_RADAR, 7: gm.DEF_GROUND, 8: gm.DEF_AA}
        if a == 1:
            self.deployed = True
        elif a in role_b and self.credits >= 500:
            self.b[role_b[a]] += 1; self.credits -= 800
        elif a == 9 and self.b[gm.PROD_VEH] and self.credits >= 900:
            self.u[gm.MAIN_BATTLE] += 1; self.credits -= 900
        elif a == 10 and self.b[gm.PROD_VEH] and self.credits >= 900:
            self.u[gm.ANTI_ARMOR] += 1; self.credits -= 900
        elif a == 11 and self.b[gm.PROD_VEH] and self.credits >= 900:
            self.u[gm.ANTI_AIR] += 1; self.credits -= 900
        elif a == 13:                                       # attack: push the enemy, chip its base
            self.attacking += 1
            if self.attacking >= 5 and (self.u[gm.MAIN_BATTLE] + self.u[gm.ANTI_ARMOR]) >= 4:
                self.enemy_hp -= 1
        else:
            self.attacking = max(0, self.attacking - 1)
        return self.enemy_hp <= 0 or self.tick >= 60


def collect(n_games=120, seed=0):
    rng = np.random.default_rng(seed)
    G, S, A = [], [], []
    for _ in range(n_games):
        sim = Sim(rng)
        for _ in range(60):
            pos = sim.position()
            a = stock_teacher(pos)
            G.append(sim.grid()); S.append(encode(pos)); A.append(a)
            if sim.step(a):
                break
    return np.asarray(G, np.float32), np.asarray(S, np.float32), np.asarray(A, np.int32)


def bc(G, S, A, steps=600, lr=2e-3, batch=512):
    p = net.init_params(random.PRNGKey(0))
    opt = optax.adam(lr); st = opt.init(p)

    def loss(pp, g, s, a):
        logits, _ = net.forward(pp, g, s)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(a.shape[0]), a])
    gfn = jax.jit(jax.value_and_grad(loss))
    n = len(A)
    rng = np.random.default_rng(1)
    for i in range(steps):
        idx = rng.integers(0, n, batch)
        l, gr = gfn(p, jnp.asarray(G[idx]), jnp.asarray(S[idx]), jnp.asarray(A[idx]))
        u, st = opt.update(gr, st, p); p = optax.apply_updates(p, u)
        if i % 100 == 0 or i == steps - 1:
            logits, _ = net.forward(p, jnp.asarray(G[idx]), jnp.asarray(S[idx]))
            acc = float((jnp.argmax(logits, -1) == jnp.asarray(A[idx])).mean())
            print(f"  step {i:4d}  loss={float(l):.3f}  acc={acc:.2f}")
    return p


if __name__ == "__main__":
    from mock_env import ACTION_NAME
    print("=== collecting offline demos from the stock-AI teacher ===")
    G, S, A = collect()
    dist = {ACTION_NAME[i]: int((A == i).sum()) for i in sorted(set(A.tolist()))}
    print(f"  {len(A)} demos from {len(A)//1}+ steps; action distribution: {dist}")
    print("=== BC the fused brain (eyes + state) offline ===")
    p = bc(G, S, A)
    with open(OUT, "wb") as f:
        pickle.dump(jax.tree.map(lambda x: np.asarray(x), p), f)
    print(f"saved warm-start -> {OUT}")
    # sanity: what does the BC'd brain do across a fresh teacher game?
    rng = np.random.default_rng(99); sim = Sim(rng)
    seq = []
    for _ in range(30):
        a, _ = net.decide(p, sim.grid(), encode(sim.position()))
        seq.append(ACTION_NAME[a])
        if sim.step(a):
            break
    print("BC'd brain greedy rollout:", " ".join(seq))
