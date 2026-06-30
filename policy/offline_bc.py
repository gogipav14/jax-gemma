"""Train the FUSED brain OFFLINE on the distilled stock-AI teacher — fast, no game.

The teacher doesn't only show the brain WHICH macro to press; the same offline states also teach
the brain the GAME'S STRUCTURE, labeled for free from game_model.py (no game needed):

    policy (pi) : copy the teacher's macro                       -- the move
    build (bld) : which roles are LEGAL to build now             -- the PREREQ tech-tree
    counter(cnt): the right answer to the dominant threat        -- unit strength / the COUNTER matrix
    threat (thr): is the base under attack right now (from eyes) -- defense / information
    eval   (ev) : position score V (material/tech sense)         -- evaluation

A lightweight sim plays the teacher (build -> mass -> counter -> attack) over many games, producing
(grid, scalar, action, aux-targets) tuples; we multi-task BC the fused net on thousands of them.
The aux heads force the shared trunk to ENCODE prereqs / counters / threat / value -- a far stronger
warm-start than copying moves alone -- which the slow live RL (and the GPU box) build on.

    python policy/offline_bc.py            # generate demos, multi-task BC, save policy/bc_fused.pkl
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
from rl_env import encode, entity_tokens                  # noqa: E402
from stock_teacher import stock_teacher                   # noqa: E402

assert len(__import__("rl_env").ENT_ROLES) + 4 == net.ENT_FEAT   # token layout must match the net

OUT = os.path.join(ROOT, "policy", "bc_fused.pkl")
ANCHOR = (44, 44)        # own base in 64-grid coords
ENEMY = (16, 16)         # enemy base across the map

# the threats the enemy can present, each with the role's correct answer (the COUNTER matrix the
# brain learns). Spanning the matrix so the counter head sees all three answers, not just one.
THREAT_MENU = [
    (gm.MAIN_BATTLE, gm.ANTI_ARMOR),    # tanks      -> Terror Drones
    (gm.ARTILLERY,   gm.ANTI_ARMOR),    # V3 / Prism -> Terror Drones (then sortie)
    (gm.SUPERUNIT,   gm.ANTI_AIR),      # Apoc/Kirov/air -> Flak (anti-air)
    (gm.ANTI_ARMOR,  gm.MAIN_BATTLE),   # enemy drones    -> main battle tanks
    (gm.ANTI_AIR,    gm.MAIN_BATTLE),   # enemy Flak Trak -> main battle tanks
]
BUILD_ROLES = [gm.POWER, gm.ECONOMY, gm.PROD_INF, gm.PROD_VEH, gm.TECH_RADAR, gm.DEF_GROUND, gm.DEF_AA]
COUNTER_IDX = {None: 0, gm.ANTI_ARMOR: 1, gm.MAIN_BATTLE: 2, gm.ANTI_AIR: 3}
COUNTER_NAME = ["none", "ANTI_ARMOR", "MAIN_BATTLE", "ANTI_AIR"]


def aux_targets(pos):
    """Label a Position with the game structure -- all derived from game_model, for free."""
    bld = np.zeros(len(BUILD_ROLES), np.float32)                   # PREREQ: legal-to-build right now
    for i, r in enumerate(BUILD_ROLES):
        if all(pos.own_buildings.get(pp, 0) > 0 for pp in gm.PREREQ.get(r, [])):
            bld[i] = 1.0
    if pos.threats:                                               # COUNTER: answer the worst threat
        dom = max(pos.threats, key=lambda t: t.severity)
        cnt = COUNTER_IDX.get(dom.counter, 2)
    else:
        cnt = 0
    thr = 1.0 if any(t.at_base for t in pos.threats) else 0.0     # INFO/DEFENSE: base under attack?
    ev = float(gm.evaluate(pos))                                  # EVAL: position score V
    return bld, cnt, np.float32(thr), np.float32(ev)


class Sim:
    """Minimal offline YR for BC data: the teacher's progression + a synthetic spatial grid."""

    def __init__(self, rng, rich=False):
        self.rng = rng
        self.rich = rich                                                 # start flush with cash?
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
        self.credits = 100000 if self.rich else 4000     # rich (skip 2nd refinery, rush) vs scarce (economy-first)
        self.tick = 0
        self.deployed = False
        self.enemy_hp = 8
        self.threats = []                      # list of (role, count, at_base, counter)
        self.attacking = 0                     # ticks our army has been pushing the enemy

    def position(self):
        p = gm.Position(prefix="NA", anchor=ANCHOR if self.deployed else None)   # no ConYard until deploy
        p.credits = self.credits                                                 # so the scalar carries cash
        n_other = sum(self.b.values()) - self.b[gm.POWER]
        p.power_surplus = self.b[gm.POWER] * 150 - n_other * 30                   # power plants feed, the rest drain
        p.own_buildings = {k: v for k, v in self.b.items() if v > 0}
        if self.deployed:
            p.own_buildings[gm.CONSTRUCTION] = 1
        p.own_units = {k: v for k, v in self.u.items() if v > 0}
        army_n = self.u[gm.MAIN_BATTLE] + self.u[gm.ANTI_ARMOR]
        p.army = [{"unique_id": i, "x": ANCHOR[0], "y": ANCHOR[1]} for i in range(army_n)]
        p.tech_tier = 2 if self.b[gm.TECH_RADAR] and self.b.get(gm.PROD_VEH) else (1 if self.b[gm.TECH_RADAR] else 0)
        p.threats = [gm.Threat(r, c, round(c * gm.ROLE_VALUE.get(r, 1.0) * (1.5 if ab else 1.0), 1), cn, ab)
                     for (r, c, ab, cn) in self.threats]
        p.enemy_belief = {r: {"count": c, "age": 0, "positions": [ENEMY]} for (r, c, ab, cn) in self.threats}
        p.enemy_seen = {r: c for (r, c, ab, cn) in self.threats}
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
        for (r, c, ab, cn) in self.threats:                 # visible enemy units
            ey, ex = (ay, ax) if ab else (ENEMY[1], ENEMY[0])
            g[4, max(0, ey - 1):ey + 2, max(0, ex - 1):ex + 2] = 1.0
        if self.attacking >= 6 and self.enemy_hp > 0:       # scouted/at the enemy base
            g[4, ENEMY[1] - 2:ENEMY[1] + 2, ENEMY[0] - 2:ENEMY[0] + 2] = 1.0
        return g

    def entities(self):
        """Synthesize the roster tokens consistent with the Sim state (own base + army + threats)."""
        ax, ay = ANCHOR[0] / 64.0, ANCHOR[1] / 64.0
        items = []
        if self.deployed:
            items.append((gm.CONSTRUCTION, False, ax, ay, 1.0))
        for role, n in self.b.items():
            for _ in range(n):
                items.append((role, False, ax, ay, 1.0))               # own buildings at base
        t = min(1.0, self.attacking / 8.0)                             # army strung toward the enemy
        ux = (ANCHOR[0] + (ENEMY[0] - ANCHOR[0]) * t) / 64.0
        uy = (ANCHOR[1] + (ENEMY[1] - ANCHOR[1]) * t) / 64.0
        for role in (gm.MAIN_BATTLE, gm.ANTI_ARMOR, gm.ANTI_AIR):
            for _ in range(self.u[role]):
                items.append((role, False, ux, uy, float(t < 0.3)))    # at base until it pushes out
        for (role, c, ab, cn) in self.threats:                          # visible enemy units
            ex, ey = (ax, ay) if ab else (ENEMY[0] / 64.0, ENEMY[1] / 64.0)
            for _ in range(min(c, 4)):
                items.append((role, True, ex, ey, float(ab)))
        return entity_tokens(items)

    def step(self, a):
        self.tick += 1
        self.credits += self.b[gm.ECONOMY] * 500 + 300
        if self.tick > 12 and self.rng.random() < 0.25:     # enemy applies pressure
            role, counter = THREAT_MENU[self.rng.integers(len(THREAT_MENU))]
            self.threats = [(role, int(self.rng.integers(1, 4)), bool(self.rng.random() < 0.6), counter)]
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
    G, S, A, BLD, CNT, THR, EV, E, EM = [], [], [], [], [], [], [], [], []
    for gi in range(n_games):
        sim = Sim(rng, rich=(gi % 3 == 0))     # ~1/3 of games start flush with cash -> different priority
        for _ in range(60):
            pos = sim.position()
            a = stock_teacher(pos)
            bld, cnt, thr, ev = aux_targets(pos)
            et, em = sim.entities()
            G.append(sim.grid()); S.append(encode(pos)); A.append(a)
            BLD.append(bld); CNT.append(cnt); THR.append(thr); EV.append(ev)
            E.append(et); EM.append(em)
            if sim.step(a):
                break
    return (np.asarray(G, np.float32), np.asarray(S, np.float32), np.asarray(A, np.int32),
            np.asarray(BLD, np.float32), np.asarray(CNT, np.int32),
            np.asarray(THR, np.float32), np.asarray(EV, np.float32),
            np.asarray(E, np.float32), np.asarray(EM, np.float32))


def bc(G, S, A, BLD, CNT, THR, EV, E, EM, steps=600, lr=2e-3, batch=512):
    p = net.init_params(random.PRNGKey(0))
    opt = optax.adam(lr); st = opt.init(p)

    def loss(pp, g, s, e, em, a, bld, cnt, thr, ev):
        h = net.heads(pp, g, s, e, em)
        n = a.shape[0]
        pi = -jnp.mean(jax.nn.log_softmax(h["pi"])[jnp.arange(n), a])              # the move
        bl = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(h["bld"], bld), -1))  # prereqs
        cl = -jnp.mean(jax.nn.log_softmax(h["cnt"])[jnp.arange(n), cnt])           # counter matrix
        tl = jnp.mean(optax.sigmoid_binary_cross_entropy(h["thr"], thr))           # threat at base
        el = jnp.mean((h["ev"] - ev) ** 2)                                         # position eval
        total = pi + 0.5 * bl + 0.5 * cl + 0.3 * tl + 0.1 * el
        return total, (pi, bl, cl, tl, el)
    gfn = jax.jit(jax.value_and_grad(loss, has_aux=True))
    n = len(A)
    rng = np.random.default_rng(1)
    for i in range(steps):
        idx = rng.integers(0, n, batch)
        bg, bs, be, bm = jnp.asarray(G[idx]), jnp.asarray(S[idx]), jnp.asarray(E[idx]), jnp.asarray(EM[idx])
        (l, parts), gr = gfn(p, bg, bs, be, bm, jnp.asarray(A[idx]),
                             jnp.asarray(BLD[idx]), jnp.asarray(CNT[idx]),
                             jnp.asarray(THR[idx]), jnp.asarray(EV[idx]))
        u, st = opt.update(gr, st, p); p = optax.apply_updates(p, u)
        if i % 100 == 0 or i == steps - 1:
            h = net.heads(p, bg, bs, be, bm)
            pacc = float((jnp.argmax(h["pi"], -1) == jnp.asarray(A[idx])).mean())
            cacc = float((jnp.argmax(h["cnt"], -1) == jnp.asarray(CNT[idx])).mean())
            bacc = float(((h["bld"] > 0) == (jnp.asarray(BLD[idx]) > 0.5)).mean())
            pi, bl, cl, tl, el = (float(x) for x in parts)
            print(f"  step {i:4d} | pi {pi:.3f}(acc {pacc:.2f})  bld {bl:.3f}(acc {bacc:.2f})  "
                  f"cnt {cl:.3f}(acc {cacc:.2f})  thr {tl:.3f}  ev {el:.2f}")
    return p


def _probe(p):
    """Show the brain DEMONSTRATING the game structure it learned (not just copying moves)."""
    print("\n=== what the brain LEARNED (probing the aux heads) ===")
    # prereqs: with only a ConYard, the only legal build is POWER
    just_cy = gm.Position(prefix="NA", anchor=ANCHOR, own_buildings={gm.CONSTRUCTION: 1})
    et, em = _pos_entities(just_cy)
    h = net.heads(p, jnp.asarray(_sim_grid_for(just_cy)[None]), jnp.asarray(encode(just_cy)[None]),
                  jnp.asarray(et[None]), jnp.asarray(em[None]))
    legal = [BUILD_ROLES[i] for i in range(len(BUILD_ROLES)) if float(h["bld"][0, i]) > 0]
    print(f"  prereqs | only a ConYard -> brain says buildable: {legal}   (truth: ['POWER'])")
    # counter matrix: read the answer for each threat kind
    for role, truth in [(gm.ARTILLERY, "ANTI_ARMOR"), (gm.SUPERUNIT, "ANTI_AIR"), (gm.ANTI_AIR, "MAIN_BATTLE")]:
        cn = dict(THREAT_MENU)[role]
        pos = gm.Position(prefix="NA", anchor=ANCHOR,
                          own_buildings={gm.CONSTRUCTION: 1, gm.POWER: 1, gm.ECONOMY: 2, gm.PROD_INF: 1, gm.PROD_VEH: 1},
                          threats=[gm.Threat(role, 2, 4.0, cn, True)],
                          enemy_belief={role: {"count": 2, "age": 0, "positions": [ENEMY]}},  # role reaches the brain here
                          enemy_seen={role: 2})
        et, em = _pos_entities(pos)
        h = net.heads(p, jnp.asarray(_sim_grid_for(pos)[None]), jnp.asarray(encode(pos)[None]),
                      jnp.asarray(et[None]), jnp.asarray(em[None]))
        ans = COUNTER_NAME[int(jnp.argmax(h["cnt"][0]))]
        atk = "yes" if float(h["thr"][0]) > 0 else "no"
        print(f"  counter | enemy {role:11s} -> brain answers {ans:11s} (truth {truth:11s}) | under attack? {atk}")


def _pos_entities(pos):
    """Roster tokens for a probe Position (own buildings at base + threats at/away from base)."""
    ax, ay = ANCHOR[0] / 64.0, ANCHOR[1] / 64.0
    items = [(role, False, ax, ay, 1.0) for role, n in pos.own_buildings.items() for _ in range(n)]
    for t in pos.threats:
        ex, ey = (ax, ay) if t.at_base else (ENEMY[0] / 64.0, ENEMY[1] / 64.0)
        items += [(t.role, True, ex, ey, float(t.at_base)) for _ in range(min(t.count, 4))]
    return entity_tokens(items)


def _sim_grid_for(pos):
    """A minimal grid consistent with a Position (own base cluster + any at-base threat)."""
    g = np.zeros((7, 64, 64), np.float32)
    ay, ax = ANCHOR[1], ANCHOR[0]
    g[2, ay - 8:ay + 8, ax - 8:ax + 8] = 1.0
    g[5, ay - 2:ay + 2, ax - 2:ax + 2] = 1.0
    for t in pos.threats:
        if t.at_base:
            g[4, ay - 1:ay + 2, ax - 1:ax + 2] = 1.0
    return g


if __name__ == "__main__":
    from mock_env import ACTION_NAME
    print("=== collecting offline demos from the stock-AI teacher (+ game-structure labels) ===")
    G, S, A, BLD, CNT, THR, EV, E, EM = collect()
    dist = {ACTION_NAME[i]: int((A == i).sum()) for i in sorted(set(A.tolist()))}
    cdist = {COUNTER_NAME[i]: int((CNT == i).sum()) for i in sorted(set(CNT.tolist()))}
    print(f"  {len(A)} demos | actions: {dist}")
    print(f"            | counters: {cdist}  | base-under-attack: {int(THR.sum())}/{len(THR)}")
    print(f"            | roster: avg {float(EM.sum(1).mean()):.1f} tokens/state (max {int(EM.sum(1).max())})")
    print("=== multi-task BC the fused brain (eyes + ROSTER + state: move + prereqs + counters + threat + eval) ===")
    p = bc(G, S, A, BLD, CNT, THR, EV, E, EM)
    with open(OUT, "wb") as f:
        pickle.dump(jax.tree.map(lambda x: np.asarray(x), p), f)
    print(f"saved warm-start -> {OUT}")
    # sanity: the BC'd brain's opening differs by economy -- scarce builds 2 refineries, rich skips
    # the 2nd and reaches production/army sooner (credit-conditional priority, learned from the obs).
    for label, rich in [("scarce (4k)", False), ("rich (100k)", True)]:
        sim = Sim(np.random.default_rng(99), rich=rich)
        seq = []
        for _ in range(16):
            et, em = sim.entities()
            a, _ = net.decide(p, sim.grid(), encode(sim.position()), entities=et, ent_mask=em)
            seq.append(ACTION_NAME[a])
            if sim.step(a):
                break
        print(f"BC'd brain opening [{label:11s}]:", " ".join(seq))
    _probe(p)
