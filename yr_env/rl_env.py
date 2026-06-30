"""RL environment — turns YR into a *learnable* problem so the agent DISCOVERS strategy from the
GOAL (win), instead of following prescribed rules.

  observation = encode(Position)        the board (the agent's eyes)
  action      = a macro index           the legal-move vocabulary (NOT a strategy)
  reward      = ΔV (dense) + terminal    +1 win / -1 loss; ΔV guides learning before it can win
  rules       = enforced by the env      illegal/ineffective macros simply don't advance V

There is NO prescriptive policy here. The agent LEARNS prereqs (an illegal build earns nothing),
unit efficiency (wasted units lower V / lose), and strategy (what raises V and wins) — via PPO
self-play (policy/, league/). This module is the substrate the learner trains on.
"""
from __future__ import annotations

import time

import numpy as np

import game_model as gm
import build_base as bb
import script_policy as sp
import baseline as bl

# --- the action vocabulary: legal moves, not strategy. The agent learns WHICH and WHEN. ---
MACROS = [
    ("noop", None), ("deploy", None),
    ("build", gm.POWER), ("build", gm.ECONOMY), ("build", gm.PROD_INF), ("build", gm.PROD_VEH),
    ("build", gm.TECH_RADAR), ("build", gm.DEF_GROUND), ("build", gm.DEF_AA),
    ("train", gm.MAIN_BATTLE), ("train", gm.ANTI_ARMOR), ("train", gm.ANTI_AIR),
    ("scout", None), ("attack", None), ("defend", None),
]
N_MACRO = len(MACROS)

# --- observation: the board folded to a fixed vector (roles, not unit IDs) ---
OWN_ROLES = [gm.POWER, gm.ECONOMY, gm.PROD_INF, gm.PROD_VEH, gm.TECH_RADAR, gm.DEF_GROUND, gm.DEF_AA,
             gm.MAIN_BATTLE, gm.ANTI_ARMOR, gm.ANTI_AIR]
ENEMY_ROLES = [gm.MAIN_BATTLE, gm.ANTI_ARMOR, gm.ANTI_AIR, gm.ARTILLERY, gm.SUPERUNIT, gm.INFANTRY]
OBS_DIM = len(OWN_ROLES) + len(ENEMY_ROLES) + 5     # + credits, power, tech, threat-severity, threats@base


def encode(pos: gm.Position) -> np.ndarray:
    own = [pos.own_buildings.get(r, 0) + pos.own_units.get(r, 0) for r in OWN_ROLES]
    enemy = [pos.enemy_belief.get(r, {}).get("count", 0) for r in ENEMY_ROLES]
    scal = [pos.credits / 20000.0, pos.power_surplus / 100.0, pos.tech_tier,
            sum(t.severity for t in pos.threats), float(sum(1 for t in pos.threats if t.at_base))]
    return np.asarray(own + enemy + scal, np.float32)


# --- the ROSTER: each visible techno as a token (per-unit WHAT) for the entity transformer ---
# token = one-hot(ENT_ROLES) + [is_enemy, x_norm, y_norm, at_base]; matches net.ENT_FEAT / net.MAX_ENT
ENT_ROLES = [gm.POWER, gm.ECONOMY, gm.PROD_INF, gm.PROD_VEH, gm.TECH_RADAR, gm.DEF_GROUND, gm.DEF_AA,
             gm.CONSTRUCTION, gm.MAIN_BATTLE, gm.ANTI_ARMOR, gm.ANTI_AIR, gm.ARTILLERY, gm.SUPERUNIT]
ENT_ROLE_IX = {r: i for i, r in enumerate(ENT_ROLES)}
ENT_FEAT = len(ENT_ROLES) + 4        # = 17 (must equal net.ENT_FEAT)
MAX_ENT = 48                         # token cap (= net.MAX_ENT)


def entity_tokens(items, max_ent=MAX_ENT):
    """items: list of (role, is_enemy, x_norm, y_norm, at_base) -> (max_ent, ENT_FEAT) + (max_ent,) mask."""
    toks = np.zeros((max_ent, ENT_FEAT), np.float32)
    mask = np.zeros((max_ent,), np.float32)
    for i, (role, is_enemy, xn, yn, atb) in enumerate(items[:max_ent]):
        ri = ENT_ROLE_IX.get(role)
        if ri is None:
            continue
        toks[i, ri] = 1.0
        toks[i, 13] = float(is_enemy)
        toks[i, 14] = float(xn); toks[i, 15] = float(yn); toks[i, 16] = float(atb)
        mask[i] = 1.0
    return toks, mask


def _execute_macro(idx, pos, obs, act, cat, ctx):
    """Carry out one macro via the non-cheating path. Illegal moves just fail (the env's 'rules')."""
    kind, role = MACROS[idx]
    if kind == "noop":
        return "noop"
    if kind == "deploy":
        return bl.execute(("deploy", None, ""), pos, obs, act, cat, ctx)
    if kind == "build":
        fid = bl.struct_id(role, pos.prefix, cat)
        return bl.execute(("build", fid, ""), pos, obs, act, cat, ctx) if fid else "no-id"
    if kind == "train":
        return bl.execute(("train", bl.unit_id(role, pos.prefix), ""), pos, obs, act, cat, ctx)
    if kind in ("scout", "attack", "defend"):
        return bl.execute((kind, None, ""), pos, obs, act, cat, ctx)
    return "?"


def terminal_reward(pos, s):
    """Win/lose from the game state. Lose: no buildings left. Win: enemy eliminated (no enemy seen
    for long + we still stand) — refined later; for now lose is the reliable terminal signal."""
    if pos.anchor is None and s.get("owned_buildings", 0) == 0:
        return -1.0, True
    return 0.0, False


class YRLearnEnv:
    """gym-style: reset() -> obs ; step(action_idx) -> (obs, reward, done, info). Requires a live match."""

    def __init__(self, launch=True):
        self.launch = launch
        self.obs_r = self.act_w = self.cat = self.lut = self.econ = None
        self.memory = {}
        self.tick = 0
        self.last_V = 0.0

    def reset(self):
        import subprocess
        from write_act import ActWriter
        from catalog import Catalog
        if self.obs_r:
            try:
                self.obs_r.close()
            except Exception:
                pass
        if self.act_w:
            try:
                self.act_w.close()
            except Exception:
                pass
        if self.launch:
            import os, sys
            sys.path.insert(0, os.path.dirname(__file__))
            from commander_build import launch_game
            # close any prior match first (multiple games can't share the OBS/ACT shared memory)
            subprocess.run(["taskkill", "/F", "/IM", "gamemd-spawn.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "Syringe.exe"], capture_output=True)
            time.sleep(1)
            launch_game()
            time.sleep(2)
        self.obs_r = bb.connect()
        self.act_w = ActWriter()
        self.cat = Catalog()
        self.lut = sp._lookup(self.cat)
        self.econ = sp.make_ctx(self.obs_r, self.cat).get("econ_ids", set())
        for _ in range(120):
            s = self.obs_r.read_state()
            if s and s["owned_units"] > 0 and s["n_enemy"] < 200:
                break
            time.sleep(1)
        self.memory, self.tick = {}, 0
        pos = gm.build_position(self.obs_r, self.cat, self.lut, self.memory, self.tick, self.econ)
        self.pos = pos
        self.last_V = pos.V
        self.prev_own_b = sum(pos.own_buildings.values())          # for the GOAL reward (survival)
        self.prev_enemy_c = sum(pos.enemy_seen.values())          # visible enemy combat (offense proxy)
        self.prev_enemy_b = self._enemy_buildings()               # enemy structures seen (destroy = the goal)
        return encode(pos)

    def _enemy_buildings(self):
        return sum(1 for e in self.obs_r.read_enemy() if e["category"] == "Building" and (e["x"] or e["y"]))

    def step(self, action_idx):
        ctx = sp.make_ctx(self.obs_r, self.cat)
        pos = gm.build_position(self.obs_r, self.cat, self.lut, self.memory, self.tick, self.econ)
        result = _execute_macro(action_idx, pos, self.obs_r, self.act_w, self.cat, ctx)
        time.sleep(1)
        self.tick += 1
        s = self.obs_r.read_state() or {}
        pos2 = gm.build_position(self.obs_r, self.cat, self.lut, self.memory, self.tick, self.econ)
        # --- GOAL-centered reward: hurt the opponent + survive. NOTHING for building the means. ---
        own_b = sum(pos2.own_buildings.values())
        enemy_c = sum(pos2.enemy_seen.values())
        enemy_b = self._enemy_buildings()
        reward = (3.0 * max(0, self.prev_enemy_b - enemy_b)      # razed an enemy STRUCTURE — the goal
                  + 0.3 * max(0, self.prev_enemy_c - enemy_c)    # killed visible enemy units (offense)
                  - 1.0 * max(0, self.prev_own_b - own_b))       # lost our own buildings (survival)
        done = (pos2.anchor is None and s.get("owned_buildings", 0) == 0)
        if done:
            reward -= 10.0                                       # base destroyed = we lost
        self.prev_own_b, self.prev_enemy_c, self.prev_enemy_b = own_b, enemy_c, enemy_b
        self.pos = pos2
        self.last_V = pos2.V
        return encode(pos2), float(reward), done, {"macro": MACROS[action_idx], "result": result, "V": pos2.V}

    def grid(self):
        """The spatial vision for the fused brain: (7, 64, 64) float in [0,1] (uint8 grid / 255)."""
        g, _ = self.obs_r.read_grid()
        return g.astype(np.float32) / 255.0

    def entities(self):
        """The roster for the entity transformer: every visible techno as a token + mask.

        Per-unit role/side/position/at-base, read straight from the (fog-honored) OBS lists -- the
        per-unit identity the role-count scalar throws away (targeting / reading enemy composition)."""
        from game_model import role_of
        anchor = self.pos.anchor if getattr(self, "pos", None) else None
        items = []
        for e in self.obs_r.read_own():
            cat = e["category"].lower()
            role = role_of(self.lut.get((cat, e["type_id"]), ""), cat)
            items.append((role, False, e["x"] / 256.0, e["y"] / 256.0, 1.0))
        for e in self.obs_r.read_enemy():
            if not (e["x"] or e["y"]):
                continue
            cat = e["category"].lower()
            role = role_of(self.lut.get((cat, e["type_id"]), ""), cat)
            atb = bool(anchor and abs(e["x"] - anchor[0]) + abs(e["y"] - anchor[1]) < 30)
            items.append((role, True, e["x"] / 256.0, e["y"] / 256.0, float(atb)))
        return entity_tokens(items)

    def close(self):
        if self.obs_r:
            self.obs_r.close()
        if self.act_w:
            self.act_w.close()


if __name__ == "__main__":
    # self-test (no game): the obs encoding + reward shaping from synthetic Positions.
    print(f"N_MACRO={N_MACRO}  OBS_DIM={OBS_DIM}")
    print("macros:", [f"{k}:{r}" if r else k for k, r in MACROS])
    p0 = gm.Position(prefix="NA", own_buildings={gm.CONSTRUCTION: 1, gm.POWER: 1, gm.ECONOMY: 1},
                     own_units={}, credits=50000, power_surplus=20, tech_tier=0)
    p0.V = gm.evaluate(p0)
    p1 = gm.Position(prefix="NA", own_buildings={gm.CONSTRUCTION: 1, gm.POWER: 1, gm.ECONOMY: 2,
                     gm.PROD_INF: 1, gm.PROD_VEH: 1}, own_units={gm.MAIN_BATTLE: 6, gm.ANTI_ARMOR: 2},
                     credits=40000, power_surplus=30, tech_tier=1)
    p1.V = gm.evaluate(p1)
    print("\nobs(p0):", encode(p0))
    print("obs(p1):", encode(p1))
    print(f"\nV: {p0.V:.2f} -> {p1.V:.2f}   reward(p0->p1) = dV = {p1.V - p0.V:+.2f}  (built up -> positive)")
    # a 'losing' transition:
    pL = gm.Position(prefix="NA", anchor=None, own_buildings={}, own_units={})
    print(f"terminal at no-base: {terminal_reward(pL, {'owned_buildings': 0})}  (lose -> -1, done)")
