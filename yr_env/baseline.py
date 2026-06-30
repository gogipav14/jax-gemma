"""Solid baseline agent that plays OVER the game model (game_model.py) — no LLM, deterministic.

Every decision reads the structured Position: it builds the prereq-correct next structure (so it can
never jam on a missing Barracks), answers the top threat with the COUNTER the model names (Terror
Drones vs artillery/armor, Flak vs air), keeps a main-battle army, scouts for the enemy base, and
pushes when strong. Harvesters/MCV are never commanded to fight. This is the reliable 'hands' the
adaptive brain and the league build on.

    PYTHONPATH=yr_env  python yr_env/baseline.py        # live match required (launches its own)
"""
from __future__ import annotations

import os
import sys
import time

from write_act import ActWriter
from catalog import Catalog
import build_base as bb
import script_policy as sp
import game_model as gm

LOG = os.path.join(os.path.dirname(__file__), "data", "baseline.log")
STRUCT_LABEL = {gm.POWER: "power", gm.ECONOMY: "refinery", gm.PROD_INF: "barracks",
                gm.PROD_VEH: "war factory", gm.TECH_RADAR: "radar", gm.TECH_LAB: "battle lab"}


def struct_id(role, prefix, cat):
    """role -> a concrete buildable structure ID for the faction."""
    if role == gm.DEF_GROUND:
        for fid in sp.DEFENSE_BLDG.get(prefix, []):
            if cat.by_id.get(fid):
                return fid
        return None
    if role == gm.DEF_AA:
        return sp.AA_BLDG.get(prefix) if cat.by_id.get(sp.AA_BLDG.get(prefix, "")) else None
    label = STRUCT_LABEL.get(role)
    if not label:
        return None
    suf = bb.name_to_suffix(label, cat, prefix)
    return prefix + suf if suf else None


def unit_id(role, prefix):
    if role == gm.ANTI_ARMOR:
        return sp.ANTI_ARMOR_UNIT.get(prefix, "DRON")
    if role == gm.ANTI_AIR:
        return sp.AA_UNIT.get(prefix, "HTK")
    return bb.MAIN_TANK.get(prefix, "HTNK")


def decide(pos, obs, cat, scout_ok):
    """Model-driven decision -> (kind, payload, label). payload: a structure/unit ID or a target."""
    prefix = pos.prefix
    if pos.anchor is None:
        return ("deploy", None, "deploy MCV")

    # 1) establish the base in PREREQ-CORRECT order (cannot jam)
    nb = pos.next_build()
    if nb:
        fid = struct_id(nb, prefix, cat)
        if fid:
            return ("build", fid, f"base: {nb} ({fid})")

    have_veh = pos.has(gm.PROD_VEH)
    # 2) answer the top BASE threat with the model's counter
    base_threats = [t for t in pos.threats if t.at_base]
    if base_threats and have_veh:
        t = base_threats[0]
        cu = unit_id(t.counter, prefix)
        if cat.by_id.get(cu) and pos.own_units.get(t.counter, 0) < max(2, t.count):
            return ("train", cu, f"COUNTER {t.role}->{t.counter} ({pos.own_units.get(t.counter,0)}/{t.count})")
        if pos.army:
            return ("engage", t, f"engage {t.role} at base")

    # 3) static defense when threatened and thin
    if pos.threats and pos.own_buildings.get(gm.DEF_GROUND, 0) < 3 and pos.has(gm.POWER):
        fid = struct_id(gm.DEF_GROUND, prefix, cat)
        if fid:
            return ("build", fid, f"base defense ({fid})")

    # 4) maintain a main-battle army
    if have_veh and pos.own_units.get(gm.MAIN_BATTLE, 0) < 8:
        return ("train", unit_id(gm.MAIN_BATTLE, prefix), f"army {pos.own_units.get(gm.MAIN_BATTLE,0)}/8")

    # 5) scout for the enemy base if we don't know where it is
    enemy_base = _enemy_base(obs)
    if enemy_base is None and scout_ok and pos.army:
        return ("scout", None, "scout for enemy base")

    # 6) push out when strong and we know where to go
    army_n = sum(pos.own_units.get(r, 0) for r in (gm.MAIN_BATTLE, gm.ANTI_ARMOR))
    if army_n >= 8:
        return ("attack", enemy_base, "attack" + (" enemy base" if enemy_base else " out"))
    return ("scout", None, "hold/scout") if pos.army else ("noop", None, "build up")


def _enemy_base(obs):
    blds = [e for e in obs.read_enemy() if e["category"] == "Building" and (e["x"] or e["y"])]
    return (blds[0]["x"], blds[0]["y"]) if blds else None


def execute(action, pos, obs, act, cat, ctx):
    kind, payload, _ = action
    if kind == "deploy":
        return sp.execute("DEPLOY_MCV", None, obs, act, cat, ctx, _snap_shim(pos))
    if kind == "build":
        return sp._build(act, obs, cat, ctx["anchor"], payload)
    if kind == "train":
        e = cat.by_id.get(payload)
        return f"{payload}: {bb.produce_retry(act, bb.UNITTYPE, e['index'])}" if e else f"no {payload}"
    if kind == "engage":
        t = payload
        tx, ty = (t.counter and None), None
        # attack-move the army at the threat's nearest position
        positions = pos.enemy_belief.get(t.role, {}).get("positions") or []
        ax, ay = ctx["anchor"]
        if positions:
            px, py = min(positions, key=lambda p: abs(p[0] - ax) + abs(p[1] - ay))
        else:
            px, py = ax, ay
        for u in pos.army:
            act.attack_move(u["unique_id"], px, py)
        return f"ENGAGE {len(pos.army)} -> ({px},{py})"
    if kind == "attack":
        ax, ay = ctx["anchor"]
        tx, ty = payload if payload else (ax // 2 if ax > 80 else ax + 90, ay)
        for u in pos.army:
            act.attack_move(u["unique_id"], tx, ty)
        return f"ATTACK {len(pos.army)} -> ({tx},{ty})"
    if kind == "scout":
        return sp.execute("SCOUT", None, obs, act, cat, ctx, _snap_shim(pos))
    return "build up"


def _snap_shim(pos):
    """sp.execute (DEPLOY/SCOUT) reads a few snapshot keys; provide them from the Position."""
    return {"units": pos.army, "anchor": pos.anchor, "enemy_arty": [], "enemy_near": []}


def main(max_ticks=80, launch=True):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(LOG, "w", encoding="utf-8"))
    if launch:
        sys.path.insert(0, os.path.dirname(__file__))
        from commander_build import launch_game
        print("launching match; SOLID BASELINE (no LLM) playing over the game model")
        launch_game()
        time.sleep(2)
    obs, act, cat = bb.connect(), ActWriter(), Catalog()
    lut = sp._lookup(cat)
    econ = sp.make_ctx(obs, cat).get("econ_ids", set())
    print("waiting for the match to initialize...")
    for _ in range(120):
        s = obs.read_state()
        if s and s["owned_units"] > 0 and s["n_enemy"] < 200:
            break
        time.sleep(1)
    print(">>> WATCH: prereq-correct base, sees threats + builds the right counter, scouts, attacks\n")

    memory, last_scout = {}, -99
    for tick in range(max_ticks):
        s = obs.read_state()
        if not s:
            time.sleep(0.5)
            continue
        ctx = sp.make_ctx(obs, cat)
        pos = gm.build_position(obs, cat, lut, memory, tick, econ_ids=econ)
        scout_ok = (tick - last_scout) >= 8
        action = decide(pos, obs, cat, scout_ok)
        if action[0] == "scout":
            last_scout = tick
        res = execute(action, pos, obs, act, cat, ctx)
        th = "; ".join(f"{t.role}x{t.count}{'@B' if t.at_base else ''}" for t in pos.threats[:3]) or "none"
        print(f"[{tick:2d}] B={s['owned_buildings']} U={s['owned_units']} V={pos.V:+.1f} "
              f"threats[{th}]  ->  {action[2]}  ->  {res}")
        time.sleep(1)
    obs.close()
    act.close()


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            st.write(x)
            st.flush()

    def flush(self):
        for st in self.s:
            st.flush()


if __name__ == "__main__":
    main()
