"""The cheat-free script PRIOR policy: the stock 'Extreme AI' condition->action brain, ported into
our non-cheating macro space (see docs/stock-ai-blueprint.md).

Unlike build_base (a fixed build sequence) and play_policy (a 1-trace NN), this is a *reactive
weighted policy*: every tick it snapshots real game state (own tech + enemy composition, read with
full ObsReader access), evaluates a rule table (condition -> macro at weight), and executes the
highest-weighted eligible macro. It KEEPS the stock AI's strategy (escalation ladder, reactive
counters, the V3 answer) and DROPS the cheats (serial production, real economy, fog — all enforced
by going through EventClass/OutList).

A STRATEGY PROFILE scales rule-category weights -> this is the knob each LLM commander turns to
produce a posterior (boom / rush / turtle) over the shared prior.

    PYTHONPATH=yr_env;commander  python yr_env/script_policy.py [profile]   # live match required
"""
from __future__ import annotations

import os
import sys
import time

from read_obs import ObsReader
from write_act import ActWriter
from catalog import Catalog
from build_base import (BUILDINGTYPE, UNITTYPE, MAIN_TANK, name_to_suffix, produce_retry,
                        find_and_place, building_ready, own_buildings, id_by_index, connect)

# --- faction toolkits (Soviet 'NA' solid; others best-effort, resolved against the catalog) ---
DEFENSE_BLDG = {"NA": ["TESLA", "NALASR"], "GA": ["ATESLA", "GAPILL"], "YA": ["YAGGUN", "NATBNK"]}
AA_BLDG = {"NA": "NAFLAK", "GA": "GASAM", "YA": "YAGGUN"}
ANTI_ARMOR_UNIT = {"NA": "DRON", "GA": "TNKD", "YA": "LTNK"}     # Terror Drone / Tank Destroyer
AA_UNIT = {"NA": "HTK", "GA": "FV", "YA": "YTNK"}                # Flak Track / IFV
ENEMY_ARTILLERY = {"V3", "SREF", "DRED", "CARRIER"}             # V3 Launcher, Prism, Dreadnought
ENEMY_AIR = {"ORCA", "BEAG", "ZEP", "JUMPJET", "KIROV", "SHAD"}
ECON_IDS = {"HARV", "HORV", "CMIN", "CMON", "SMIN", "AMCV", "SMCV", "PCV", "YMCV"}  # harvesters + MCVs: never fight/scout

PROFILES = {                                                     # category weight multipliers
    "balanced": {},
    "boom":   {"economy": 1.5, "tech": 1.4, "attack": 0.7},
    "rush":   {"army": 1.5, "attack": 1.6, "economy": 0.8, "defense": 0.8},
    "turtle": {"defense": 1.6, "aa": 1.5, "counter": 1.5, "attack": 0.5},
}


def _lookup(cat):
    return {(e["category"], e["index"]): e["id"] for e in cat.by_id.values()}


def _suffix_present(blds, cat, suffix):
    return any((id_by_index(cat, "building", b["type_id"]) or "")[2:6] == suffix for b in blds)


def snapshot(obs, cat, lut, ctx):
    s = obs.read_state() or {}
    own, enemies = obs.read_own(), obs.read_enemy()
    blds = [u for u in own if u["category"] == "Building"]
    econ = ctx.get("econ_ids", set())                       # harvesters + MCVs: NEVER send to fight/scout
    combat = [u for u in own if u["category"] == "Unit" and u["type_id"] not in econ]
    units = [u for u in combat if (u["x"] or u["y"])]       # positioned combat units (commandable)
    prefix = ctx.get("prefix")

    def own_has(suf):
        return _suffix_present(blds, cat, suf) if prefix else False

    # classify enemies by role via the (category,index)->id reverse map (catalog keys are lowercase)
    def role(e):
        eid = lut.get((e["category"].lower(), e["type_id"]), "")
        if eid in ENEMY_ARTILLERY:
            return "artillery"
        if eid in ENEMY_AIR:
            return "air"
        return "other"
    enemy_arty = [e for e in enemies if role(e) == "artillery"]
    enemy_air = [e for e in enemies if role(e) == "air"]
    anchor = ctx.get("anchor")
    near = []
    if anchor:
        ax, ay = anchor
        near = [e for e in enemies if abs(e["x"] - ax) + abs(e["y"] - ay) < 35]   # threat near the base

    return {
        "credits": s.get("credits", 0),
        "power_surplus": s.get("power_output", 0) - s.get("power_drain", 0),
        "n_bld": len(blds), "n_army": len(combat), "n_enemy": len(enemies),
        "has_power": own_has("POWR"), "n_refn": sum(1 for b in blds if (id_by_index(cat, "building", b["type_id"]) or "")[2:6] == "REFN"),
        "has_barracks": own_has("HAND") or own_has("PILE") or own_has("BRCK"),
        "has_weap": own_has("WEAP"), "has_radar": own_has("RADR") or own_has("AIRC"),
        "n_defense": sum(1 for b in blds if (id_by_index(cat, "building", b["type_id"]) or "") in
                         (DEFENSE_BLDG.get(prefix, []) + [AA_BLDG.get(prefix, "")])),
        "enemy_arty": enemy_arty, "enemy_air": enemy_air, "enemy_near": near,
        "enemy_buildings": [e for e in enemies if e.get("category") == "Building" and (e["x"] or e["y"])],
        "units": units, "anchor": anchor, "prefix": prefix,
    }


# rule table: (name, category, condition(state)->bool, macro, payload_key, base_weight)
# Principle: ESTABLISH the economy first (ambient enemies don't stop the build-order); respond to
# REAL threats (detected artillery, or a genuine push AT the base once a War Factory exists); keep
# an army; attack only when strong and safe. Higher effective weight wins among eligible rules.
RULES = [
    ("deploy",       "deploy",  lambda s: s["n_bld"] == 0,                                   "DEPLOY_MCV",        None,        1000),
    # --- artillery is THE killer: detected V3/Prism gets the top reactive response at any range ---
    ("sortie_v3",    "counter", lambda s: s["enemy_arty"] and s["n_army"] >= 3,              "ANTI_ARTY_SORTIE",  "artillery", 200),
    ("train_drone",  "counter", lambda s: s["enemy_arty"] and s["has_weap"],                 "TRAIN_ANTIARMOR",   None,        190),
    ("build_aa",     "aa",      lambda s: s["enemy_air"] and s["has_power"],                 "BUILD_AA",          None,        130),
    # --- economy / build-order: establish the base; NOT preempted by ambient enemies ---
    ("power",        "economy", lambda s: not s["has_power"] or s["power_surplus"] < 50,      "BUILD_POWER",      None,        120),
    ("refinery",     "economy", lambda s: s["n_refn"] < 2,                                    "BUILD_REFINERY",   None,        110),
    ("barracks",     "economy", lambda s: not s["has_barracks"],                              "BUILD_BARRACKS",   None,        100),
    ("warfactory",   "tech",    lambda s: not s["has_weap"],                                  "BUILD_WARFACTORY", None,        98),
    ("radar",        "tech",    lambda s: s["has_weap"] and not s["has_radar"],               "BUILD_RADAR",      None,        80),
    # --- base under a REAL push (only after a War Factory exists, so economy isn't starved) ---
    ("base_defense", "defense", lambda s: s["has_weap"] and len(s["enemy_near"]) >= 3 and s["n_defense"] < 4, "BUILD_DEFENSE", None, 95),
    ("engage",       "defense", lambda s: s["has_weap"] and len(s["enemy_near"]) >= 5 and s["n_army"] >= 4, "DEFEND", None,        95),
    # --- scout to FIND the enemy base (so we can actually push to win), then mass + attack ---
    ("scout",        "army",    lambda s: s["has_weap"] and s["n_army"] >= 3 and not s["enemy_buildings"]
                                          and len(s["enemy_near"]) < 5 and s.get("scout_ok"), "SCOUT", None, 92),
    ("army",         "army",    lambda s: s["has_weap"] and s["n_army"] < 12,                 "TRAIN_TANK",       None,        90),
    ("attack",       "attack",  lambda s: s["has_weap"] and ((s["enemy_buildings"] and s["n_army"] >= 8)
                                                             or s["n_army"] >= 12),           "ATTACK",   None,        100),
]

_MEM = {"tick": -1, "last_scout": -99}     # scout-cooldown memory (decide() is otherwise stateless)


def decide(state, profile):
    mult = PROFILES.get(profile, {})
    _MEM["tick"] += 1
    state = dict(state)
    state["scout_ok"] = (_MEM["tick"] - _MEM["last_scout"]) >= 8     # scout at most ~every 8 ticks
    best = None
    for name, cat_, cond, macro, payload, w in RULES:
        try:
            ok = cond(state)
        except Exception:
            ok = False
        if not ok:
            continue
        eff = w * mult.get(cat_, 1.0)
        if best is None or eff > best[0]:
            best = (eff, name, macro, payload)
    if not best:
        return "NOOP", None, "idle"
    if best[2] == "SCOUT":
        _MEM["last_scout"] = _MEM["tick"]
    return best[2], best[3], best[1]


# ---- executors (reuse build_base helpers; new logic for defense / AA / counter / sortie) ----
def _build(act, obs, cat, anchor, full_id):
    e = cat.by_id.get(full_id)
    if not e:
        return f"no {full_id}"
    pr = produce_retry(act, BUILDINGTYPE, e["index"])
    if not pr or pr[0] != 0:
        return f"produce-fail {pr}"
    for _ in range(40):
        if building_ready(obs):
            break
        time.sleep(1)
    return f"built {full_id} @ {find_and_place(act, e['index'], anchor)}"


def _train(act, cat, full_id):
    e = cat.by_id.get(full_id)
    if not e:
        return f"no {full_id}"
    return f"train {full_id}: {produce_retry(act, UNITTYPE, e['index'])}"


def execute(macro, payload, obs, act, cat, ctx, state):
    p = ctx.get("prefix")
    if macro == "NOOP":
        return "idle"
    if macro == "DEPLOY_MCV":
        mcv = next((u for u in obs.read_own() if u["category"] == "Unit" and u["type_id"] in ctx["mcv_ids"]), None)
        if not mcv:
            return "no-mcv"
        act.deploy(mcv["unique_id"])
        for _ in range(30):
            if own_buildings(obs):
                return "deployed"
            time.sleep(1)
        return "deploy-timeout"
    if macro in ("BUILD_POWER", "BUILD_REFINERY", "BUILD_BARRACKS", "BUILD_WARFACTORY", "BUILD_RADAR"):
        label = {"BUILD_POWER": "power", "BUILD_REFINERY": "refinery", "BUILD_BARRACKS": "barracks",
                 "BUILD_WARFACTORY": "war factory", "BUILD_RADAR": "radar"}[macro]
        suf = name_to_suffix(label, cat, p)
        return _build(act, obs, cat, ctx["anchor"], p + suf) if suf else "no-suffix"
    if macro == "BUILD_DEFENSE":
        for full in DEFENSE_BLDG.get(p, []):
            if cat.by_id.get(full) and (full != "TESLA" or state["has_radar"]):   # tesla needs radar
                return _build(act, obs, cat, ctx["anchor"], full)
        return "no-defense-type"
    if macro == "BUILD_AA":
        return _build(act, obs, cat, ctx["anchor"], AA_BLDG.get(p, ""))
    if macro == "TRAIN_TANK":
        return _train(act, cat, MAIN_TANK.get(p, "HTNK"))
    if macro == "TRAIN_ANTIARMOR":
        return _train(act, cat, ANTI_ARMOR_UNIT.get(p, "DRON"))
    if macro == "DEFEND":
        # ACTIVE defense: send the army to DESTROY the nearest attackers (attack-move = move + fire),
        # not passively huddle at the ConYard. This is visible combat on screen.
        units = state["units"]
        if not units:
            return "no-army"
        near = state["enemy_near"] or obs.read_enemy()
        if near:
            ax, ay = ctx["anchor"]
            t = min(near, key=lambda e: abs(e["x"] - ax) + abs(e["y"] - ay))
            for u in units:
                act.attack_move(u["unique_id"], t["x"], t["y"])
            return f"ENGAGE {len(units)} units -> attackers at ({t['x']},{t['y']})"
        ax, ay = ctx["anchor"]
        for u in units:
            act.attack_move(u["unique_id"], ax, ay)
        return f"hold base x{len(units)}"
    if macro == "ANTI_ARTY_SORTIE":
        arty = state["enemy_arty"]
        if not arty or not state["units"]:
            return "no-target/army"
        ax, ay = ctx["anchor"]
        t = min(arty, key=lambda e: abs(e["x"] - ax) + abs(e["y"] - ay))
        for u in state["units"]:
            act.attack_move(u["unique_id"], t["x"], t["y"])
        return f"SORTIE {len(state['units'])} units -> V3 at ({t['x']},{t['y']})"
    if macro == "SCOUT":
        units = state["units"]
        if not units:
            return "no-unit"
        ax, ay = ctx["anchor"]
        enemies = obs.read_enemy()
        if enemies:                                        # probe BEYOND the skirmishers toward their base
            ex = sum(e["x"] for e in enemies) / len(enemies)
            ey = sum(e["y"] for e in enemies) / len(enemies)
            tx, ty = int(ax + (ex - ax) * 2.2), int(ay + (ey - ay) * 2.2)
        else:                                              # nothing seen -> probe the far map
            tx, ty = (ax + 100 if ax < 128 else ax - 100), (ay + 100 if ay < 128 else ay - 100)
        tx, ty = max(2, tx), max(2, ty)
        for u in units[:2]:                                # send a couple of scouts, not the whole army
            act.move(u["unique_id"], tx, ty)
        return f"SCOUT {min(2, len(units))} -> ({tx},{ty})"
    if macro == "ATTACK":
        units = state["units"]
        if not units:
            return "no-army"
        ax, ay = ctx["anchor"]
        ebld, enemies = state.get("enemy_buildings") or [], obs.read_enemy()
        if ebld:                                           # we found the enemy BASE -> march on it (the win)
            t = min(ebld, key=lambda e: abs(e["x"] - ax) + abs(e["y"] - ay))
            tx, ty, what = t["x"], t["y"], "enemy BASE"
        elif enemies:
            t = min(enemies, key=lambda e: abs(e["x"] - ax) + abs(e["y"] - ay))
            tx, ty, what = t["x"], t["y"], "enemy force"
        else:
            tx, ty, what = (ax // 2 if ax > 80 else ax + 90), ay, "map interior"
        for u in units:
            act.attack_move(u["unique_id"], tx, ty)
        return f"ATTACK {what} ({tx},{ty}) x{len(units)}"
    return "unknown"


def make_ctx(obs, cat):
    bld = own_buildings(obs)
    return {"anchor": (bld[0]["x"], bld[0]["y"]) if bld else None,
            "prefix": (id_by_index(cat, "building", bld[0]["type_id"]) or "GA")[:2] if bld else None,
            "mcv_ids": {e["index"] for e in cat.by_id.values()
                        if e["category"] == "unit" and ("MCV" in e["id"] or "MCV" in e["ui_name"])},
            "econ_ids": {e["index"] for e in cat.by_id.values()
                         if e["category"] == "unit" and (e["id"] in ECON_IDS or "MCV" in e["id"])}}


def main(profile="balanced", max_ticks=45, launch=True):
    if launch:
        sys.path.insert(0, os.path.dirname(__file__))
        from commander_build import launch_game
        print(f"launching match; script-prior policy (profile={profile}) will play...")
        launch_game()
        time.sleep(2)
    obs, act, cat = connect(), ActWriter(), Catalog()
    lut = _lookup(cat)
    print("waiting for the match to initialize...")
    for _ in range(120):
        s = obs.read_state()
        if s and s["owned_units"] > 0 and s["n_enemy"] < 200:
            break
        time.sleep(1)
    print(f">>> WATCH: cheat-free STOCK-AI BRAIN drives (profile={profile}) — it builds, defends, "
          f"and answers artillery with Terror Drones\n")
    for tick in range(max_ticks):
        s = obs.read_state()
        if not s:
            time.sleep(0.5)
            continue
        ctx = make_ctx(obs, cat)
        st = snapshot(obs, cat, lut, ctx)
        macro, payload, why = decide(st, profile)
        res = execute(macro, payload, obs, act, cat, ctx, st)
        threat = f" THREAT[arty={len(st['enemy_arty'])} air={len(st['enemy_air'])} near={len(st['enemy_near'])}]" if (st["enemy_arty"] or st["enemy_near"] or st["enemy_air"]) else ""
        print(f"[{tick:2d}] B={s['owned_buildings']} U={s['owned_units']} E={s['n_enemy']} "
              f"cr={s['credits']}{threat}  ->  {macro} ({why})  ->  {res}")
        time.sleep(1)
    obs.close()
    act.close()


if __name__ == "__main__":
    prof = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PROFILES else "balanced"
    main(profile=prof)
