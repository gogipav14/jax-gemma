"""Discrete macro-action space — the non-cheating action vocabulary the NN policy picks from,
and a state-grounded executor that carries each macro out via the proven build_base helpers.

This is the bridge between the learned policy and the game: the policy outputs ONE macro id per
decision; execute_macro() turns it into concrete EventClass/OutList actions (produce->wait->place,
train, attack), with all parameters resolved from current game state. The SAME space is what the
Phase-5 self-play RL will explore (small, executable, provably non-cheating).

relabel() maps a recorded fine-grained (action_type, category, type_id) decision back to its macro,
so we can behaviorally-clone the macro policy from the existing traces.
"""
from __future__ import annotations

import time

from contract import ActionType, RTTIType
from build_base import (BUILDINGTYPE, UNITTYPE, MAIN_TANK, name_to_suffix, produce_retry,
                        find_and_place, building_ready, own_buildings, id_by_index)

MACROS = ["NOOP", "DEPLOY_MCV", "BUILD_POWER", "BUILD_REFINERY", "BUILD_BARRACKS",
          "BUILD_WARFACTORY", "BUILD_RADAR", "BUILD_LAB", "TRAIN_TANK", "ATTACK"]
N_MACRO = len(MACROS)
M_INDEX = {n: i for i, n in enumerate(MACROS)}

# macro -> readable structure label that name_to_suffix understands (faction prefix applied in-game)
BUILD_LABEL = {"BUILD_POWER": "power", "BUILD_REFINERY": "refinery", "BUILD_BARRACKS": "barracks",
               "BUILD_WARFACTORY": "war factory", "BUILD_RADAR": "radar", "BUILD_LAB": "battle lab"}
# building ID suffix -> macro (for relabeling recorded build actions)
SUFFIX2MACRO = {"POWR": "BUILD_POWER", "REFN": "BUILD_REFINERY", "HAND": "BUILD_BARRACKS",
                "PILE": "BUILD_BARRACKS", "BRCK": "BUILD_BARRACKS", "WEAP": "BUILD_WARFACTORY",
                "RADR": "BUILD_RADAR", "TECH": "BUILD_LAB"}


def relabel(cat, action_row):
    """Map a recorded action row [atype, category_rtti, type_id, cx, cy, target] -> macro id."""
    atype, crtti, type_id = int(action_row[0]), int(action_row[1]), int(action_row[2])
    if atype == ActionType.DEPLOY:
        return M_INDEX["DEPLOY_MCV"]
    if atype in (ActionType.GROUP_ATTACK, ActionType.GROUP_MOVE):
        return M_INDEX["ATTACK"]
    if atype in (ActionType.PRODUCE, ActionType.PLACE):
        if crtti == RTTIType.UNIT_TYPE:
            return M_INDEX["TRAIN_TANK"]
        if crtti == RTTIType.BUILDING_TYPE:
            bid = id_by_index(cat, "building", type_id)
            suf = bid[2:6] if bid else ""
            return M_INDEX[SUFFIX2MACRO.get(suf, "NOOP")]
    return M_INDEX["NOOP"]


def build_context(obs, cat):
    """Resolve faction prefix + ConYard anchor + MCV type ids from current state."""
    bld = own_buildings(obs)
    anchor = (bld[0]["x"], bld[0]["y"]) if bld else None
    prefix = (id_by_index(cat, "building", bld[0]["type_id"]) or "GA")[:2] if bld else None
    mcv_ids = {e["index"] for e in cat.by_id.values()
               if e["category"] == "unit" and ("MCV" in e["id"] or "MCV" in e["ui_name"])}
    return {"anchor": anchor, "prefix": prefix, "mcv_ids": mcv_ids}


def execute_macro(mid, obs, act, cat, ctx):
    """Carry out one macro to completion via build_base helpers; return a short result string."""
    name = MACROS[mid]
    if name == "NOOP":
        return "noop"

    if name == "DEPLOY_MCV":
        mcv = next((u for u in obs.read_own()
                    if u["category"] == "Unit" and u["type_id"] in ctx["mcv_ids"]), None)
        if not mcv:
            return "no-mcv"
        act.deploy(mcv["unique_id"])
        for _ in range(30):
            if own_buildings(obs):
                return "deployed"
            time.sleep(1)
        return "deploy-timeout"

    if name == "ATTACK":
        units = [u for u in obs.read_own() if u["category"] == "Unit" and (u["x"] or u["y"])]
        if not units:
            return "no-army"
        ax, ay = ctx["anchor"] or (units[0]["x"], units[0]["y"])
        enemies = [e for e in obs.read_enemy() if (e["x"] or e["y"])]
        if enemies:
            t = min(enemies, key=lambda e: abs(e["x"] - ax) + abs(e["y"] - ay))
            tx, ty = t["x"], t["y"]
        else:
            tx, ty = (ax // 2 if ax > 80 else ax + 80), ay
        for u in units:
            act.attack_move(u["unique_id"], tx, ty)
        return f"attack ({tx},{ty}) x{len(units)}"

    if name == "TRAIN_TANK":
        if not ctx["prefix"]:
            return "no-base"
        tank = MAIN_TANK.get(ctx["prefix"], "HTNK")
        e = cat.by_id.get(tank)
        if not e:
            return "no-tank-type"
        return f"train {tank}: {produce_retry(act, UNITTYPE, e['index'])}"

    if name in BUILD_LABEL:
        if not ctx["prefix"]:
            return "no-base"
        suffix = name_to_suffix(BUILD_LABEL[name], cat, ctx["prefix"])
        if not suffix:
            return "no-suffix"
        e = cat.by_id.get(ctx["prefix"] + suffix)
        if not e:
            return "not-in-catalog"
        pr = produce_retry(act, BUILDINGTYPE, e["index"])
        if not pr or pr[0] != 0:
            return f"produce-fail {pr}"
        for _ in range(40):
            if building_ready(obs):
                break
            time.sleep(1)
        return f"built {ctx['prefix'] + suffix} @ {find_and_place(act, e['index'], ctx['anchor'])}"

    return "unknown"
