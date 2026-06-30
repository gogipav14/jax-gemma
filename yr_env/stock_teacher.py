"""The BC teacher = the DISTILLED STOCK AI (docs/stock-ai-blueprint.md), in our non-cheating macro
space. NOT a hand-coded guess — the stock AI's own decision logic: the escalation ladder (build
order), the always-on REACTIVE COUNTER layer (artillery -> Terror Drones + sortie; air/heavy ->
AA + drones; armor -> drones), defenses on a real push, and tech-gated offense (mass, then attack
waves). The agent BC-clones this expert opening, then self-play RL improves past it toward the goal.

macro indices (rl_env.MACROS): 0 noop 1 deploy 2 POWER 3 ECONOMY 4 PROD_INF 5 PROD_VEH 6 RADAR
7 DEF_GROUND 8 DEF_AA 9 train MAIN 10 train ANTI 11 train AA 12 scout 13 attack 14 defend
"""
from __future__ import annotations

import game_model as gm

ROLE2BUILD = {gm.POWER: 2, gm.ECONOMY: 3, gm.PROD_INF: 4, gm.PROD_VEH: 5, gm.TECH_RADAR: 6}


def stock_teacher(pos: gm.Position) -> int:
    if pos.anchor is None:
        return 1                                          # deploy MCV -> Construction Yard
    nb = pos.next_build()
    if nb in ROLE2BUILD:
        return ROLE2BUILD[nb]                             # escalation ladder: power->2 refn->barracks->war factory->radar

    have_weap = pos.has(gm.PROD_VEH)
    army = pos.own_units.get(gm.MAIN_BATTLE, 0) + pos.own_units.get(gm.ANTI_ARMOR, 0)
    arty = [t for t in pos.threats if t.role == gm.ARTILLERY]
    superu = [t for t in pos.threats if t.role == gm.SUPERUNIT]      # aircraft fold to SUPERUNIT + heavies
    armor = [t for t in pos.threats if t.role == gm.MAIN_BATTLE]
    near = [t for t in pos.threats if t.at_base]

    # --- always-on REACTIVE COUNTER layer (answer the enemy's actual composition) ---
    if arty and have_weap:                                # artillery (V3/Prism) -> Terror Drones, then SORTIE
        return 10 if pos.own_units.get(gm.ANTI_ARMOR, 0) < 4 else 13
    if superu and pos.has(gm.POWER) and pos.own_buildings.get(gm.DEF_AA, 0) < 1:
        return 8                                          # air / heavy -> Flak Cannon (anti-air)
    if (superu or armor) and have_weap and pos.own_units.get(gm.ANTI_ARMOR, 0) < 3:
        return 10                                         # enemy armor / heavies -> Terror Drones

    # --- defenses on a REAL push (the DefenseRatio layer), then engage ---
    if len(near) >= 3:
        if pos.own_buildings.get(gm.DEF_GROUND, 0) < 4 and pos.has(gm.POWER):
            return 7
        if army >= 2:
            return 14

    # --- offense scales with tech: mass a force, then send attack waves at the opponent ---
    if have_weap and army < 6:
        return 9
    if army >= 4:
        return 13
    return 0


if __name__ == "__main__":
    A = ["noop", "deploy", "POWER", "ECONOMY", "BARRACKS", "WAR_FACTORY", "RADAR", "DEF_G",
         "DEF_AA", "train_MAIN", "train_ANTI", "train_AA", "scout", "ATTACK", "defend"]
    DONE = {gm.CONSTRUCTION: 1, gm.POWER: 1, gm.ECONOMY: 2, gm.PROD_INF: 1, gm.PROD_VEH: 1, gm.TECH_RADAR: 1}
    army = lambda n: [{"unique_id": i, "x": 50, "y": 50} for i in range(n)]

    def P(**kw):
        p = gm.Position(prefix="NA", anchor=(50, 50))
        for k, v in kw.items():
            setattr(p, k, v)
        return p
    ARTY = gm.Threat(gm.ARTILLERY, 3, 4.8, gm.ANTI_ARMOR, True)
    SUPER = gm.Threat(gm.SUPERUNIT, 2, 6.0, gm.ANTI_ARMOR, True)
    PUSH = [gm.Threat(gm.MAIN_BATTLE, 1, 1.0, gm.ANTI_ARMOR, True)] * 4
    cases = [
        ("no base", gm.Position(prefix="NA", anchor=None)),
        ("need power", P(own_buildings={gm.CONSTRUCTION: 1})),
        ("refineries -> barracks before war factory", P(own_buildings={gm.CONSTRUCTION: 1, gm.POWER: 1, gm.ECONOMY: 2})),
        ("V3 at base -> Terror Drones", P(own_buildings=DONE, own_units={}, army=army(1), threats=[ARTY])),
        ("V3 + drones ready -> SORTIE", P(own_buildings=DONE, own_units={gm.ANTI_ARMOR: 4}, army=army(4), threats=[ARTY])),
        ("air/heavy -> Flak", P(own_buildings=DONE, own_units={}, army=army(2), threats=[SUPER])),
        ("ground push -> defense", P(own_buildings=DONE, own_units={gm.MAIN_BATTLE: 3}, army=army(3), threats=PUSH)),
        ("safe, small army -> mass", P(own_buildings=DONE, own_units={gm.MAIN_BATTLE: 3}, army=army(3))),
        ("safe, big army -> ATTACK", P(own_buildings=DONE, own_units={gm.MAIN_BATTLE: 7}, army=army(7))),
    ]
    print("stock-AI teacher (the distillation) decisions:")
    for label, pos in cases:
        print(f"  {label:42s} -> {A[stock_teacher(pos)]}")
