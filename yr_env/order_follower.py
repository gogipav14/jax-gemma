"""Order-follower executor: carry out the LLM commander's SPECIFIC orders move-for-move.

Instead of mapping the LLM to a coarse strategy profile, this resolves the directive's
`priority_build_order` (named structures) and `army_composition` (named units -> counts) to real
catalog entries and produces them through the non-cheating command path, then executes the `stance`.
This is what makes the discovered playbook actually COMMAND the units. A thin safety layer still
guarantees: deploy the MCV, keep power + a refinery (economy floor), and answer a real push.

    PYTHONPATH=yr_env;commander  python yr_env/order_follower.py [model]    # live match required
"""
from __future__ import annotations

import os
import sys
import threading
import time

from write_act import ActWriter
from catalog import Catalog
import build_base as bb
import script_policy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "commander"))
from commander import command                          # noqa: E402

# --- name resolution: LLM words -> catalog IDs (Soviet 'NA' solid; others best-effort) ---
DEFENSE_NAMES = {"tesla coil": "TESLA", "tesla": "TESLA", "flak cannon": "NAFLAK", "sentry gun": "NALASR",
                 "sentry": "NALASR", "battle bunker": "NABNKR", "bunker": "NABNKR",
                 "gattling cannon": "YAGGUN", "pillbox": "GAPILL", "prism tower": "ATESLA"}
UNIT_ALIASES = {"terror drone": "DRON", "drone": "DRON", "flak track": "HTK", "mobile flak": "HTK",
                "apocalypse": "APOC", "apocalypse tank": "APOC", "apoc": "APOC", "v3": "V3",
                "v3 launcher": "V3", "rhino": "HTNK", "rhino tank": "HTNK", "heavy tank": "HTNK",
                "conscript": "E2", "flak trooper": "FLAKT", "flak infantry": "FLAKT",
                "dreadnought": "DRED", "kirov": "KIROV", "grizzly": "MTNK", "lasher": "LTNK"}
MAIN_TANK_NAMES = {"tank", "tanks", "battle tank", "main battle tank", "main tank", "mbt"}
CAT_RTTI = {"building": bb.BUILDINGTYPE, "unit": bb.UNITTYPE, "infantry": 16, "aircraft": 3}


def resolve(name, cat, prefix):
    """LLM-named structure/unit -> (category, full_id) or None. category in {building, unit, infantry}."""
    n = (name or "").strip().lower()
    if not n:
        return None
    suf = bb.name_to_suffix(n, cat, prefix)                     # standard structures (power/refn/weap/...)
    if suf and cat.by_id.get(prefix + suf):
        return ("building", prefix + suf)
    if n in DEFENSE_NAMES and cat.by_id.get(DEFENSE_NAMES[n]):  # named defenses / AA
        return ("building", DEFENSE_NAMES[n])
    if n in MAIN_TANK_NAMES:                                    # faction main battle tank
        t = bb.MAIN_TANK.get(prefix, "HTNK")
        if cat.by_id.get(t):
            return ("unit", t)
    if n in UNIT_ALIASES and cat.by_id.get(UNIT_ALIASES[n]):    # named units
        e = cat.by_id[UNIT_ALIASES[n]]
        return (e["category"], UNIT_ALIASES[n])
    return None


def own_suffixes(obs, cat):
    return {(bb.id_by_index(cat, "building", b["type_id"]) or "")[2:6] for b in bb.own_buildings(obs)}


def own_count(obs, cat, full_id):
    e = cat.by_id[full_id]
    return sum(1 for u in obs.read_own()
               if u["category"].lower() == e["category"] and u["type_id"] == e["index"])


def has_factory(kind, have):
    if kind == "unit":
        return "WEAP" in have
    if kind == "infantry":
        return bool({"HAND", "PILE", "BRCK"} & have)
    return True


def decide(directive, obs, cat, ctx, snap):
    """Pick the next concrete action to advance the LLM's orders. Returns (kind, full_id, label)."""
    prefix = ctx["prefix"]
    if snap["n_bld"] == 0:
        return ("deploy", None, "deploy MCV")
    if len(snap["enemy_near"]) >= 5 and snap["n_army"] >= 3:           # reactive safety: real push
        return ("engage", None, "defend base (reactive)")
    have = own_suffixes(obs, cat)
    if "POWR" not in have:                                             # economy floor: power then a refinery
        return ("build", prefix + "POWR", "Power Plant (floor)")
    if "REFN" not in have:
        return ("build", prefix + "REFN", "Refinery (floor)")
    # --- FOLLOW the ordered build list (structures the LLM named, in order) ---
    for name in (directive.get("priority_build_order") or []):
        r = resolve(name, cat, prefix)
        if r and r[0] == "building" and r[1][2:6] not in have:
            return ("build", r[1], name)
    # --- FOLLOW the army composition (named units -> target counts) ---
    for name, target in (directive.get("army_composition") or {}).items():
        if not isinstance(target, int) or target <= 0:
            continue
        r = resolve(name, cat, prefix)
        if r and r[0] in ("unit", "infantry") and has_factory(r[0], have) and own_count(obs, cat, r[1]) < target:
            return ("train", r[1], f"{name} {own_count(obs, cat, r[1])}/{target}")
    # --- orders fulfilled -> execute stance ---
    stance = (directive.get("stance") or "").lower()
    if stance == "aggressive":
        return ("attack", None, "push out (stance)")
    if stance == "defensive":
        return ("defend", None, "hold (stance)")
    return ("scout", None, "scout/expand (stance)")


def execute(action, obs, act, cat, ctx, snap):
    kind, fid, label = action
    if kind == "deploy":
        return sp.execute("DEPLOY_MCV", None, obs, act, cat, ctx, snap)
    if kind in ("engage", "defend"):
        return sp.execute("DEFEND", None, obs, act, cat, ctx, snap)
    if kind == "attack":
        return sp.execute("ATTACK", None, obs, act, cat, ctx, snap)
    if kind == "scout":
        return sp.execute("SCOUT", None, obs, act, cat, ctx, snap)
    if kind == "build":
        return sp._build(act, obs, cat, ctx["anchor"], fid)
    if kind == "train":
        e = cat.by_id[fid]
        return f"{fid}: {bb.produce_retry(act, CAT_RTTI[e['category']], e['index'])}"
    return "noop"


def print_orders(model, tick, brief, d):
    print("\n" + "=" * 80)
    print(f" COMMANDER ({model}) -- ORDERS @ tick {tick}")
    print("-" * 80)
    print("  ASSESSMENT: " + str(d.get("assessment", "?")))
    print("  BUILD ORDER: " + str(d.get("priority_build_order", "?")))
    print("  ARMY COMP  : " + str(d.get("army_composition", "?")))
    print("  STANCE     : " + str(d.get("stance", "?")) + "   STRATEGY: " + str(d.get("strategy", "?")))
    print("  REASONING  : " + str(d.get("reasoning", "?")))
    print("=" * 80 + "\n")


def briefing(s, snap):
    side = {0: "Allied", 1: "Soviet", 4: "Yuri"}.get(s.get("side_index", -1), "?")
    return "\n".join([
        f"Faction: {side}   Credits: {s.get('credits', 0)}   Power surplus: {snap['power_surplus']}",
        f"Your base: {snap['n_bld']} buildings (war factory={snap['has_weap']} radar={snap['has_radar']} "
        f"defenses={snap['n_defense']}); army: {snap['n_army']} vehicles",
        f"Enemy VISIBLE: {snap['n_enemy']} (artillery={len(snap['enemy_arty'])} air={len(snap['enemy_air'])} "
        f"at-base={len(snap['enemy_near'])} enemy-buildings-seen={len(snap['enemy_buildings'])})",
    ])


LOG = os.path.join(os.path.dirname(__file__), "data", "order_follower.log")


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def main(model="gemma4", max_ticks=60, launch=True):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(LOG, "w", encoding="utf-8"))
    if launch:
        sys.path.insert(0, os.path.dirname(__file__))
        from commander_build import launch_game
        print(f"launching match; LLM commander = {model}; executor = ORDER-FOLLOWER (carries out the directive)")
        launch_game()
        time.sleep(2)
    obs, act, cat = bb.connect(), ActWriter(), Catalog()
    lut = sp._lookup(cat)
    print("waiting for the match to initialize...")
    for _ in range(120):
        s = obs.read_state()
        if s and s["owned_units"] > 0 and s["n_enemy"] < 200:
            break
        time.sleep(1)

    # a sensible default directive to FOLLOW until the LLM's (slow, playbook-grounded) first order lands
    shared = {"directive": {"priority_build_order": ["Power Plant", "Refinery", "Barracks", "War Factory", "Radar"],
                            "army_composition": {"Rhino Tank": 8, "Terror Drone": 3}, "stance": "defensive"},
              "pending": None}
    lock = threading.Lock()
    stop = threading.Event()

    def commander_loop():
        while not stop.is_set():
            s2 = obs.read_state()
            if s2:
                snap2 = sp.snapshot(obs, cat, lut, sp.make_ctx(obs, cat))
                try:
                    d = command(briefing(s2, snap2), model=model)
                    with lock:
                        shared["directive"] = d
                        shared["pending"] = (s2.get("frame_seq"), d)
                except Exception as e:
                    print(f"  [commander error: {e}]")
            stop.wait(5)        # the LLM is slow w/ the full playbook; re-ask as fast as it can answer

    th = threading.Thread(target=commander_loop, daemon=True)
    th.start()
    print(">>> WATCH: executor FOLLOWS gemma4's orders — it builds exactly what the directive names "
          "and fields the army composition it specifies\n")

    for tick in range(max_ticks):
        s = obs.read_state()
        if not s:
            time.sleep(0.5)
            continue
        ctx = sp.make_ctx(obs, cat)
        snap = sp.snapshot(obs, cat, lut, ctx)
        with lock:
            directive = dict(shared["directive"])
            pending = shared["pending"]
            shared["pending"] = None
        if pending:
            print_orders(model, tick, None, pending[1])
        action = decide(directive, obs, cat, ctx, snap)
        res = execute(action, obs, act, cat, ctx, snap)
        threat = (f" THREAT[arty={len(snap['enemy_arty'])} air={len(snap['enemy_air'])} "
                  f"near={len(snap['enemy_near'])}]") if (snap["enemy_arty"] or snap["enemy_near"] or snap["enemy_air"]) else ""
        print(f"[{tick:2d}] B={s['owned_buildings']} U={s['owned_units']} E={s['n_enemy']}{threat}"
              f"  ->  FOLLOW: {action[2]}  ->  {res}")
        time.sleep(1)

    stop.set()
    th.join(timeout=2)
    obs.close()
    act.close()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        cat = Catalog()
        print("resolve() name->catalog (Soviet prefix NA):")
        for nm in ["Power Plant", "War Factory", "Radar", "Tesla Coil", "Flak Cannon", "Sentry Gun",
                   "Terror Drone", "Rhino Tank", "battle tank", "Flak Track", "Apocalypse", "V3", "Conscript"]:
            print(f"  {nm:14s} -> {resolve(nm, cat, 'NA')}")
    else:
        main(model=sys.argv[1] if len(sys.argv) > 1 else "gemma4")
