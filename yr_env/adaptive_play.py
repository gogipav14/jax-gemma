"""Adaptive agent: a FAST small LLM (qwen3.5:0.8b) reads the live battle every ~15s and ADAPTS —
{read, focus, counter, stance} — and the executor adjusts a sensible baseline accordingly. The key
move: when the LLM names a COUNTER (Terror Drone vs V3, Flak vs air), the executor BUILDS it. This
is adaptation with insight, not a script — and it's fast enough to actually run in the loop.

    PYTHONPATH=yr_env;commander  python yr_env/adaptive_play.py [model]     # live match required
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
import order_follower as of

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "commander"))
from commander import adapt                              # noqa: E402

LOG = os.path.join(os.path.dirname(__file__), "data", "adaptive_play.log")


def briefing(s, snap):
    side = {0: "Allied", 1: "Soviet", 4: "Yuri"}.get(s.get("side_index", -1), "?")
    return (f"{side}. Base {snap['n_bld']} buildings (war factory={snap['has_weap']} radar={snap['has_radar']}). "
            f"Army {snap['n_army']}. Enemy visible {snap['n_enemy']} (artillery={len(snap['enemy_arty'])} "
            f"air={len(snap['enemy_air'])} at-base={len(snap['enemy_near'])}).")


def decide_adaptive(a, obs, cat, ctx, snap):
    """Adjust a baseline by the LLM's adaptation a={read,focus,counter,stance}. Returns (kind, fid, label)."""
    prefix = ctx["prefix"]
    if snap["n_bld"] == 0:
        return ("deploy", None, "deploy MCV")
    have = of.own_suffixes(obs, cat)
    if "POWR" not in have:
        return ("build", prefix + "POWR", "power (floor)")
    if "REFN" not in have:
        return ("build", prefix + "REFN", "refinery (floor)")
    if "WEAP" not in have:
        return ("build", prefix + "WEAP", "war factory (floor)")

    # --- ADAPT: build the COUNTER the LLM named for the current threat ---
    counter = (a.get("counter") or "").strip()
    if counter and counter.lower() not in ("none", "null", "n/a", ""):
        r = of.resolve(counter, cat, prefix)
        if r:
            if r[0] == "building" and r[1][2:6] not in have:
                return ("build", r[1], f"COUNTER {counter}")
            if r[0] in ("unit", "infantry") and of.has_factory(r[0], have) and of.own_count(obs, cat, r[1]) < 4:
                return ("train", r[1], f"COUNTER {counter} {of.own_count(obs, cat, r[1])}/4")

    # --- reactive safety: a real push gets engaged regardless ---
    if len(snap["enemy_near"]) >= 5 and snap["n_army"] >= 3:
        return ("engage", None, "defend base (reactive)")

    # --- FOCUS shapes the baseline build ---
    focus = (a.get("focus") or "").lower()
    if focus == "defense" and snap["n_defense"] < 3:
        return ("build_defense", None, "focus: base defense")
    if focus == "tech" and not snap["has_radar"]:
        return ("build", prefix + "RADR", "focus: radar")
    if focus == "economy" and snap["n_refn"] < 3:
        return ("build", prefix + "REFN", "focus: economy")
    if snap["n_army"] < 10:
        return ("train", bb.MAIN_TANK.get(prefix, "HTNK"), f"army {snap['n_army']}/10")

    # --- STANCE drives what the standing army does ---
    stance = (a.get("stance") or "").lower()
    if stance == "attack":
        return ("attack", None, "stance: attack")
    if stance == "defend":
        return ("defend", None, "stance: defend")
    return ("scout", None, "stance: expand/scout")


def execute_adaptive(action, obs, act, cat, ctx, snap):
    if action[0] == "build_defense":
        return sp.execute("BUILD_DEFENSE", None, obs, act, cat, ctx, snap)
    return of.execute(action, obs, act, cat, ctx, snap)


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


def main(model="qwen3.5:0.8b", max_ticks=70, launch=True):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(LOG, "w", encoding="utf-8"))
    if launch:
        sys.path.insert(0, os.path.dirname(__file__))
        from commander_build import launch_game
        print(f"launching match; ADAPTIVE commander = {model} (fast, in-the-loop); executor adapts to it")
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

    shared = {"a": {"read": "(thinking...)", "focus": "economy", "counter": "none", "stance": "expand"}}
    lock = threading.Lock()
    stop = threading.Event()

    def commander_loop():
        while not stop.is_set():
            s2 = obs.read_state()
            if s2:
                snap2 = sp.snapshot(obs, cat, lut, sp.make_ctx(obs, cat))
                try:
                    a = adapt(briefing(s2, snap2), model=model)
                    with lock:
                        shared["a"] = a
                    print(f"\n  >>> ADAPT [{a.get('focus')}/{a.get('stance')}] counter={a.get('counter')}"
                          f"  ::  {a.get('read')}\n")
                except Exception as e:
                    print(f"  [adapt error: {e}]")
            stop.wait(3)

    th = threading.Thread(target=commander_loop, daemon=True)
    th.start()
    print(">>> WATCH: a fast 0.8B reads the battle and ADAPTS; the executor builds the COUNTER it "
          "names (Terror Drones vs artillery, Flak vs air) and acts on its stance\n")

    for tick in range(max_ticks):
        s = obs.read_state()
        if not s:
            time.sleep(0.5)
            continue
        ctx = sp.make_ctx(obs, cat)
        snap = sp.snapshot(obs, cat, lut, ctx)
        with lock:
            a = dict(shared["a"])
        action = decide_adaptive(a, obs, cat, ctx, snap)
        res = execute_adaptive(action, obs, act, cat, ctx, snap)
        threat = (f" THREAT[arty={len(snap['enemy_arty'])} air={len(snap['enemy_air'])} "
                  f"near={len(snap['enemy_near'])}]") if (snap["enemy_arty"] or snap["enemy_near"] or snap["enemy_air"]) else ""
        print(f"[{tick:2d}] B={s['owned_buildings']} U={s['owned_units']} E={s['n_enemy']}{threat}"
              f"  ->  {action[2]}  ->  {res}")
        time.sleep(1)

    stop.set()
    th.join(timeout=2)
    obs.close()
    act.close()


if __name__ == "__main__":
    main(model=sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:0.8b")
