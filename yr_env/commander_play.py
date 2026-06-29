"""Hierarchical agent: a slow LLM COMMANDER (Ollama) sets the STRATEGY — the *posterior* over the
stock-AI prior — and the fast script_policy executor carries it out tick-by-tick. The commander's
ORDERS are printed in plain language so you can see what the AI is thinking and deciding.

The commander runs in a BACKGROUND THREAD so its ~60-90s CPU latency never freezes the executor:
the agent plays immediately under a default profile, and each time the LLM returns a directive the
executor's profile updates live. The LLM's strategy maps to a script_policy PROFILE (reweighting
the prior's rule categories) — that mapping IS the per-commander posterior. Swap model= for a
different posterior / playstyle over the same shared prior.

    PYTHONPATH=yr_env;commander  python yr_env/commander_play.py [model]    # live match required
"""
from __future__ import annotations

import os
import sys
import threading
import time

from write_act import ActWriter
from catalog import Catalog
from build_base import connect
import script_policy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "commander"))
from commander import command                          # noqa: E402

STRAT2PROFILE = {"rush": "rush", "harass": "rush", "boom": "boom", "tech": "boom",
                 "defend": "turtle", "balanced": "balanced"}
COMMANDER_SECONDS = 20          # re-issue strategic orders ~every N seconds (wall clock)
LOG = os.path.join(os.path.dirname(__file__), "data", "commander_play.log")


class _Tee:
    """Mirror stdout to the console AND a log file, flushing each write (so a watcher window
    and an external reader both see orders live)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def briefing(s, snap):
    side = {0: "Allied", 1: "Soviet", 4: "Yuri"}.get(s.get("side_index", -1), "?")
    base = "complete" if snap["has_weap"] else "still building"
    return "\n".join([
        f"Faction: {side}   Credits: {s.get('credits', 0)}   Power surplus: {snap['power_surplus']}",
        f"Your base ({base}): {snap['n_bld']} buildings | war factory={snap['has_weap']} "
        f"radar={snap['has_radar']} base-defenses={snap['n_defense']}",
        f"Your army: {snap['n_army']} vehicles",
        f"Enemy VISIBLE: {snap['n_enemy']} total — artillery(V3/Prism)={len(snap['enemy_arty'])}, "
        f"aircraft={len(snap['enemy_air'])}, AT YOUR BASE={len(snap['enemy_near'])}",
        "Tactics note: artillery out-ranges base defenses; if you see it build Terror Drones and sortie.",
    ])


def print_orders(model, tick, brief, d, profile):
    w = sp.PROFILES.get(profile, {})
    wt = " ".join(f"{k}x{v}" for k, v in w.items()) or "(even weights)"
    lines = brief.split("\n")
    print("\n" + "=" * 80)
    print(f" COMMANDER ({model}) -- ORDERS @ tick {tick}")
    print("-" * 80)
    print("  SEES:  " + lines[0])
    for ln in lines[1:]:
        print("         " + ln)
    print(f"  ASSESSMENT: {d.get('assessment', '?')}")
    print(f"  STRATEGY  : {d.get('strategy', '?')}   STANCE: {d.get('stance', '?')}"
          f"   ==> executor profile = {profile}  [{wt}]")
    print(f"  OBJECTIVES: {d.get('objectives', '?')}")
    print(f"  REASONING : {d.get('reasoning', '?')}")
    print("=" * 80 + "\n")


def main(model="gemma4", max_ticks=55, launch=True):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(LOG, "w", encoding="utf-8"))
    if launch:
        sys.path.insert(0, os.path.dirname(__file__))
        from commander_build import launch_game
        print(f"launching match; LLM commander = {model}; executor = cheat-free script-prior policy")
        launch_game()
        time.sleep(2)
    obs, act, cat = connect(), ActWriter(), Catalog()
    lut = sp._lookup(cat)
    print("waiting for the match to initialize...")
    for _ in range(120):
        s = obs.read_state()
        if s and s["owned_units"] > 0 and s["n_enemy"] < 200:
            break
        time.sleep(1)

    shared = {"profile": "balanced", "pending": None}      # pending = (brief, directive, profile) to print
    lock = threading.Lock()
    stop = threading.Event()

    def commander_loop():
        """Background: think strategy without ever blocking the executor."""
        while not stop.is_set():
            s2 = obs.read_state()
            if s2:
                ctx2 = sp.make_ctx(obs, cat)
                snap2 = sp.snapshot(obs, cat, lut, ctx2)
                brief = briefing(s2, snap2)
                try:
                    d = command(brief, model=model)
                    strat = (d.get("strategy") or "").lower()
                    prof = STRAT2PROFILE.get(strat, "turtle" if (d.get("stance") or "") == "defensive" else "balanced")
                    with lock:
                        shared["profile"] = prof
                        shared["pending"] = (brief, d, prof)
                except Exception as e:
                    print(f"  [commander error: {e}]")
            stop.wait(COMMANDER_SECONDS)

    th = threading.Thread(target=commander_loop, daemon=True)
    th.start()
    print(">>> WATCH: executor plays NOW under a default plan; gemma4's ORDERS update the strategy "
          "as they arrive (it thinks in the background)\n")

    for tick in range(max_ticks):
        s = obs.read_state()
        if not s:
            time.sleep(0.5)
            continue
        ctx = sp.make_ctx(obs, cat)
        snap = sp.snapshot(obs, cat, lut, ctx)
        with lock:
            profile = shared["profile"]
            pending = shared["pending"]
            shared["pending"] = None
        if pending:                                        # new orders arrived -> show them
            print_orders(model, tick, pending[0], pending[1], pending[2])
        macro, payload, why = sp.decide(snap, profile)
        res = sp.execute(macro, payload, obs, act, cat, ctx, snap)
        threat = (f" THREAT[arty={len(snap['enemy_arty'])} air={len(snap['enemy_air'])} "
                  f"near={len(snap['enemy_near'])}]") if (snap["enemy_arty"] or snap["enemy_near"] or snap["enemy_air"]) else ""
        print(f"[{tick:2d}] ({profile}) B={s['owned_buildings']} U={s['owned_units']} "
              f"E={s['n_enemy']}{threat}  ->  {macro} ({why})  ->  {res}")
        time.sleep(1)

    stop.set()
    th.join(timeout=2)
    obs.close()
    act.close()


if __name__ == "__main__":
    main(model=sys.argv[1] if len(sys.argv) > 1 else "gemma4")
