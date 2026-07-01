"""Drive a LIVE Yuri's Revenge game with the v2 spectral-wired brain (non-cheating, player path).

Loads the v2 warm-start (bc_fused_v2.pkl), launches a skirmish, and each step reads the fog-honored
obs (grid + roster + scalars), derives the world-model bits (buildable/power/threat), lets the brain
DECIDE a macro, and injects it via the player event path. Prints a live readout of what the brain
sees and does -- so you can watch its exact world-model line up with the screen.

    PYTHONPATH=yr_env  python policy/play_live_v2.py [n_steps]
"""
from __future__ import annotations

import os
import pickle
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "yr_env"))
sys.path.insert(0, os.path.join(ROOT, "policy"))
sys.path.insert(0, os.path.join(ROOT, "commander"))       # the env's launcher imports commander
import net_spectral as ns                                 # noqa: E402
import spectral_logic as sl                               # noqa: E402
from rl_env import YRLearnEnv, encode                     # noqa: E402
from mock_env import ACTION_NAME                          # noqa: E402

PKL = os.path.join(ROOT, "policy", "bc_fused_v2.pkl")


def main(n_steps=60):
    if not os.path.exists(PKL):
        print("no bc_fused_v2.pkl -- run policy/offline_bc_v2.py first"); return
    with open(PKL, "rb") as f:
        p = pickle.load(f)
    pcoef, ccoef = sl.build_spectrum(), sl.build_counter_spectrum()
    cname = ["none", "ANTI_ARMOR", "MAIN_BATTLE", "ANTI_AIR"]

    print("=== launching live YR; the v2 spectral brain takes the player house ===")
    env = YRLearnEnv(launch=True)
    obs = env.reset()
    won = razed = 0
    try:
        for t in range(n_steps):
            pos = env.pos
            grid, (et, em) = env.grid(), env.entities()
            bb = sl.bits_from_buildings(pos.own_buildings)
            eb = sl.bits_from_threats(pos)
            pw = 1.0 if pos.power_surplus >= 0 else 0.0
            if pos.anchor is None:
                a, val = 1, 0.0                    # LAW: the MCV must deploy before anything (forced)
            else:
                a, val = ns.decide(p, grid, obs, et, em, bb, eb, pw)

            # the brain's EXACT world-model (the frozen laws) for the readout
            buildable = [sl.BUILD_ROLES[i] for i in range(len(sl.BUILD_ROLES))
                         if sl.predict(pcoef, bb[None])[0, i] > 0.5]
            threats = [t_.role for t_ in pos.threats]
            answer = cname[int(sl.predict_counter(ccoef, eb[None])[0])] if threats else "-"
            power = "OK " if pw else "LOW"
            print(f"  t{t:2d} {ACTION_NAME[a]:12s} | power {power} surplus {pos.power_surplus:+4d} "
                  f"| buildable {','.join(buildable) or '-':22s} | threats {','.join(threats) or '-':16s}->{answer:11s} "
                  f"| V={pos.V:+.2f} val={val:+.2f}")

            obs, r, done, info = env.step(a)
            razed += max(0, r)
            if done:
                won = r > 0
                print(f"  -- game over at t{t}: {'WON' if won else 'lost'} --")
                break
    finally:
        env.close()
    print(f"=== done: reward-so-far {razed:+.1f} ===")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
