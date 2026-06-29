"""Phase 4c: the LEARNED policy drives a live YR match.

Each decision tick: read OBS -> normalize -> the macro policy picks a macro -> execute_macro carries
it out (non-cheating, via the player command path). The NN is in the driver's seat for the macro
decisions; placement/timing/targeting are resolved from state.

v1: trained on limited traces, so it largely reproduces the learned build — the point is the CLOSED
LOOP (policy -> action -> game), which is the exact inference path Phase-5 self-play RL will drive.

    PYTHONPATH=yr_env;commander  python yr_env/play_policy.py    # live match required
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import jax.numpy as jnp

from write_act import ActWriter
from catalog import Catalog
from env import encode_obs, normalize
from macro import MACROS, N_MACRO, execute_macro, build_context
from build_base import connect

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "policy"))
from policy import forward                           # noqa: E402

POLICY = os.path.join(os.path.dirname(__file__), "..", "policy", "macro_policy.pkl")


def pick_macro(params, ob):
    logits, _ = forward(params, jnp.asarray(ob))
    return int(jnp.argmax(logits[:N_MACRO]))         # restrict to the macro outputs


def main(max_macros=20, launch=True):
    if not os.path.exists(POLICY):
        print(f"no trained macro policy at {POLICY}; run policy/train_macro.py first")
        return
    with open(POLICY, "rb") as f:
        params = pickle.load(f)
    if launch:
        sys.path.insert(0, os.path.dirname(__file__))
        from commander_build import launch_game
        print("launching a fresh match for the policy to play...")
        launch_game()
        time.sleep(2)
    obs, act, cat = connect(), ActWriter(), Catalog()
    # Wait for the match to initialize (player units spawned) so the FIRST decision sees the same
    # distribution the trace started from (B=0, U>0). Otherwise step 0 reads U=0/E=256 transients.
    print("waiting for the match to initialize (units to spawn)...")
    for _ in range(120):
        s = obs.read_state()
        if s and s["owned_units"] > 0 and s["n_enemy"] < 200:
            break
        time.sleep(1)
    print(">>> WATCH: the NEURAL POLICY now drives — deploy, build, train, attack — all its choices\n")
    attacked, stuck, last = False, 0, None
    for step in range(max_macros):
        s = obs.read_state()
        if not s:
            time.sleep(0.5)
            continue
        ob = normalize(encode_obs(s, obs.read_factories()))
        mid = pick_macro(params, ob)
        ctx = build_context(obs, cat)
        res = execute_macro(mid, obs, act, cat, ctx)
        print(f"[{step:2d}] B={s['owned_buildings']} U={s['owned_units']} "
              f"E={s['n_enemy']} cr={s['credits']}  ->  NN: {MACROS[mid]}  ->  {res}")
        if MACROS[mid] == "ATTACK" and res.startswith("attack"):
            attacked = True
        if attacked and s["owned_buildings"] >= 6:
            print("\nbase built + army committed — NN-driven episode complete.")
            break
        sig = (mid, s["owned_buildings"], s["owned_units"])   # distribution-shift / stuck guard
        stuck = stuck + 1 if sig == last else 0
        last = sig
        if stuck >= 4:
            print(f"\npolicy stuck on {MACROS[mid]} (state unchanged x{stuck}) — stopping. "
                  f"(BC distribution-shift: 1 trace isn't enough; more traces / RL fixes this.)")
            break
        time.sleep(1)
    obs.close()
    act.close()


if __name__ == "__main__":
    main()
