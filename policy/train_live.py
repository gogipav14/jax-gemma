"""Live training start on the real game (rl_env). Step 1 of the plan: warm-start the learner by BC
from the baseline-derived TEACHER (collected on the LIVE env), then WATCH the learner play its first
games. Slow (real matches) but the machinery is proven on the mock (policy/learner.py).

    PYTHONPATH=yr_env;commander  python policy/train_live.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import jax.numpy as jnp
from jax import random

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "yr_env"))
sys.path.insert(0, os.path.join(ROOT, "policy"))
import game_model as gm                                  # noqa: E402
from rl_env import YRLearnEnv, MACROS                    # noqa: E402
from learner import init_params, forward, bc_train       # noqa: E402

LOG = os.path.join(ROOT, "yr_env", "data", "train_live.log")
# macro index for each role-build (mirrors rl_env.MACROS order)
ROLE2BUILD = {gm.POWER: 2, gm.ECONOMY: 3, gm.PROD_INF: 4, gm.PROD_VEH: 5, gm.TECH_RADAR: 6}


def teacher_action(pos: gm.Position) -> int:
    """The nudge (BC initial condition): an AGGRESSIVE baseline — build, mass, then ATTACK. It must
    point toward the GOAL (destroy the opponent), so it only defends a REAL push, never turtles."""
    if pos.anchor is None:
        return 1                                          # deploy MCV
    nb = pos.next_build()
    if nb in ROLE2BUILD:
        return ROLE2BUILD[nb]                             # prereq-correct base
    army = pos.own_units.get(gm.MAIN_BATTLE, 0) + pos.own_units.get(gm.ANTI_ARMOR, 0)
    if any(t.role == gm.ARTILLERY for t in pos.threats) and pos.own_units.get(gm.ANTI_ARMOR, 0) < 3 and pos.has(gm.PROD_VEH):
        return 10                                         # counter artillery with Terror Drones
    if len([t for t in pos.threats if t.at_base]) >= 3 and army >= 2:
        return 14                                         # only a REAL push at the base -> defend
    if pos.has(gm.PROD_VEH) and army < 8:
        return 9                                          # mass a main-battle army
    if army >= 4:
        return 13                                         # ATTACK the opponent (the goal)
    return 0


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


def log_step(tag, tick, pos, macro_idx, reward, info):
    th = "; ".join(f"{t.role}x{t.count}{'@B' if t.at_base else ''}" for t in pos.threats[:2]) or "-"
    print(f"  [{tag} t{tick:2d}] B={pos.own_buildings.get(gm.CONSTRUCTION,0)+sum(v for k,v in pos.own_buildings.items() if k!=gm.CONSTRUCTION)} "
          f"army={sum(pos.own_units.values())} V={pos.V:+.1f} threats[{th}]  ->  {MACROS[macro_idx][0]}:{MACROS[macro_idx][1] or ''}"
          f"  r={reward:+.2f}  ({info.get('result','')})")


def main(bc_games=1, watch_games=2, max_steps=60):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    sys.stdout = _Tee(sys.__stdout__, open(LOG, "w", encoding="utf-8"))
    env = YRLearnEnv(launch=True)

    print("=== STEP 1: the TEACHER (nudge) plays the live game — watch + collect BC demos ===")
    O, A = [], []
    for g in range(bc_games):
        obs = env.reset()
        print(f"\n-- teacher game {g + 1} --")
        for t in range(max_steps):
            a = teacher_action(env.pos)
            O.append(obs)
            A.append(a)
            obs, r, done, info = env.step(a)
            log_step("T", t, env.pos, a, r, info)
            if done:
                print("  (game over)")
                break

    print(f"\n=== STEP 2: BC the learner from {len(O)} teacher decisions (the initial condition) ===")
    from rl_env import OBS_DIM, N_MACRO
    params = bc_train(init_params(random.PRNGKey(0), OBS_DIM, N_MACRO),
                      np.asarray(O, np.float32), np.asarray(A), steps=400)
    logits, _ = forward(params, jnp.asarray(O[0]))
    print(f"  BC done. (sanity: first-state action dist argmax = {MACROS[int(jnp.argmax(logits))][0]})")

    print("\n=== STEP 3: WATCH THE LEARNER play its first games (sampled policy) ===")
    key = random.PRNGKey(1)
    for g in range(watch_games):
        obs = env.reset()
        print(f"\n-- learner game {g + 1} --")
        for t in range(max_steps):
            lg, _ = forward(params, jnp.asarray(obs))
            key, sk = random.split(key)
            a = int(random.categorical(sk, lg))
            obs, r, done, info = env.step(a)
            log_step("L", t, env.pos, a, r, info)
            if done:
                print("  (game over)")
                break
    env.close()


if __name__ == "__main__":
    main(bc_games=1, watch_games=1)
