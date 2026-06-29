"""Train the macro-action policy by behavioral cloning: relabel recorded traces into macro ids
(yr_env/macro.py) and fit obs -> macro. Saves macro_policy.pkl for the live inference loop
(yr_env/play_policy.py). Same JAX trainer as train_bc; the policy's 11 outputs cover the 10 macros.

    python policy/train_macro.py [--steps 800]
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "yr_env"))
from train_bc import train                          # noqa: E402
from macro import relabel, MACROS                   # noqa: E402
from catalog import Catalog                         # noqa: E402

TRACES = os.path.join(os.path.dirname(__file__), "..", "yr_env", "data", "traces", "trace_*.npz")
OUT = os.path.join(os.path.dirname(__file__), "macro_policy.pkl")


def load_macro(paths, cat):
    O, Y = [], []
    for p in paths:
        d = np.load(p)
        mask = d["result"] == 0
        obs, act = np.asarray(d["obs"])[mask], np.asarray(d["act"])[mask]
        for i in range(len(obs)):
            O.append(obs[i])
            Y.append(relabel(cat, act[i].tolist()))
    return jnp.asarray(np.array(O), jnp.float32), jnp.asarray(np.array(Y), "int32")


def main():
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 600
    paths = sorted(glob.glob(TRACES))
    if not paths:
        print(f"no traces at {TRACES}; record with yr_env/record_traces.py")
        return
    cat = Catalog()
    obs, y = load_macro(paths, cat)
    dist = {MACROS[i]: int((y == i).sum()) for i in sorted(set(int(v) for v in y))}
    print(f"pooled {obs.shape[0]} decisions from {len(paths)} trace(s); macro distribution: {dist}")
    params = train(steps=steps, data=(obs, y))
    with open(OUT, "wb") as f:
        pickle.dump(jax.tree.map(lambda a: np.asarray(a), params), f)   # np leaves keep pytree structure
    print(f"macro policy saved -> {OUT}")


if __name__ == "__main__":
    main()
