"""Train the warm-start policy from ALL recorded real traces (Phase 4b -> 4a bridge).

Globs yr_env/data/traces/trace_*.npz (the live recordings from record_traces.py), pools their
legal (result==OK) decisions, trains the JAX policy by behavioral cloning, and saves the params.
This is the competent, non-cheating warm start the self-play league (Phase 5) starts from.

    python policy/bc_from_traces.py            # train on every recorded trace
    python policy/bc_from_traces.py --steps 800
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import jax

sys.path.insert(0, os.path.dirname(__file__))
from train_bc import load_traces, train, accuracy   # noqa: E402

TRACES = os.path.join(os.path.dirname(__file__), "..", "yr_env", "data", "traces", "trace_*.npz")
OUT = os.path.join(os.path.dirname(__file__), "bc_policy.pkl")


def main():
    steps = 400
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    paths = sorted(glob.glob(TRACES))
    if not paths:
        print(f"no recorded traces at {TRACES}\n"
              f"  record one first:  PYTHONPATH=yr_env;commander python yr_env/record_traces.py")
        return
    obs, y = load_traces(paths)
    classes = sorted(set(int(v) for v in y))
    print(f"pooled {obs.shape[0]} legal decisions from {len(paths)} trace(s); action_types={classes}")
    params = train(steps=steps, data=(obs, y))
    with open(OUT, "wb") as f:
        pickle.dump(jax.tree.map(lambda a: a.tolist(), params), f)
    print(f"warm-start policy saved -> {OUT}  (final train acc on real data above)")


if __name__ == "__main__":
    main()
