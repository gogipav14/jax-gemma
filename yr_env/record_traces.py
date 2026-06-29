"""Phase 4b: record REAL (obs -> action) traces while the scripted/LLM-driven agent plays,
producing a behavioral-cloning dataset.

A RecordingActWriter subclasses ActWriter and overrides send() — the single funnel every action
(produce/place/deploy/attack_move/...) passes through — to snapshot the current OBS, the action
taken, and the result. We then run the existing build_base agent with this writer injected, so the
agent's logic is untouched and we capture its entire decision stream. The trace is saved to an
.npz that policy/train_bc.py imitates -> a competent, non-cheating warm start.

    PYTHONPATH=yr_env;commander  python yr_env/record_traces.py [model]    # needs a live match
    python yr_env/record_traces.py --selftest                              # offline serialization test
"""
from __future__ import annotations

import os
import struct
import sys
import time

import numpy as np

import contract
from env import encode_obs, OBS_DIM
from write_act import ActWriter
import build_base

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "traces")
# act-row columns kept for BC (col 0 is the action_type the policy head predicts):
ACT_COLS = ("action_type", "category_rtti", "type_id", "cell_x", "cell_y", "target_unique")


def action_row(action_bytes: bytes):
    """Unpack a BridgeAction into the 6 BC columns."""
    a = struct.unpack("<" + contract.ACTION_FMT, action_bytes)
    # a = (atype, category_rtti, is_naval, stance, type_id, cell_x, cell_y, target_unique, group_id)
    return [a[0], a[1], a[4], a[5], a[6], a[7]]


def save_trace(obs_log, act_log, res_log, path):
    obs = np.asarray(obs_log, np.float32).reshape(-1, OBS_DIM)
    act = np.asarray(act_log, np.int32).reshape(-1, len(ACT_COLS))
    res = np.asarray(res_log, np.int32).reshape(-1)
    np.savez(path, obs=obs, act=act, result=res)
    return obs, act, res


class RecordingActWriter(ActWriter):
    """ActWriter that logs (obs, action, result) for every action sent. Inject into build_base."""

    def __init__(self, obs_reader):
        super().__init__()
        self.obs_r = obs_reader
        self.obs_log, self.act_log, self.res_log = [], [], []

    def _snapshot_obs(self):
        for _ in range(20):
            s = self.obs_r.read_state()
            if s:
                return encode_obs(s, self.obs_r.read_factories())
            time.sleep(0.01)
        return np.zeros(OBS_DIM, np.float32)

    def send(self, action_bytes, wait_s=2.0):
        ob = self._snapshot_obs()                       # state at decision time (pre-action)
        result = super().send(action_bytes, wait_s)
        self.obs_log.append(ob)
        self.act_log.append(action_row(action_bytes))
        self.res_log.append(result[0] if result else -1)
        return result

    def save(self, path):
        _, _, res = save_trace(self.obs_log, self.act_log, self.res_log, path)
        print(f"  saved {len(res)} transitions ({int((res == 0).sum())} OK) -> {path}")
        return path


def next_trace_path():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len([f for f in os.listdir(OUT_DIR) if f.startswith("trace_") and f.endswith(".npz")])
    return os.path.join(OUT_DIR, f"trace_{n + 1:04d}.npz")


def main():
    model = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "gemma4"
    from commander import think, SAMPLE_OBS            # commander/ on PYTHONPATH
    from commander_build import launch_game, ROSTER

    print(f"=== Commander ({model}) deciding (grounded), then recording the agent's trace ===")
    d = think(SAMPLE_OBS, model=model, roster=ROSTER)
    order = d.get("priority_build_order", d.get("build_order", [])) or []
    army = d.get("army_composition", {}) or {}
    n_tanks = min(6, sum(v for v in army.values() if isinstance(v, int) and v > 0)) or 4
    attack = (d.get("stance") or "").lower() != "defensive"
    print(f"  order={order}  n_tanks={n_tanks}  attack={attack}")

    launch_game()
    time.sleep(2)
    obs = build_base.connect()
    rec = RecordingActWriter(obs)
    build_base.main(order_override=order, n_tanks=n_tanks, attack=attack, obs=obs, act=rec)
    rec.save(next_trace_path())
    rec.close()
    obs.close()


def selftest():
    """Offline: simulate a few decisions, save, reload, sanity-check (no game / no mapping)."""
    fake_obs = np.arange(OBS_DIM, dtype=np.float32)
    pack = lambda *a, **k: ActWriter._pack(None, *a, **k)   # _pack ignores self
    samples = [
        (pack(contract.ActionType.DEPLOY, target_unique=123), 0),
        (pack(contract.ActionType.PRODUCE, category_rtti=7, type_id=303), 2),   # rejected (prereq)
        (pack(contract.ActionType.PRODUCE, category_rtti=7, type_id=303), 0),
        (pack(contract.ActionType.PLACE, category_rtti=6, type_id=303, cell_x=50, cell_y=60), 0),
        (pack(contract.ActionType.GROUP_ATTACK, target_unique=99, cell_x=200, cell_y=110), 0),
    ]
    obs_log = [fake_obs for _ in samples]
    act_log = [action_row(b) for b, _ in samples]
    res_log = [c for _, c in samples]
    path = os.path.join(OUT_DIR, "selftest.npz")
    os.makedirs(OUT_DIR, exist_ok=True)
    save_trace(obs_log, act_log, res_log, path)
    d = np.load(path)
    print(f"saved + reloaded: obs{d['obs'].shape} act{d['act'].shape} result{d['result'].shape}")
    print(f"  action_types: {d['act'][:, 0].tolist()}  (10=DEPLOY 1=PRODUCE 2=PLACE 6=ATTACK)")
    print(f"  results:      {d['result'].tolist()}  (0=OK 2=REJECTED)")
    ok = int((d["result"] == 0).sum())
    print(f"  BC keeps OK transitions: {ok}/{len(d['result'])}  -> trace recorder serialization OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
