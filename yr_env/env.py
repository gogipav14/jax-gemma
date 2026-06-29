"""Gym-style environment wrapping the YR bridge (OBS read + ACT write) — the Phase-3
foundation for RL. Observation = a fixed feature vector; action = a BridgeAction dict.

NOTE on compute: this CPU-only, non-headless box is for *building/validating* the env, not
large-scale training. Real self-play needs a GPU + many parallel (minimized) instances; the env
is written to be reusable there. See docs/ for the training plan.
"""
from __future__ import annotations

import time

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except Exception:                       # env still usable as a plain reset/step object
    _HAS_GYM = False

from read_obs import ObsReader
from write_act import ActWriter

GLOBAL_KEYS = ("credits", "power_output", "power_drain", "side_index", "owned_units",
               "owned_buildings", "owned_infantry", "owned_aircraft", "owned_navy")
FAC_CATS = ("Building", "Unit", "Infantry", "Aircraft", "Naval")
OBS_DIM = len(GLOBAL_KEYS) + 3 + len(FAC_CATS)   # globals + (n_own,n_enemy,n_factory) + active-factory/category


def encode_obs(state: dict, factories: list) -> np.ndarray:
    g = [float(state.get(k, 0)) for k in GLOBAL_KEYS]
    counts = [float(state.get("n_own", 0)), float(state.get("n_enemy", 0)), float(state.get("n_factory", 0))]
    cats = {c: 0 for c in FAC_CATS}
    for f in factories:
        if f.get("active") and f.get("category") in cats:
            cats[f["category"]] += 1
    return np.array(g + counts + [float(cats[c]) for c in FAC_CATS], dtype=np.float32)


def score(state: dict) -> float:
    """A simple non-cheating score proxy: economy + military strength."""
    return (state.get("owned_buildings", 0) * 100.0 + state.get("owned_units", 0) * 50.0
            + state.get("owned_infantry", 0) * 20.0 + state.get("credits", 0) * 0.001)


class YREnv(gym.Env if _HAS_GYM else object):
    """Requires a live skirmish (Release bridge DLL). reset() connects; step() injects an action.
    action is a dict with BridgeAction fields: type, category_rtti, is_naval, type_id, cell_x,
    cell_y, target_unique."""

    metadata = {"render_modes": []}

    def __init__(self):
        if _HAS_GYM:
            self.observation_space = spaces.Box(-1e12, 1e12, (OBS_DIM,), np.float32)
            self.action_space = spaces.Dict({
                "type": spaces.Discrete(11),
                "category_rtti": spaces.Discrete(64),
                "is_naval": spaces.Discrete(2),
                "type_id": spaces.Discrete(512),
                "cell_x": spaces.Box(0, 512, (), np.int32),
                "cell_y": spaces.Box(0, 512, (), np.int32),
                "target_unique": spaces.Box(-1, 2**31 - 1, (), np.int64),
            })
        self.obs_r = None
        self.act_w = None
        self._last = 0.0

    def _read(self):
        for _ in range(50):
            s = self.obs_r.read_state()
            if s:
                return s, self.obs_r.read_factories()
            time.sleep(0.02)
        return None, []

    def reset(self, *, seed=None, options=None):
        self.obs_r = ObsReader()
        self.act_w = ActWriter()
        s, f = self._read()
        self._last = score(s) if s else 0.0
        ob = encode_obs(s, f) if s else np.zeros(OBS_DIM, np.float32)
        return (ob, {}) if _HAS_GYM else ob

    def step(self, action: dict):
        pkt = self.act_w._pack(
            int(action.get("type", 0)), int(action.get("category_rtti", 0)),
            int(action.get("is_naval", 0)), 0, int(action.get("type_id", 0)),
            int(action.get("cell_x", 0)), int(action.get("cell_y", 0)),
            int(action.get("target_unique", -1)))
        result = self.act_w.send(pkt)
        s, f = self._read()
        if not s:                                   # match ended / process gone
            z = np.zeros(OBS_DIM, np.float32)
            return (z, 0.0, True, False, {"result": result}) if _HAS_GYM else (z, 0.0, True, {"result": result})
        ob = encode_obs(s, f)
        sc = score(s)
        reward = sc - self._last
        self._last = sc
        done = bool(s.get("status", 0) & 0x4)       # loss bit (Phase-3 reward wiring)
        info = {"result": result, "score": sc}
        return (ob, reward, done, False, info) if _HAS_GYM else (ob, reward, done, info)

    def close(self):
        if self.obs_r:
            self.obs_r.close()
        if self.act_w:
            self.act_w.close()


if __name__ == "__main__":
    # self-test (no game): validate obs encoding + reward shaping on synthetic states
    print(f"gymnasium available: {_HAS_GYM}; OBS_DIM={OBS_DIM}")
    s0 = {"credits": 10000, "power_output": 0, "power_drain": 0, "side_index": 6,
          "owned_units": 6, "owned_buildings": 0, "owned_infantry": 0, "owned_aircraft": 0,
          "owned_navy": 0, "n_own": 6, "n_enemy": 2, "n_factory": 0, "status": 0}
    s1 = dict(s0, owned_buildings=5, owned_units=12, n_factory=5)
    f1 = [{"active": True, "category": "Unit"}, {"active": True, "category": "Building"}]
    print("obs(s0):", encode_obs(s0, []))
    print("obs(s1):", encode_obs(s1, f1))
    print(f"reward s0->s1: {score(s1) - score(s0):.1f}  (built 5 buildings + 6 units => positive)")
