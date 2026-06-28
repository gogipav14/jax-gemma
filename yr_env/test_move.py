"""Live smoke test of the ACT path: remotely move one of the agent's units.

Reads an own vehicle from OBS, injects a GROUP_MOVE toward another own unit's cell
(guaranteed valid ground), and confirms the unit relocates. Proves DLL action
injection end-to-end. Run while a skirmish with the bridge DLL is live.
"""
import time
from read_obs import ObsReader
from write_act import ActWriter


def connect(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return ObsReader()
        except OSError:
            time.sleep(1)
    raise OSError("OBS mapping never appeared")


def main():
    print("waiting for the match to load...")
    obs, act = connect(), ActWriter()
    units, own = [], []
    for _ in range(90):                       # wait for the agent house to populate units
        own = obs.read_own()
        units = [u for u in own if u["category"] == "Unit" and (u["x"] or u["y"])]
        if units and len(own) >= 2:
            break
        time.sleep(1)
    if len(units) < 1 or len(own) < 2:
        print("not enough own units:", own[:4]); return
    mover = units[0]
    # target = some OTHER own unit's cell (valid, reachable terrain), a few cells away
    target = next((u for u in own if u["unique_id"] != mover["unique_id"]
                   and (abs(u["x"] - mover["x"]) + abs(u["y"] - mover["y"])) > 3), own[-1])
    tx, ty = target["x"], target["y"]
    start = (mover["x"], mover["y"])
    print(f"Mover uid={mover['unique_id']} type={mover['type_id']} at {start} -> target ({tx},{ty})")

    r = act.move(mover["unique_id"], tx, ty)
    print("inject GROUP_MOVE result:", r)

    moved = False
    for _ in range(24):
        time.sleep(0.5)
        cur = next((e for e in obs.read_own() if e["unique_id"] == mover["unique_id"]), None)
        if not cur:
            continue
        if (cur["x"], cur["y"]) != start:
            print(f"  MOVED to ({cur['x']},{cur['y']}) from {start}  -> action injection WORKS")
            moved = True
            break
    if not moved:
        print("  unit did not move (target may be blocked, or not the agent's unit)")
    obs.close(); act.close()


if __name__ == "__main__":
    main()
