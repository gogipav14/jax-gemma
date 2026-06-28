"""Live test: deploy the agent's MCV into a Construction Yard via the ACT bridge.

Tests the new DEPLOY action (a hand-built EventType::Deploy event). Watch the screen:
the MCV should transform into a Construction Yard.
"""
import time

from read_obs import ObsReader
from write_act import ActWriter
from catalog import Catalog


def connect(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return ObsReader()
        except OSError:
            time.sleep(1)
    raise OSError("OBS mapping never appeared")


def n_buildings(obs):
    return len([u for u in obs.read_own() if u["category"] == "Building"])


def main():
    obs, act, cat = connect(), ActWriter(), Catalog()
    mcv_ids = {e["index"] for e in cat.by_id.values()
               if e["category"] == "unit" and ("MCV" in e["id"] or "MCV" in e["ui_name"])}
    print("MCV type_ids in catalog:", sorted(mcv_ids))

    mcv = None
    for _ in range(90):
        for u in obs.read_own():
            if u["category"] == "Unit" and u["type_id"] in mcv_ids:
                mcv = u
                break
        if mcv:
            break
        time.sleep(1)
    if not mcv:
        print("no MCV among own units yet"); return

    b0 = n_buildings(obs)
    print(f"MCV uid={mcv['unique_id']} type={mcv['type_id']} at ({mcv['x']},{mcv['y']}); buildings now={b0}")
    print(">>> WATCH: the MCV should deploy into a Construction Yard (a building appears).")
    r = act.deploy(mcv["unique_id"])
    print("deploy result:", r)

    for _ in range(40):
        time.sleep(0.5)
        b = n_buildings(obs)
        if b > b0:
            print(f"  DEPLOYED — buildings {b0} -> {b}. Construction Yard is up! DEPLOY works.")
            break
    else:
        print("  no new building appeared (MCV did not deploy — needs a different deploy mechanism)")
    obs.close(); act.close()


if __name__ == "__main__":
    main()
