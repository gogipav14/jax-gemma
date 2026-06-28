"""Autonomous base-builder (non-cheating). Deploy the MCV (if needed), then build
Power Plant -> Refinery -> War Factory: produce via ACT/OutList, wait until the
building factory is ready (IsSuspended), then PLACE at a valid cell near the ConYard.
Faction-adaptive: reads the building-ID prefix from the agent's own Construction Yard.

Run while a skirmish with the Release bridge DLL is live.
WATCH: buildings appear one at a time.
"""
import time

from read_obs import ObsReader
from write_act import ActWriter
from catalog import Catalog

BUILDINGTYPE = 7  # AbstractType::BuildingType


def connect(timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            return ObsReader()
        except OSError:
            time.sleep(1)
    raise OSError("OBS never appeared")


def own_buildings(obs):
    return [u for u in obs.read_own() if u["category"] == "Building"]


def id_by_index(cat, category, index):
    for e in cat.by_id.values():
        if e["category"] == category and e["index"] == index:
            return e["id"]
    return None


def building_ready(obs):
    for f in obs.read_factories():
        if f["category"] == "Building" and (f["suspended"] or f["progress"] >= 53):
            return f
    return None


def find_and_place(act, idx, anchor):
    cx, cy = anchor
    for r in range(2, 11):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                if max(abs(dx), abs(dy)) != r:
                    continue
                res = act.place(idx, cx + dx, cy + dy)
                if res and res[0] == 0:           # ACT_OK
                    return (cx + dx, cy + dy)
    return None


def main():
    obs, act, cat = connect(), ActWriter(), Catalog()

    # 1) ensure a Construction Yard (deploy MCV if we have none)
    if not own_buildings(obs):
        mcv_ids = {e["index"] for e in cat.by_id.values()
                   if e["category"] == "unit" and ("MCV" in e["id"] or "MCV" in e["ui_name"])}
        mcv = None
        for _ in range(60):
            for u in obs.read_own():
                if u["category"] == "Unit" and u["type_id"] in mcv_ids:
                    mcv = u; break
            if mcv:
                break
            time.sleep(1)
        if not mcv:
            print("no MCV found"); return
        print(f">>> WATCH: deploying MCV at ({mcv['x']},{mcv['y']}) -> Construction Yard")
        act.deploy(mcv["unique_id"])
        for _ in range(40):
            if own_buildings(obs):
                break
            time.sleep(1)

    bld = own_buildings(obs)
    if not bld:
        print("no Construction Yard"); return
    anchor = (bld[0]["x"], bld[0]["y"])
    prefix = (id_by_index(cat, "building", bld[0]["type_id"]) or "GA")[:2]
    print(f"Construction Yard at {anchor}; faction prefix '{prefix}'")
    print(">>> WATCH: Power Plant -> Refinery -> War Factory will appear one by one.\n")

    # 2) build sequence
    for suffix, label in [("POWR", "Power Plant"), ("REFN", "Refinery"), ("WEAP", "War Factory")]:
        bid = prefix + suffix
        e = cat.by_id.get(bid)
        if not e:
            print(f"  {label} ({bid}): not in catalog, skip"); continue
        idx = e["index"]
        before = len(own_buildings(obs))
        print(f"  producing {label} ({bid}, idx {idx})...")
        print("    produce:", act.produce(BUILDINGTYPE, idx))
        # wait until ready to place
        ready = False
        for _ in range(90):
            if building_ready(obs):
                ready = True; break
            time.sleep(1)
        if not ready:
            print(f"    {label}: never became ready (CanBuild / power / funds?)"); continue
        cell = find_and_place(act, idx, anchor)
        if not cell:
            print(f"    {label}: no valid placement cell found"); continue
        print(f"    placed at {cell}")
        for _ in range(20):
            if len(own_buildings(obs)) > before:
                break
            time.sleep(0.5)
        print(f"    -> buildings now = {len(own_buildings(obs))}\n")

    print("BASE BUILT. building type_ids:", sorted(b["type_id"] for b in own_buildings(obs)))
    obs.close(); act.close()


if __name__ == "__main__":
    main()
