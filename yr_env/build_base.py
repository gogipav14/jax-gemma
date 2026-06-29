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

# readable building name -> ID suffix (prefix is the faction, resolved in-game). Most suffixes are
# faction-consistent (POWR/REFN/WEAP/TECH); Barracks varies (HAND/PILE/BRCK) and is tried in order.
SUFFIX_MAP = {"power": "POWR", "refinery": "REFN", "ore": "REFN", "war factory": "WEAP",
              "battle lab": "TECH", "science": "TECH", "tech": "TECH", "radar": "RADR"}


UNITTYPE = 40  # AbstractType::UnitType
MAIN_TANK = {"NA": "HTNK", "GA": "GTNK", "YA": "LTNK"}  # faction main battle tank ID


def name_to_suffix(name, cat, prefix):
    n = name.lower()
    if "barrack" in n:
        for suf in ("HAND", "PILE", "BRCK"):
            if cat.by_id.get(prefix + suf):
                return suf
        return None
    for key, suf in SUFFIX_MAP.items():
        if key in n and cat.by_id.get(prefix + suf):
            return suf
    return None


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


def produce_retry(act, rtti, idx, timeout=18):
    """Produce, retrying on REJECTED_CANBUILD (prereq still constructing) until it sticks."""
    t0 = time.time()
    r = None
    while time.time() - t0 < timeout:
        r = act.produce(rtti, idx)
        if r and r[0] == 0:                  # ACT_OK
            return r
        if r and r[0] == 2:                  # REJECTED_CANBUILD -> prereq not ready yet
            time.sleep(2); continue
        return r                              # NOFACTORY / BAD_* -> don't retry
    return r


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


def main(order_override=None, n_tanks=0, attack=False):
    """order_override: optional list of readable building names (e.g. from an LLM build order).
    n_tanks: after the base, produce this many main battle tanks from the War Factory.
    attack: after building the army, attack-move it toward a visible enemy (or scout out)."""
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

    # 2) build sequence — order is a list of READABLE names (default or from the LLM commander)
    order = order_override or ["Power Plant", "Refinery", "Barracks", "War Factory"]
    # HARD PREREQUISITE the LLM may not know: a Power Plant must come first (everything needs power).
    # The commander supplies strategic intent; the executor enforces buildability.
    powers = [x for x in order if ("power" in x.lower() or "reactor" in x.lower())]
    rest = [x for x in order if x not in powers]
    order = ([powers[0]] if powers else ["Power Plant"]) + rest
    print("  effective build order (executor enforces power-first):", order)
    for label in order:
        suffix = name_to_suffix(label, cat, prefix)
        if not suffix:
            print(f"  {label}: not a mappable {prefix} structure, skip"); continue
        bid = prefix + suffix
        e = cat.by_id.get(bid)
        if not e:
            print(f"  {label} ({bid}): not in catalog, skip"); continue
        idx = e["index"]
        before = len(own_buildings(obs))
        print(f"  producing {label} ({bid}, idx {idx})...")
        pr = produce_retry(act, BUILDINGTYPE, idx)
        print(f"    produce: {pr}")
        if not pr or pr[0] != 0:
            print(f"    {label}: could not start ({pr}); skipping"); continue
        # wait until ready to place
        ready = False
        for _ in range(40):
            if building_ready(obs):
                ready = True; break
            time.sleep(1)
        if not ready:
            print(f"    {label}: produced but never ready to place"); continue
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

    # 3) produce an army of the faction's main battle tank from the War Factory (serial / non-cheating)
    if n_tanks > 0:
        tank = MAIN_TANK.get(prefix, "HTNK")
        e = cat.by_id.get(tank)
        if not e:
            print(f"  no main tank '{tank}' in catalog; skip army")
        else:
            print(f"\n>>> WATCH: producing {n_tanks}x {tank} (tanks) from the War Factory, one at a time")
            u0 = sum(1 for u in obs.read_own() if u["category"] == "Unit")
            for i in range(n_tanks):
                r = produce_retry(act, UNITTYPE, e["index"])
                print(f"  tank {i + 1}: {r}")
                time.sleep(2)
            for _ in range(45):
                time.sleep(1)
                u = sum(1 for ent in obs.read_own() if ent["category"] == "Unit")
                if u >= u0 + n_tanks:
                    break
            u = sum(1 for ent in obs.read_own() if ent["category"] == "Unit")
            print(f"  ARMY: own units {u0} -> {u} (+{u - u0} tanks built serially via OutList)")

    # 4) send the army to attack — toward a VISIBLE enemy (fog-honored), else push out to scout
    if attack:
        time.sleep(2)
        tanks = [u for u in obs.read_own() if u["category"] == "Unit" and (u["x"] or u["y"])]
        enemies = [e for e in obs.read_enemy() if (e["x"] or e["y"])]
        ax, ay = anchor
        if enemies:
            tgt = min(enemies, key=lambda e: abs(e["x"] - ax) + abs(e["y"] - ay))
            tx, ty = tgt["x"], tgt["y"]
            print(f"\n>>> WATCH: attacking with {len(tanks)} units toward a VISIBLE enemy at ({tx},{ty})")
        else:
            tx, ty = (ax // 2 if ax > 80 else ax + 80), ay  # no enemy seen -> push toward map interior
            print(f"\n>>> WATCH: no enemy in sight; pushing {len(tanks)} units toward ({tx},{ty}) to scout/attack")
        for u in tanks:
            act.attack_move(u["unique_id"], tx, ty)
        print(f"  ordered {len(tanks)} units to attack-move.")

    obs.close(); act.close()


if __name__ == "__main__":
    main()
