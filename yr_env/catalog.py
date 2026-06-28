"""Type catalog: map the commander's readable build names -> (rtti, type_id) actions,
and provide a grounded roster so the LLM stops hallucinating non-YR units.

Source: data/type_catalog.csv, dumped by the bridge DLL (Bridge.Catalog.cpp) from the
engine's real *TypeClass arrays — ground truth for whatever rules/mod are loaded.
Each row: category, type_rtti, index, id, ui_name.

resolve("War Factory") -> (7, 303)   # BuildingType, YAWEAP index
"""
from __future__ import annotations

import csv
import difflib
import os

from contract import RTTIType

_HERE = os.path.dirname(__file__)
_CSV = os.path.join(_HERE, "data", "type_catalog.csv")

# Readable alias -> internal ID, curated for the Yuri faction (from the dumped catalog).
# The alias KEYS double as the grounded vocabulary we feed the commander.
YURI_ALIASES = {
    # buildings
    "Construction Yard": "YACNST", "Power Plant": "YAPOWR", "Bio Reactor": "YAPOWR",
    "Refinery": "YAREFN", "Slave Miner": "YAREFN", "Barracks": "YABRCK",
    "War Factory": "YAWEAP", "Naval Yard": "YAYARD", "Battle Lab": "YATECH",
    "Radar": "NAPSIS", "Gatling Cannon": "YAGGUN", "Psychic Tower": "YAPSYT",
    "Grinder": "YAGRND", "Tank Bunker": "YAGNTC", "Psychic Sensor": "YAPSYT",
    # infantry / units (extend as needed)
    "Yuri Clone": "YURI", "Initiate": "INIT", "Brute": "BRUTE",
}


def load_catalog(path: str = _CSV):
    """Return dict: id -> {category, rtti, index, id, ui_name}, plus by-name views."""
    by_id = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_id[row["id"]] = {
                "category": row["category"], "rtti": int(row["type_rtti"]),
                "index": int(row["index"]), "id": row["id"], "ui_name": row["ui_name"],
            }
    return by_id


class Catalog:
    def __init__(self, path: str = _CSV, aliases: dict | None = None):
        self.by_id = load_catalog(path)
        self.aliases = aliases if aliases is not None else YURI_ALIASES
        # searchable text -> id (id itself, ui_name without 'Name:' prefix)
        self._search = {}
        for cid, e in self.by_id.items():
            self._search[cid.lower()] = cid
            ui = e["ui_name"].replace("Name:", "").lower()
            self._search.setdefault(ui, cid)

    def resolve(self, name: str):
        """Readable name/ID -> (rtti, index) or None."""
        if not name:
            return None
        key = name.strip()
        # strip annotations like "War Factory (build)" the LLM sometimes adds
        for sep in (" (", " -", ":"):
            if sep in key:
                key = key.split(sep)[0].strip()
        # 1) curated alias
        cid = self.aliases.get(key) or self.aliases.get(key.title())
        # 2) exact id
        if not cid and key.upper() in self.by_id:
            cid = key.upper()
        # 3) fuzzy against id + ui_name
        if not cid:
            m = difflib.get_close_matches(key.lower(), list(self._search.keys()), n=1, cutoff=0.72)
            if m:
                cid = self._search[m[0]]
        if not cid or cid not in self.by_id:
            return None
        e = self.by_id[cid]
        return e["rtti"], e["index"]

    def roster(self):
        """Readable names to ground the commander (the alias vocabulary it should use)."""
        return sorted(self.aliases.keys())


if __name__ == "__main__":
    cat = Catalog()
    print(f"catalog: {len(cat.by_id)} types loaded; roster has {len(cat.roster())} aliases\n")
    for n in ["Power Plant", "Refinery", "War Factory", "Battle Lab", "Yuri Clone",
              "War Factory (build)", "Science Center", "Banshee Fighter"]:
        r = cat.resolve(n)
        rtti = {7: "BuildingType", 40: "UnitType", 16: "InfantryType", 3: "AircraftType"}.get(r[0]) if r else None
        print(f"  {n!r:24} -> {r}  {rtti or '(unresolved)'}")
