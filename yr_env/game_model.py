"""Operationalized game model (see docs/game-model.md): the ROLE ontology, the PREREQ tech-tree,
the COUNTER graph, and build_position() — which folds the raw OBS into a structured Position the
brain reasons OVER (instead of reacting to keywords).

This is the board. The solid baseline plays over it; the adaptive brain reads Position.brief();
the league learns Position.V. Faction-invariant: everything is in terms of roles, not unit IDs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --- roles (the strategic ontology) ---
CONSTRUCTION, POWER, ECONOMY = "CONSTRUCTION", "POWER", "ECONOMY"
PROD_INF, PROD_VEH, TECH_RADAR, TECH_LAB = "PROD_INF", "PROD_VEH", "TECH_RADAR", "TECH_LAB"
DEF_GROUND, DEF_AA, SUPERWEAPON = "DEF_GROUND", "DEF_AA", "SUPERWEAPON"
HARVESTER, MCV = "HARVESTER", "MCV"
MAIN_BATTLE, ANTI_ARMOR, ANTI_AIR, ARTILLERY = "MAIN_BATTLE", "ANTI_ARMOR", "ANTI_AIR", "ARTILLERY"
SCOUT, ENGINEER, INFANTRY, SUPERUNIT, OTHER = "SCOUT", "ENGINEER", "INFANTRY", "SUPERUNIT", "OTHER"

NONCOMBAT = {HARVESTER, MCV}                     # never commanded to fight
COMBAT_ROLES = {MAIN_BATTLE, ANTI_ARMOR, ANTI_AIR, ARTILLERY, SUPERUNIT, INFANTRY, SCOUT}

# --- ontology: catalog ID -> role ---
BLDG_SUFFIX_ROLE = {"CNST": CONSTRUCTION, "POWR": POWER, "NRCT": POWER, "APWR": POWER,
                    "REFN": ECONOMY, "HAND": PROD_INF, "PILE": PROD_INF, "BRCK": PROD_INF,
                    "WEAP": PROD_VEH, "RADR": TECH_RADAR, "AIRC": TECH_RADAR, "DOME": TECH_RADAR,
                    "TECH": TECH_LAB, "PSIS": TECH_RADAR}
DEF_GROUND_IDS = {"TESLA", "ATESLA", "NALASR", "GAPILL", "NABNKR", "YAGGUN", "NATBNK", "GTGCAN", "YAPSYT"}
DEF_AA_IDS = {"NAFLAK", "GASAM", "NASAM"}
SUPERWEAPON_IDS = {"NAMISL", "GAWEAT", "GACSPH", "NAIRON", "YAPPET", "GAGAP"}
UNIT_ROLE = {"HARV": HARVESTER, "HORV": HARVESTER, "CMIN": HARVESTER, "CMON": HARVESTER, "SMIN": HARVESTER,
             "AMCV": MCV, "SMCV": MCV, "PCV": MCV,
             "HTNK": MAIN_BATTLE, "MTNK": MAIN_BATTLE, "LTNK": MAIN_BATTLE, "MGTK": MAIN_BATTLE,
             "DRON": ANTI_ARMOR, "TNKD": ANTI_ARMOR, "CAOS": ANTI_ARMOR,
             "HTK": ANTI_AIR, "FV": ANTI_AIR,
             "V3": ARTILLERY, "SREF": ARTILLERY, "DRED": ARTILLERY, "CARRIER": ARTILLERY,
             "APOC": SUPERUNIT, "KIROV": SUPERUNIT, "ZEP": SUPERUNIT}
INF_ROLE = {"E1": INFANTRY, "E2": INFANTRY, "INIT": INFANTRY, "GGI": INFANTRY, "CONS": INFANTRY,
            "FLAKT": ANTI_AIR, "DOG": SCOUT, "ADOG": SCOUT, "TERROR": ANTI_ARMOR,
            "ENGINEER": ENGINEER, "SENGINEER": ENGINEER, "YENGINEER": ENGINEER, "SPY": SCOUT, "SNIPE": INFANTRY}


def role_of(cat_id: str, category: str) -> str:
    """Map a catalog ID (+ its category) to a strategic role."""
    if not cat_id:
        return OTHER
    if category == "building":
        if cat_id in DEF_GROUND_IDS:
            return DEF_GROUND
        if cat_id in DEF_AA_IDS:
            return DEF_AA
        if cat_id in SUPERWEAPON_IDS:
            return SUPERWEAPON
        return BLDG_SUFFIX_ROLE.get(cat_id[2:6], OTHER)
    if category == "infantry":
        return INF_ROLE.get(cat_id, INFANTRY)
    # unit / aircraft
    if "MCV" in cat_id:
        return MCV
    return UNIT_ROLE.get(cat_id, MAIN_BATTLE if category == "unit" else SUPERUNIT)


# --- logic: prerequisite tech-tree (role -> roles it needs) -> a correct build order ---
PREREQ = {POWER: [CONSTRUCTION], ECONOMY: [POWER], PROD_INF: [POWER],
          PROD_VEH: [PROD_INF, POWER], TECH_RADAR: [PROD_VEH], TECH_LAB: [TECH_RADAR],
          DEF_GROUND: [POWER], DEF_AA: [POWER]}
# canonical opening (topologically valid): the order the baseline establishes
BASE_ORDER = [POWER, ECONOMY, ECONOMY, PROD_INF, PROD_VEH, TECH_RADAR]
# above this much cash, money is NOT the constraint -- the one-at-a-time queue + build TIME is.
# A 2nd refinery is then wasted queue: bank fewer economy buildings, reach production/army sooner.
RICH_CREDITS = 50000

# --- logic: counter graph (enemy role -> the role that best answers it) ---
COUNTER = {ARTILLERY: ANTI_ARMOR, MAIN_BATTLE: ANTI_ARMOR, SUPERUNIT: ANTI_ARMOR,
           ANTI_AIR: MAIN_BATTLE, ANTI_ARMOR: MAIN_BATTLE, ARTILLERY + "_air": ANTI_AIR}
AIR_ROLES = {ANTI_AIR}  # placeholder; aircraft come through as SUPERUNIT/role below


def counter_for(enemy_role: str) -> str:
    return COUNTER.get(enemy_role, MAIN_BATTLE)


# combat value per role (for the military term of V and threat severity)
ROLE_VALUE = {MAIN_BATTLE: 1.0, ANTI_ARMOR: 1.2, ANTI_AIR: 0.8, ARTILLERY: 1.6,
              SUPERUNIT: 3.0, INFANTRY: 0.3, SCOUT: 0.1, ENGINEER: 0.2}


@dataclass
class Threat:
    role: str
    count: int
    severity: float
    counter: str
    at_base: bool


@dataclass
class Position:
    prefix: str = "NA"
    anchor: tuple = None
    credits: int = 0
    power_surplus: int = 0
    own_buildings: dict = field(default_factory=dict)   # role -> count
    own_units: dict = field(default_factory=dict)       # role -> count (combat only)
    army: list = field(default_factory=list)            # commandable combat units (no harvesters/MCV)
    tech_tier: int = 0                                   # 0 none, 1 radar, 2 lab
    enemy_seen: dict = field(default_factory=dict)       # role -> count (currently visible)
    enemy_belief: dict = field(default_factory=dict)     # role -> {count, age, confidence}
    threats: list = field(default_factory=list)
    V: float = 0.0

    def has(self, role):
        return self.own_buildings.get(role, 0) > 0

    def next_build(self):
        """Prereq-correct next structure role to establish the base (None if base complete).

        Economy is credit-conditional: flush with cash (>= RICH_CREDITS) one refinery is enough and
        a second only clogs the queue, so we want fewer -- reaching production/army sooner. Scarce,
        we want two refineries to ramp income. The brain reads credits in its obs and learns this."""
        econ_want = 1 if self.credits >= RICH_CREDITS else 2
        for r in BASE_ORDER:
            need = self.own_buildings.get(r, 0)
            want = econ_want if r == ECONOMY else 1
            if need < want and all(self.own_buildings.get(p, 0) > 0 for p in PREREQ.get(r, [])):
                return r
        return None

    def brief(self):
        """Render the Position as the board description the brain reads."""
        eb = ", ".join(f"{r}x{b['count']}(age{b['age']})" for r, b in self.enemy_belief.items()) or "none known"
        th = "; ".join(f"{t.role}x{t.count}->{t.counter}{'@BASE' if t.at_base else ''}" for t in self.threats) or "none"
        return (f"Faction {self.prefix} | credits {self.credits} power {self.power_surplus:+d} | "
                f"tech tier {self.tech_tier}\n"
                f"My buildings: {dict(self.own_buildings)}\n"
                f"My army (combat): {dict(self.own_units)}\n"
                f"Enemy (belief): {eb}\n"
                f"THREATS: {th}\n"
                f"Position score V={self.V:+.2f} ({'ahead' if self.V > 0 else 'behind'})")


def fold_roles(entities, lut, want_combat=False):
    """Fold an entity list into role -> [entities]. lut: (category_lower, index) -> catalog id."""
    out = {}
    for e in entities:
        cat = e["category"].lower()
        rid = lut.get((cat, e["type_id"]), "")
        role = role_of(rid, cat)
        out.setdefault(role, []).append(e)
    return out


def update_belief(memory, enemy_roles, tick):
    """Update the decaying belief about the (mostly hidden) enemy from what's currently visible."""
    for role, ents in enemy_roles.items():
        if role in NONCOMBAT or role == OTHER:
            continue
        memory[role] = {"count": len(ents), "tick": tick,
                        "positions": [(e["x"], e["y"]) for e in ents if e["x"] or e["y"]]}
    # age + drop stale beliefs (no sighting in a while -> forget)
    belief = {}
    for role, b in memory.items():
        age = tick - b["tick"]
        if age <= 30:                                   # forget after ~30 ticks unseen
            belief[role] = {"count": b["count"], "age": age, "positions": b.get("positions", [])}
    return belief


def assess_threats(enemy_belief, anchor):
    """Turn beliefs about the enemy into concrete threats with counters + base-proximity."""
    threats = []
    for role, b in enemy_belief.items():
        if role not in COMBAT_ROLES:
            continue
        at_base = False
        if anchor and b.get("positions"):
            ax, ay = anchor
            at_base = any(abs(x - ax) + abs(y - ay) < 30 for x, y in b["positions"])
        sev = b["count"] * ROLE_VALUE.get(role, 0.5) * (1.5 if at_base else 1.0)
        threats.append(Threat(role=role, count=b["count"], severity=round(sev, 1),
                              counter=counter_for(role), at_base=at_base))
    threats.sort(key=lambda t: -t.severity)
    return threats


def evaluate(pos: Position) -> float:
    """The chess score: am I ahead, and why? (heuristic V; the league learns the real one.)"""
    economy = pos.own_buildings.get(ECONOMY, 0) * 1.0 + pos.credits / 20000.0
    military = sum(ROLE_VALUE.get(r, 0.5) * n for r, n in pos.own_units.items())
    tech = pos.tech_tier * 1.0
    mapc = pos.own_buildings.get(CONSTRUCTION, 0) * 1.0
    threat = sum(t.severity for t in pos.threats)
    return round(0.8 * economy + 0.5 * military + 0.6 * tech + 0.4 * mapc - 0.5 * threat, 2)


def build_position(obs, cat, lut, memory, tick, econ_ids=None) -> Position:
    """Fold the live OBS into a Position (the board), updating the belief memory in place."""
    s = obs.read_state() or {}
    own = obs.read_own()
    enemy = [e for e in obs.read_enemy() if (e["x"] or e["y"])]   # sanitize the n_enemy overflow
    own_roles = fold_roles(own, lut)
    enemy_roles = fold_roles(enemy, lut)

    blds = {r: len(v) for r, v in own_roles.items() if r in BLDG_SUFFIX_ROLE.values()
            or r in (CONSTRUCTION, POWER, ECONOMY, PROD_INF, PROD_VEH, TECH_RADAR, TECH_LAB, DEF_GROUND, DEF_AA, SUPERWEAPON)}
    units = {r: len(v) for r, v in own_roles.items() if r in COMBAT_ROLES}
    army = [e for r, v in own_roles.items() if r in COMBAT_ROLES and r != SCOUT for e in v if (e["x"] or e["y"])]

    bld_list = own_roles.get(CONSTRUCTION, []) or [b for v in own_roles.values() for b in v if b["category"] == "Building"]
    anchor = None
    prefix = "NA"
    own_blds = [b for b in own if b["category"] == "Building"]
    if own_blds:
        anchor = (own_blds[0]["x"], own_blds[0]["y"])
        from build_base import id_by_index
        prefix = (id_by_index(cat, "building", own_blds[0]["type_id"]) or "GA")[:2]

    tech = 2 if blds.get(TECH_LAB) else (1 if blds.get(TECH_RADAR) else 0)
    belief = update_belief(memory, enemy_roles, tick)
    threats = assess_threats(belief, anchor)

    pos = Position(prefix=prefix, anchor=anchor, credits=s.get("credits", 0),
                   power_surplus=s.get("power_output", 0) - s.get("power_drain", 0),
                   own_buildings=blds, own_units=units, army=army, tech_tier=tech,
                   enemy_seen={r: len(v) for r, v in enemy_roles.items() if r in COMBAT_ROLES},
                   enemy_belief=belief, threats=threats)
    pos.V = evaluate(pos)
    return pos


if __name__ == "__main__":
    # self-test (no game): fold synthetic entities -> roles, threats, V, and the brief.
    LUT = {("unit", 1): "HARV", ("unit", 14): "V3", ("unit", 16): "DRON", ("building", 9): "NAPOWR",
           ("building", 15): "NAREFN", ("building", 14): "NAWEAP", ("unit", 2): "APOC"}
    print("role_of checks:",
          role_of("NAWEAP", "building"), role_of("V3", "unit"), role_of("DRON", "unit"),
          role_of("HARV", "unit"), role_of("TESLA", "building"), role_of("NAFLAK", "building"))
    print("BASE_ORDER:", BASE_ORDER)
    print("counter_for(ARTILLERY):", counter_for(ARTILLERY), " counter_for(MAIN_BATTLE):", counter_for(MAIN_BATTLE))

    own = [{"category": "Building", "type_id": 9, "x": 50, "y": 50},
           {"category": "Building", "type_id": 15, "x": 52, "y": 50},
           {"category": "Unit", "type_id": 1, "x": 48, "y": 50},      # harvester (noncombat)
           {"category": "Unit", "type_id": 16, "x": 51, "y": 51}]     # terror drone (anti-armor)
    enemy = [{"category": "Unit", "type_id": 14, "x": 60, "y": 55},   # V3 near base
             {"category": "Unit", "type_id": 14, "x": 61, "y": 55},
             {"category": "Unit", "type_id": 2, "x": 200, "y": 200}]  # APOC far
    own_r = fold_roles(own, LUT)
    enemy_r = fold_roles(enemy, LUT)
    print("\nown roles:", {r: len(v) for r, v in own_r.items()})
    print("enemy roles:", {r: len(v) for r, v in enemy_r.items()})
    mem = {}
    belief = update_belief(mem, enemy_r, tick=5)
    threats = assess_threats(belief, anchor=(50, 50))
    print("\nthreats (severity-sorted):")
    for t in threats:
        print(f"  {t.role} x{t.count}  sev={t.severity}  counter={t.counter}  at_base={t.at_base}")
    pos = Position(prefix="NA", anchor=(50, 50), credits=80000, power_surplus=40,
                   own_buildings={CONSTRUCTION: 1, POWER: 1, ECONOMY: 1},
                   own_units={ANTI_ARMOR: 1}, tech_tier=0, enemy_belief=belief, threats=threats)
    pos.V = evaluate(pos)
    print("\nnext_build (prereq-correct):", pos.next_build())
    print("\n" + pos.brief())
