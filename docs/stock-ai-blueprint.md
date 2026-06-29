# Stock YR "Extreme AI" — Decision-Logic Blueprint

Source: `…/Command & Conquer Red Alert II/INI/Game Options/AI/Extreme AI.ini` (8,526 lines, by
modder "RAZER"). This is the transplantable *strategy* brain. The Brutal difficulty file is almost
pure cheat-tuning (see "Non-cheat boundary" below). We copy the **decision logic**, strip the cheats.

## The four-layer architecture

```
AITriggerType   "WHEN <side> & <game-state condition> THEN make TeamType X at weight W"
      │ produces
      ▼
TeamType        TaskForce + Script + flags (Priority, Max=1, Autocreate, Aggressive, AvoidThreats…)
   ┌──┴──┐
TaskForce      ScriptType
(unit list)    (opcode list: pick-target / attack / move-to-waypoint / guard)
```

- **135 TaskForces** — small unit groups (typically 3–9): armor balls (4–9 tanks), infantry pokes,
  AA escorts (+1–2 Flak), artillery teams, capture squads (1–2 engineers), superunit deathballs.
- **89 ScriptTypes** — opcode programs. Dominant offensive pattern: `54 (pick best target) → 53
  (acquire) → 0,<quarry> (attack) → 49 (rescan) → 0,1 (attack anything) → loop`. Defensive pattern:
  `58 (go to base waypoint) → 11 (guard forever)`. Plus patrol loops and transport/infiltration ops.
- **169 TeamTypes** — bind TaskForce+Script+flags. `Max=1` almost always (trigger weight controls
  re-creation). Key flags: `Aggressive`, `IsBaseDefense`, and **`AvoidThreats=yes`** (kiting — set on
  artillery so it never trades).
- **173 AITriggerTypes** — the conditional "what to build when" engine (see below).

## What killed our agent: the Radar-gated kiting V3 wave

- `RZRAITT63`: **WHEN Soviet AI owns a Radar (NARADR), difficulty=Hard → build "Soviet Bombard Hard"
  (RAZERTT62 = 3 Rhino + 6 V3 + 2 Boris), weight 70.** (Easy/Med variants: 3 V3 / 5 V3+2 Boris.)
- Bound to script `RAZERST9`: `pick-best-target → attack BUILDINGS (0,2) → attack BASE DEFENSES
  (0,7) → rescan (49) → loop`. With `AvoidThreats=yes`, the V3s **kite** — they out-range tanks and
  base defenses, shell structures from max range, and retreat from anything that closes.
- **Unlocks at Radar tier (early).** A static defensive base *cannot* beat it. The counter is mobile:
  out-range it, or fast flankers to kill the squishy V3s, or AA-mobile+armor that closes distance.
  Our agent had no army, no defense, and no concept of "sortie to kill artillery." It had to lose.

## The brain = a weighted condition→action table (this is why it feels smart)

There is **no single "attack now" flag.** Every cycle the AI recruits whatever teams are currently
*eligible*, biased by weight. Eligibility keys on five condition families:

| Condition family | Examples | Triggers… |
|---|---|---|
| **Own tech tier** | owns Radar / War Factory / Tech lab | escalating offense (Radar→V3, Tech→Apoc/Kirov) |
| **Enemy superweapon** | enemy Nuke / Weather / Grand Cannon / Chronosphere | reactive counter teams (anti-nuke transport, anti-cannon artillery) |
| **Enemy composition** | enemy has Grizzly/Rhino, air, Spy, Disc | hard counters (Terror Drones vs armor, **AA vs air @ weight 5000**, dogs vs spies) |
| **Neutral map tech** | Oil Derrick, Airport, Hospital | engineer capture, **weight 5000 (near-always)** |
| **Difficulty gate** | Easy/Med/Hard enable flags | Hard turns on big armor, V3, Apoc, Kirov, MCV re-expansion |

AITrigger record (load-bearing fields): `Name, TeamType, House, Priority, TechLevel, CondObject,
CondFlags(subject+comparator), Min/Max/CurWeight, …, SecondTeamType, EnabledEasy, Med, Hard`.
`SecondTeamType` is usually `RAZERTT88` (Chrono-Miner economy) — "also reinforce the economy."

## Escalation ladder (the spine to replicate)

0. **No tech:** cheap infantry/light-tank harassment + base guards; engineer-grab neutral oil (w5000).
1. **Radar:** **V3/artillery bombardment waves go live** (the breakpoint that beat us).
2. **War Factory / Tech:** armor balls (6–9), then superunits (Apoc/Kirov/Boomer/Master-Mind).
3. **Always-on reactive layer:** counters to enemy composition fire regardless of tier (AA, Terror
   Drones, dogs, anti-superweapon).

Defense = always-on + reactive (high weight). Offense = proportional to own tech + difficulty.

## Economy / build-order ratios (from the sibling Beta `[AI]` block)

```
PowerSurplus=50   RefineryRatio=.16 RefineryLimit=2   WarRatio=.2 WarLimit=3
BarracksRatio=.16 BarracksLimit=2   DefenseRatio=.4 DefenseLimit=40   AARatio=.07 AALimit=15
```
Order: ConYard → Power(+50 surplus) → Refinery(≤2) → Barracks → War Factory(≤3) → Radar, with
harvester/Chrono-Miner economy teams spawned constantly.

## Non-cheat boundary (what we STRIP vs KEEP)

- **STRIP (cheats):** `AIVirtualPurifiers=10,10,10` (free economy multiplier), engine parallel
  production queues, build-limit bypass, difficulty economy/build-speed multipliers.
- **KEEP (strategy):** the condition→action trigger table, TaskForce compositions, scripts (kite /
  guard / counter), the economy ratios, the escalation ladder. Execute it all under the human
  one-queue, real-economy, fog constraints.

## Translation hooks → non-cheating macro policy

1. **Expanded macro action space** (current agent has only DEPLOY/BUILD_*/TRAIN_TANK/ATTACK):
   add **DEFEND/GUARD**, **BUILD_AA**, **BUILD_BASE_DEFENSE**, **COUNTER_ARMOR** (Terror-Drone /
   tank-destroyer sortie), **ANTI_ARTILLERY_SORTIE** (fast flankers to kill kiting V3s),
   **ENGINEER_GRAB** (capture neutral tech), **EXPAND_MCV**.
2. **Rule-based PRIOR policy** = port §"condition→action table" directly: an obs-conditioned weighted
   recruiter. This is the immediately-competent, non-cheating baseline (and the BC teacher).
3. **The economy ratios** are a ready-made build-order controller (Refinery≤2, War≤3, +50 power).
4. **Per-commander POSTERIORS:** each LLM sets a *strategy profile* = a reweighting of the trigger
   weights (boom→economy/tech triggers; rush→early-attack triggers; turtle→defense triggers).
   `P(strategy | commander)` = a reweighting of the shared prior. → diverse seed population for the
   self-play league.
