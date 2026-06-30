# A Logical, Epistemological & Ontological Model of Yuri's Revenge
*The foundation for a non-cheating, chess-like agent. State, actions, evaluation, and reward all derive from this.*

## Why this comes first
Every failure so far came from an agent that **reacts** instead of **seeing the board** — it answered keywords ("artillery → drone") with no model of the position. Chess strength needs three things made explicit: what **exists** (ontology), what can be **known** (epistemology), and the **rules + valid inferences** (logic). Chess has all three for free (32 pieces, perfect information, fixed rules). YR hides them under 569 unit types, fog of war, and real-time chaos. This document makes them explicit so the brain has a *position* to evaluate, not a feed to react to.

---

## I. ONTOLOGY — what exists

### 1. Substances (top-level kinds)
- **Resources:** `credits`, `power` (output − drain), `ore` (the source credits come from).
- **Technos** (engine `WhatAmI`): `Building`, `Unit` (vehicle), `Infantry`, `Aircraft`. (Naval = a Unit subset.)
- **Map:** a grid of cells; terrain (passable / impassable / water), `ore` fields, base sites, choke points.
- **Houses:** players, each a faction — Allied (`GA`), Soviet (`NA`), Yuri (`YA`) — with role-equivalent entities.

### 2. The ROLE taxonomy — the strategic ontology (the key abstraction)
The 569 raw types collapse into a handful of **roles**; strategy operates on roles, not IDs.

**Structures by role**
| role | does | examples (NA / GA / YA) | hard need |
|---|---|---|---|
| CONSTRUCTION | root; enables all building | `*CNST` (MCV deploys to it) | — |
| POWER | enables everything | `NAPOWR` / `GAPOWR` / `YAPOWR` | — |
| ECONOMY | credits (+ harvester) | `NAREFN` / `GAREFN` / `YAREFN` | Power |
| PROD-INFANTRY | trains infantry | `NAHAND` / `GAPILE` / `YABRCK` | Power |
| PROD-VEHICLE | builds vehicles | `NAWEAP` / `GAWEAP` / `YAWEAP` | **Barracks** + Power |
| TECH-RADAR | vision + unlocks | `NARADR` / `GAAIRC` / `AMRADR` | War Factory |
| TECH-LAB | superunits/SW | `NATECH` / `GATECH` / `YATECH` | Radar |
| DEF-GROUND | static anti-ground | `TESLA`,`NALASR` / `GAPILL`,`ATESLA` / `YAGGUN` | Power (Tesla: Radar) |
| DEF-AA | static anti-air | `NAFLAK` / `GASAM` / `YAGGUN` | Power |
| SUPERWEAPON | game-enders | nuke / weather / iron curtain / chrono | Lab |

**Units by role**
| role | property | examples | counters / countered |
|---|---|---|---|
| HARVESTER | economy, NEVER fights | `HARV`,`CMIN`,`SMIN` | (protect; do not command) |
| MCV | mobile ConYard | `AMCV`,`SMCV`,`PCV` | (deploy / expand) |
| MAIN-BATTLE | core army | `HTNK`,`MTNK`,`LTNK` | beats infantry; loses to anti-armor swarm |
| ANTI-ARMOR | fast vehicle-killer | **`DRON`** (Terror Drone), `TNKD` | **beats artillery + vehicles**; dies to infantry/AA |
| ANTI-AIR | kills aircraft | `HTK` (Flak Track), `FV` | beats air; weak vs ground |
| ARTILLERY | long-range siege, **kites** | **`V3`**, `SREF`, `DRED` | beats static defense + buildings; dies to fast flankers |
| SCOUT | cheap, fast, sees | dogs, cheap infantry | (information) |
| ENGINEER | capture / repair | `*ENGINEER` | — |
| SUPERUNIT | expensive heavy | `APOC`, `KIROV` | beats most; slow/costly |

### 3. Relations (the structure that makes it a system)
- **PREREQUISITE (tech tree):** `ConYard → Power → Refinery`; `Barracks` then `Power+Barracks → War Factory`; `War Factory → Radar`; `Radar → (V3, Tesla, advanced)`; `Radar → Battle Lab → superunits/SW`. **(War Factory requires a Barracks — the exact rule the agent jammed on.)**
- **PRODUCES:** factory → its unit category, **one at a time** (serial = the non-cheat rule).
- **COUNTERS (rock-paper-scissors):** artillery ▷ static-defense & buildings; fast anti-armor ▷ artillery & vehicles; anti-air ▷ aircraft; main-battle ▷ infantry; infantry-mass / engineers ▷ over-extended armor. *No unit is universal; advantage is positional and compositional.*
- **ECONOMY loop:** `Refinery + Harvester → credits`; `credits → production`; `Power → enables`. Choke any link and the system stalls.
- **ANALOGUES:** roles are faction-invariant (`HTNK ≈ MTNK ≈ LTNK` = MAIN-BATTLE), so reasoning is written once over roles.

---

## II. EPISTEMOLOGY — what can be known (and how)

### 1. The knowledge state
- **Per cell:** `SHROUDED` (never seen) · `FOGGED` (seen before, now stale) · `VISIBLE` (current).
- **Own technos:** fully known.
- **Enemy technos:** known **only where currently VISIBLE** (fog-gated). At any moment the agent sees a *fraction* of the enemy.
- **Hidden:** the enemy's base, production, tech, and most of its army are unknown most of the time.

### 2. Belief & inference (reasoning under uncertainty — the missing piece)
The agent must hold **beliefs** about the hidden state, inferred from sparse evidence:
- saw an enemy **Radar** ⇒ infer **V3 artillery waves are coming** (the blueprint's trigger), *before* seeing a V3;
- saw unit type X ⇒ infer the enemy's **tech tier** and likely composition;
- **no recent scouting** ⇒ high uncertainty, beliefs decay.
Maintain a **memory** of last-seen enemy positions/counts with **decaying confidence**. *(This is precisely what was missing when the agent declared "no threats" with 18 enemies in its base — it had no belief state, only the current flicker, and a count that had overflowed to garbage.)*

### 3. Knowledge acquisition — scouting
Scouting is the **action that converts SHROUDED/FOGGED → VISIBLE**. It has a *value* (uncertainty reduced) and a *cost* (a unit's time / risk) — an explore/exploit decision, not an afterthought.

### 4. The non-cheat boundary is epistemological
The agent may use **only VISIBLE enemy information** (fog-honored) — no maphack. This is the epistemic constraint that makes it *fair*, distinct from the stock AI which reads the whole map. (Our OBS already enforces fog; this is why it matters.)

---

## III. LOGIC — rules & inference

### 1. Hard rules (the game's laws = the non-cheat constraints)
serial production · prerequisite chains · power dependency · economy dependency · placement (build radius, valid cells) · fog. These are simultaneously the *rules of the world* and the *fairness constraints*.

### 2. Strategic inference (the playbook's logic, formalized)
- **COUNTER:** threat role `T` ⇒ counter role `C(T)` (artillery⇒fast-anti-armor + sortie; air⇒anti-air; armor-mass⇒anti-armor; etc.).
- **TEMPO:** tech windows — out-teching vs out-massing; the Radar timing that unlocks the enemy's V3s is a clock.
- **POSITION:** range (out-range artillery or *close on it*), flanking (kill kiting units), choke control, base layout.
- **ECONOMY ↔ MILITARY** tradeoff: income now vs army now; over-investing in either loses.

### 3. Position evaluation — the chess "score"
A value over the position — *what winning looks like*, and what the learned evaluator will approximate:
```
V = w_eco·economy(income, refineries, credits)
  + w_mil·military(Σ army value by role, weighted by counter-fit vs the enemy)
  + w_tech·tech_tier
  + w_map·map_control(bases, territory, scouted area)
  + w_def·defense_coverage
  − w_threat·threats_against(incoming/uncovered, by belief)
  + w_init·initiative
```
This is how the agent knows whether it is **ahead or behind, and why** — the thing it has never had.

### 4. Win / lose
**Lose:** ConYard + production destroyed (can't rebuild). **Win:** enemy eliminated. **Proximate objectives:** protect the economy, keep army ≥ threats, deny the enemy's economy, find and destroy the enemy base.

---

## IV. From model → agent (how we "then adapt")
- **STATE (the board)** = ontology (own + enemy entities folded into *roles*) + epistemology (fog status, belief/memory of the hidden) → a structured **Position**.
- **ACTIONS** = ontology under the rules: produce role `R`, build structure `S` (prereq-checked), move/attack a group, scout, expand. A clean, legal, faction-invariant action space.
- **REASONING / ADAPTATION** = inference over the Position: counter the *believed* threats, exploit weaknesses, manage the tech/army tempo — evaluate and plan, not keyword-reflex.
- **REWARD / VALUE** = the position evaluation `V` + win/lose — exactly what self-play RL maximizes. Chess-strength = a learned `V` + planning, trained by self-play (AlphaZero/AlphaStar). The LLM is the opening coach; the league grows the chess brain.

---

## V. Operationalize (next)
`yr_env/game_model.py`:
1. **ROLE map** — `catalog ID → role` (from §I.2), so the agent reasons over roles.
2. **PREREQ graph** — structure → prerequisites (from §I.3), so it *never* jams on a missing Barracks again.
3. **COUNTER graph** — role → counter role (from §I.3 / III.2).
4. **`build_position(obs, memory) → Position`** — fold OBS into roles, apply fog + belief, score `V`, list threats & opportunities. *This is the board the brain (LLM now, learned net later) evaluates.*

Solid baseline plays over this model; the adaptive brain reasons over it; the league learns `V` on it. One foundation, three layers.
