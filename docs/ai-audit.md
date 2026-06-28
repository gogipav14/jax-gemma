# How the Current Yuri's Revenge AI Is Scripted & Set Up — And Exactly How It Cheats

> Produced by the `yr-ai-audit` multi-agent workflow (6 agents) and **adversarially verified**.
> Sources: (a) the AI override INIs in the Steam install (`INI\Game Options\AI\{Brutal AI,Extreme AI,
> Extreme AI Beta}.ini`), (b) engine headers + spawner source under `bridge/yrpp-spawner/`, (c) the
> CnCNet client config (`Resources\GameLobbyBase.ini`, `GameOptions.ini`). Quantitative claims the
> verifier could not anchor in local files are **explicitly flagged `uncertain`**.

---

## 1. The scripting model

The YR AI is **data-driven** — its whole behavioral repertoire lives in INI sections; the compiled
engine is just the interpreter. Four pillars chain "when → what → how":

```
AITriggerType   →   TeamType   →   TaskForce        ScriptType
 (condition)         (binds)       (composition)  +  (behavior)
   WHEN              GLUE            WHAT              HOW
```

1. **`[AITriggerTypes]` — the "when".** Each entry tests game state and, if satisfied (and enabled for
   the current difficulty, by weight), requests a team. Decoded fields:
   `Name, Team1, OwnerHouse, TechLevel, ConditionType, ComparisonObject, Comparator(64-hex),
   StartingWeight, MinimumWeight, MaximumWeight, IsForSkirmish, <unused>, Side, IsBaseDefense, Team2,
   EnabledInEasy, EnabledInNormal, EnabledInHard`.
   Example `RZRAITT0` "Allied Anti-Nuke 1": Team1 `RAZERTT39`, ComparisonObject `NAMISL` (nuke),
   weights `70/10/70`, enabled `1,1,1`. Engine checks `Enabled_Easy/Normal/Hard` in
   `AITriggerTypeClass::ConditionMet()` (`0x41E720`).
   > **Corrected:** the original pass said "19 fields"; the real entry has **17–18**. The 64-hex
   > `Comparator` bit layout and the `ConditionType` code table are **uncertain** (not in local files).
2. **`[TeamTypes]` — the glue.** Flat `RAZERTEAM0…168` → `RAZERTT#`; in-engine `TeamTypeClass` holds a
   `ScriptTypeClass* ScriptType` and a `TaskForceClass* TaskForce`.
3. **`[TaskForces]` — the "what".** `INDEX=COUNT,UNIT_CODE`, e.g. `RAZERTF0`: `0=4,E1` `1=3,GGI`
   `2=2,ADOG` `3=3,JUMPJET` `Group=-1` (4 GIs, 3 Guardian GIs, 2 dogs, 3 Rocketeers).
4. **`[ScriptTypes]` — the "how".** `N=ACTION_CODE,PARAM`, stored as `ScriptActions[50]`. Codes:
   `0`=AttackQuarryType, `5`=GuardArea, `6`=JumpToLine(loop), `53`=Gather, `54`=Regroup,
   `58`=MoveToFriendlyStructure. E.g. `RAZERST0` "Refinery Guard": `58,1 → 5,75 → … → 6,1` (loop).

**The loop:** condition met → spawn Team1/Team2 → TeamType resolves TaskForce + ScriptType → actions
run 0→1→2… with JumpToLine loops until the team dies.

**`[General]` tunables (verified, `Easy,Normal,Hard` triples):** `TeamDelays=100,500,1000`,
`AISafeDistance=34`, `AIFriendlyDistance=18`, `TotalAITeamCap=100,75,50`,
`AIVirtualPurifiers=50,24,12`, `MultiplayerAICM=4800,2400,1200` (*meaning uncertain*),
`AITriggerSuccessWeightDelta=5/-5` (re-weights triggers after success/failure).

---

## 2. This install's hard-AI mods — Brutal vs Extreme

Both are **map-code overlays** on the base rules. Opposite ends of effort:

- **`Brutal AI.ini`** — ~6 data lines, `[General]` only. **No** triggers/teams/taskforces/scripts of
  its own (inherits them). Changes only tempo/economy: `TeamDelays=500,1000,1500`,
  `AIVirtualPurifiers=10,10,10`, `Min/MaxAIDefensiveTeams=0,0,0`, `DissolveUnfilledTeamDelay=2500`.
  "Brutal" = a tempo/economy re-tune.
- **`Extreme AI.ini`** — a full rewrite. **Verified counts:** `[AITriggerTypes]` **173**,
  `[TeamTypes]` **169**, `[TaskForces]` **135**, `[ScriptTypes]` **89**. Notably, its three difficulty
  blocks are **flattened to 1.0× stats** (`Groundspeed/Airspeed/Armor/ROF/Cost=1.0`), with
  `BuildTime=0.6` and delays `0.01` — so it expresses difficulty through **economy + behavior**, not
  stat inflation.

---

## 3. How it cheats

Cheats are properties of the **AI code path**, not skill. AI houses route through internal factory
APIs; human houses go through the UI/event queue (§5).

### ★ Parallel production (the big one)

- **Mechanism.** A human owns ~one shared queue per category, serialized into `EventClass::OutList`
  and dequeued one at a time. The AI does not use that path — each AI factory can be its own primary
  and produce **independently and simultaneously**.
- **In code (verified in-repo):** `HouseClass` has nine `Primary_For*` factory pointers
  (`HouseClass.h:925–933`); the AI drives them via `HouseClass::AI_BaseConstructionUpdate()`
  (`0x4FE3E0`) and `AI_VehicleConstructionUpdate()` (`0x4FEA60`), calling
  `FactoryClass::DemandProduction()` (`FactoryClass.h:41`) **directly — bypassing `EventClass::OutList`**
  (`EventClass.h:16`). The fork is gated by `HouseClass::IsHumanPlayer` (`HouseClass.h:818`). The INIs
  enable it via `AllowParallelAIQueues=yes` (`Extreme AI.ini:12`) + `AllowBypassBuildLimit=yes,yes,yes`.
  > **Uncertain:** the precise "3 factories = 3× output" multiplier isn't in local files. The *setting*
  > and *engine wiring* are confirmed; the exact rate multiplier is not. Say "parallel, faster" — not 3×.

### AIVirtualPurifiers — phantom economy

Free refinery income for purifiers it never built: `10,10,10` (Brutal); `50,24,12` (Extreme — *inverse*:
more on Easy). **Raw counts are fact.** The "+25%/purifier → +250%" math is **uncertain** (per-purifier
bonus constant not in local files).

### `[Difficulty]` multipliers (AI-only stat knobs)

Applied per-house via `AssignHandicap` → `*Multiplier` fields. `BuildTime=0.6` ≈ **40% faster** builds
(verified). `Cost/ROF/Armor/Firepower/...` exist as `HouseClass::*Multiplier` (`HouseClass.h:802–808`) —
all `1.0` in this mod. **Engine caveat:** the header marks `GroundspeedMultiplier/AirspeedMultiplier/
BuildTimeMultiplier` **"unused"** in this build (only `FirepowerMultiplier` "used") — so some keys may be
inert regardless of INI value.

### Other (mechanism named; magnitude not locally verified — ModEnc-sourced)

- **Perfect map knowledge / no fog** in AI targeting (`PickTargetByType` `0x50D170` reads true state).
- **Prereq/adjacency relaxation** (`AIBaseSpacing` vs human `Adjacent`).
- `ContentScan=yes` is **inert** (vestigial Tiberian Sun flag) — *not* a cheat.

---

## 4. Difficulty wiring on this machine

**Dropdown → overlay** (`Resources\GameLobbyBase.ini`, `cmbAIModifier`, `DataWriteMode=MapCode`):

| Label | File | Nature |
|---|---|---|
| Vanilla AI | `No Change.ini` | empty stub |
| Brutal AI | `Brutal AI.ini` | tiny `[General]` tweak |
| Extreme AI | `Extreme AI.ini` | full rewrite (173/169/135/89) |
| **Nightmare AI (Beta)** | `Extreme AI Beta.ini` | largest override |

> **"Nightmare" is not a vanilla tier** — it's the client mapping the 4th dropdown slot to
> `Extreme AI Beta.ini`. **"Hard"** is the engine's top *base* difficulty (integer 2), optionally
> combined with an overlay. (`Extreme AI Beta.ini` line count **unverified** this pass.)

**Per-house difficulty → spawn.ini.** `[HouseHandicaps] Multi1…Multi8` integers, interpreted through the
**reversed** `AIDifficulty` enum (`HouseClass.h:801`: *"Hard == 0 and Easy == 2"*).

**Spawner applies it** (`Spawner.cpp:170–171`):
```cpp
if (pHousesConfig->HandicapDifficulty != -1)
    pHouse->AssignHandicap(pHousesConfig->HandicapDifficulty);   // HouseClass::AssignHandicap @ 0x4F6EC0
```
plus `AINamesByDifficulty` (`Spawner.cpp:121`) and the global `AIDifficulty` default (`Spawner.cpp:215`).
> **Open question:** precedence when global `AIDifficulty` disagrees with a per-house `AssignHandicap`
> is unresolved in the available source.

---

## 5. Implications for our non-cheating agent

Our agent plays as a **human house** — orders go through `EventClass::OutList` (`EventClass.h:16`), the
same serialized queue the sidebar uses. That one fact makes it fair. The contrast:

| # | Stock AI cheat (mechanism + symbol) | Constraint our agent accepts (human/`OutList` path) | Skill that must replace it |
|---|---|---|---|
| 1 | **Parallel per-factory production** — `Primary_For*` × `AI_*ConstructionUpdate` → `FactoryClass::DemandProduction()`, gated `!IsHumanPlayer`; `AllowParallelAIQueues=yes` | **Serial production per queue** — one in-flight build per category via `OutList` | Build-order optimization; build **extra factories** to legitimately parallelize; never idle a queue |
| 2 | **Virtual economy** — `AIVirtualPurifiers` (10/10/10; 50/24/12): free income | **Real economy** — must build refineries/purifiers, defend harvesters | Expansion, harvester protection, ore denial, faster real tech |
| 3 | **Free build speed** — `BuildTime=0.6` via `AssignHandicap`→`BuildTimeMultiplier` (field marked "unused") | **Baseline `1.0×`** build time | Tighter macro; tempo from decisions, not a clock |
| 4 | **Free stat multipliers** — `Firepower/ROF/Armor/Cost` (1.0 here, but a live cheat surface) | **Book stats** | Focus fire, counters, positioning, retreat timing |
| 5 | **Perfect information** — targeting reads true positions, no shroud | **Fog-limited sight** (our Phase-1 bridge enforces this) | Scouting, map control, prediction |
| 6 | **Placement/prereq relaxation** — `AIBaseSpacing` vs `Adjacent` | **Full placement + prereq rules** | Smart layout within adjacency; correct tech order |

**Bottom line.** "Brutal/Extreme/Nightmare" is **not a smarter brain** — it's the *same* scripted brain
(AITrigger → TeamType → TaskForce → ScriptType) on a **privileged execution path**: parallel factories,
phantom economy, per-house handicaps. Our agent gets none of that, because it speaks to the game only
through `EventClass::OutList` as a player. It must win on **decision quality** — scouting vs fog,
real-economy efficiency, build-order tempo, tactical micro.

---

### Reference: verified symbols & keys

`HouseClass.h` — `AI_BaseConstructionUpdate() 0x4FE3E0`, `AI_VehicleConstructionUpdate() 0x4FEA60`,
`AssignHandicap() 0x4F6EC0`, `Primary_For*` (925–933), `IsHumanPlayer` (818), reversed `AIDifficulty`
(801), `*Multiplier` (802–808), `IQLevel` (811). `FactoryClass.h:41` `DemandProduction()`.
`EventClass.h:16` `OutList`. `AITriggerTypeClass::ConditionMet() 0x41E720` + `Enabled_*`.
`ScriptTypeClass::ScriptActions[50]`. `TeamTypeClass::{ScriptType,TaskForce}`.
`Spawner.cpp:170–171/121/215`. INI: `AIVirtualPurifiers`, `AllowParallelAIQueues`,
`AllowBypassBuildLimit`, `TeamDelays`, `TotalAITeamCap`, `MultiplayerAICM`, `BuildTime/RepairDelay/
BuildDelay`, `HouseHandicaps`, `HandicapDifficulty`, `AINamesByDifficulty`, `cmbAIModifier`,
`DataWriteMode=MapCode`.
