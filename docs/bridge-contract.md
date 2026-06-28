# Bridge Contract (v0 draft)

The C++ bridge DLL and the Python `yr_env` must agree on this shared-memory layout and action
schema. This is the single source of truth for the IPC boundary. **Subject to revision once
Phase 1/2 pin the exact YRpp field offsets.**

## Transport

- **Mechanism:** a named shared-memory segment (Windows `CreateFileMapping`/`MapViewOfFile`),
  plus two named events (or a double-buffer + sequence counter) for frame sync.
- Two regions: `OBS` (DLL → Python, written each logic frame) and `ACT` (Python → DLL, read each
  logic frame). A monotonically increasing `frame_seq` in the header lets Python detect new frames
  and lets the DLL detect a fresh action.
- **Sync model (training):** the DLL's per-frame hook can *block* until Python writes an action
  for `frame_seq`, giving deterministic frame-stepping. (Real-time mode: don't block; reuse last
  action.) Decided per-run via a flag in `spawn.ini` / bridge config.

## Header (both regions start with)

| field | type | notes |
|-------|------|-------|
| `magic` | u32 | `'YRBR'` sanity check |
| `version` | u32 | bumped on layout change |
| `frame_seq` | u64 | logic frame number |
| `house_index` | i32 | which house this OBS is for / this ACT controls |
| `status` | u32 | bitflags: game_over, win, loss, paused |

## OBS region (DLL → Python)

Fog-of-war honored: **only the agent house's discovered cells / visible enemy technos are written.**
Reading undiscovered state would be a maphack — explicitly forbidden.

### Global scalars (the agent house)
- `credits` (i32, from `HouseClass::Balance`)
- `power_output`, `power_drain` (i32, `HouseClass::PowerOutput/PowerDrain`)
- `side_index` (i32, `HouseClass::SideIndex`)
- owned counts: `owned_units/infantry/aircraft/navy/buildings` (i32)
- superweapon timers (array of {type_id, ready_frame})

### Spatial map grid (multi-channel, H×W per cell)
Channels (uint8 unless noted):
- terrain/passability, ore/gems amount, shroud/fog (0=shrouded,1=discovered,2=visible),
  own-unit occupancy, visible-enemy occupancy, own-building footprint, height.
- Map dims from the engine's `MapClass`; clamp/pad to a fixed (H_MAX, W_MAX) for the NN.

### Entity lists (fixed-cap, padded)
For own technos and *visible* enemy technos, up to `N_OWN` / `N_ENEMY`:
- `type_id` (i32), `x`,`y` (cell coords), `hp_frac` (u8), `state` (u8: idle/move/attack/build),
  `group_id` (u8, for group commands), `cooldown` (u8).

### Production state (per factory category: building/unit/infantry/aircraft/navy)
- `current_type_id` (i32, from `FactoryClass::Object`), `progress_frac` (u8),
  `queue` (array of type_ids), `primary_factory_present` (u8).
- `buildable_mask`: for the agent's tech tree, which `type_id`s pass `HouseClass::CanBuild()` now.

## ACT region (Python → DLL)

One action per frame (v1 = **macro + group commands**; per-unit micro is engine-scripted).
The DLL translates each action into one or more `EventClass` events enqueued on the player path.

### Action schema (factored / auto-regressive)
```
action = {
  type: enum {
    NOOP,
    PRODUCE,        # start producing a type in its factory queue
    PLACE,          # place a completed building at a cell
    SET_PRIMARY,    # set primary factory
    SELL,           # sell a building
    GROUP_MOVE,     # move a group_id to a cell (attack-move scripted)
    GROUP_ATTACK,   # order a group_id to attack a target entity/cell
    GROUP_FORM,     # (re)assign selected/visible units to a group_id
    SUPERWEAPON,    # fire a ready superweapon at a cell
    STANCE,         # set group stance (guard/aggressive/hold)
  },
  type_id: i32,     # for PRODUCE/SELL/PLACE/SUPERWEAPON
  cell_x, cell_y: i32,   # for PLACE/GROUP_MOVE/GROUP_ATTACK/SUPERWEAPON
  group_id: u8,     # for GROUP_*
  target_entity: i32,    # for GROUP_ATTACK (index into enemy entity list, or -1)
}
```

### EventClass mapping (CONFIRMED against YRpp `EventClass.h`)

**Injection point:** `EventClass::OutList.Add(ev)` — `OutList` is `QueueClass<EventClass,128>` @
`0x00A802C8`, the *same outgoing queue the human UI Adds to*. (`EventClass::AddEvent` is deprecated
and just forwards to `OutList.Add`.) Events carry an explicit `char HouseIndex`, so we issue "as"
the agent's house. The engine drains OutList → DoList and executes on the player command path, so
the shared production queue / economy / fog apply automatically. **This is the non-cheat guarantee.**

Each event is built via a documented constructor (hardcoded `JMP_THIS` thunk address):

| action | EventClass ctor (houseIndex first) | addr | payload struct |
|--------|-----------------------------------|------|----------------|
| PRODUCE | `(hidx, EventType::Produce, rtti, heapId, BOOL isNaval)` | `0x4C6970` | `Produce{RTTIType, HeapID, IsNaval}` |
| SUSPEND/ABANDON | same shape, EventType::Suspend/Abandon | `0x4C6970` | `Suspend`/`Abandon` |
| PLACE | `(hidx, EventType::Place, AbstractType rtti, heapId, isNaval, CellStruct cell)` | `0x4C6AE0` | `Place{RTTIType,HeapID,IsNaval,Location}` |
| SET_PRIMARY | `(hidx, EventType::Primary, id, rtti)` (Target ctor) | `0x4C65E0` | `Primary{Whom}` |
| SELL | `(hidx, EventType::Sell, id, rtti)` (Target ctor) | `0x4C65E0` | `Sell{Whom}` |
| SELLCELL (walls) | `(hidx, EventType::SellCell, CellStruct cell)` | `0x4C6650` | `SellCell{Location}` |
| GROUP_MOVE/ATTACK | `(hidx, TargetClass src, Mission, TargetClass target, dest, follow)` (MegaMission) | `0x4C6860` | `MegaMission{Whom,Mission,Target,Destination,Follow}` |
| SUPERWEAPON | `(hidx, EventType::SpecialPlace, id, CellStruct cell)` | `0x4C6B60` | `SpecialPlace{ID,Location}` |

`EventClass` is `#pragma pack(1)`, `sizeof==111`: `EventType Type; bool IsExecuted; char HouseIndex;
uint Frame;` then a 104-byte union of payloads. `Mission` enum (Move/Attack/Guard/Enter/…) drives
GROUP_* via MegaMission. Group commands = one MegaMission per member unit (`Whom` = unit target).

> **Non-cheat invariant:** only ever enqueue via `OutList.Add` with the agent's `HouseIndex`. The
> bridge must NOT call AI-only production helpers (e.g. `HouseClass::AI*`) or write engine memory to
> fabricate units/credits. Phase 2 asserts: with ≥2 war factories, PRODUCE events still yield
> strictly serial production (identical to a human).

## Open items (pin during implementation)
- Exact `EventClass` struct/union field names + the enqueue static/global (re-confirm `EventClass.h`
  against the repo default branch; the raw fetch 404'd on a guessed path).
- Per-house shroud accessor in YRpp (`MapClass`/`HouseClass`).
- Group abstraction: maintain group→unit membership in the DLL or in Python? (Draft: DLL keeps a
  `group_id` map keyed by object IDs so commands survive unit attrition.)

## Phase 1 implementation notes (as built)

- **Hook site:** `DEFINE_HOOK(0x55DDA0, Bridge_AfterFrame, 0x5)` (MainLoop_AfterRender), returns 0 —
  chains cleanly with ProtocolZero's return-0 hook at the same address. Guarded by
  `ScenarioClass::Instance` (the hook also fires on menu/score screens where state is stale).
- **Enemy visibility = CURRENT, not "ever seen".** Enemies are emitted only when their cell is
  `!IsShrouded() && !IsFogged()` (via `ObjectClass::GetCell()`). We deliberately do **not** use
  `DiscoveredBy` (it's sticky → would leak live positions through fog = maphack). Phase 1 assumes
  the agent house == `HouseClass::CurrentPlayer`, so the global cell shroud/fog *is* the agent's
  view; per-house visibility for an arbitrary agent house is deferred to Phase 3 (self-play).
- **Torn-read protection:** the DLL writes the whole OBS body, then `MemoryBarrier()`, then publishes
  `header.frame_seq` LAST. The Python reader should: read `frame_seq` → read body → re-read
  `frame_seq`; if it changed, retry (seqlock pattern). Phase 1 validation reads via the `-LOG`
  `debug.log` `[BRIDGE]` dump, not the mapping, so this only matters from Phase 3 on.
- **`type_id` = `GetArrayIndex()`** is per-RTTI-category (infantry/unit/aircraft/building indexed
  separately) — NOT globally unique yet. A category byte will be added in Phase 1b so Python can use
  it as a global key.
- **Crash guards:** skip `InLimbo` technos and any with a null `GetTechnoType()`; bound writes by
  `N_OWN`/`N_ENEMY`. (Reviewed adversarially; 4 issues found and fixed before first run.)
