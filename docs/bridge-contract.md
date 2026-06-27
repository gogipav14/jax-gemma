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

### EventClass mapping (to confirm in Phase 2 against YRpp `EventClass.h`)
| action | EventType | notes |
|--------|-----------|-------|
| PRODUCE | `PRODUCE` | house index = agent; respects the shared queue automatically |
| PLACE | `PLACE` | building must be ready in factory |
| SELL | `SELL` / `SELLCELL` | |
| GROUP_MOVE | `MEGAMISSION` (Move) | issued per unit in the group |
| GROUP_ATTACK | `MEGAMISSION` (Attack/Guard) | target = TechnoClass or cell |
| SUPERWEAPON | `SPECIAL_PLACE` | super must be charged |

> **Non-cheat invariant:** every event is enqueued with the agent's `house_index` on the same
> outgoing event list the human UI uses. The bridge must NOT call AI-only production helpers or
> write engine memory to fabricate units/credits. Phase 2 asserts: with ≥2 war factories, PRODUCE
> events still yield strictly serial production.

## Open items (pin during implementation)
- Exact `EventClass` struct/union field names + the enqueue static/global (re-confirm `EventClass.h`
  against the repo default branch; the raw fetch 404'd on a guessed path).
- Per-house shroud accessor in YRpp (`MapClass`/`HouseClass`).
- Group abstraction: maintain group→unit membership in the DLL or in Python? (Draft: DLL keeps a
  `group_id` map keyed by object IDs so commands survive unit attrition.)
