# Phase 1 Implementation Plan: OBSERVATION Bridge for the CnCNet YR Spawner DLL

> Produced by the `yr-phase1-design` multi-agent workflow (6 agents) and **adversarially verified**
> against the local YRpp headers + spawner sources. Every address/field below was re-checked; the
> verifier flagged **zero crash risks**. One research claim was **rejected** and corrected (see ⚠️).

**Scope:** OBS-only (DLL → Python). Write the agent house's **fog-honored** state to shared memory
once per logic frame. ACT/event injection (`EventClass::OutList`) is **Phase 2**.

> ⚠️ **CORRECTION — `Main::ExeRun` @ `0x7CD810` is NOT per-frame.** `src/Main.cpp:30–36` shows it only
> runs `Patch::ApplyStatic()` once at startup. Do **not** drive `WriteOBS()` from it. The per-frame
> site is the `QueueAIMultiplayer` hook below.

---

## 1. Per-frame hook (piggyback existing spawner infrastructure)

The spawner already owns a confirmed per-logic-frame hook at `QueueAIMultiplayer` entry (`0x647BEB`,
used by ProtocolZero). Chain a second stub via `DEFINE_HOOK_AGAIN` so both run.

```cpp
// src/Bridge/Bridge.Hook.cpp
// ProtocolZero already defines DEFINE_HOOK(0x647BEB, ..., 0x9) at ProtocolZero.Hook.cpp:38.
// DEFINE_HOOK_AGAIN chains a second stub at the SAME address/size. Do NOT change addr/size;
// Spawner.Hook.cpp:244 warns about Phobos coordination on spawner hook addresses.
DEFINE_HOOK_AGAIN(0x647BEB, Bridge_PerFrame, 0x9)
{
    Bridge::OnLogicFrame();   // writes OBS for the agent house
    return 0;                 // fall through to engine / ProtocolZero stub
}
```

- **Address:** `0x647BEB` (`QueueAIMultiplayer` entry), size `0x9` (matches existing confirmed hook).
- **Read no registers** — `OnLogicFrame()` re-derives state from globals (avoids register coupling).
- **frame_seq** = `Unsorted::CurrentFrame` (`int` @ `0xA8ED84`, `Fundamentals.h`; used at `ProtocolZero.Hook.cpp:53`).
- Not `0x64C598` (`ExecuteDoList`, per-frame, size `0x6`) — reserve that for Phase-2 ACT injection.
- Not `0x55DDA0` (`MainLoop_AfterRender`) — post-render, **not** logic-frame (confirmed).

---

## 2. OBS shared-memory structs (match `docs/bridge-contract.md`) — all `#pragma pack(1)`

Agent house (Phase 1): `HouseClass::CurrentPlayer` (@ `0xA83D4Cu`, `HouseClass.h:172`), guarded (§6).

```cpp
#pragma pack(push, 1)
struct BridgeHeader {          // static_assert(sizeof==24)
    uint32_t magic;            // 0x59524252 'YRBR'
    uint32_t version;          // = 1
    uint64_t frame_seq;        // (uint64_t)Unsorted::CurrentFrame  (0xA8ED84)
    int32_t  house_index;      // pAgent->ArrayIndex                (HouseClass.h:781)
    uint32_t status;           // bit0 game_over, bit1 win, bit2 loss(=pAgent->Defeated), bit3 paused
};
struct BridgeGlobals {
    int32_t credits;           // HouseClass::Balance        (HouseClass.h:898)
    int32_t power_output;      // HouseClass::PowerOutput    (HouseClass.h:923)
    int32_t power_drain;       // HouseClass::PowerDrain     (HouseClass.h:924)
    int32_t side_index;        // HouseClass::SideIndex      (HouseClass.h:817)
    int32_t owned_units;       // HouseClass::OwnedUnits     (HouseClass.h:892)
    int32_t owned_buildings;   // HouseClass::OwnedBuildings (HouseClass.h:894)
    int32_t owned_infantry;    // HouseClass::OwnedInfantry  (HouseClass.h:895)
    int32_t owned_aircraft;    // HouseClass::OwnedAircraft  (HouseClass.h:896)
    int32_t owned_navy;        // derived: own technos whose GetTechnoType()->Naval == true
};
struct BridgeEntity {
    int32_t type_id;           // GetTechnoType()->ArrayIndex
    int16_t x, y;              // GetMapCoords() -> CellStruct.X/.Y
    uint8_t hp_frac;           // GetHealthPercentage()*255
    uint8_t state;             // Phase 1: coarse (0 idle) — correctness bar is type/pos/hp
    uint8_t group_id;          // 0 in Phase 1 (groups are Phase 2)
    uint8_t cooldown;          // 0 in Phase 1
};
#define N_OWN   256
#define N_ENEMY 256
struct BridgeOBS {             // static_assert(sizeof) — bump version on any change
    BridgeHeader  header;
    BridgeGlobals globals;
    uint16_t      n_own, n_enemy;
    BridgeEntity  own[N_OWN];
    BridgeEntity  enemy[N_ENEMY];
};
#pragma pack(pop)
```

Per-entity accessors (via `TechnoClass`/inherited `ObjectClass`): `type_id` = `GetTechnoType()->ArrayIndex`
(`ObjectClass.h:86`; `ArrayIndex` is the first member of each `*TypeClass`). `x,y` = `GetMapCoords()`
(`ObjectClass.h:233`). `hp_frac` = `GetHealthPercentage()*255` (`ObjectClass.h:203`).

> **Phase 1b** (follow-on): production state (`FactoryClass::Object`/`GetProgress()` 0..54/`QueuedObjects`;
> house `Primary_For*` @ `HouseClass.h:925–933`) and the multi-channel spatial grid. Kept out of the
> first cut to stay crash-minimal.

---

## 3. Enumeration: OWN + VISIBLE-ENEMY, fog honored

No per-house "all owned" list exists → scan the global array.

```cpp
HouseClass* pAgent = HouseClass::CurrentPlayer;          // 0xA83D4Cu, guarded (§6)
for (int i = 0; i < TechnoClass::Array->Count; ++i) {
    TechnoClass* pT = TechnoClass::Array->Items[i];      // 0xA8EC78u (TechnoClass.h:187)
    if (!pT || pT->InLimbo) continue;                    // ObjectClass.h:296
    if (pT->Owner == pAgent) {                           // TechnoClass.h:639
        emitOwn(pT);                                     // own units: no fog gate
    } else if (!pAgent->IsAlliedWith(pT->Owner)) {
        if (pT->DiscoveredBy(pAgent))                    // ObjectClass.h:171 (virtual) — fog gate
            emitEnemy(pT);
    }
}
```

**Fog correctness (non-maphack guarantee):** gate enemies **only** with the virtual
`DiscoveredBy(pAgent)`. Do **not** use cached `DiscoveredByCurrentPlayer/Computer`
(`TechnoClass.h:746–747`) — they lag and are relative to current player/computer, not an arbitrary
agent house.

---

## 4. New files + exact `Spawner.vcxproj` change

New files: `src/Bridge/Bridge.h` (facade + structs + Win32 handles), `src/Bridge/Bridge.cpp`
(`CreateMapping`/`DestroyMapping`/`OnLogicFrame`), `src/Bridge/Bridge.Hook.cpp` (the hook + lifecycle).

The vcxproj uses **explicit** `ClCompile`/`ClInclude` items (no globs — verified lines 18–96). Add:

```xml
    <!-- Bridge -->  (compile ItemGroup, ~after Utilities block)
    <ClCompile Include="$(ThisDir)\src\Bridge\Bridge.cpp" />
    <ClCompile Include="$(ThisDir)\src\Bridge\Bridge.Hook.cpp" />
```
```xml
    <!-- Bridge -->  (header ItemGroup)
    <ClInclude Include="$(ThisDir)\src\Bridge\Bridge.h" />
```

No `Spawner.props` change needed (already C++20, `/O2`, SSE, no-exceptions, StdCall, MultiThreaded).
Include `<windows.h>` in `Bridge.cpp` only. **Lifecycle:** create mapping lazily on first
`OnLogicFrame()` (static-bool guard) — not in `DllMain`, not in `ExeRun`; destroy on
`DLL_PROCESS_DETACH` (`Main.Hook.cpp:23`).

---

## 5. Validation (log dump vs. on-screen)

Use the spawner's existing `Debug::Log(...)` (`src/Utilities/Debug.h:35`), throttled
(`if (Unsorted::CurrentFrame % 60 == 0)`), to dump the same frame it writes:

```
[BRIDGE] f=%d house=%d credits=%d powerOut=%d powerDrain=%d ownUnits=%d ownBldg=%d ownInf=%d ownAir=%d nOwn=%d nEnemyVis=%d
```

Run a skirmish and compare: `credits` vs on-screen counter; `power_output/drain` vs power tooltip;
`owned_buildings` vs a manual building count; `n_own` ≈ sum of owned counts (±limbo). **Fog spot-check:**
an enemy unit in shroud must be **absent** from `enemy[]`; scouting it into sight must make it **appear**.
Pass = scalars match across ≥3 frames AND fog toggles correctly. Only then expose to Python.

---

## 6. Risks + guards

| Risk | Guard |
|------|-------|
| Agent house null / spectator | `if (!HouseClass::CurrentPlayer) return;` at top of `OnLogicFrame()` |
| `ExeRun` misuse (rejected claim) | Drive only from the verified `0x647BEB` hook; never from `ExeRun` |
| Stale/limbo/null array entries | `if (!pT) continue; if (pT->InLimbo) continue;` before deref |
| Fixed-cap overflow | bounds-check `n_own < N_OWN` / `n_enemy < N_ENEMY`; clamp, never overrun |
| Maphack | enemy gate = virtual `DiscoveredBy(pAgent)` only; own units bypass intentionally |
| Packing mismatch w/ Python | all OBS structs `#pragma pack(1)` + `static_assert(sizeof...)`; bump `version` on change |
| Hook coordination w/ ProtocolZero/Phobos | `DEFINE_HOOK_AGAIN` at `0x647BEB` (don't move/resize); stub reads no regs, returns 0 |
| Handle leak | idempotent `CreateMapping`; `DestroyMapping` on `DLL_PROCESS_DETACH` |

**Verified-fact ledger:** `0x647BEB`/`0x9` (`ProtocolZero.Hook.cpp:38`); `Unsorted::CurrentFrame`@`0xA8ED84`;
`TechnoClass::Array`@`0xA8EC78u`; `TechnoClass::Owner` (`TechnoClass.h:639`); `ObjectClass::DiscoveredBy`
(`ObjectClass.h:171`); `InLimbo` (`ObjectClass.h:296`); `GetMapCoords/GetHealthPercentage/GetTechnoType`
(`ObjectClass.h:233/203/86`); `HouseClass::CurrentPlayer`@`0xA83D4Cu`; `Balance/PowerOutput/PowerDrain/
SideIndex/Owned*` (`HouseClass.h:898/923/924/817/892–896`); `ArrayIndex` (`HouseClass.h:781`);
`FactoryClass::Object/GetProgress/QueuedObjects/Production` (`FactoryClass.h:119/61/118/117`);
`EventClass::OutList`@`0x00A802C8` + PRODUCE ctor @`0x4C6970`, `EventType::Produce=0xE` (Phase 2).
**Rejected:** `ExeRun`@`0x7CD810` as per-frame.
