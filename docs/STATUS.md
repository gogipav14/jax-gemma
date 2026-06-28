# Status

| Phase | Description | State |
|-------|-------------|-------|
| 0 | Setup & baseline spawner build (inject + launch skirmish) | **in progress** |
| 1 | Observation bridge (per-frame state → shared memory) | not started |
| 2 | Action injection via `EventClass` + non-cheat proof | not started |
| 3 | Python Gym env (`yr_env`) over the bridge | not started |
| 4 | LLM seed (×3) + behavioral cloning warm start | not started |
| 5 | Self-play league (PPO) | not started |
| 6 | Evaluation vs Brutal/Extreme + non-cheat audit | not started |

## Phase 0 checklist
- [x] Scaffold repo + git init
- [x] Clone `CnCNet/yrpp-spawner` (+ `YRpp` submodule, HTTPS) into `bridge/yrpp-spawner`
- [x] Confirm C++ toolchain — VS2022 BuildTools (MSBuild + MSVC 14.43.34808 + Win10 SDK 22621 + v143). cl.exe x86 present.
- [x] Pin non-cheat injection API — `EventClass::OutList.Add()` @0x00A802C8 + ctor addrs (see bridge-contract.md)
- [x] ATL component installed (via VS Installer GUI)
- [x] Build `CnCNet-Spawner.dll` (Debug + Debug-CnCNetYR | Win32) via `scripts/build.ps1` — 0 warnings/errors
- [x] **Injection PROVEN** — `syringe.log`: "Recognized DLL: CnCNet-Spawner.dll" + "Done (2736 hooks added)", game process ran, no crash/except. Original DLL restored after.

### Phase 0 findings (important)
- This install runs **Ares.dll + Phobos.dll + CnCNet-Spawner.dll** together (live spawner = "CnCNet YR, hardened" 0.0.0.16, Phobos 0.4.0.2). Our Phase-1 hook must avoid colliding with Phobos hooks → use `DEFINE_HOOK_AGAIN`.
- **Correct launch command** (from `Client\client.log` / `QuickMatch.ini`) — game args go inside `--args="..."`:
  `Syringe.exe -i=Ares.dll -i=CnCNet-Spawner.dll -i=Phobos.dll gamemd-spawn.exe --args="-SPAWN -LOG -CD -Include -Inheritance"`
  (A bare `-SPAWN` is eaten by SyringeEx → "game started incorrectly" banner. `-LOG` enables debug.log.)
  Now wired into `scripts/deploy-and-test.ps1 -Launch`.

### Phase 1 — DESIGNED (see docs/phase1-plan.md)
Verified per-frame hook `DEFINE_HOOK_AGAIN(0x647BEB, ...)`; fog-honored enumeration via
`TechnoClass::Array` + `ObjectClass::DiscoveredBy(pAgent)`; OBS structs + exact YRpp accessors;
`src/Bridge/Bridge.{h,cpp,Hook.cpp}` + vcxproj edits; zero crash risks flagged.

## Environment (verified)
- Game: `C:\Program Files (x86)\Steam\steamapps\common\Command & Conquer Red Alert II\` (vanilla YR / CnCNet)
- Baseline AIs: `...\INI\Game Options\AI\{Brutal AI,Extreme AI,Extreme AI Beta}.ini`
- Python 3.13.5, JAX 0.9.2 (CPU only), Git 2.45.1
- VS2019 present; **VS2022 + C++ workload needed for yrpp-spawner** (TODO)
- No NVIDIA GPU (AMD Radeon) → JAX CPU-only; GPU decision deferred to Phase 5

## Decisions
- Action space v1: macro + group commands (per-unit micro scripted)
- Seed commanders: `google/gemma-4-E2B-it`, `HuggingFaceTB/SmolLM3-3B`, `Qwen/Qwen3.5-0.8B` (Unsloth 4-bit/GGUF, offline, one-time)
- Non-cheat principle: drive a player house via `EventClass` on the player path; never the AI path
