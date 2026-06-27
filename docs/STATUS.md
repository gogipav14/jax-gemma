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
- [ ] Scaffold repo + git init
- [ ] Clone `CnCNet/yrpp-spawner` (+ `YRpp` submodule) into `bridge/`
- [ ] Confirm C++ toolchain (VS2022 Build Tools + C++ workload)
- [ ] Build unmodified `CnCNet-Spawner.dll`
- [ ] Inject into a local skirmish and confirm it launches

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
