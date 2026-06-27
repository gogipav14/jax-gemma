# jax-gemma — Non-Cheating RL/NN AI for C&C Yuri's Revenge

A neural-net agent that plays **Command & Conquer: Yuri's Revenge** (vanilla, via CnCNet)
*stronger* than the stock "Brutal/Extreme" AI but **without cheating** — bound by the exact
same rules a human player is.

## Why this is "non-cheating"

The stock YR AI cheats: it produces from multiple war factories in parallel, bypassing the
one-unit-at-a-time shared production queue that constrains human players. We avoid this by
**driving a player house through the engine's normal player command path** (`EventClass`
events), *not* the engine's AI/TeamType/Trigger code path. Because the agent issues the same
events the human UI issues, the shared production queue, economy, build times, and fog of war
all apply to it automatically.

## Architecture

```
gamemd.exe (vanilla YR, lockstep)
  └─ Syringe-injected bridge DLL (fork of CnCNet/yrpp-spawner + YRpp)
        ├─ per-logic-frame hook → serialize ONE house's view → shared memory
        └─ command intake → build EventClass → enqueue on the PLAYER path
                      │  shared memory / local socket
                      ▼
  Python / JAX harness (this repo)
        ├─ yr_env/   Gym-style reset()/step() over the shared-mem bridge
        ├─ seed/     3 small LLMs emit rule-legal macro decisions (offline, one-time)
        ├─ bc/       behavioral cloning: distill seed traces → NN warm start
        ├─ policy/   JAX NN policy (CNN + entity encoder + factored action heads)
        └─ league/   self-play PPO; agents co-adapt ("learn from one another")
```

## Layout

| Dir | Purpose |
|-----|---------|
| `bridge/`  | C++ Syringe DLL: game-state export + `EventClass` action injection (fork of yrpp-spawner) |
| `yr_env/`  | Python Gym wrapper over the shared-memory IPC |
| `seed/`    | 3 seed commanders — `google/gemma-4-E2B-it`, `HuggingFaceTB/SmolLM3-3B`, `Qwen/Qwen3.5-0.8B` (Unsloth 4-bit/GGUF, CPU) |
| `bc/`      | behavioral-cloning / distillation trainer (JAX) |
| `policy/`  | NN policy modules (JAX) |
| `league/`  | self-play / PPO league runner |
| `replays/` | optional CnCNet `.rep` ingestion → extra BC data |
| `docs/`    | bridge contract (shared-mem layout, action schema) + notes |
| `scripts/` | build / launch helpers |

## Status

See [`docs/STATUS.md`](docs/STATUS.md). Implemented phase-by-phase; current focus: **Phase 0**
(build the unmodified spawner DLL and confirm injection).

## Key constraints

- **Target = vanilla YR via CnCNet** at `C:\Program Files (x86)\Steam\steamapps\common\Command & Conquer Red Alert II\`. NOT Mental Omega (Ares-based, incompatible with yrpp-spawner).
- **CPU-only** through Phase 4 (no NVIDIA GPU on this machine; JAX runs CPU). GPU/cloud decision deferred to the self-play league (Phase 5).
- Train/run **offline only**. Never point the bot at the ranked CnCNet ladder — that would be cheating against humans.

## Plan

Full plan: `C:\Users\Lenovo\.claude\plans\does-it-have-to-jazzy-crane.md`.
