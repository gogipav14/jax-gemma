"""LLM 'commander' — the slow strategic thinking layer (Ollama-backed).

Reads a game-state summary, asks a local Ollama model to REASON about strategy,
and returns a structured directive (JSON) that the fast executor turns into
EventClass/OutList actions (the non-cheating player path). Runs at a slow tick
(~5-15s), NOT per frame, so we can afford a capable model.

Decoupled from JAX/RL: talks to Ollama over HTTP (localhost:11434). Swap models
by changing `model=` (gemma4 now; qwen2.5-coder:7b / qwen3.6:27b once pulled).

Demo:  python commander/commander.py            # uses a sample early-game state
       python commander/commander.py qwen2.5-coder:7b
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

OLLAMA_CHAT = "http://localhost:11434/api/chat"


def _load_playbook() -> str:
    """The reverse-engineered stock-AI decision table that the analysis agent DISCOVERED, loaded
    AS-IS and fed to the commander — so the LLM reasons over the real playbook, not a hand-summary."""
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "stock-ai-blueprint.md")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


PLAYBOOK = _load_playbook()


def _messages(user_content: str):
    """System prompt + the discovered playbook (as-is) + the live battlefield query."""
    msgs = [{"role": "system", "content": SYSTEM}]
    if PLAYBOOK:
        msgs.append({"role": "system", "content":
                     "DISCOVERED ENEMY AI PLAYBOOK (reverse-engineered from the game files; "
                     "use it as-is to anticipate and counter):\n\n" + PLAYBOOK})
    msgs.append({"role": "user", "content": user_content})
    return msgs

# The commander plays FAIRLY — this mirrors the constraints our agent accepts vs.
# the stock AI's cheats (see docs/ai-audit.md): serial production per queue, real
# economy, fog of war. The model must win on strategy, not engine privileges.
SYSTEM = """You are the strategic commander for a Command & Conquer: Yuri's Revenge player.
You play FAIRLY, exactly like a human — NOT like the cheating AI:
- ONE unit at a time per production queue (a War Factory builds one vehicle at a time).
- REAL economy: you must build refineries and protect harvesters to earn credits.
- FOG OF WAR: you only know enemy units you can currently see; scout to learn more.

Below this message you are given the ENEMY stock AI's ACTUAL decision table — reverse-engineered
from the game files (the "discovered playbook"). Treat it as ground truth: anticipate which teams
its triggers will build, counter its real unit compositions, and hold YOURSELF to equally strong,
non-cheating play drawn from it. Read the battlefield report NUMBERS precisely; do not assume
threats that are not listed (if artillery=0, there is no artillery).

You think at a high level and emit a short strategic DIRECTIVE. A fast executor will
carry it out. Be decisive and concrete. Output ONLY JSON matching this schema:
{
  "assessment": "<one sentence: read of the situation>",
  "strategy": "rush | boom | tech | defend | harass",
  "priority_build_order": ["<structure or unit>", "..."],   // next 4-6 items, in order
  "army_composition": {"<unit>": <count>, "...": 0},          // the army to aim for
  "stance": "aggressive | defensive | expand",
  "objectives": ["<short actionable goal>", "..."],           // 2-4 goals this tick
  "reasoning": "<2-3 sentences of chain-of-thought>"
}"""


def summarize_state(obs: dict) -> str:
    """Turn an OBS dict (header + globals + counts) into a concise text briefing."""
    g = obs.get("globals", obs)  # accept flat or nested
    side = {0: "Allied", 1: "Soviet", 4: "Yuri"}.get(obs.get("side_index", g.get("side_index", -1)), "Unknown")
    lines = [
        f"Frame: {obs.get('frame_seq', '?')}   Faction: {side}   Map: {obs.get('map', '1v7 skirmish')}",
        f"Credits: {g.get('credits', '?')}   Power: {g.get('power_output', 0)}/{g.get('power_drain', 0)} (out/drain)",
        f"Your forces: {g.get('owned_buildings', 0)} buildings, {g.get('owned_units', 0)} vehicles, "
        f"{g.get('owned_infantry', 0)} infantry, {g.get('owned_aircraft', 0)} aircraft, {g.get('owned_navy', 0)} naval",
        f"Enemy units currently VISIBLE to you: {obs.get('n_enemy', g.get('n_enemy', 0))} "
        f"(others are hidden by fog).",
    ]
    return "\n".join(lines)


def think(obs: dict, model: str = "gemma4", timeout: int = 300, roster=None) -> dict:
    """Ask the model for a strategic directive. Returns a parsed dict.

    roster: optional list of buildable structure names to GROUND the model (prevents
    hallucinated units; the agent can only build/map these)."""
    briefing = summarize_state(obs)
    ground = ""
    if roster:
        ground = ("\n\nYou may ONLY order these structures (use these EXACT names in "
                  "priority_build_order): " + ", ".join(roster))
    payload = {
        "model": model,
        "messages": _messages(f"Current game state:\n{briefing}{ground}\n\nGive your directive as JSON."),
        "stream": False,
        "format": "json",   # force structured output — key for a reliable agent
        # think=False: gemma4/Qwen3.x are reasoning models — without this they spend the
        # whole budget in the hidden 'thinking' channel and return empty content. We want
        # the directive directly. (For a deliberate slow tick you could enable it + raise
        # num_predict to let it reason first.)
        "think": False,
        "keep_alive": "10m",  # hold the model resident between ticks (no cold reload)
        "options": {"temperature": 0.4, "num_predict": 600},
    }
    req = urllib.request.Request(
        OLLAMA_CHAT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    content = resp["message"]["content"]
    return json.loads(content)


def command(briefing: str, model: str = "gemma4", timeout: int = 300) -> dict:
    """Like think(), but takes a pre-built battlefield briefing string (so the live hierarchical
    loop can include threat detail). Returns the parsed strategic directive dict."""
    payload = {
        "model": model,
        "messages": _messages(f"Current battlefield report:\n{briefing}\n\nGive your directive as JSON."),
        "stream": False, "format": "json", "think": False, "keep_alive": "10m",
        "options": {"temperature": 0.4, "num_predict": 600},
    }
    req = urllib.request.Request(OLLAMA_CHAT, data=json.dumps(payload).encode(),
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read())
    return json.loads(resp["message"]["content"])


# A representative early-game OBS (the 1v7 Andalusia state our Phase-1 bridge actually read).
SAMPLE_OBS = {
    "frame_seq": 300, "side_index": 4, "map": "[8] Andalusia (1 human vs 7 AI)",
    "globals": {
        "credits": 10000, "power_output": 0, "power_drain": 0, "side_index": 4,
        "owned_units": 6, "owned_buildings": 0, "owned_infantry": 7,
        "owned_aircraft": 0, "owned_navy": 0,
    },
    "n_enemy": 3,
}


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4"
    print(f"=== Commander thinking with model '{model}' ===\n")
    print("STATE BRIEFING:\n" + summarize_state(SAMPLE_OBS) + "\n")
    try:
        directive = think(SAMPLE_OBS, model=model)
    except Exception as e:
        print(f"[error] {e}\n(Is Ollama running and the model pulled? `ollama list`)")
        sys.exit(1)
    print("DIRECTIVE (the 'thinking' output):\n")
    print(json.dumps(directive, indent=2))
