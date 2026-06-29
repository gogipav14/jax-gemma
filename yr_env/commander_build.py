"""Full read->think->act loop: a GROUNDED LLM commander picks the build order, the game is
launched, and the autonomous builder executes it (deploy MCV + build the chosen structures) —
all via the non-cheating ACT/OutList path.

We think FIRST (no game running), then launch, so the match isn't idle while the (slow, CPU)
LLM reasons. Requires the Release bridge DLL deployed + a calm spawn.ini.

    PYTHONPATH=yr_env;commander  python yr_env/commander_build.py [model]
"""
import os
import subprocess
import sys
import time

from commander import think, SAMPLE_OBS   # commander/ on PYTHONPATH
import build_base                          # yr_env/ on PYTHONPATH

GAME = r"C:\Program Files (x86)\Steam\steamapps\common\Command & Conquer Red Alert II"
ROSTER = ["Power Plant", "Refinery", "Barracks", "War Factory", "Battle Lab", "Radar"]


def launch_game():
    syringe = os.path.join(GAME, "Syringe.exe")
    subprocess.Popen(
        [syringe, "-i=Ares.dll", "-i=CnCNet-Spawner.dll", "-i=Phobos.dll", "gamemd-spawn.exe",
         "--args=-SPAWN -LOG -CD -Include -Inheritance"],
        cwd=GAME)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "gemma4"
    print(f"=== Commander ({model}) deciding the build order (grounded) ===")
    d = think(SAMPLE_OBS, model=model, roster=ROSTER)
    order = d.get("priority_build_order", d.get("build_order", [])) or []
    print(f"  strategy:  {d.get('strategy')}")
    print(f"  reasoning: {(d.get('reasoning') or '')[:180]}")
    print(f"  >>> BUILD ORDER (from the LLM): {order}\n")

    print("=== Launching the match, then executing the build order ===")
    launch_game()
    time.sleep(2)
    build_base.main(order_override=order)


if __name__ == "__main__":
    main()
