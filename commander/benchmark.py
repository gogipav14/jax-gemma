"""Benchmark commander models on the same game state: latency + directive quality.

Runs each Ollama model through commander.think() on the sample early-game OBS and
prints (total seconds, strategy, build order, army). Total time includes model load
(Ollama swaps models that don't co-fit in RAM), which is realistic for a slow tick.
"""
import time

from commander import think, SAMPLE_OBS

# ready now: gemma4, qwen3.5:4b, qwen2.5-coder:7b/3b. (9b / 27b still downloading.)
MODELS = ["qwen3.5:4b", "qwen2.5-coder:7b", "qwen2.5-coder:3b", "gemma4"]


def main():
    for m in MODELS:
        t0 = time.time()
        try:
            d = think(SAMPLE_OBS, model=m, timeout=400)
            dt = time.time() - t0
            bo = d.get("priority_build_order", d.get("build_order", []))
            print(f"\n=== {m}   ({dt:.1f}s) ===")
            print(f"  strategy: {d.get('strategy', '?')}")
            print(f"  build:    {bo[:5]}")
            print(f"  army:     {d.get('army_composition', {})}")
        except Exception as e:
            print(f"\n=== {m} ===   FAILED: {e}")


if __name__ == "__main__":
    main()
