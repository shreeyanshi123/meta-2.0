#!/usr/bin/env python3
"""Smoke test: starts env, runs 5 random-verdict rounds via HTTP API,
prints total reward and breakdown, exits non-zero on any exception.

Usage:
    python scripts/smoke_test.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Tribunal smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    client = httpx.Client(base_url=base, timeout=30.0)

    print("=" * 60)
    print("  🏛️  Tribunal Smoke Test")
    print("=" * 60)

    # ── Health check ──
    print("\n[1/4] Health check...")
    r = client.get("/health")
    r.raise_for_status()
    health = r.json()
    print(f"  ✓ ok={health['ok']}  version={health['version']}")

    # ── Info ──
    print("\n[2/4] Environment info...")
    r = client.get("/info")
    r.raise_for_status()
    info = r.json()
    print(f"  ✓ {info['name']}  theme={info['theme']}")

    # ── Reset ──
    print("\n[3/4] Reset...")
    r = client.post("/reset")
    r.raise_for_status()
    obs = r.json()
    print(f"  ✓ round_id={obs['round_id']}  workers={len(obs['worker_outputs'])}")

    # ── Step loop (5 random verdicts) ──
    print("\n[4/4] Running 5 step rounds...")
    rng = random.Random(42)
    total_reward = 0.0
    failure_types = ["HALLUCINATION", "COLLUSION", "MANIPULATION", "SILENCE"]

    for i in range(5):
        # Generate random verdict
        num_accused = rng.randint(0, 3)
        accused = rng.sample(range(4), num_accused)
        ft = {str(wid): rng.choice(failure_types) for wid in accused}
        conf = {str(wid): round(rng.uniform(0.1, 0.9), 2) for wid in range(4)}
        explanation = (
            f"Round {i}: I suspect workers {accused} because their outputs "
            f"show signs of manipulation and inconsistency."
        )

        verdict = {
            "verdict": {
                "accused": accused,
                "failure_types": ft,
                "explanation": explanation,
                "per_worker_confidence": conf,
            }
        }

        r = client.post("/step", json=verdict)
        if r.status_code != 200:
            print(f"  ✗ Round {i}: HTTP {r.status_code} — {r.text[:200]}")
            # If episode is done, reset and try again
            if "done" in r.text.lower() or "reset" in r.text.lower():
                print("    → Resetting for next round...")
                r2 = client.post("/reset")
                r2.raise_for_status()
                r = client.post("/step", json=verdict)
                r.raise_for_status()
            else:
                return 1

        result = r.json()
        reward = result["reward"]
        done = result["done"]
        total_reward += reward

        breakdown = result["info"]["breakdown"]
        print(
            f"  Round {i}: reward={reward:+.4f}  done={done}  "
            f"id={breakdown['identification']:.3f}  "
            f"type={breakdown['type_classification']:.3f}  "
            f"expl={breakdown['explanation_quality']:.3f}  "
            f"fp={breakdown['false_positive_penalty']:.3f}  "
            f"calib={breakdown['calibration']:.3f}  "
            f"hack={breakdown['anti_hack_penalty']:.3f}"
        )

        if done:
            print("  → Episode done, resetting...")
            r = client.post("/reset")
            r.raise_for_status()

    # ── State ──
    r = client.get("/state")
    r.raise_for_status()
    state = r.json()

    print("\n" + "=" * 60)
    print(f"  Total reward: {total_reward:+.4f}")
    print(f"  State: round={state['round_index']}  cumulative={state['cumulative_reward']:.4f}")
    print("  ✅ Smoke test PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n❌ Smoke test FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
