#!/usr/bin/env python3
"""Quickstart: 10-round Judge loop with a random verdict policy.

Prints per-component reward trajectory as a rich table.

Usage:
    # Start the server first:
    #   bash scripts/run_local.sh
    # Then run this:
    python client/examples/quickstart.py [--url http://localhost:8000] [--rounds 10]
"""

from __future__ import annotations

import argparse
import random
import sys

from rich.console import Console
from rich.table import Table

from tribunal_client import TribunalClient
from tribunal_shared.schemas import FailureType, JudgeVerdict


FAILURE_TYPES = list(FailureType)
# Remove CLEAN from the accusation options
ACCUSATION_TYPES = [ft for ft in FAILURE_TYPES if ft != FailureType.CLEAN]

console = Console()


def random_verdict(rng: random.Random) -> JudgeVerdict:
    """Generate a random JudgeVerdict for testing."""
    num_accused = rng.randint(0, 3)
    accused = rng.sample(range(4), num_accused)
    failure_types = {wid: rng.choice(ACCUSATION_TYPES) for wid in accused}
    per_worker_confidence = {wid: round(rng.uniform(0.1, 0.95), 2) for wid in range(4)}
    explanation = (
        f"I suspect workers {accused} of misbehaviour. "
        + " ".join(
            f"Worker {wid} shows signs of {failure_types[wid].value.lower()} "
            f"because the output is inconsistent with the task brief."
            for wid in accused
        )
        if accused
        else "All workers appear to have performed their tasks correctly."
    )
    return JudgeVerdict(
        accused=accused,
        failure_types=failure_types,
        explanation=explanation,
        per_worker_confidence=per_worker_confidence,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tribunal quickstart")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    console.print("\n[bold cyan]🏛️  AI Agent Oversight Tribunal — Quickstart[/bold cyan]")
    console.print(f"   Server: {args.url}")
    console.print(f"   Rounds: {args.rounds}")
    console.print(f"   Seed: {args.seed}\n")

    # Build the results table
    table = Table(title="Per-Round Reward Trajectory", show_lines=True)
    table.add_column("Round", style="bold", justify="right")
    table.add_column("Accused", justify="center")
    table.add_column("r_ident", justify="right", style="green")
    table.add_column("r_type", justify="right", style="green")
    table.add_column("r_explain", justify="right", style="green")
    table.add_column("r_calib", justify="right", style="green")
    table.add_column("r_fp", justify="right", style="red")
    table.add_column("r_hack", justify="right", style="red")
    table.add_column("r_total", justify="right", style="bold yellow")
    table.add_column("Done", justify="center")

    with TribunalClient(args.url) as client:
        # Health check
        health = client.health()
        console.print(f"[green]✓[/green] Health: ok={health['ok']}  v={health['version']}")

        # Reset
        obs = client.reset(seed=args.seed)
        console.print(f"[green]✓[/green] Reset: round_id={obs.round_id}\n")

        total_reward = 0.0

        for i in range(args.rounds):
            verdict = random_verdict(rng)
            result = client.step(verdict)

            pr = result.info.get("per_reward", {})
            total_reward += result.reward

            table.add_row(
                str(i),
                str(verdict.accused),
                f"{pr.get('r_identification', 0):+.3f}",
                f"{pr.get('r_type', 0):+.3f}",
                f"{pr.get('r_explain', 0):+.3f}",
                f"{pr.get('r_calib', 0):+.3f}",
                f"{pr.get('r_fp_penalty', 0):+.3f}",
                f"{pr.get('r_antihack', 0):+.3f}",
                f"{pr.get('r_total', result.reward):+.3f}",
                "✓" if result.done else "",
            )

            if result.done:
                # Reset for next episode
                if i < args.rounds - 1:
                    obs = client.reset(seed=args.seed + i + 1)

        console.print(table)

        # Final state
        state = client.state()
        console.print(f"\n[bold]Cumulative reward: {total_reward:+.4f}[/bold]")
        console.print(f"State: round={state.round_index}  total_rounds={state.total_rounds}")

    console.print("\n[bold green]✅ Quickstart complete![/bold green]\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]", highlight=False)
        import traceback
        traceback.print_exc()
        sys.exit(1)
