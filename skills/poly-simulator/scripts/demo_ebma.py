#!/usr/bin/env python3
"""
Demo script showing EBMA weight updates in action.

Simulates 3 agents making predictions on 5 markets, then calculates EBMA weights.
"""

import json
from datetime import datetime, timezone, timedelta
from learning import (
    update_agent_weights,
    select_strategy,
    update_strategy_prior,
    calculate_weight_divergence,
    check_groupthink,
)


def demo_ebma_weights():
    """Demo EBMA weight calculation."""
    print("\n" + "=" * 70)
    print("  EBMA WEIGHT UPDATE DEMO")
    print("=" * 70)

    # Initialize state
    state = {
        "agent_weights": {},
        "strategy_priors": {},
    }

    # Simulate 5 resolved markets
    markets = [
        {
            "name": "Will Bitcoin hit $100K by EOY?",
            "market_price": 0.35,
            "outcome": 1.0,  # YES
            "agent_predictions": {
                "sentinel": 0.65,  # Correct contrarian
                "oracle": 0.40,    # OK
                "maverick": 0.25,  # Wrong
            },
        },
        {
            "name": "Will Trump win 2024?",
            "market_price": 0.60,
            "outcome": 0.0,  # NO
            "agent_predictions": {
                "sentinel": 0.45,  # Correct
                "oracle": 0.55,    # OK
                "maverick": 0.70,  # Wrong
            },
        },
        {
            "name": "Will GDP grow >3% this quarter?",
            "market_price": 0.50,
            "outcome": 0.0,  # NO
            "agent_predictions": {
                "sentinel": 0.30,  # Correct
                "oracle": 0.48,    # Very close
                "maverick": 0.55,  # Slightly wrong
            },
        },
        {
            "name": "Will Ukraine war end by March?",
            "market_price": 0.20,
            "outcome": 0.0,  # NO
            "agent_predictions": {
                "sentinel": 0.15,  # Correct
                "oracle": 0.18,    # Correct
                "maverick": 0.40,  # Wrong
            },
        },
        {
            "name": "Will Ethereum break $5000?",
            "market_price": 0.45,
            "outcome": 1.0,  # YES
            "agent_predictions": {
                "sentinel": 0.70,  # Correct
                "oracle": 0.50,    # OK
                "maverick": 0.30,  # Wrong
            },
        },
    ]

    print("\nProcessing 5 resolved markets...\n")

    for i, market in enumerate(markets, 1):
        # Calculate consensus
        preds = list(market["agent_predictions"].values())
        consensus = sum(preds) / len(preds)

        print(f"[Market {i}] {market['name']}")
        print(f"  Market price: {market['market_price']:.2f} | Outcome: {market['outcome']:.2f}")
        print(f"  Agent predictions:")

        for agent, pred in market["agent_predictions"].items():
            error = abs(pred - market["outcome"])
            print(f"    {agent:12s}: {pred:.2f} (error: {error:.2f})")

        # Update weights
        state = update_agent_weights(
            state=state,
            agent_predictions=market["agent_predictions"],
            outcome=market["outcome"],
            consensus_prob=consensus,
            market_price=market["market_price"],
        )

        # Show current weights
        weights = {name: data["weight"] for name, data in state["agent_weights"].items()}
        print(f"  Updated weights: {json.dumps({k: round(v, 3) for k, v in weights.items()})}")
        print()

    # Final summary
    print("=" * 70)
    print("  FINAL WEIGHTS (after 5 markets)")
    print("=" * 70)

    weights = {name: data["weight"] for name, data in state["agent_weights"].items()}
    divergence = calculate_weight_divergence(weights)

    for name in sorted(weights.keys(), key=lambda x: -weights[x]):
        weight = weights[name]
        bar = "#" * int(weight * 100)
        pred_count = len(state["agent_weights"][name]["predictions"])
        avg_brier = sum(p["brier_score"] for p in state["agent_weights"][name]["predictions"]) / pred_count
        print(f"  {name:12s} {weight:.3f} {bar}")
        print(f"               Avg Brier: {avg_brier:.4f} | {pred_count} predictions")

    print(f"\n  Weight divergence: {divergence:.3f} (max - min)")

    # Check groupthink
    last_market_preds = list(markets[-1]["agent_predictions"].values())
    groupthink = check_groupthink(last_market_preds)
    print(f"  Groupthink detected: {groupthink}")

    print("=" * 70)


def demo_thompson_sampling():
    """Demo Thompson Sampling strategy selection."""
    print("\n" + "=" * 70)
    print("  THOMPSON SAMPLING DEMO")
    print("=" * 70)

    state = {
        "strategy_priors": {
            "edge_hunter": {"alpha": 1.0, "beta": 1.0},
            "whale_follower": {"alpha": 1.0, "beta": 1.0},
            "contrarian": {"alpha": 1.0, "beta": 1.0},
        }
    }

    print("\nSimulating 10 trading cycles with performance feedback...\n")

    # Simulate 10 cycles
    # edge_hunter performs well, whale_follower poor, contrarian mixed
    performance = {
        "edge_hunter": [True, True, False, True, True, True, False, True, True, True],
        "whale_follower": [False, False, True, False, False, False, True, False, False, False],
        "contrarian": [True, False, True, False, True, True, False, True, False, True],
    }

    for cycle in range(10):
        # Select strategy
        chosen = select_strategy(state)
        print(f"[Cycle {cycle+1}] Selected: {chosen}")

        # Get outcome for this cycle
        success = performance[chosen][cycle]
        print(f"          Outcome: {'SUCCESS' if success else 'FAILURE'}")

        # Update prior
        state = update_strategy_prior(state, chosen, success)

        # Show current priors
        print(f"          Priors:")
        for strat, prior in state["strategy_priors"].items():
            alpha = prior["alpha"]
            beta = prior["beta"]
            mean = alpha / (alpha + beta)
            print(f"            {strat:15s} α={alpha:.1f}, β={beta:.1f} | Mean={mean:.2%}")
        print()

    # Final summary
    print("=" * 70)
    print("  FINAL STRATEGY PRIORS")
    print("=" * 70)

    for strat, prior in state["strategy_priors"].items():
        alpha = prior["alpha"]
        beta = prior["beta"]
        mean = alpha / (alpha + beta)
        total_samples = int(alpha + beta - 2)
        success_rate = (alpha - 1) / total_samples if total_samples > 0 else 0
        print(f"  {strat:15s} α={alpha:.1f}, β={beta:.1f} | Mean={mean:.2%} | Success: {success_rate:.0%} ({total_samples} samples)")

    print("=" * 70)


if __name__ == "__main__":
    demo_ebma_weights()
    demo_thompson_sampling()

    print("\n✅ EBMA and Thompson Sampling systems operational!\n")
