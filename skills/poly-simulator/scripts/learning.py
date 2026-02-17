#!/usr/bin/env python3
"""
Upgraded Self-Learning System for PolyClaw — EBMA + Thompson Sampling + Reflexion.

Replaces linear weight scaling with Bayesian ensemble methods:
- EBMA (Ensemble Bayesian Model Averaging): accuracy² * uniqueness * recency
- Thompson Sampling: strategy selection via Beta distributions
- Reflexion: meta-learning from mistakes (stub for future)

Usage:
    python learning.py --test   # Run self-tests with mock data
"""

import argparse
import json
import math
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


# --- Dataclasses ---

@dataclass
class AgentPerformance:
    """Tracks per-agent prediction history for EBMA weighting."""
    agent_name: str
    predictions: list = field(default_factory=list)  # {timestamp, prob, outcome, brier_score}
    weight: float = 0.2
    uniqueness_bonus: float = 1.0
    last_update: Optional[str] = None


@dataclass
class StrategyPrior:
    """Beta distribution priors for Thompson Sampling."""
    strategy_name: str
    alpha: float = 1.0  # Success count
    beta: float = 1.0   # Failure count
    last_sampled: Optional[str] = None


@dataclass
class Reflection:
    """Meta-learning reflection after a prediction resolves."""
    agent_name: str
    market_category: str
    prediction: float
    outcome: float
    reasoning: str
    lesson: str
    timestamp: str


# --- EBMA Weight Calculator ---

def calculate_brier_score(predictions: list) -> float:
    """
    Calculate mean Brier score for a list of predictions.

    Brier score: mean((p_i - outcome_i)^2)
    Lower is better (perfect = 0.0, random = 0.25)

    Args:
        predictions: List of {prob, outcome} dicts

    Returns:
        Mean Brier score (0.0 = perfect, 0.25 = random)
    """
    if not predictions:
        return 0.25  # Random baseline

    scores = [(p["prob"] - p["outcome"]) ** 2 for p in predictions]
    return sum(scores) / len(scores)


def calculate_recency_factor(days_since: float, half_life: int = 30) -> float:
    """
    Exponential decay for old predictions.

    Formula: 2^(-days_since / half_life)

    Args:
        days_since: Days since prediction
        half_life: Half-life in days (default 30)

    Returns:
        Recency factor (1.0 = today, 0.5 = half_life days ago)
    """
    return 2 ** (-days_since / half_life)


def calculate_uniqueness_bonus(
    agent_correct: bool,
    agent_prob: float,
    consensus_prob: float,
    threshold: float = 0.10
) -> float:
    """
    Reward correct predictions when agent disagreed with consensus.

    Args:
        agent_correct: True if agent prediction was closer to outcome than market
        agent_prob: Agent's predicted probability
        consensus_prob: Consensus probability from other agents
        threshold: Minimum divergence to count as "disagreement" (default 0.10)

    Returns:
        Uniqueness bonus multiplier (1.10 for contrarian success, 1.0 otherwise)
    """
    disagreed = abs(agent_prob - consensus_prob) > threshold

    if agent_correct and disagreed:
        return 1.10  # 10% bonus for contrarian success
    return 1.0


def calculate_ebma_weight(
    agent_predictions: list,
    current_time: Optional[datetime] = None,
    weight_ceiling: float = 0.45,
    weight_floor: float = 0.05,
) -> float:
    """
    Calculate EBMA weight for an agent.

    Formula:
        weight = (accuracy_score² * uniqueness_bonus * recency_factor)

    Where:
        - accuracy_score = 1 / (1 + brier_score)
        - brier_score = mean((p - outcome)²)
        - uniqueness_bonus = 1.10 if contrarian success, else 1.0
        - recency_factor = exp(-days_since / 30)

    Args:
        agent_predictions: List of {timestamp, prob, outcome, uniqueness_bonus} dicts
        current_time: Current time (defaults to now)
        weight_ceiling: Maximum allowed weight (default 0.45)
        weight_floor: Minimum allowed weight (default 0.05)

    Returns:
        EBMA weight (clamped to [weight_floor, weight_ceiling])
    """
    if not agent_predictions:
        return weight_floor

    now = current_time or datetime.now(timezone.utc)

    # Calculate weighted components
    total_weighted = 0
    total_recency = 0

    for pred in agent_predictions:
        # Parse timestamp
        try:
            pred_time = datetime.fromisoformat(pred["timestamp"].replace("Z", "+00:00"))
            days_since = (now - pred_time).days
        except (ValueError, TypeError, KeyError):
            days_since = 0

        # Recency factor (exponential decay)
        recency = calculate_recency_factor(days_since, half_life=30)

        # Brier score for this prediction
        brier = (pred["prob"] - pred["outcome"]) ** 2
        accuracy = 1 / (1 + brier)

        # Uniqueness bonus (stored in prediction)
        uniqueness = pred.get("uniqueness_bonus", 1.0)

        # Combine: accuracy² * uniqueness * recency
        weighted_score = (accuracy ** 2) * uniqueness * recency

        total_weighted += weighted_score
        total_recency += recency

    # Average weighted score
    if total_recency > 0:
        raw_weight = total_weighted / total_recency
    else:
        raw_weight = weight_floor

    # Clamp to [floor, ceiling]
    return max(weight_floor, min(raw_weight, weight_ceiling))


def update_agent_weights(
    state: dict,
    agent_predictions: dict,  # {agent_name: prob}
    outcome: float,
    consensus_prob: float,
    market_price: float,
) -> dict:
    """
    Update agent weights after a market resolves using EBMA.

    Args:
        state: Current simulator state (contains agent_weights)
        agent_predictions: {agent_name: probability} dict
        outcome: Actual outcome (0.0 = NO, 1.0 = YES, or value in between)
        consensus_prob: Weighted consensus probability
        market_price: Market's implied probability

    Returns:
        Updated state dict with new weights
    """
    now = datetime.now(timezone.utc).isoformat()

    # Initialize agent weights if missing
    if "agent_weights" not in state:
        state["agent_weights"] = {}

    # Update each agent's prediction history
    for agent_name, prob in agent_predictions.items():
        if agent_name not in state["agent_weights"]:
            state["agent_weights"][agent_name] = {
                "predictions": [],
                "weight": 0.2,
                "uniqueness_bonus": 1.0,
                "last_update": None,
            }

        agent_data = state["agent_weights"][agent_name]

        # Calculate uniqueness bonus
        agent_error = abs(prob - outcome)
        market_error = abs(market_price - outcome)
        agent_correct = agent_error < market_error

        uniqueness_bonus = calculate_uniqueness_bonus(
            agent_correct=agent_correct,
            agent_prob=prob,
            consensus_prob=consensus_prob,
            threshold=0.10,
        )

        # Add prediction to history
        agent_data["predictions"].append({
            "timestamp": now,
            "prob": prob,
            "outcome": outcome,
            "uniqueness_bonus": uniqueness_bonus,
            "brier_score": (prob - outcome) ** 2,
        })

        # Keep last 100 predictions
        agent_data["predictions"] = agent_data["predictions"][-100:]

        # Recalculate EBMA weight
        new_weight = calculate_ebma_weight(agent_data["predictions"])
        agent_data["weight"] = new_weight
        agent_data["last_update"] = now

    # Normalize weights to sum = 1.0
    total_weight = sum(
        state["agent_weights"][name]["weight"]
        for name in agent_predictions.keys()
    )

    if total_weight > 0:
        for agent_name in agent_predictions.keys():
            state["agent_weights"][agent_name]["weight"] /= total_weight

    return state


# --- Thompson Sampling Strategy Selector ---

def sample_beta(alpha: float, beta: float) -> float:
    """
    Sample from Beta distribution using built-in betavariate.

    Args:
        alpha: Success count (α parameter)
        beta: Failure count (β parameter)

    Returns:
        Sample from Beta(α, β) distribution
    """
    return random.betavariate(alpha, beta)


def select_strategy(state: dict, strategies: list = None) -> str:
    """
    Select trading strategy using Thompson Sampling.

    Each strategy is modeled as Beta(α, β) distribution.
    Sample from each, choose the one with highest sample.

    Args:
        state: Simulator state (contains strategy_priors)
        strategies: List of strategy names (defaults to edge_hunter, whale_follower, contrarian)

    Returns:
        Selected strategy name
    """
    if strategies is None:
        strategies = ["edge_hunter", "whale_follower", "contrarian"]

    # Initialize priors if missing
    if "strategy_priors" not in state:
        state["strategy_priors"] = {}

    for strat in strategies:
        if strat not in state["strategy_priors"]:
            state["strategy_priors"][strat] = {
                "alpha": 1.0,
                "beta": 1.0,
                "last_sampled": None,
            }

    # Sample from each strategy's Beta distribution
    samples = {}
    for strat in strategies:
        prior = state["strategy_priors"][strat]
        samples[strat] = sample_beta(prior["alpha"], prior["beta"])

    # Choose strategy with highest sample
    chosen = max(samples, key=samples.get)

    # Update last_sampled timestamp
    state["strategy_priors"][chosen]["last_sampled"] = datetime.now(timezone.utc).isoformat()

    return chosen


def update_strategy_prior(
    state: dict,
    strategy_name: str,
    success: bool,
    decay: float = 0.98,
) -> dict:
    """
    Update strategy prior after observing performance.

    Args:
        state: Simulator state
        strategy_name: Name of strategy to update
        success: True if reward exceeded threshold
        decay: Daily decay factor for non-stationarity (default 0.98)

    Returns:
        Updated state dict
    """
    if "strategy_priors" not in state or strategy_name not in state["strategy_priors"]:
        return state

    prior = state["strategy_priors"][strategy_name]

    # Bayesian update
    if success:
        prior["alpha"] += 1.0
    else:
        prior["beta"] += 1.0

    # Apply decay (for non-stationary markets)
    prior["alpha"] *= decay
    prior["beta"] *= decay

    return state


# --- Reflexion Framework (Stub) ---

def generate_reflection(
    agent_name: str,
    prediction: float,
    outcome: float,
    reasoning: str,
    market_category: str = "unknown",
) -> Reflection:
    """
    Generate a reflection after prediction resolves.

    This is a STUB — in production, would use LLM to analyze mistake.

    Args:
        agent_name: Name of agent
        prediction: Agent's predicted probability
        outcome: Actual outcome
        reasoning: Agent's reasoning string
        market_category: Market category

    Returns:
        Reflection object
    """
    error = abs(prediction - outcome)

    # Simple heuristic lesson (would use LLM in production)
    if error > 0.4:
        lesson = f"High error ({error:.2f}) — re-examine base rates and priors"
    elif error > 0.2:
        lesson = f"Moderate error ({error:.2f}) — check recency bias"
    else:
        lesson = f"Low error ({error:.2f}) — calibration good, continue approach"

    return Reflection(
        agent_name=agent_name,
        market_category=market_category,
        prediction=prediction,
        outcome=outcome,
        reasoning=reasoning[:200],  # Truncate
        lesson=lesson,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def store_reflection(
    reflection: Reflection,
    data_dir: Path = Path(__file__).parent.parent.parent.parent / "data" / "reflections",
):
    """
    Store reflection to disk.

    Args:
        reflection: Reflection object
        data_dir: Directory to store reflections
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    agent_file = data_dir / f"{reflection.agent_name}.json"

    # Load existing reflections
    if agent_file.exists():
        reflections = json.loads(agent_file.read_text())
    else:
        reflections = []

    # Append new reflection
    reflections.append(asdict(reflection))

    # Keep last 100 reflections
    reflections = reflections[-100:]

    # Save
    agent_file.write_text(json.dumps(reflections, indent=2))


def get_relevant_reflections(
    agent_name: str,
    market_category: str,
    limit: int = 5,
    data_dir: Path = Path(__file__).parent.parent.parent.parent / "data" / "reflections",
) -> list:
    """
    Retrieve relevant reflections for an agent in a category.

    Args:
        agent_name: Name of agent
        market_category: Market category
        limit: Max reflections to return
        data_dir: Directory containing reflections

    Returns:
        List of reflection dicts (most recent first)
    """
    agent_file = data_dir / f"{agent_name}.json"

    if not agent_file.exists():
        return []

    reflections = json.loads(agent_file.read_text())

    # Filter by category
    filtered = [r for r in reflections if r["market_category"] == market_category]

    # Sort by timestamp descending
    filtered.sort(key=lambda r: r["timestamp"], reverse=True)

    return filtered[:limit]


# --- Diversity Metrics ---

def calculate_weight_divergence(weights: dict) -> float:
    """
    Calculate max - min of agent weights.

    High divergence (> 0.30) suggests one agent is dominating.

    Args:
        weights: {agent_name: weight} dict

    Returns:
        Weight divergence (max - min)
    """
    if not weights:
        return 0.0

    values = list(weights.values())
    return max(values) - min(values)


def calculate_consensus_spread(predictions: list) -> float:
    """
    Calculate standard deviation of agent predictions.

    Low spread (< 0.05) suggests groupthink.

    Args:
        predictions: List of probability values

    Returns:
        Standard deviation of predictions
    """
    if len(predictions) < 2:
        return 0.0

    mean = sum(predictions) / len(predictions)
    variance = sum((p - mean) ** 2 for p in predictions) / (len(predictions) - 1)
    return math.sqrt(variance)


def check_groupthink(predictions: list, threshold: float = 0.05) -> bool:
    """
    Check if agent predictions show groupthink (too similar).

    Args:
        predictions: List of probability values
        threshold: Std dev threshold below which is groupthink (default 0.05)

    Returns:
        True if groupthink detected
    """
    spread = calculate_consensus_spread(predictions)
    return spread < threshold


# --- Integration Helpers ---

def evolve_with_ebma(state: dict) -> Optional[dict]:
    """
    Suggest parameter evolution based on EBMA-weighted performance.

    Replaces simulator.py's suggest_evolution() with EBMA-aware version.

    Args:
        state: Simulator state

    Returns:
        Suggestion dict or None
    """
    # This is a wrapper — actual logic stays in simulator.py for now
    # Future: Move all evolution logic here

    # Calculate diversity metrics
    if "agent_weights" in state:
        weights = {name: data["weight"] for name, data in state["agent_weights"].items()}
        divergence = calculate_weight_divergence(weights)

        # If one agent is dominating (weight > 0.40), log warning
        max_weight = max(weights.values()) if weights else 0
        if max_weight > 0.40:
            print(f"  [WARNING] Agent dominance detected: max weight = {max_weight:.2f}")

    return None  # Placeholder for future full implementation


# --- CLI for Self-Tests ---

def run_self_tests():
    """Run self-tests with mock data."""
    print("\n" + "=" * 60)
    print("  LEARNING SYSTEM SELF-TESTS")
    print("=" * 60)

    # Test 1: Brier score calculation
    print("\n[TEST 1] Brier Score Calculation")
    predictions = [
        {"prob": 0.7, "outcome": 1.0},  # Good prediction
        {"prob": 0.3, "outcome": 0.0},  # Good prediction
        {"prob": 0.8, "outcome": 0.0},  # Bad prediction
    ]
    brier = calculate_brier_score(predictions)
    print(f"  Brier score: {brier:.4f} (expected ~0.23)")
    assert 0.20 < brier < 0.30, "Brier score out of expected range"

    # Test 2: Recency factor
    print("\n[TEST 2] Recency Factor (30-day half-life)")
    today = calculate_recency_factor(0, half_life=30)
    month_ago = calculate_recency_factor(30, half_life=30)
    three_months_ago = calculate_recency_factor(90, half_life=30)
    print(f"  Today: {today:.4f} (expected 1.0)")
    print(f"  30 days ago: {month_ago:.4f} (expected 0.5)")
    print(f"  90 days ago: {three_months_ago:.4f} (expected 0.125)")
    assert abs(today - 1.0) < 0.01
    assert abs(month_ago - 0.5) < 0.05

    # Test 3: Uniqueness bonus
    print("\n[TEST 3] Uniqueness Bonus")
    bonus_contrarian = calculate_uniqueness_bonus(
        agent_correct=True,
        agent_prob=0.7,
        consensus_prob=0.4,
        threshold=0.10,
    )
    bonus_conformist = calculate_uniqueness_bonus(
        agent_correct=True,
        agent_prob=0.7,
        consensus_prob=0.68,
        threshold=0.10,
    )
    print(f"  Contrarian success: {bonus_contrarian:.2f} (expected 1.10)")
    print(f"  Conformist success: {bonus_conformist:.2f} (expected 1.00)")
    assert bonus_contrarian == 1.10
    assert bonus_conformist == 1.00

    # Test 4: EBMA weight calculation
    print("\n[TEST 4] EBMA Weight Calculation")
    now = datetime.now(timezone.utc)
    agent_preds = [
        {
            "timestamp": (now - timedelta(days=5)).isoformat(),
            "prob": 0.8,
            "outcome": 1.0,
            "uniqueness_bonus": 1.10,
        },
        {
            "timestamp": (now - timedelta(days=10)).isoformat(),
            "prob": 0.6,
            "outcome": 1.0,
            "uniqueness_bonus": 1.0,
        },
    ]
    weight = calculate_ebma_weight(agent_preds, current_time=now)
    print(f"  EBMA weight: {weight:.4f} (expected > 0.20)")
    assert weight > 0.10, "Weight too low"

    # Test 5: Thompson Sampling
    print("\n[TEST 5] Thompson Sampling Strategy Selection")
    state = {
        "strategy_priors": {
            "edge_hunter": {"alpha": 10.0, "beta": 5.0},  # 67% success rate
            "whale_follower": {"alpha": 5.0, "beta": 10.0},  # 33% success rate
            "contrarian": {"alpha": 1.0, "beta": 1.0},  # Uniform prior
        }
    }

    # Sample 100 times, edge_hunter should win most often
    selections = {"edge_hunter": 0, "whale_follower": 0, "contrarian": 0}
    for _ in range(100):
        chosen = select_strategy(state.copy())
        selections[chosen] += 1

    print(f"  edge_hunter: {selections['edge_hunter']}%")
    print(f"  whale_follower: {selections['whale_follower']}%")
    print(f"  contrarian: {selections['contrarian']}%")
    assert selections["edge_hunter"] > 40, "Thompson sampling not working"

    # Test 6: Diversity metrics
    print("\n[TEST 6] Diversity Metrics")
    weights = {"sentinel": 0.15, "oracle": 0.40, "maverick": 0.25, "fundamental": 0.20}
    divergence = calculate_weight_divergence(weights)
    print(f"  Weight divergence: {divergence:.2f} (max - min)")

    predictions = [0.5, 0.52, 0.48, 0.51, 0.49]
    spread = calculate_consensus_spread(predictions)
    groupthink = check_groupthink(predictions, threshold=0.05)
    print(f"  Consensus spread: {spread:.4f}")
    print(f"  Groupthink: {groupthink} (expected True)")
    assert groupthink, "Groupthink detection failed"

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PolyClaw Learning System")
    parser.add_argument("--test", action="store_true", help="Run self-tests")

    args = parser.parse_args()

    if args.test:
        run_self_tests()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
