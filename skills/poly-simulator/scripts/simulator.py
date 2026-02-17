#!/usr/bin/env python3
"""
Polymarket Paper Trading Simulator with Self-Learning.

Usage:
    python simulator.py --start --strategy edge_hunter    # Start simulation
    python simulator.py --status                          # Show dashboard
    python simulator.py --evolve                          # Force learning cycle
    python simulator.py --graduation-check                # Check Web3 readiness
    python simulator.py --evolution-log                   # View parameter changes
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

# Local imports
try:
    from learning import (
        calculate_ebma_weight,
        update_agent_weights,
        select_strategy,
        update_strategy_prior,
        calculate_weight_divergence,
        calculate_consensus_spread,
        check_groupthink,
        generate_reflection,
        store_reflection,
    )
    LEARNING_MODULE_AVAILABLE = True
except ImportError:
    LEARNING_MODULE_AVAILABLE = False

# --- Config ---
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
SIM_DIR = DATA_DIR / "simulator"
STATE_FILE = SIM_DIR / "state.json"
EVOLUTION_LOG_FILE = SIM_DIR / "evolution_log.json"

# Simulation defaults
INITIAL_BANKROLL = 10_000
MAX_POSITION_PCT = 0.10
MIN_POSITION_USD = 10
MAX_OPEN_POSITIONS = 20

# Self-learning safety rails
MIN_TRADES_FOR_SUGGESTION = 10
MAX_PARAM_SHIFT_RATIO = 0.10
SHARPE_FLOOR = 0.5
WINDOW_DAYS = 30

# Strategy defaults
STRATEGY_DEFAULTS = {
    "edge_hunter": {
        "edge_threshold": 50,
        "position_size_pct": 0.05,
        "stop_loss_pct": 0.30,
        "min_confidence": 0.60,
        "min_agents_agree": 3,
    },
    "whale_follower": {
        "edge_threshold": 30,
        "position_size_pct": 0.03,
        "stop_loss_pct": 0.25,
        "min_confidence": 0.50,
        "min_agents_agree": 2,
    },
    "contrarian": {
        "edge_threshold": 40,
        "position_size_pct": 0.04,
        "stop_loss_pct": 0.35,
        "min_confidence": 0.55,
        "min_agents_agree": 2,
    },
    "new_listing": {
        "edge_threshold": 30,
        "position_size_pct": 0.02,
        "stop_loss_pct": 0.40,
        "min_confidence": 0.50,
        "min_agents_agree": 2,
    },
}

# Graduation requirements
GRADUATION = {
    "min_trades": 50,
    "min_days": 30,
    "min_win_rate": 0.55,
    "min_sharpe": 1.0,
    "max_drawdown": 0.15,
    "min_profit_pct": 0.05,
    "consecutive_profitable_weeks": 3,
}


def load_state() -> dict:
    """Load simulator state."""
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def save_state(state: dict):
    """Save simulator state."""
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def load_evolution_log() -> list:
    """Load evolution log."""
    if EVOLUTION_LOG_FILE.exists():
        return json.loads(EVOLUTION_LOG_FILE.read_text())
    return []


def save_evolution_log(log: list):
    """Save evolution log."""
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    EVOLUTION_LOG_FILE.write_text(json.dumps(log[-100:], indent=2))


def kelly_size(
    edge: float,
    confidence: float,
    market_price: float,
    bankroll: float,
    half_kelly: bool = True,
) -> float:
    """Calculate Kelly Criterion position size."""
    if market_price <= 0 or market_price >= 1 or edge <= 0:
        return 0

    # Implied odds
    b = (1.0 / market_price) - 1
    p = confidence
    q = 1 - p

    # Kelly fraction
    kelly_f = (b * p - q) / b if b > 0 else 0

    if half_kelly:
        kelly_f /= 2

    # Clamp
    position_pct = max(min(kelly_f, MAX_POSITION_PCT), 0)
    position_usd = bankroll * position_pct

    return max(min(position_usd, bankroll * MAX_POSITION_PCT), 0)


def calculate_metrics(positions: list, window_days: int = 30, offset_days: int = 0) -> dict:
    """Calculate strategy performance metrics over a rolling window."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(days=window_days + offset_days)
    window_end = now - timedelta(days=offset_days)

    # Filter closed positions in window
    closed = []
    for p in positions:
        if p.get("status") != "closed":
            continue
        closed_at = p.get("closed_at", "")
        if not closed_at:
            continue
        try:
            dt = datetime.fromisoformat(closed_at.replace("Z", "+00:00"))
            if window_start <= dt <= window_end:
                closed.append(p)
        except (ValueError, TypeError):
            continue

    if not closed:
        return {
            "total_trades": 0, "wins": 0, "losses": 0,
            "total_pnl": 0, "win_rate": 0, "avg_edge": 0,
            "sharpe_ratio": 0, "max_drawdown": 0,
        }

    wins = [p for p in closed if p.get("pnl", 0) > 0]
    losses = [p for p in closed if p.get("pnl", 0) <= 0]
    pnls = [p.get("pnl", 0) for p in closed]
    total_pnl = sum(pnls)
    win_rate = len(wins) / len(closed) if closed else 0
    avg_edge = sum(p.get("edge_score", 0) for p in closed) / len(closed) if closed else 0

    # Sharpe ratio (annualized, simplified)
    if len(pnls) > 1:
        mean_pnl = total_pnl / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.01
        sharpe = (mean_pnl / std_pnl) * math.sqrt(252 / max(window_days, 1))
    else:
        sharpe = 0

    # Max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for pnl in pnls:
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = (peak - cumulative) / max(peak, 1)
        max_dd = max(max_dd, drawdown)

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 4),
        "avg_edge": round(avg_edge, 2),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
    }


def suggest_evolution(state: dict, use_ebma: bool = True) -> Optional[dict]:
    """
    Compare recent vs prior performance and suggest parameter adjustments.

    Args:
        state: Simulator state
        use_ebma: Use EBMA-weighted metrics (default True if learning module available)

    Returns:
        Suggestion dict or None
    """
    strategy = state.get("strategy", "edge_hunter")
    params = state.get("params", STRATEGY_DEFAULTS.get(strategy, {}))
    all_positions = state.get("positions", {}).get("closed", [])

    # Check for agent weight dominance (if EBMA enabled)
    if use_ebma and LEARNING_MODULE_AVAILABLE and "agent_weights" in state:
        weights = {name: data["weight"] for name, data in state["agent_weights"].items()}
        divergence = calculate_weight_divergence(weights)
        max_weight = max(weights.values()) if weights else 0

        if max_weight > 0.40:
            print(f"  [LEARNING] Agent dominance detected: max weight = {max_weight:.2f}, divergence = {divergence:.2f}")

        # Check for groupthink
        recent = calculate_metrics(all_positions, window_days=15, offset_days=0)
        if recent["total_trades"] >= 5:
            # Would need agent predictions per trade to check groupthink
            # For now, just log the warning
            pass

    # Recent window (last 15 days)
    recent = calculate_metrics(all_positions, window_days=15, offset_days=0)
    # Prior window (days 16-30)
    prior = calculate_metrics(all_positions, window_days=15, offset_days=15)

    if recent["total_trades"] < MIN_TRADES_FOR_SUGGESTION:
        return None

    # Auto-revert check
    if recent["sharpe_ratio"] < SHARPE_FLOOR:
        return {
            "action": "REVERT",
            "reason": f"Sharpe {recent['sharpe_ratio']:.2f} < {SHARPE_FLOOR} floor",
            "defaults": STRATEGY_DEFAULTS.get(strategy, {}),
        }

    if recent["max_drawdown"] > 0.25:
        return {
            "action": "REVERT",
            "reason": f"Max drawdown {recent['max_drawdown']:.1%} > 25% limit",
            "defaults": STRATEGY_DEFAULTS.get(strategy, {}),
        }

    # Skip if not enough prior data
    if prior["total_trades"] < 5:
        return None

    suggestions = {}

    # If win rate improved → tighten edge threshold
    if recent["win_rate"] > prior["win_rate"] + 0.02:
        new_thresh = params.get("edge_threshold", 50) * 1.05
        suggestions["edge_threshold"] = min(int(new_thresh), 80)

    # If win rate dropped → loosen edge threshold
    elif recent["win_rate"] < prior["win_rate"] - 0.05:
        new_thresh = params.get("edge_threshold", 50) * 0.95
        suggestions["edge_threshold"] = max(int(new_thresh), 20)

    # If Sharpe improved → can be slightly more aggressive on sizing
    if recent["sharpe_ratio"] > prior.get("sharpe_ratio", 0) + 0.2:
        new_size = params.get("position_size_pct", 0.05) * 1.05
        suggestions["position_size_pct"] = round(min(new_size, 0.10), 4)

    if not suggestions:
        return None

    # Enforce max 10% shift per param
    clamped = {}
    for param, new_value in suggestions.items():
        current = params.get(param, STRATEGY_DEFAULTS.get(strategy, {}).get(param, new_value))
        max_shift = abs(current * MAX_PARAM_SHIFT_RATIO)
        clamped[param] = round(
            max(current - max_shift, min(new_value, current + max_shift)),
            4 if isinstance(current, float) else 0,
        )

    return {
        "action": "EVOLVE",
        "changes": {k: {"from": params.get(k), "to": v} for k, v in clamped.items()},
        "metrics_recent": recent,
        "metrics_prior": prior,
        "reason": "Recent window outperforms prior window",
    }


def apply_evolution(state: dict, suggestion: dict) -> dict:
    """Apply evolution suggestion to strategy params."""
    strategy = state.get("strategy", "edge_hunter")

    if suggestion["action"] == "REVERT":
        state["params"] = STRATEGY_DEFAULTS.get(strategy, {}).copy()
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "action": "REVERT",
            "reason": suggestion["reason"],
            "reverted_to": state["params"],
        }
    else:
        for param, change in suggestion.get("changes", {}).items():
            state["params"][param] = change["to"]

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "action": "EVOLVE",
            "changes": suggestion.get("changes", {}),
            "reason": suggestion.get("reason", ""),
            "metrics_recent": suggestion.get("metrics_recent"),
            "metrics_prior": suggestion.get("metrics_prior"),
        }

    # Log the evolution
    log = load_evolution_log()
    log.append(entry)
    save_evolution_log(log)

    state["last_evolution"] = datetime.now(timezone.utc).isoformat()
    save_state(state)

    return state


def check_graduation(state: dict) -> dict:
    """Check if strategy meets graduation criteria for Web3."""
    all_positions = state.get("positions", {}).get("closed", [])
    metrics = calculate_metrics(all_positions, window_days=90)
    started = state.get("started_at", "")
    bankroll = state.get("bankroll", INITIAL_BANKROLL)

    # Calculate days running
    days_running = 0
    if started:
        try:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            days_running = (datetime.now(timezone.utc) - start_dt).days
        except (ValueError, TypeError):
            pass

    # Calculate weekly P&L for consecutive check
    weekly_pnl = []
    now = datetime.now(timezone.utc)
    for week in range(4):
        week_start = now - timedelta(weeks=week + 1)
        week_end = now - timedelta(weeks=week)
        week_closed = [
            p for p in all_positions
            if p.get("status") == "closed"
            and p.get("closed_at")
            and week_start <= datetime.fromisoformat(p["closed_at"].replace("Z", "+00:00")) <= week_end
        ]
        week_pnl = sum(p.get("pnl", 0) for p in week_closed)
        weekly_pnl.append(week_pnl)

    consecutive_profitable = 0
    for wpnl in weekly_pnl:
        if wpnl > 0:
            consecutive_profitable += 1
        else:
            break

    profit_pct = (bankroll - INITIAL_BANKROLL) / INITIAL_BANKROLL

    checks = {
        "min_trades": (metrics["total_trades"], GRADUATION["min_trades"], metrics["total_trades"] >= GRADUATION["min_trades"]),
        "min_days": (days_running, GRADUATION["min_days"], days_running >= GRADUATION["min_days"]),
        "min_win_rate": (metrics["win_rate"], GRADUATION["min_win_rate"], metrics["win_rate"] >= GRADUATION["min_win_rate"]),
        "min_sharpe": (metrics["sharpe_ratio"], GRADUATION["min_sharpe"], metrics["sharpe_ratio"] >= GRADUATION["min_sharpe"]),
        "max_drawdown": (metrics["max_drawdown"], GRADUATION["max_drawdown"], metrics["max_drawdown"] <= GRADUATION["max_drawdown"]),
        "min_profit_pct": (profit_pct, GRADUATION["min_profit_pct"], profit_pct >= GRADUATION["min_profit_pct"]),
        "consecutive_weeks": (consecutive_profitable, GRADUATION["consecutive_profitable_weeks"],
                              consecutive_profitable >= GRADUATION["consecutive_profitable_weeks"]),
    }

    passed = all(c[2] for c in checks.values())
    passed_count = sum(1 for c in checks.values() if c[2])
    total_checks = len(checks)
    progress = passed_count / total_checks

    return {
        "graduated": passed,
        "progress": round(progress, 2),
        "checks": {k: {"current": v[0], "required": v[1], "passed": v[2]} for k, v in checks.items()},
        "message": "Strategy ready for Web3!" if passed else f"Still learning... ({passed_count}/{total_checks} criteria met)",
    }


def init_simulation(strategy: str = "edge_hunter", bankroll: float = INITIAL_BANKROLL):
    """Initialize a new paper trading simulation."""
    if STATE_FILE.exists():
        existing = load_state()
        if existing.get("active"):
            print(f"Simulation already running ({existing.get('strategy')}). Use --status to check.")
            return

    state = {
        "active": True,
        "strategy": strategy,
        "bankroll": bankroll,
        "initial_bankroll": bankroll,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "params": STRATEGY_DEFAULTS.get(strategy, STRATEGY_DEFAULTS["edge_hunter"]).copy(),
        "positions": {"open": [], "closed": []},
        "last_evolution": None,
    }

    save_state(state)
    print(f"\nSimulation started!")
    print(f"  Strategy: {strategy}")
    print(f"  Bankroll: ${bankroll:,.2f}")
    print(f"  Params: {json.dumps(state['params'], indent=4)}")


def show_status():
    """Display simulator dashboard."""
    state = load_state()

    if not state.get("active"):
        print("No active simulation. Use --start to begin.")
        return

    strategy = state.get("strategy", "?")
    bankroll = state.get("bankroll", 0)
    initial = state.get("initial_bankroll", INITIAL_BANKROLL)
    started = state.get("started_at", "?")
    pnl = bankroll - initial
    pnl_pct = (pnl / initial) * 100 if initial > 0 else 0

    # Calculate days
    days = 0
    if started != "?":
        try:
            start_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
            days = (datetime.now(timezone.utc) - start_dt).days
        except (ValueError, TypeError):
            pass

    open_positions = state.get("positions", {}).get("open", [])
    closed_positions = state.get("positions", {}).get("closed", [])
    all_positions = open_positions + closed_positions

    metrics = calculate_metrics(closed_positions, window_days=30)

    print(f"\n{'='*60}")
    print(f"  SIMULATOR STATUS")
    print(f"{'='*60}")
    print(f"  Strategy: {strategy} (Day {days})")
    sign = "+" if pnl >= 0 else ""
    print(f"  Bankroll: ${bankroll:,.2f} ({sign}{pnl_pct:.2f}%)")

    # Show EBMA agent weights if available
    if LEARNING_MODULE_AVAILABLE and "agent_weights" in state:
        weights = {name: data["weight"] for name, data in state["agent_weights"].items()}
        if weights:
            divergence = calculate_weight_divergence(weights)
            print(f"\n  AGENT WEIGHTS (EBMA):")
            for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
                bar = "#" * int(weight * 100)
                pred_count = len(state["agent_weights"][name].get("predictions", []))
                print(f"    {name:15s} {weight:.3f} {bar} ({pred_count} preds)")
            print(f"  Divergence: {divergence:.2f} (max - min)")

    # Show Thompson Sampling strategy priors if available
    if LEARNING_MODULE_AVAILABLE and "strategy_priors" in state:
        print(f"\n  STRATEGY PRIORS (Thompson Sampling):")
        for strat, prior in state["strategy_priors"].items():
            alpha = prior["alpha"]
            beta = prior["beta"]
            mean = alpha / (alpha + beta)
            total = alpha + beta
            print(f"    {strat:15s} α={alpha:.1f}, β={beta:.1f} | Mean={mean:.2%} ({int(total)-2} samples)")


    if open_positions:
        print(f"\n  OPEN POSITIONS ({len(open_positions)}):")
        for i, p in enumerate(open_positions[:10], 1):
            q = p.get("question", "?")[:50]
            side = p.get("side", "?")
            entry = p.get("entry_price", 0)
            size = p.get("size_usd", 0)
            print(f"    {i}. {q}")
            print(f"       {side} @ ${entry:.2f} | Size: ${size:.2f}")

    print(f"\n  METRICS (30-day rolling):")
    print(f"    Trades: {metrics['total_trades']} | Wins: {metrics['wins']} | Losses: {metrics['losses']}")
    print(f"    Win Rate: {metrics['win_rate']:.1%} | Avg Edge: {metrics['avg_edge']}")
    print(f"    Sharpe: {metrics['sharpe_ratio']:.2f} | Max Drawdown: {metrics['max_drawdown']:.1%}")

    print(f"\n  CURRENT PARAMS:")
    params = state.get("params", {})
    for k, v in params.items():
        print(f"    {k}: {v}")

    last_evo = state.get("last_evolution")
    if last_evo:
        print(f"\n  Last evolution: {last_evo[:10]}")

    # Graduation progress
    grad = check_graduation(state)
    passed = sum(1 for c in grad["checks"].values() if c["passed"])
    total = len(grad["checks"])
    grad_status = 'Ready!' if grad['graduated'] else f"{grad['progress']:.0%} ready"
    print(f"\n  GRADUATION: {grad_status} ({passed}/{total})")
    print(f"{'='*60}")


def run_evolution():
    """Force a learning cycle."""
    state = load_state()
    if not state.get("active"):
        print("No active simulation.")
        return

    suggestion = suggest_evolution(state)

    if not suggestion:
        print("Not enough data for evolution, or no improvements suggested.")
        return

    print(f"\nEvolution suggestion: {suggestion['action']}")
    print(f"  Reason: {suggestion.get('reason', 'N/A')}")

    if suggestion["action"] == "REVERT":
        print("  Reverting to defaults...")
    else:
        for param, change in suggestion.get("changes", {}).items():
            print(f"  {param}: {change['from']} -> {change['to']}")

    apply_evolution(state, suggestion)
    print("\nEvolution applied!")


def show_evolution_log():
    """Display parameter evolution history."""
    log = load_evolution_log()

    if not log:
        print("No evolution history yet.")
        return

    print(f"\n{'='*60}")
    print(f"  EVOLUTION LOG ({len(log)} entries)")
    print(f"{'='*60}")

    for entry in log[-10:]:  # Last 10 entries
        ts = entry.get("timestamp", "?")[:10]
        action = entry.get("action", "?")
        reason = entry.get("reason", "")

        print(f"\n  [{ts}] {action}")
        print(f"  Reason: {reason}")

        if action == "EVOLVE":
            for param, change in entry.get("changes", {}).items():
                print(f"    {param}: {change.get('from')} -> {change.get('to')}")
        elif action == "REVERT":
            print("    Reverted to defaults")

    print(f"\n{'='*60}")


def show_graduation():
    """Display graduation progress."""
    state = load_state()
    if not state.get("active"):
        print("No active simulation.")
        return

    grad = check_graduation(state)

    print(f"\n{'='*60}")
    print(f"  GRADUATION CHECK")
    print(f"{'='*60}")

    for name, check in grad["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}: {check['current']} (need: {check['required']})")

    print(f"\n  {grad['message']}")
    print(f"  Progress: {grad['progress']:.0%}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Paper Trading Simulator")
    parser.add_argument("--start", action="store_true", help="Start new simulation")
    parser.add_argument("--strategy", type=str, default="edge_hunter",
                        choices=list(STRATEGY_DEFAULTS.keys()),
                        help="Trading strategy to simulate")
    parser.add_argument("--bankroll", type=float, default=INITIAL_BANKROLL,
                        help="Starting virtual bankroll")
    parser.add_argument("--status", action="store_true", help="Show simulator dashboard")
    parser.add_argument("--evolve", action="store_true", help="Force learning cycle")
    parser.add_argument("--evolution-log", action="store_true", help="Show evolution history")
    parser.add_argument("--graduation-check", action="store_true", help="Check Web3 readiness")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.start:
        init_simulation(strategy=args.strategy, bankroll=args.bankroll)
    elif args.status:
        show_status()
    elif args.evolve:
        run_evolution()
    elif args.evolution_log:
        show_evolution_log()
    elif args.graduation_check:
        show_graduation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
