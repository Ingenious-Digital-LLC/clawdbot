---
name: poly-simulator
description: Paper trading simulator with self-learning feedback loop. Use when simulating trades, backtesting strategies, checking P&L, reviewing learning metrics, or adjusting strategy parameters. Triggers on "simulate", "paper trade", "backtest", "how am I doing", "strategy performance", "learning metrics", "auto-correct".
metadata: { "openclaw": { "emoji": "💰", "primaryEnv": "OPENROUTER_API_KEY" } }
---

# Poly Simulator

Paper trading engine with self-learning feedback loop. Simulates real trades without risking capital, tracks performance, and auto-evolves strategy parameters based on outcomes.

## Capabilities

1. **Paper Trading** — Execute simulated trades with virtual bankroll
2. **Kelly Criterion Sizing** — Optimal position sizing based on edge and confidence
3. **P&L Tracking** — Real-time profit/loss with proper cost basis
4. **Strategy Evolution** — Auto-adjust parameters based on rolling performance
5. **Safety Rails** — Auto-revert strategies that degrade, cap parameter shifts
6. **Resolution Tracking** — Monitor markets for outcomes, score predictions
7. **Graduation Signal** — Flag when a strategy is consistently profitable for Web3 connection

## Paper Trading Engine

### Virtual Bankroll
```python
INITIAL_BANKROLL = 10000  # $10,000 virtual USD
MAX_POSITION_PCT = 0.10   # Max 10% of bankroll per position
MIN_POSITION_USD = 10     # Minimum $10 per trade
MAX_OPEN_POSITIONS = 20   # Limit concurrent positions
```

### Position Lifecycle
```
Signal (from poly-research)
  → Kelly Sizing (edge + confidence → optimal bet size)
  → Paper Execute (record entry price, size, timestamp)
  → Monitor (track price changes via CLOB API)
  → Resolution (market resolves → record P&L)
  → Score (update agent weights + strategy metrics)
  → Evolve (adjust strategy parameters if enough data)
```

### Kelly Criterion Sizing

```python
def kelly_size(edge, confidence, bankroll):
    """
    Kelly fraction = (bp - q) / b
    Where:
      b = odds (payout ratio)
      p = probability of winning (our AI consensus)
      q = 1 - p

    We use HALF Kelly for safety (less volatile than full Kelly)
    """
    p = confidence  # Our estimated win probability
    q = 1 - p
    b = (1.0 / market_price) - 1  # Implied odds from market price

    kelly_fraction = (b * p - q) / b
    half_kelly = kelly_fraction / 2  # Half Kelly for safety

    # Cap at max position size
    position_pct = min(half_kelly, MAX_POSITION_PCT)
    position_usd = bankroll * max(position_pct, 0)

    return max(min(position_usd, bankroll * MAX_POSITION_PCT), 0)
```

## Self-Learning Feedback Loop

Ported from PolyHuntr's strategy_evolver.py — the brain that makes the bot smarter over time.

### Rolling Window Metrics

Every strategy is evaluated over a rolling 30-day window:

```python
@dataclass
class StrategyPerformance:
    strategy_name: str
    total_trades: int
    wins: int
    losses: int
    total_pnl: float
    win_rate: float         # wins / total_trades
    avg_edge: float         # average edge score at entry
    sharpe_ratio: float     # risk-adjusted return
    max_drawdown: float     # worst peak-to-trough decline
    avg_hold_time_hours: float
```

### Strategy Parameters (Evolvable)

```python
EVOLVABLE_PARAMS = {
    "edge_threshold": {
        "default": 50,      # Minimum edge score to trade
        "min": 20,
        "max": 80,
        "description": "Minimum edge score required to open position"
    },
    "position_size_pct": {
        "default": 0.05,    # 5% of bankroll (half Kelly)
        "min": 0.01,
        "max": 0.10,
        "description": "Base position size as % of bankroll"
    },
    "stop_loss_pct": {
        "default": 0.30,    # Close if price drops 30% from entry
        "min": 0.10,
        "max": 0.50,
        "description": "Stop loss threshold"
    },
    "min_confidence": {
        "default": 0.60,    # Minimum consensus confidence
        "min": 0.40,
        "max": 0.90,
        "description": "Minimum agent consensus confidence"
    },
    "min_agents_agree": {
        "default": 3,       # At least 3 of 5 agents must agree
        "min": 2,
        "max": 5,
        "description": "Minimum agents agreeing on direction"
    }
}
```

### Evolution Algorithm

```python
def suggest_parameter_adjustments(strategy, window_days=30):
    """
    Compare recent performance (last 15 days) vs prior window (days 16-30).
    Suggest parameter shifts based on what's working.

    SAFETY RAILS (ported from polyhuntr):
    - MIN_TRADES_FOR_SUGGESTION = 10 (need enough data)
    - MAX_PARAM_SHIFT_RATIO = 0.10 (max 10% change per cycle)
    - SHARPE_FLOOR = 0.5 (auto-revert if Sharpe drops below)
    """
    recent = get_metrics(strategy, days=15)
    prior = get_metrics(strategy, days=30, offset=15)

    if recent.total_trades < 10:
        return None  # Not enough data

    suggestions = {}

    # If win rate improved with tighter edge threshold → tighten further
    if recent.win_rate > prior.win_rate and recent.avg_edge > prior.avg_edge:
        suggestions["edge_threshold"] = min(
            current.edge_threshold * 1.05,  # 5% tighter
            EVOLVABLE_PARAMS["edge_threshold"]["max"]
        )

    # If Sharpe dropped below floor → AUTO-REVERT to defaults
    if recent.sharpe_ratio < 0.5:
        return {"action": "REVERT_TO_DEFAULTS", "reason": f"Sharpe {recent.sharpe_ratio:.2f} < 0.5"}

    # Cap all changes at 10% shift
    for param, new_value in suggestions.items():
        current_value = get_current_param(strategy, param)
        max_shift = current_value * MAX_PARAM_SHIFT_RATIO
        suggestions[param] = clamp(
            new_value,
            current_value - max_shift,
            current_value + max_shift
        )

    return suggestions
```

### Learning Cycle

```
Every 24 hours (or after 10 new resolutions):
  1. Score all resolved positions → update agent weights
  2. Calculate rolling metrics per strategy
  3. Compare recent vs prior window
  4. If enough data → suggest parameter adjustments
  5. Apply changes (max 10% shift per param)
  6. If Sharpe < 0.5 → auto-revert to defaults
  7. Log all changes to data/simulator/evolution_log.json
```

## Graduation System

When a strategy proves consistently profitable, it's "graduated" — ready for real money.

### Graduation Criteria

```python
GRADUATION_REQUIREMENTS = {
    "min_trades": 50,           # At least 50 paper trades
    "min_days": 30,             # At least 30 days of history
    "min_win_rate": 0.55,       # > 55% win rate
    "min_sharpe": 1.0,          # Sharpe ratio > 1.0
    "max_drawdown": 0.15,       # Max drawdown < 15%
    "min_profit_pct": 0.05,     # > 5% total return
    "consecutive_profitable_weeks": 3  # 3 straight winning weeks
}

def check_graduation(strategy):
    metrics = get_metrics(strategy, days=90)
    weekly = get_weekly_pnl(strategy, weeks=4)

    passed = all([
        metrics.total_trades >= 50,
        metrics.win_rate >= 0.55,
        metrics.sharpe_ratio >= 1.0,
        metrics.max_drawdown <= 0.15,
        metrics.total_pnl / INITIAL_BANKROLL >= 0.05,
        all(w > 0 for w in weekly[-3:])
    ])

    return {
        "graduated": passed,
        "metrics": metrics,
        "message": "🎓 Strategy ready for Web3!" if passed else "📚 Still learning..."
    }
```

## How to Use

### Start Simulation
```bash
python scripts/simulator.py --start --strategy edge_hunter
# Begins paper trading with edge_hunter strategy defaults
```

### Check Performance
```bash
python scripts/simulator.py --status
# Shows bankroll, open positions, P&L, win rate, Sharpe
```

### View Learning Log
```bash
python scripts/simulator.py --evolution-log
# Shows parameter changes over time with before/after metrics
```

### Check Graduation
```bash
python scripts/simulator.py --graduation-check
# Shows progress toward Web3 readiness
```

### Force Learning Cycle
```bash
python scripts/simulator.py --evolve
# Manually trigger strategy evolution (normally runs automatically)
```

## Output Format

### Status Dashboard
```
💰 SIMULATOR STATUS
─────────────────────────────────────────
Strategy: edge_hunter (Day 23 of simulation)
Bankroll: $10,847 (+8.47%)

OPEN POSITIONS (4):
  1. "Will X happen?" — YES @ $0.42 | Now: $0.58 | +$80
  2. "Will Y occur?"  — NO  @ $0.35 | Now: $0.30 | +$25
  3. "Will Z pass?"   — YES @ $0.71 | Now: $0.68 | -$15
  4. "Will W win?"    — YES @ $0.55 | Now: $0.55 | $0

METRICS (30-day rolling):
  Trades: 34 | Wins: 21 | Losses: 13
  Win Rate: 61.8% | Avg Edge: 0.12
  Sharpe: 1.34 | Max Drawdown: 8.2%
  Avg Hold: 4.2 days

LEARNING STATUS:
  Last evolution: 2 days ago
  Changes applied: edge_threshold 50→52 (+4%)
  Agent weights: Oracle 0.28 (+) | Sentinel 0.19 (-)

GRADUATION: 📚 68% ready (need 16 more trades, 7 more days)
─────────────────────────────────────────
```

## Cross-Skill Integration

- **poly-research** → Provides signals with edge scores and agent consensus
- **poly-scanner** → Discovers markets for analysis pipeline
- **poly-whale** → Whale activity boosts edge score (factor bonus)
- **poly-watchlist** → Alerts when simulated positions hit thresholds

## Storage

All simulation data in `data/simulator/`:
```json
{
  "bankroll": 10847.00,
  "initial_bankroll": 10000.00,
  "strategy": "edge_hunter",
  "started_at": "2026-01-15T00:00:00Z",
  "positions": {
    "open": [...],
    "closed": [...]
  },
  "metrics": {
    "total_trades": 34,
    "wins": 21,
    "losses": 13,
    "total_pnl": 847.00,
    "sharpe_ratio": 1.34,
    "max_drawdown": 0.082
  },
  "evolution_log": [
    {
      "timestamp": "2026-02-01T00:00:00Z",
      "changes": {"edge_threshold": {"from": 50, "to": 52}},
      "reason": "Win rate improved 58%→62% with higher edge threshold",
      "metrics_before": {...},
      "metrics_after": {...}
    }
  ],
  "agent_weights": {...},
  "graduation_progress": {...}
}
```

## API Endpoints Used

- **CLOB API**: `GET https://clob.polymarket.com/prices` — Monitor position prices
- **Gamma API**: `GET https://gamma-api.polymarket.com/markets?id={id}` — Check resolution status
- **Data API**: `GET https://data-api.polymarket.com/trades` — Verify resolution outcomes
