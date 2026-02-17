---
name: poly-research
description: Multi-agent intelligence layer for Polymarket analysis. Use when analyzing a market, getting AI predictions, running edge scoring, researching market context, or comparing agent opinions. Triggers on "analyze market", "what do agents think", "research this", "edge score", "intelligence report", "deep analysis".
metadata: { "openclaw": { "emoji": "🧠", "primaryEnv": "OPENROUTER_API_KEY" } }
---

# Poly Research

Multi-agent intelligence layer that runs parallel AI analysis on Polymarket markets, calculates edge scores, and produces actionable trading signals.

## Capabilities

1. **Multi-Agent Analysis** — 3-5 AI agents analyze markets from different perspectives simultaneously
2. **Edge Scoring** — 7-factor scoring system (ported from PolyHuntr edge engine)
3. **Web Research** — Fetch real-time context via web search before analysis
4. **Consensus Engine** — Aggregate agent opinions into a weighted final prediction
5. **Category Efficiency** — Adjust edge thresholds by market category (Becker research)

## Multi-Agent System

### Agent Roster

| Agent | Role | Personality | Focus |
|-------|------|-------------|-------|
| **Sentinel** | Contrarian | Challenges consensus | "What if the crowd is wrong?" |
| **Oracle** | Neutral | Uses base rates | Historical precedent, statistical analysis |
| **Maverick** | Aggressive | Spots trends early | Momentum, sentiment shifts, early signals |
| **Fundamental** | Deep Value | Analyzes underlying factors | Economics, incentives, structural analysis |
| **Technical** | Pattern Recognition | Reads market microstructure | Volume patterns, price action, orderbook |

### Agent Prompt Template

Each agent receives:
```
Market: [question]
Current YES Price: $[price]
Volume 24h: $[volume]
Category: [category]
Time to Resolution: [days]

Research Context: [web search results]

You are [Agent Name], a [personality] analyst.
Analyze this market and provide:
1. probability: Your estimated probability (0.0-1.0)
2. confidence: How confident you are (0.0-1.0)
3. reasoning: 2-3 sentence explanation
4. edge_direction: "YES" or "NO" (which side has edge)
```

### Orchestration Pattern

```python
# Run all agents in parallel (ported from polyhuntr orchestration.py)
# Each agent has a 60-second timeout
# Failed agents produce fallback signals (prior price ± small random offset)
# Minimum 2 successful agents required for valid consensus

import asyncio

async def run_analysis(market):
    agents = [sentinel, oracle, maverick, fundamental, technical]
    results = await asyncio.gather(
        *[agent.analyze(market) for agent in agents],
        return_exceptions=True
    )
    # Filter out failures, use fallback for failed agents
    valid = [r for r in results if not isinstance(r, Exception)]
    if len(valid) < 2:
        return None  # Insufficient consensus
    return aggregate(valid)
```

## Edge Scoring (7 Factors)

Ported from PolyHuntr's edge engine. Each market gets a 0-100 edge score.

### Base Score (0-50 points)
```
edge_magnitude = abs(ai_consensus_probability - market_yes_price)
base_score = min(edge_magnitude * 500, 50)
```

### Factor Bonuses (up to 10 points each)

| Factor | Condition | Points |
|--------|-----------|--------|
| **Volume Spike** | 24h volume > 3x 7-day average | +10 |
| **Whale Activity** | Whale trade in same direction as edge | +10 |
| **Consensus Strong** | All agents agree on direction, avg confidence > 0.7 | +10 |
| **Time Pressure** | < 7 days to resolution + large edge | +8 |
| **Tail Event** | Market price < 0.10 or > 0.90 | +5 |
| **Edge Large** | Edge magnitude > 0.15 | +7 |
| **Liquidity Depth** | Best bid/ask > $500 | +5 |

### Category Efficiency Multiplier (Becker Research)

Markets in different categories have different efficiency levels. Less efficient categories offer more exploitable edge.

```python
CATEGORY_EFFICIENCY = {
    "sports": 0.70,        # High bias → 1/0.70 = 1.43x edge multiplier
    "entertainment": 0.75, # Celebrity/culture bias
    "science": 0.80,       # Moderate efficiency
    "politics": 0.85,      # Heavily traded, more efficient
    "crypto": 0.95,        # Very efficient
    "finance": 1.20,       # Extremely efficient → 0.83x edge reduction
}

# Final edge score = raw_score * (1.0 / category_efficiency)
# Sports market with raw 40 → 40 * 1.43 = 57 (boosted)
# Finance market with raw 40 → 40 * 0.83 = 33 (reduced)
```

### Signal Weights (Bayesian Updates)

Each agent's weight is updated based on prediction accuracy:
```python
# After market resolution:
# If agent was correct (within 0.1 of outcome):
#   weight *= 1.05 (increase by 5%)
# If agent was wrong:
#   weight *= 0.95 (decrease by 5%)
# Weights are normalized to sum to 1.0

# Consensus = weighted average of agent probabilities
consensus_prob = sum(agent.prob * agent.weight for agent in agents) / sum(weights)
```

## How to Use

### Quick Analysis
```bash
python scripts/research.py --market "Will X happen?" --mode quick
# Uses 3 agents (Sentinel, Oracle, Maverick), no web search
```

### Deep Analysis
```bash
python scripts/research.py --market "Will X happen?" --mode deep
# Uses all 5 agents, includes web search context
```

### Batch Analysis
```bash
python scripts/research.py --from-scanner --top 10
# Analyze top 10 opportunities from poly-scanner
```

## Output Format

```
🧠 INTELLIGENCE REPORT
─────────────────────────────────────────
Market: "Will X happen by Y?"
Current: YES $0.45 | NO $0.55
Category: politics | Efficiency: 0.85

AGENT ANALYSIS:
  Sentinel (contrarian):  P=0.62  C=0.75  → YES edge
  Oracle (neutral):       P=0.58  C=0.80  → YES edge
  Maverick (aggressive):  P=0.70  C=0.65  → YES edge
  Fundamental (value):    P=0.55  C=0.70  → YES edge
  Technical (patterns):   P=0.50  C=0.60  → Neutral

CONSENSUS: P=0.59 (weighted) | Direction: YES
EDGE: +0.14 (14 cents vs market)

EDGE SCORE: 72/100
  Base: 35/50 (edge magnitude)
  + Volume Spike: +10
  + Consensus Strong: +10
  + Whale Activity: +10
  + Liquidity: +5
  × Category Multiplier: 1.18x

SIGNAL: STRONG BUY YES @ $0.45
  Suggested Size: See poly-simulator for Kelly sizing
─────────────────────────────────────────
```

## Cross-Skill Integration

- **poly-scanner** → Feeds markets to analyze (batch mode)
- **poly-whale** → Whale activity factor in edge scoring
- **poly-watchlist** → Alert when watched market gets high edge score
- **poly-simulator** → Passes signals for paper trading + Kelly sizing

## Storage

Agent weights and history in `data/research/`:
```json
{
  "agent_weights": {
    "sentinel": 0.22,
    "oracle": 0.25,
    "maverick": 0.18,
    "fundamental": 0.20,
    "technical": 0.15
  },
  "analysis_history": [
    {
      "market_id": "...",
      "timestamp": "...",
      "consensus_prob": 0.59,
      "edge_score": 72,
      "outcome": null
    }
  ]
}
```

## API Endpoints Used

- **Gamma API**: `GET https://gamma-api.polymarket.com/markets?id={id}` — Market details + category
- **CLOB API**: `GET https://clob.polymarket.com/prices` — Current prices
- **CLOB API**: `GET https://clob.polymarket.com/book` — Orderbook depth for liquidity factor
- **Data API**: `GET https://data-api.polymarket.com/trades` — Volume history for spike detection
- **Web Search**: OpenRouter LLM with web search tool for real-time context
