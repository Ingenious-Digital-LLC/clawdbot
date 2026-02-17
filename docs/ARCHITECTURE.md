# PolyClaw Architecture

This document details the internal architecture of PolyClaw — how OpenClaw orchestrates autonomous trading through cron scheduling, heartbeat monitoring, and skill discovery.

## System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATION LAYER                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────┐          ┌──────────────────────┐          │
│  │  Cron Scheduler    │          │  Heartbeat Monitor   │          │
│  │  (config/cron)     │          │  (5min interval)     │          │
│  └────────┬───────────┘          └──────────┬───────────┘          │
│           │                                  │                       │
│           └──────────────┬───────────────────┘                       │
│                          │                                           │
│                 ┌────────▼────────┐                                  │
│                 │  Agent Session  │                                  │
│                 │  (LLM brain)    │                                  │
│                 └────────┬────────┘                                  │
│                          │                                           │
└──────────────────────────┼───────────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────────┐
│                         EXECUTION LAYER                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────────┐ │
│  │ poly-      │  │ poly-      │  │ poly-      │  │ poly-        │ │
│  │ scanner    │  │ research   │  │ simulator  │  │ whale/watch  │ │
│  │            │  │            │  │            │  │              │ │
│  │ Python     │  │ Python     │  │ Python     │  │ Python       │ │
│  │ scripts    │  │ scripts    │  │ scripts    │  │ scripts      │ │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └──────┬───────┘ │
│        │               │               │                │          │
└────────┼───────────────┼───────────────┼────────────────┼──────────┘
         │               │               │                │
┌────────▼───────────────▼───────────────▼────────────────▼──────────┐
│                       POLYMARKET APIs                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Gamma API (markets)  │  CLOB API (prices)  │  Data API (trades)  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. OpenClaw Gateway

**Technology:** Node.js 22 + Python 3 hybrid container

The gateway is the orchestration layer that:
- Hosts the AI agent's brain (LLM with persistent context)
- Manages skill discovery and execution
- Runs cron jobs (scheduled agent turns)
- Monitors heartbeats (continuous background checks)
- Handles message delivery (Telegram, WhatsApp, etc.)

**Key files:**
- `Dockerfile` — Multi-stage build (Node + Python)
- `entrypoint.sh` — Syncs poly-* skills to workspace on startup
- `config/openclaw.json` — Gateway settings (heartbeat interval, cron enabled)

### 2. Agent Brain (LLM)

**Technology:** Claude AI (via OpenClaw's provider system)

The agent is NOT a static script — it's a language model with memory that:
- Receives context from workspace files (AGENTS.md, SOUL.md, HEARTBEAT.md)
- Makes decisions based on current market state
- Invokes skills via shell commands (`python3 skills/poly-scanner/...`)
- Adapts its behavior based on trading outcomes

**Context files (workspace/):**
- `AGENTS.md` — Complete trading instructions, skill reference, decision rules
- `SOUL.md` — Agent personality (data-driven, risk-aware, patient)
- `HEARTBEAT.md` — Monitoring checklist (what to check every 5 minutes)

**Why LLM orchestration?**
- Traditional bots follow rigid if/then rules
- PolyClaw can adapt to unexpected market conditions
- Natural language context is easier to maintain than code
- Agent can reason about trade-offs (e.g., "high edge but low liquidity — skip")

### 3. Cron Scheduler

**Technology:** OpenClaw's built-in cron system (defined in `config/cron/jobs.json`)

Scheduled agent turns trigger at specific intervals with custom prompts.

**Four cron jobs:**

| Job | Cron Expression | What Happens |
|-----|-----------------|--------------|
| `poly-research-cycle` | `*/30 * * * *` (every 30min) | Agent wakes up, scans markets, researches top opportunities, paper trades if edge ≥ 50 |
| `poly-learning-cycle` | `0 0 * * *` (midnight UTC) | Agent scores resolutions, updates agent weights, evolves strategy parameters |
| `poly-daily-pnl` | `55 23 * * *` (23:55 UTC) | Agent generates daily trading summary, delivers to Telegram |
| `poly-weekly-review` | `0 12 * * 0` (Sunday noon UTC) | Agent deep-dives into 7-day performance, graduation progress |

**Flow:**
```
Cron job fires
  → OpenClaw creates isolated agent session
  → Agent receives prompt (e.g., "Run the research cycle")
  → Agent reads AGENTS.md for instructions
  → Agent invokes skills: python3 skills/poly-scanner/scripts/scan_markets.py --mode quick
  → Agent parses output, makes decisions
  → Agent invokes poly-research for top opportunities
  → Agent invokes poly-simulator to execute paper trades
  → Agent logs findings, session ends
```

**Why isolated sessions?**
- Each cron job gets a fresh agent context
- Prevents context pollution (long-running agents accumulate noise)
- Failed jobs don't corrupt agent state

### 4. Heartbeat Monitor

**Technology:** OpenClaw's heartbeat system (configured in `config/openclaw.json`)

Every 5 minutes, the agent wakes up and checks critical conditions:
- Open positions with stop-loss triggers (price dropped 30%+ from entry)
- Watchlist markets that have resolved
- Newly resolved markets that need scoring

**Heartbeat flow:**
```
Heartbeat timer fires (every 5min)
  → Agent session created
  → Agent reads workspace/HEARTBEAT.md for checklist
  → Agent queries poly-simulator: python3 scripts/simulator.py --status
  → Agent checks each open position against current CLOB prices
  → Agent queries poly-watchlist: python3 scripts/watchlist.py --status
  → Agent checks for resolution events
  → If action needed:
       → Close positions (stop-loss hit)
       → Score resolved markets (update agent weights)
       → Alert user (Telegram notification)
  → If nothing needs action:
       → Agent replies "HEARTBEAT_OK" (silent)
       → Session ends
```

**Why heartbeat vs polling?**
- Heartbeat is event-driven (not continuous CPU burn)
- Agent only wakes when needed (cost-efficient)
- Graceful degradation (if heartbeat fails, cron jobs keep running)

### 5. Skill System

**Technology:** Python 3 scripts (installed via Dockerfile, synced via entrypoint.sh)

Skills are **NOT** running daemons — they're CLI scripts that:
- Accept arguments (e.g., `--mode quick`, `--market "Question"`)
- Print structured output (JSON or formatted text)
- Exit after execution (no persistent processes)

**Skill discovery:**
- Each skill has a `SKILL.md` file with metadata
- OpenClaw scans `skills/*/SKILL.md` on startup
- Skill descriptions include trigger phrases (e.g., "scan markets", "analyze market")
- Agent can invoke skills by name or by natural language intent

**Skill dependency chain:**
```
poly-scanner (market discovery)
  ↓ Outputs: List of opportunities with edge scores
poly-research (multi-agent analysis)
  ↓ Inputs: Market questions from scanner
  ↓ Outputs: Consensus probability, edge score, trade signal
poly-simulator (paper trading)
  ↓ Inputs: Trade signals from research
  ↓ Outputs: Position entries, P&L tracking
poly-whale (whale monitoring)
  ↓ Inputs: Market IDs from scanner/research
  ↓ Outputs: Whale activity signals (fed back into edge scoring)
poly-watchlist (alert system)
  ↓ Inputs: Market IDs to track
  ↓ Outputs: Price alerts, resolution notifications
```

**Why Python for skills?**
- Rich ecosystem (httpx for API calls, asyncio for parallelism)
- Portable (runs in Docker, locally, or on cloud runners)
- Easy for contributors (most AI/trading devs know Python)
- Isolated from Node.js gateway (failure in one doesn't crash the other)

## Data Flow

### Research Cycle (Every 30 Minutes)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CRON FIRES                                               │
│    "Run the research cycle"                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. AGENT SESSION STARTS                                     │
│    - Reads workspace/AGENTS.md                              │
│    - Knows: "Research cycle = scan → analyze → trade"      │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. SCAN MARKETS                                             │
│    python3 skills/poly-scanner/scripts/scan_markets.py      │
│    --mode quick                                             │
│                                                             │
│    Output: Top 10 opportunities by edge score              │
│    [                                                        │
│      {                                                      │
│        "market": "Will X happen?",                          │
│        "edge_score": 0.034,                                 │
│        "signal": "VOLUME_SPIKE"                             │
│      },                                                     │
│      ...                                                    │
│    ]                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 4. RESEARCH TOP OPPORTUNITIES                               │
│    python3 skills/poly-research/scripts/research.py         │
│    --from-scanner --top 3                                   │
│                                                             │
│    For each market:                                         │
│      → Web search for context                               │
│      → Run 5 AI agents in parallel                          │
│      → Calculate consensus probability                      │
│      → Calculate edge score (7 factors)                     │
│      → Apply category efficiency multiplier                 │
│                                                             │
│    Output:                                                  │
│    [                                                        │
│      {                                                      │
│        "market": "Will X happen?",                          │
│        "consensus_prob": 0.68,                              │
│        "edge_score": 72,                                    │
│        "signal": "STRONG BUY YES"                           │
│      },                                                     │
│      ...                                                    │
│    ]                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. DECISION GATE                                            │
│    Agent evaluates each signal:                             │
│      - Edge score >= 50? ✓                                  │
│      - Confidence >= 0.60? ✓                                │
│      - At least 3 agents agree? ✓                           │
│      - Position size within 10% of bankroll? ✓              │
│      - No correlation with existing positions? ✓            │
│                                                             │
│    If ALL checks pass: TRADE                                │
│    If ANY check fails: SKIP                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 6. PAPER TRADE                                              │
│    python3 skills/poly-simulator/scripts/simulator.py       │
│    --trade "Will X happen?" --side YES --edge 72            │
│    --confidence 0.68                                        │
│                                                             │
│    Simulator:                                               │
│      → Calculates Kelly position size                       │
│      → Records entry price, timestamp                       │
│      → Updates bankroll                                     │
│      → Stores position in data/simulator/bankroll.json      │
│                                                             │
│    Output:                                                  │
│    {                                                        │
│      "position_id": "pos_123",                              │
│      "market": "Will X happen?",                            │
│      "side": "YES",                                         │
│      "entry_price": 0.42,                                   │
│      "position_usd": 523.50,                                │
│      "bankroll_remaining": 9476.50                          │
│    }                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 7. LOG FINDINGS                                             │
│    Agent writes summary to session log:                     │
│      "Scanned 100 markets, researched 3 opportunities,      │
│       executed 1 trade: YES on 'Will X happen?' @ $0.42     │
│       (edge score 72, Kelly size $523.50)"                  │
│                                                             │
│    Session ends, next cycle in 30 minutes.                  │
└─────────────────────────────────────────────────────────────┘
```

### Learning Cycle (Daily at Midnight UTC)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. CRON FIRES                                               │
│    "Run the daily learning cycle"                           │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 2. SCORE RESOLUTIONS                                        │
│    python3 skills/poly-simulator/scripts/learning.py        │
│                                                             │
│    For each resolved market:                                │
│      → Fetch actual outcome (0.0 or 1.0)                    │
│      → Compare to agent predictions                         │
│      → Update agent weights (EBMA):                         │
│          if |prediction - outcome| < 0.1:                   │
│            weight *= 1.05  # Correct                        │
│          else:                                              │
│            weight *= 0.95  # Incorrect                      │
│      → Normalize weights to sum to 1.0                      │
│                                                             │
│    Output:                                                  │
│    {                                                        │
│      "agent_weights": {                                     │
│        "sentinel": 0.19,  # Was wrong, weight decreased     │
│        "oracle": 0.28,    # Was right, weight increased     │
│        "maverick": 0.18,                                    │
│        "fundamental": 0.20,                                 │
│        "technical": 0.15                                    │
│      }                                                      │
│    }                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 3. EVOLVE STRATEGY PARAMETERS                               │
│    python3 skills/poly-simulator/scripts/simulator.py       │
│    --evolve                                                 │
│                                                             │
│    Compare recent performance (last 15 days) vs             │
│    prior window (days 16-30):                               │
│      - Win rate improved with higher edge threshold?        │
│        → Increase edge_threshold by 5%                      │
│      - Sharpe dropped below 0.5?                            │
│        → AUTO-REVERT to defaults                            │
│      - Max drawdown exceeded safety limit?                  │
│        → Tighten stop_loss_pct by 5%                        │
│                                                             │
│    Cap all changes at 10% shift per parameter.              │
│                                                             │
│    Output:                                                  │
│    {                                                        │
│      "changes": {                                           │
│        "edge_threshold": {"from": 50, "to": 52},            │
│        "stop_loss_pct": {"from": 0.30, "to": 0.28}          │
│      },                                                     │
│      "reason": "Win rate improved 58%→62% with higher edge" │
│    }                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 4. CHECK GRADUATION                                         │
│    python3 skills/poly-simulator/scripts/simulator.py       │
│    --graduation-check                                       │
│                                                             │
│    Evaluate against criteria:                               │
│      [✓] 50+ trades (current: 67)                           │
│      [✓] 30+ days (current: 42)                             │
│      [✓] Win rate > 55% (current: 61%)                      │
│      [✓] Sharpe > 1.0 (current: 1.34)                       │
│      [✓] Max drawdown < 15% (current: 8.2%)                 │
│      [✗] 3 consecutive profitable weeks (current: 2)        │
│                                                             │
│    Output:                                                  │
│    {                                                        │
│      "graduated": false,                                    │
│      "progress": "83%",                                     │
│      "message": "Need 1 more profitable week"               │
│    }                                                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ 5. LOG CHANGES                                              │
│    All parameter changes → data/simulator/evolution_log.json│
│    Agent weights → data/research/agent_weights.json         │
│                                                             │
│    Agent summary: "Updated agent weights (Oracle +0.03),    │
│    evolved 2 parameters (edge_threshold 50→52,              │
│    stop_loss 30%→28%), graduation progress 83%"             │
│                                                             │
│    Session ends, next cycle tomorrow at midnight.           │
└─────────────────────────────────────────────────────────────┘
```

## Decision Rationale

### Why OpenClaw Instead of Standalone Python?

**Option A (Standalone Python bot):**
- Rigid if/then logic
- Hard to adapt to new market conditions
- Requires code changes for strategy tweaks
- No natural language context

**Option B (OpenClaw + LLM orchestration):**
- Agent can reason about trade-offs
- Natural language instructions (easier to maintain)
- Adapts to unexpected conditions (e.g., "high edge but market closes in 1 hour — skip")
- Built-in cron, heartbeat, and messaging infrastructure

**Trade-off:** OpenClaw adds LLM cost per agent turn, but saves engineering time and provides better adaptability.

### Why Cron vs Continuous Polling?

**Cron:**
- Event-driven (only runs when scheduled)
- Predictable resource usage
- Isolated sessions (failures don't corrupt state)
- Easy to reason about (4 distinct jobs, each with clear purpose)

**Continuous polling:**
- Wastes CPU cycles checking unchanged data
- Context pollution (long-running agents accumulate noise)
- Harder to debug (which iteration failed?)

**Trade-off:** Cron has fixed latency (max 30min before research cycle), but PolyClaw is not a high-frequency trader — 30min is acceptable.

### Why Heartbeat for Monitoring?

**Heartbeat:**
- Wakes agent only when needed (not continuous)
- Graceful degradation (if heartbeat fails, cron keeps running)
- Silent operation (only alerts on action needed)

**Alternative (webhook subscriptions):**
- Polymarket doesn't offer webhooks for resolution events
- Polling CLOB API every 5min is acceptable load

**Trade-off:** Heartbeat is simpler and more reliable than building a webhook relay.

### Why Python for Skills Instead of TypeScript?

**Python:**
- Richer ecosystem (httpx, asyncio, data science libs)
- Most AI/trading devs know Python
- Skills can run standalone (no Node.js dependency)

**TypeScript:**
- Would integrate tighter with OpenClaw gateway
- Better type safety

**Trade-off:** Python skills are more portable and easier for contributors. Gateway orchestration handles the glue logic.

## Polymarket API Integration

PolyClaw uses three Polymarket APIs:

### 1. Gamma API (Market Discovery)
- **Endpoint:** `https://gamma-api.polymarket.com/markets`
- **Purpose:** Discover active markets, filter by category, sort by volume
- **Auth:** None required (public API)
- **Used by:** poly-scanner, poly-research

### 2. CLOB API (Prices + Order Book)
- **Endpoint:** `https://clob.polymarket.com/prices`, `https://clob.polymarket.com/book`
- **Purpose:** Current YES/NO prices, order book depth (liquidity analysis)
- **Auth:** None required for read-only
- **Used by:** poly-scanner, poly-research, poly-simulator (price monitoring)

### 3. Data API (Trade History)
- **Endpoint:** `https://data-api.polymarket.com/trades`
- **Purpose:** Historical volume, whale trades, resolution outcomes
- **Auth:** None required
- **Used by:** poly-scanner (volume spike detection), poly-whale (whale tracking)

**Rate limits:** Polymarket doesn't publish official limits, but PolyClaw respects conservative thresholds (1 req/sec per endpoint) to avoid throttling.

## Data Persistence Model

All trading state lives in `data/` subdirectories (Docker volume mount):

```
data/
├── scanner_history.json      # Markets already scanned (deduplication)
│   Schema: { "seen": ["market_id_1", "market_id_2", ...] }
│
├── research/
│   ├── agent_weights.json    # EBMA weights per agent
│   │   Schema: {
│   │     "sentinel": 0.19,
│   │     "oracle": 0.28,
│   │     "maverick": 0.18,
│   │     "fundamental": 0.20,
│   │     "technical": 0.15
│   │   }
│   │
│   └── analysis_history.json # Past research results
│       Schema: [
│         {
│           "market_id": "...",
│           "timestamp": "...",
│           "consensus_prob": 0.68,
│           "edge_score": 72,
│           "outcome": null  # Filled on resolution
│         }
│       ]
│
├── simulator/
│   ├── bankroll.json         # Current bankroll + open/closed positions
│   │   Schema: {
│   │     "bankroll": 10847.00,
│   │     "initial_bankroll": 10000.00,
│   │     "positions": {
│   │       "open": [...],
│   │       "closed": [...]
│   │     }
│   │   }
│   │
│   ├── metrics.json          # Rolling 30-day performance metrics
│   │   Schema: {
│   │     "total_trades": 67,
│   │     "wins": 41,
│   │     "losses": 26,
│   │     "sharpe_ratio": 1.34,
│   │     "max_drawdown": 0.082,
│   │     ...
│   │   }
│   │
│   └── evolution_log.json    # Parameter change history
│       Schema: [
│         {
│           "timestamp": "...",
│           "changes": {"edge_threshold": {"from": 50, "to": 52}},
│           "reason": "Win rate improved",
│           "metrics_before": {...},
│           "metrics_after": {...}
│         }
│       ]
│
├── whale_history.json        # Tracked wallets + large trades
│   Schema: {
│     "whales": ["0xABC...", "0xDEF..."],
│     "trades": [
│       {
│         "wallet": "0xABC...",
│         "market_id": "...",
│         "side": "YES",
│         "amount_usd": 50000,
│         "timestamp": "..."
│       }
│     ]
│   }
│
└── watchlist.json            # Watched markets + alerts
    Schema: [
      {
        "market_id": "...",
        "question": "Will X happen?",
        "alerts": [
          {"type": "price_above", "threshold": 0.70},
          {"type": "resolution"}
        ]
      }
    ]
```

**Why JSON files instead of a database?**
- Simpler deployment (no Postgres container)
- Human-readable state (easy to debug)
- Git-friendly (can track changes in version control if desired)
- Sufficient scale (PolyClaw won't exceed 10K trades in paper trading phase)

**When to migrate to a database:**
- After graduation (live trading requires audit trail)
- When backtesting historical data (need complex queries)
- When scaling to multiple strategies simultaneously

## Security Considerations

### No Private Keys in Repo
- All credentials via environment variables (`.env` file gitignored)
- No Web3 wallet connection until graduated
- API keys never logged or committed

### Non-Root Container
- Runs as `node` user (UID 1000)
- Reduces attack surface (container escape prevention)

### Loopback Binding
- Gateway binds to `127.0.0.1` by default
- Prevents external access without explicit `OPENCLAW_GATEWAY_BIND=lan`

### Rate Limiting
- Skills respect Polymarket API limits (1 req/sec per endpoint)
- No DoS risk

### Isolated Skills
- Python scripts run as subprocesses (sandboxed from Node.js gateway)
- Skill failure doesn't crash gateway

## Performance Characteristics

### Latency
- Research cycle: ~30min interval (cron scheduling)
- Heartbeat checks: 5min interval
- Skill execution: ~2-10 seconds per skill (network-bound)

### Throughput
- Max ~20 research cycles per day (24h / 30min × 3 markets = 144 analyses/day)
- Max 20 open positions at once (configurable limit)

### Resource Usage
- Gateway: ~200MB RAM idle, ~500MB during cron job
- Python skills: ~50MB per skill execution (ephemeral)
- Disk: <100MB for data/ (scales with trade history)

### Cost Breakdown (Monthly Estimate)
- LLM API calls: ~1440 agent turns/month × $0.01 = $14.40
- Polymarket API: Free (no auth required for read-only)
- Hosting: $5-10 (Fly.io, Railway, or self-hosted)

**Total: ~$20-25/month** for autonomous paper trading

## Failure Modes and Recovery

### Skill Execution Failure
- Symptom: Python script crashes or returns malformed output
- Detection: Agent sees error in stdout/stderr
- Recovery: Agent logs error, skips that skill, continues session
- Mitigation: Skills validate input arguments, handle API errors gracefully

### API Rate Limiting
- Symptom: Polymarket API returns HTTP 429
- Detection: Skills retry with exponential backoff (built into httpx)
- Recovery: Agent skips current opportunity, retries next cycle
- Mitigation: Conservative request rate (1 req/sec)

### Heartbeat Missed
- Symptom: Heartbeat doesn't fire at 5min mark
- Detection: Next heartbeat notices gap in timestamps
- Recovery: Check for stop-loss triggers since last successful heartbeat
- Mitigation: Heartbeat failures don't corrupt state (stateless checks)

### Cron Job Overlap
- Symptom: Research cycle takes >30min, next cycle fires before completion
- Detection: OpenClaw prevents concurrent sessions (queues next job)
- Recovery: Next cycle waits for previous session to end
- Mitigation: Isolated sessions (each cron job is independent)

### Data File Corruption
- Symptom: JSON parse error when reading data/simulator/bankroll.json
- Detection: Skill crashes with parse error
- Recovery: Agent logs error, uses fallback defaults
- Mitigation: Atomic writes (write to .tmp, rename on success)

## Future Architecture Enhancements

### Multi-Strategy Support
- Current: Single "edge_hunter" strategy
- Future: Run multiple strategies in parallel (value, momentum, arbitrage)
- Challenge: Position correlation, bankroll allocation

### Web3 Integration (Live Trading)
- Current: Paper trading only
- Future: Connect to Web3 wallet, execute real trades on Polymarket
- Challenge: Gas fees, MEV protection, slippage handling

### Portfolio Rebalancing
- Current: Independent positions
- Future: Portfolio-level optimization (risk-parity, correlation limits)
- Challenge: Multi-dimensional optimization (edge, risk, correlation)

### Multi-Market Arbitrage
- Current: Single market analysis
- Future: Cross-market arbitrage (same outcome, different markets)
- Challenge: Execution latency, liquidity coordination

### Telegram Bot Interface
- Current: Delivery-only (daily P&L reports)
- Future: Two-way control (manual overrides, parameter tweaks)
- Challenge: Authentication, command parsing

---

**Last updated:** 2026-02-16

For questions or contributions, see the main README.
