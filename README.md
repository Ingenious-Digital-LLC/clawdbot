# PolyClaw — Autonomous Polymarket Trading Bot

> **AI-powered prediction market trader that learns from every bet**

PolyClaw is an autonomous trading agent built on the OpenClaw AI agent platform. It continuously scans Polymarket for opportunities, analyzes them with multi-agent intelligence, executes paper trades, and evolves its own strategy parameters through self-learning feedback loops.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      OpenClaw Gateway                           │
│  (Node.js + Python hybrid — AI orchestration layer)            │
└──────┬────────────────────────────────────┬────────────────────┘
       │                                    │
  ┌────▼────────┐                     ┌────▼──────────┐
  │   Cron      │                     │  Heartbeat    │
  │  Scheduler  │                     │   Monitor     │
  │ (4 jobs)    │                     │ (every 5min)  │
  └────┬────────┘                     └────┬──────────┘
       │                                    │
       └──────────────┬─────────────────────┘
                      │
            ┌─────────▼──────────┐
            │  PolyClaw Agent    │
            │  (LLM brain with   │
            │   context files)   │
            └─────────┬──────────┘
                      │
         ┌────────────┼────────────┐
         │            │            │
    ┌────▼───┐   ┌───▼────┐   ┌──▼────┐
    │Scanner │   │Research│   │Simulator│
    │Skill   │   │Skill   │   │Skill   │
    └────┬───┘   └───┬────┘   └──┬────┘
         │           │           │
    ┌────▼───────────▼───────────▼────┐
    │     Polymarket APIs              │
    │  (Gamma, CLOB, Data)             │
    └──────────────────────────────────┘
```

**Key principle:** The LLM agent IS the brain — not Python scripts. OpenClaw's cron scheduler triggers agent turns with prompts like "run the research cycle", and the agent decides which skills to invoke and in what order based on its instructions.

## Trading Skills

| Skill | Purpose | Key Commands |
|-------|---------|--------------|
| **poly-scanner** 🔍 | Market discovery | `scan_markets.py --mode quick/deep/new` |
| **poly-research** 🧠 | Multi-agent analysis (5 AI agents) | `research.py --market "Question"` |
| **poly-simulator** 💰 | Paper trading + self-learning | `simulator.py --status/--evolve` |
| **poly-whale** 🐋 | Whale tracker | `whale_monitor.py --threshold 10000` |
| **poly-watchlist** 📌 | Price alerts | `watchlist.py --status` |

All skills are Python scripts that communicate with Polymarket APIs. The agent orchestrates them via shell commands.

## Quick Start

### Prerequisites
- Docker + Docker Compose
- OpenClaw account with API access
- Polymarket API access (no auth required for read-only)
- LLM provider API keys (see `.env.example`)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/clawdbot.git
cd clawdbot

# 2. Set up environment
cp .env.example .env
# Edit .env with your API keys:
#   - OPENCLAW_GATEWAY_TOKEN
#   - CLAUDE_AI_SESSION_KEY (or CLAUDE_WEB_SESSION_KEY + CLAUDE_WEB_COOKIE)
#   - MINIMAX_API_KEY (primary LLM)
#   - Optional: ZHIPU_API_KEY, GROQ_API_KEY (for multi-agent research)

# 3. Build the container
docker compose build openclaw-gateway

# 4. Start the gateway
docker compose up -d openclaw-gateway

# 5. Verify it's running
docker compose logs -f openclaw-gateway
# Should see "Gateway listening on 0.0.0.0:18789"

# 6. Attach to the CLI
docker compose run --rm openclaw-cli
# You're now in an interactive OpenClaw session
```

### Test the Skills

```bash
# Inside the OpenClaw CLI:
python3 skills/poly-scanner/scripts/scan_markets.py --mode quick
# Should return top 10 market opportunities

python3 skills/poly-simulator/scripts/simulator.py --status
# Shows paper trading bankroll and metrics
```

## How Autopilot Works

PolyClaw runs autonomously via **cron jobs** (scheduled agent turns) and **heartbeat monitoring** (continuous background checks).

### Cron Schedule

| Job | Frequency | What It Does |
|-----|-----------|--------------|
| **Research Cycle** | Every 30 minutes | Scans markets → analyzes top 3 opportunities → paper trades if edge ≥ 50 |
| **Learning Cycle** | Daily (midnight UTC) | Scores resolutions → updates agent weights → evolves strategy parameters |
| **Daily P&L** | 23:55 UTC | Generates trading summary for Telegram delivery |
| **Weekly Review** | Sunday 12:00 UTC | Deep performance analysis + graduation progress |

All jobs are defined in `config/cron/jobs.json`.

### Heartbeat Monitoring

Every 5 minutes, the agent wakes up and checks:
- Open positions for stop-loss triggers (price dropped 30%+ from entry)
- Watchlist markets for resolution events
- Newly resolved markets to score predictions

If nothing needs action, the agent replies `HEARTBEAT_OK` and goes dormant.

This pattern is defined in `workspace/HEARTBEAT.md`.

## The Self-Learning System

PolyClaw gets smarter over time through three feedback mechanisms:

### 1. Agent Weight Updates (EBMA — Ensemble Bayesian Model Averaging)

Five AI agents analyze each market:
- **Sentinel** (contrarian) — Challenges consensus
- **Oracle** (neutral) — Uses base rates and historical precedent
- **Maverick** (aggressive) — Spots trends early
- **Fundamental** (value) — Analyzes underlying factors
- **Technical** (patterns) — Reads market microstructure

After each market resolves, agents are scored:
```python
if abs(agent_prediction - actual_outcome) < 0.1:
    agent.weight *= 1.05  # Increase by 5%
else:
    agent.weight *= 0.95  # Decrease by 5%
```

Weights are normalized to sum to 1.0, so the consensus shifts toward accurate agents.

### 2. Thompson Sampling (Edge Threshold Tuning)

Different market categories have different efficiency levels. PolyClaw tracks which categories it wins on and adjusts edge thresholds accordingly:
```python
CATEGORY_EFFICIENCY = {
    "sports": 0.70,        # High bias → 1.43x edge multiplier
    "entertainment": 0.75,
    "science": 0.80,
    "politics": 0.85,
    "crypto": 0.95,
    "finance": 1.20,       # Very efficient → 0.83x edge reduction
}
```

### 3. Strategy Evolution (Parameter Auto-Tuning)

Every 24 hours, PolyClaw compares recent performance (last 15 days) vs prior window (days 16-30) and adjusts evolvable parameters:

| Parameter | Default | Range | Purpose |
|-----------|---------|-------|---------|
| `edge_threshold` | 50 | 20-80 | Minimum edge score to trade |
| `position_size_pct` | 5% | 1-10% | Base position size (half-Kelly) |
| `stop_loss_pct` | 30% | 10-50% | Exit trigger |
| `min_confidence` | 0.60 | 0.40-0.90 | Minimum agent consensus confidence |
| `min_agents_agree` | 3 | 2-5 | Minimum agents agreeing on direction |

**Safety rails:**
- Max 10% shift per parameter per cycle
- Auto-reverts to defaults if Sharpe ratio drops below 0.5
- Requires minimum 10 trades before suggesting changes

All changes are logged to `data/simulator/evolution_log.json`.

## Graduation Criteria (Live Trading Readiness)

PolyClaw starts in **paper trading mode** and must "graduate" before connecting to a Web3 wallet for real trades.

**Requirements:**
- ✅ Minimum 50 paper trades
- ✅ Minimum 30 days of history
- ✅ Win rate > 55%
- ✅ Sharpe ratio > 1.0
- ✅ Max drawdown < 15%
- ✅ Total return > 5%
- ✅ Three consecutive profitable weeks

Check progress:
```bash
python3 skills/poly-simulator/scripts/simulator.py --graduation-check
```

## Configuration

### OpenClaw Gateway (`config/openclaw.json`)
```json
{
  "agents": {
    "defaults": {
      "heartbeat": { "every": "5m", "target": "none" }
    }
  },
  "cron": { "enabled": true },
  "gateway": { "mode": "local" }
}
```

### Agent Identity (`workspace/SOUL.md`)
Defines PolyClaw's personality:
- Data-driven (no gut feelings, no FOMO)
- Risk-aware (half-Kelly sizing, stop losses)
- Patient (doing nothing is a valid strategy)
- Self-improving (track every outcome, evolve parameters)
- Concise (dashboard-style reports)

### Agent Instructions (`workspace/AGENTS.md`)
Complete trading pipeline instructions, skill reference, decision rules, and data storage paths.

### Heartbeat Checklist (`workspace/HEARTBEAT.md`)
What to check every 5 minutes — stop-losses, resolutions, alerts.

### Cron Jobs (`config/cron/jobs.json`)
Scheduled agent turns with prompts for each trading cycle.

## Environment Variables

```bash
# OpenClaw Gateway
OPENCLAW_GATEWAY_TOKEN=           # Gateway auth token
CLAUDE_AI_SESSION_KEY=            # Claude AI session key (primary)
# OR use web session:
CLAUDE_WEB_SESSION_KEY=           # Web session key
CLAUDE_WEB_COOKIE=                # Web session cookie

# LLM Providers
MINIMAX_API_KEY=                  # Primary LLM (Anthropic-compatible)
ZHIPU_API_KEY=                    # Multi-agent research (native web search)
GROQ_API_KEY=                     # Fast inference subagents

# Optional
FMP_API_KEY=                      # Financial Modeling Prep (macro data)
TWILIO_ACCOUNT_SID=               # WhatsApp notifications
TWILIO_AUTH_TOKEN=
TWILIO_WHATSAPP_FROM=
```

## Data Persistence

All trading state lives in `data/` subdirectories (mounted as Docker volumes):

```
data/
├── scanner_history.json      # Seen opportunities (deduplication)
├── research/
│   ├── agent_weights.json    # EBMA weights per agent
│   └── analysis_history.json # Past research results
├── simulator/
│   ├── bankroll.json         # Current bankroll + positions
│   ├── metrics.json          # Rolling performance metrics
│   └── evolution_log.json    # Parameter change history
├── whale_history.json        # Tracked wallets + large trades
└── watchlist.json            # Watched markets + alerts
```

## Project Structure

```
clawdbot/
├── config/                  # OpenClaw configuration
│   ├── openclaw.json       # Gateway settings
│   └── cron/jobs.json      # Scheduled trading cycles
├── workspace/               # Agent context files
│   ├── AGENTS.md           # Trading instructions
│   ├── HEARTBEAT.md        # Monitoring checklist
│   └── SOUL.md             # Agent identity
├── skills/                  # Python trading skills
│   ├── poly-scanner/       # Market discovery
│   ├── poly-research/      # Multi-agent analysis
│   ├── poly-simulator/     # Paper trading engine
│   ├── poly-whale/         # Whale tracker
│   └── poly-watchlist/     # Price alerts
├── data/                    # Persistent trading state (gitignored)
├── Dockerfile               # Node + Python container
├── docker-compose.yml       # Service orchestration
├── entrypoint.sh            # Skill sync on startup
├── .env.example             # Environment template
└── README.md                # This file
```

## Development

### Logs

```bash
# Gateway logs
docker compose logs -f openclaw-gateway

# CLI session
docker compose run --rm openclaw-cli

# Python script output
docker compose exec openclaw-gateway python3 skills/poly-scanner/scripts/scan_markets.py --mode quick
```

### Adding New Skills

1. Create `skills/your-skill/` directory
2. Add `SKILL.md` with metadata and documentation
3. Add `scripts/` with Python executables
4. Update `workspace/AGENTS.md` with skill reference
5. Rebuild container: `docker compose build openclaw-gateway`

### Modifying Cron Jobs

Edit `config/cron/jobs.json` and restart:
```bash
docker compose restart openclaw-gateway
```

### Testing Without Docker

```bash
# Install Python dependencies
pip3 install httpx

# Run skills directly
python3 skills/poly-scanner/scripts/scan_markets.py --mode quick

# Simulate a research cycle
python3 skills/poly-scanner/scripts/scan_markets.py --mode quick
python3 skills/poly-research/scripts/research.py --from-scanner --top 3
```

## Security

- **No private keys in the repo** — All credentials via environment variables
- **Paper trading only** — No Web3 wallet connection until graduated
- **Rate limiting** — Respects Polymarket API limits (built into skills)
- **Non-root container** — Runs as `node` user (UID 1000)
- **Loopback binding** — Gateway binds to `127.0.0.1` by default (override with `OPENCLAW_GATEWAY_BIND=lan`)

## Roadmap

- [ ] Graduation to live trading (Web3 integration)
- [ ] Multi-outcome market support (3+ options)
- [ ] Portfolio correlation analysis
- [ ] Risk-parity position sizing
- [ ] Conditional markets (outcome dependencies)
- [ ] Liquidity provision strategies
- [ ] Telegram bot interface for manual overrides

## Credits

Built on:
- [OpenClaw](https://github.com/openclaw/openclaw) — AI agent platform with cron + heartbeat orchestration
- [PolyHuntr](https://github.com/yourusername/polyhuntr) — Edge scoring engine + multi-agent research system (ported)

## License

MIT
