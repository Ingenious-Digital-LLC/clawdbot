# Heartbeat — Autonomous Monitoring Loop

Run this checklist every heartbeat. Act on anything that needs attention. If nothing needs action, reply `HEARTBEAT_OK`.

## 1. Position Monitor
Run `python3 skills/poly-simulator/scripts/simulator.py --status` and check:
- Any position dropped 30%+ from entry? → Log the stop-loss close
- Any position with unrealized P&L > +50%? → Consider taking profit
- Bankroll drawdown > 15%? → Pause new trades, flag for review

## 2. Resolution Check
Check if any open positions' markets have resolved. If resolved:
- Run `python3 skills/poly-simulator/scripts/learning.py` to score predictions and update agent weights
- Report the outcome briefly

## 3. Watchlist Scan
Run `python3 skills/poly-watchlist/scripts/watchlist.py --status` and check:
- Any price alerts triggered?
- Any watched markets resolved?
- Any unusual volume spikes on watched markets?

## 4. Quick Opportunity Scan
If there are fewer than 5 open positions and no circuit breakers active:
- Run `python3 skills/poly-scanner/scripts/scan_markets.py --mode quick`
- If a new high-edge opportunity appears (edge > 0.05), flag it for the next research cycle

## Priority
Positions first (protect capital), then resolutions (learn), then opportunities (grow).
If nothing needs action: `HEARTBEAT_OK`
