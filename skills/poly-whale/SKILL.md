---
name: poly-whale
description: Polymarket whale watching and large trade detection. Use when tracking whale wallets, detecting large trades, monitoring insider activity, or finding smart money signals. Triggers on "whale watch", "large trades", "smart money", "insider activity", "track wallet", "who's buying".
metadata: { "openclaw": { "emoji": "🐋" } }
---

# Poly Whale

Monitor whale activity on Polymarket — large trades, wallet tracking, insider pattern detection.

## Capabilities

1. **Large Trade Detection** — Flag trades above configurable threshold ($10K+ default)
2. **Wallet Tracking** — Monitor specific whale wallets for new positions
3. **Insider Pattern Detection** — Fresh wallets + large bets + niche markets = suspicious
4. **Leaderboard Monitoring** — Track top traders' moves
5. **Copy Trade Signals** — Generate signals when tracked whales enter positions

## Data Sources

### Polymarket Data API (free, no auth)
- `GET https://data-api.polymarket.com/activity` — Recent trading activity
- `GET https://data-api.polymarket.com/positions?user={address}` — Wallet positions
- `GET https://data-api.polymarket.com/trades` — Trade history with filters

### Subgraph (Goldsky, free GraphQL)
- **Activity**: `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/activity-subgraph/0.0.4/gn`
- **Positions**: `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/positions-subgraph/0.0.7/gn`
- **PnL**: `https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn`

### CLOB WebSocket (real-time, free)
- `wss://ws-subscriptions-clob.polymarket.com/ws/market` — Subscribe to market price changes

## How to Use

### Monitor Large Trades
Run `scripts/whale_monitor.py --threshold 10000` to watch for trades above $10K.

### Track Specific Wallet
Run `scripts/whale_monitor.py --track 0xABC...` to monitor a whale wallet.

### Insider Detection
Run `scripts/whale_monitor.py --insider-scan` to scan for suspicious patterns:
- Wallet age < 7 days
- First trade > $5K
- Market has < $50K total volume (niche)

## Whale Classification

| Category | Trade Size | Emoji |
|----------|-----------|-------|
| Shrimp | < $1K | 🦐 |
| Fish | $1K - $10K | 🐟 |
| Dolphin | $10K - $50K | 🐬 |
| Whale | $50K - $250K | 🐋 |
| Mega Whale | > $250K | 🐳 |

## Alert Format

```
🐋 WHALE ALERT
Wallet: [address] (Known: [alias or "Unknown"])
Market: [question]
Side: [YES/NO] @ $[price]
Size: $[amount]
Wallet Age: [days]
Previous Trades: [count]
Win Rate: [rate]% (if known)
Signal Strength: [LOW | MEDIUM | HIGH]
```

Signal strength is HIGH when: known profitable trader + large size + early market entry.

## Watchlist Integration

When a whale trade is detected, check if the market is on the user's poly-watchlist.
If not, suggest adding it. If yes, flag it as a confirmation signal.

## Storage

Store whale data in `data/whale_history.json`:
- Tracked wallets and their aliases
- Trade history per wallet
- Win rate calculations
- Alert history (prevent duplicate alerts)
