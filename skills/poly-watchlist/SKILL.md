---
name: poly-watchlist
description: Personal Polymarket watchlist with price and volume alerts. Use when adding markets to watchlist, setting alerts, checking watched markets, or managing tracked positions. Triggers on "watchlist", "watch this market", "set alert", "track market", "my markets", "remove from watchlist".
metadata: { "openclaw": { "emoji": "📋" } }
---

# Poly Watchlist

Personal watchlist for tracking specific Polymarket markets with configurable alerts.

## Capabilities

1. **Add/Remove Markets** — Maintain a personal watchlist of interesting markets
2. **Price Alerts** — Trigger when YES/NO price crosses a threshold
3. **Volume Alerts** — Trigger when 24h volume exceeds a threshold
4. **Resolution Alerts** — Notify when a watched market resolves
5. **Watchlist Dashboard** — Summary view of all watched markets with current prices

## How to Use

### Add a Market
Provide the market question or condition ID. The skill will look up the market on Gamma API
and add it to `data/watchlist.json`.

### Set Alert
```
Alert types:
- price_above: Trigger when YES price > threshold
- price_below: Trigger when YES price < threshold
- volume_spike: Trigger when 24h volume > threshold
- resolution: Trigger when market resolves
```

### Check Watchlist
Run `scripts/watchlist.py --status` to display all watched markets with current prices.

### Remove Market
Specify market question or ID to remove from watchlist.

## API Endpoints

- **Gamma API**: `GET https://gamma-api.polymarket.com/markets?id={condition_id}` — Market details
- **CLOB API**: `GET https://clob.polymarket.com/prices?token_ids={ids}` — Current prices
- **Data API**: `GET https://data-api.polymarket.com/trades?market={slug}` — Recent trades

## Dashboard Output

```
📋 WATCHLIST (5 markets)
─────────────────────────────────────────
1. "Will X happen by Y?"
   YES: $0.72 (+0.05) | NO: $0.28 (-0.05)
   Vol 24h: $125K | Alert: price_above $0.80 ⏳

2. "Will Z occur?"
   YES: $0.45 (-0.12) | NO: $0.55 (+0.12)
   Vol 24h: $89K | Alert: price_below $0.40 ⏳

   🐋 Whale activity detected (see poly-whale)
─────────────────────────────────────────
```

## Cross-Skill Integration

- **poly-scanner**: When scanner finds an opportunity, suggest adding to watchlist
- **poly-whale**: When whale trades in a watched market, flag it
- **poly-simulator**: When simulator has a position in a watched market, show sim P&L

## Storage

`data/watchlist.json`:
```json
{
  "markets": [
    {
      "condition_id": "0x...",
      "question": "Will X happen?",
      "added_at": "2026-02-17T00:00:00Z",
      "alerts": [
        {"type": "price_above", "threshold": 0.80, "triggered": false}
      ],
      "notes": "User notes about why this is interesting"
    }
  ]
}
```
