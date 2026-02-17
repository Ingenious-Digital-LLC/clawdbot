---
name: poly-scanner
description: Polymarket market scanner and opportunity detector. Use when scanning for new markets, volume spikes, mispriced markets, arbitrage opportunities, or trending topics. Triggers on "scan markets", "find opportunities", "what's hot", "new listings", "mispriced", "arbitrage".
metadata: { "openclaw": { "emoji": "🔍" } }
---

# Poly Scanner

Autonomous market scanner that continuously monitors Polymarket for trading opportunities.

## Capabilities

1. **New Market Detection** — Discover newly listed markets before they get attention
2. **Volume Spike Detection** — Flag markets with unusual volume increases (>3x 24h average)
3. **Mispricing Detection** — Find markets where YES + NO prices sum to significantly less than $1.00
4. **Trending Topics** — Correlate news events with available markets
5. **Liquidity Analysis** — Identify markets with sufficient depth for entry/exit

## How to Scan

### Quick Scan (top opportunities right now)
Run `scripts/scan_markets.py --mode quick` to get the top 10 opportunities sorted by edge score.

### Deep Scan (comprehensive analysis)
Run `scripts/scan_markets.py --mode deep` to analyze all active markets with full scoring.

### New Listings
Run `scripts/scan_markets.py --mode new` to find markets created in the last 24 hours.

## API Endpoints Used

- **Gamma API**: `GET https://gamma-api.polymarket.com/markets` — Market discovery with filtering
  - Params: `active=true`, `closed=false`, `order=volume24hr`, `ascending=false`, `limit=100`
- **CLOB API**: `GET https://clob.polymarket.com/prices` — Current YES/NO prices per market
- **CLOB API**: `GET https://clob.polymarket.com/book` — Order book depth

## Opportunity Scoring

```
edge_score = (1.0 - (yes_price + no_price)) * volume_weight * liquidity_weight
```

Where:
- `volume_weight` = log(volume_24h) / 10 (normalized)
- `liquidity_weight` = min(best_bid_size, best_ask_size) / 100 (normalized)

Markets with `edge_score > 0.02` are flagged as opportunities.

## Output Format

Report each opportunity as:
```
Market: [question]
Category: [category]
YES: $[price] | NO: $[price] | Spread: [spread]
Volume 24h: $[volume] | Liquidity: $[liquidity]
Edge Score: [score]
Signal: [MISPRICING | VOLUME_SPIKE | NEW_LISTING | TRENDING]
```

## Scheduling

For continuous monitoring, scan every 5 minutes. Report only NEW opportunities (not previously seen).
Store seen opportunities in `data/scanner_history.json` to avoid duplicates.
