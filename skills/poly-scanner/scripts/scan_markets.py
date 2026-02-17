#!/usr/bin/env python3
"""
Polymarket Market Scanner - Discovers trading opportunities.

Usage:
    python scan_markets.py --mode quick    # Top 10 opportunities
    python scan_markets.py --mode deep     # Full market analysis
    python scan_markets.py --mode new      # New listings (24h)
    python scan_markets.py --arb           # Pair arbitrage scanner
    python scan_markets.py --bonds         # Bonding strategy scanner
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# --- Config ---
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
HISTORY_FILE = DATA_DIR / "scanner_history.json"

# Rate limiting
REQUEST_DELAY = 1.0  # seconds between requests


@dataclass
class Opportunity:
    market_id: str
    question: str
    category: str
    yes_price: float
    no_price: float
    spread: float
    volume_24h: float
    liquidity: float
    edge_score: float
    signal_type: str  # MISPRICING | VOLUME_SPIKE | NEW_LISTING | TRENDING
    discovered_at: str


@dataclass
class ArbitrageOpportunity:
    """Pair arbitrage opportunity where YES + NO != $1.00"""
    market_id: str
    question: str
    slug: str
    yes_price: float
    no_price: float
    spread: float
    profit_per_100: float  # Guaranteed profit per $100 invested
    volume_24h: float
    end_date: str
    discovered_at: str


@dataclass
class BondingOpportunity:
    """Near-certain outcome for bonding strategy"""
    market_id: str
    question: str
    slug: str
    near_certain_side: str  # YES or NO
    price: float
    return_pct: float  # Return percentage (e.g., 3.0 for 3%)
    days_to_resolution: int
    annualized_return: float  # Annualized return percentage
    volume_24h: float
    end_date: str
    discovered_at: str


def fetch_markets(mode: str = "quick", limit: int = 100) -> list[dict]:
    """Fetch markets from Gamma API."""
    params = {
        "active": "true",
        "closed": "false",
        "limit": str(limit),
    }

    if mode == "new":
        params["order"] = "startDate"
        params["ascending"] = "false"
    else:
        params["order"] = "volume24hr"
        params["ascending"] = "false"

    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{GAMMA_API}/markets", params=params)
        resp.raise_for_status()
        return resp.json()


def fetch_prices(token_ids: list[str]) -> dict:
    """Fetch current prices from CLOB API."""
    if not token_ids:
        return {}

    ids_str = ",".join(token_ids)
    with httpx.Client(timeout=15) as client:
        resp = client.get(f"{CLOB_API}/prices", params={"token_ids": ids_str})
        resp.raise_for_status()
        return resp.json()


def fetch_orderbook(token_id: str) -> dict:
    """Fetch orderbook depth for liquidity analysis."""
    with httpx.Client(timeout=15) as client:
        resp = client.get(f"{CLOB_API}/book", params={"token_id": token_id})
        resp.raise_for_status()
        return resp.json()


def calculate_interest_score(
    yes_price: float,
    no_price: float,
    volume_24h: float,
    liquidity: float,
) -> float:
    """
    Calculate market interest/tradability score (0-100).

    This is NOT an edge score — it ranks markets by how tradable and
    interesting they are for AI analysis. Actual edge detection happens
    in poly-research when agents estimate true probability.

    Factors:
    - Price uncertainty: Mid-range prices (0.3-0.7) are more interesting
      than near-resolved extremes (0.01 or 0.99)
    - Volume: Higher volume = more liquid, easier to trade
    - Liquidity: Deeper orderbook = less slippage
    """
    # Price uncertainty: peaks at 0.50, drops toward 0 and 1
    # Uses entropy-like measure: p * log(p) + (1-p) * log(1-p)
    p = max(0.01, min(0.99, yes_price))
    uncertainty = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    # uncertainty is 0-1, peaks at 0.50

    # Volume weight: log-scaled, 0-1
    volume_weight = min(math.log10(max(volume_24h, 1) + 1) / 6, 1.0)

    # Liquidity weight: 0-1
    liquidity_weight = min(liquidity / 200_000, 1.0)

    # Combine: uncertainty dominates (50%), volume (30%), liquidity (20%)
    raw = (uncertainty * 50) + (volume_weight * 30) + (liquidity_weight * 20)
    return round(raw, 2)


def detect_signal_type(market: dict, yes_price: float, no_price: float) -> str:
    """Classify the type of opportunity signal."""
    spread = 1.0 - (yes_price + no_price)

    # Check market age
    start_date = market.get("startDate", "")
    if start_date:
        try:
            created = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - created).total_seconds() / 3600
            if age_hours < 24:
                return "NEW_LISTING"
        except (ValueError, TypeError):
            pass

    # Mispricing: spread > 2 cents
    if spread > 0.02:
        return "MISPRICING"

    # Volume spike would require historical comparison (simplified here)
    volume_24h = float(market.get("volume24hr", 0))
    if volume_24h > 50_000:
        return "TRENDING"

    return "VOLUME_SPIKE"


def load_history() -> set:
    """Load previously seen opportunity IDs."""
    if HISTORY_FILE.exists():
        data = json.loads(HISTORY_FILE.read_text())
        return set(data.get("seen_ids", []))
    return set()


def save_history(seen_ids: set):
    """Save seen opportunity IDs to prevent duplicates."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "seen_ids": list(seen_ids),
        "last_scan": datetime.now(timezone.utc).isoformat(),
    }
    HISTORY_FILE.write_text(json.dumps(data, indent=2))


def scan(mode: str = "quick", show_seen: bool = False) -> list[Opportunity]:
    """Run a full market scan."""
    print(f"Scanning Polymarket ({mode} mode)...")

    limit = 20 if mode == "quick" else 100
    markets = fetch_markets(mode=mode, limit=limit)
    print(f"  Fetched {len(markets)} markets")

    seen = load_history() if not show_seen else set()
    opportunities = []

    for market in markets:
        market_id = market.get("conditionId", market.get("id", ""))
        if not market_id or (market_id in seen and not show_seen):
            continue

        # Parse prices from the market data
        prices_str = market.get("outcomePrices", "[]")
        try:
            prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
            yes_price = float(prices[0]) if len(prices) > 0 else 0.5
            no_price = float(prices[1]) if len(prices) > 1 else 0.5
        except (json.JSONDecodeError, IndexError, TypeError):
            yes_price, no_price = 0.5, 0.5

        volume_24h = float(market.get("volume24hr", 0))
        liquidity = float(market.get("liquidity", 0))

        interest = calculate_interest_score(yes_price, no_price, volume_24h, liquidity)
        signal_type = detect_signal_type(market, yes_price, no_price)

        # Include all markets with reasonable interest or new listings
        if interest > 10 or signal_type == "NEW_LISTING":
            opp = Opportunity(
                market_id=market_id,
                question=market.get("question", "Unknown"),
                category=market.get("category", "unknown"),
                yes_price=yes_price,
                no_price=no_price,
                spread=round(1.0 - (yes_price + no_price), 4),
                volume_24h=volume_24h,
                liquidity=liquidity,
                edge_score=interest,  # Now "interest score" — AI edge comes from poly-research
                signal_type=signal_type,
                discovered_at=datetime.now(timezone.utc).isoformat(),
            )
            opportunities.append(opp)
            seen.add(market_id)

        time.sleep(0.1)  # Light rate limiting

    # Sort by edge score
    opportunities.sort(key=lambda o: o.edge_score, reverse=True)

    save_history(seen)
    return opportunities


def scan_arbitrage(threshold: float = 0.025, limit: int = 100, quiet: bool = False) -> list[ArbitrageOpportunity]:
    """
    Scan for pair arbitrage opportunities where YES + NO != $1.00.

    When the spread is positive (YES + NO < 1.00), buying both sides
    guarantees profit at resolution.
    """
    if not quiet:
        print(f"Scanning for arbitrage opportunities (threshold: {threshold*100:.1f}%)...")

    opportunities = []
    offset = 0

    while offset < limit:
        params = {
            "active": "true",
            "closed": "false",
            "limit": "100",
            "offset": str(offset),
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(f"{GAMMA_API}/markets", params=params)
                resp.raise_for_status()
                markets = resp.json()

                if not markets:
                    break

                for market in markets:
                    # Parse prices
                    prices_str = market.get("outcomePrices", "[]")
                    try:
                        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                        if not prices or len(prices) < 2:
                            continue
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
                        continue

                    # Calculate spread
                    spread = 1.0 - (yes_price + no_price)

                    # Filter by threshold (looking for positive spreads)
                    if abs(spread) > threshold:
                        volume_24h = float(market.get("volume24hr", 0))
                        profit_per_100 = spread * 100

                        opp = ArbitrageOpportunity(
                            market_id=market.get("conditionId", market.get("id", "")),
                            question=market.get("question", "Unknown"),
                            slug=market.get("slug", ""),
                            yes_price=yes_price,
                            no_price=no_price,
                            spread=spread,
                            profit_per_100=profit_per_100,
                            volume_24h=volume_24h,
                            end_date=market.get("endDate", ""),
                            discovered_at=datetime.now(timezone.utc).isoformat(),
                        )
                        opportunities.append(opp)

                offset += len(markets)
                time.sleep(0.2)  # Rate limiting

        except httpx.HTTPError as e:
            if not quiet:
                print(f"API error at offset {offset}: {e}", file=sys.stderr)
            break

    # Sort by absolute spread (largest arbitrage first)
    opportunities.sort(key=lambda o: abs(o.spread), reverse=True)

    if not quiet:
        print(f"  Found {len(opportunities)} arbitrage opportunities")
    return opportunities


def scan_bonding(min_prob: float = 0.95, max_days: int = 90, limit: int = 100, quiet: bool = False) -> list[BondingOpportunity]:
    """
    Scan for bonding opportunities: near-certain outcomes (>95%) with clear resolution paths.

    These are low-risk, bond-like positions where you lock in small returns
    by betting on near-certain outcomes.
    """
    if not quiet:
        print(f"Scanning for bonding opportunities (min prob: {min_prob*100:.0f}%)...")

    opportunities = []
    offset = 0
    now = datetime.now(timezone.utc)

    while offset < limit:
        params = {
            "active": "true",
            "closed": "false",
            "limit": "100",
            "offset": str(offset),
        }

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.get(f"{GAMMA_API}/markets", params=params)
                resp.raise_for_status()
                markets = resp.json()

                if not markets:
                    break

                for market in markets:
                    # Parse prices
                    prices_str = market.get("outcomePrices", "[]")
                    try:
                        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                        if not prices or len(prices) < 2:
                            continue
                        yes_price = float(prices[0])
                        no_price = float(prices[1])
                    except (json.JSONDecodeError, IndexError, TypeError, ValueError):
                        continue

                    # Check if either side is near-certain
                    near_certain_side = None
                    price = 0.0

                    if yes_price >= min_prob:
                        near_certain_side = "YES"
                        price = yes_price
                    elif no_price >= min_prob:
                        near_certain_side = "NO"
                        price = no_price
                    else:
                        continue

                    # Parse end date
                    end_date_str = market.get("endDate", "")
                    if not end_date_str:
                        continue

                    try:
                        end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                        days_to_resolution = (end_date - now).days

                        # Filter by max_days
                        if days_to_resolution <= 0 or days_to_resolution > max_days:
                            continue

                    except (ValueError, TypeError):
                        continue

                    # Calculate returns
                    return_pct = (1.0 - price) * 100
                    annualized_return = (return_pct / days_to_resolution) * 365

                    volume_24h = float(market.get("volume24hr", 0))

                    opp = BondingOpportunity(
                        market_id=market.get("conditionId", market.get("id", "")),
                        question=market.get("question", "Unknown"),
                        slug=market.get("slug", ""),
                        near_certain_side=near_certain_side,
                        price=price,
                        return_pct=return_pct,
                        days_to_resolution=days_to_resolution,
                        annualized_return=annualized_return,
                        volume_24h=volume_24h,
                        end_date=end_date_str,
                        discovered_at=datetime.now(timezone.utc).isoformat(),
                    )
                    opportunities.append(opp)

                offset += len(markets)
                time.sleep(0.2)  # Rate limiting

        except httpx.HTTPError as e:
            if not quiet:
                print(f"API error at offset {offset}: {e}", file=sys.stderr)
            break

    # Sort by annualized return (highest first)
    opportunities.sort(key=lambda o: o.annualized_return, reverse=True)

    if not quiet:
        print(f"  Found {len(opportunities)} bonding opportunities")
    return opportunities


def print_opportunities(opportunities: list[Opportunity], top_n: int = 10):
    """Pretty-print opportunities."""
    if not opportunities:
        print("\nNo new opportunities found.")
        return

    print(f"\n{'='*60}")
    print(f"  SCAN RESULTS — {len(opportunities)} opportunities")
    print(f"{'='*60}")

    for i, opp in enumerate(opportunities[:top_n], 1):
        print(f"\n{i}. {opp.question}")
        print(f"   Category: {opp.category}")
        print(f"   YES: ${opp.yes_price:.2f} | NO: ${opp.no_price:.2f} | Spread: {opp.spread:.4f}")
        vol_str = f"${opp.volume_24h:,.0f}" if opp.volume_24h else "N/A"
        liq_str = f"${opp.liquidity:,.0f}" if opp.liquidity else "N/A"
        print(f"   Volume 24h: {vol_str} | Liquidity: {liq_str}")
        print(f"   Interest Score: {opp.edge_score:.1f}/100")
        print(f"   Signal: {opp.signal_type}")

    if len(opportunities) > top_n:
        print(f"\n... and {len(opportunities) - top_n} more")


def print_arbitrage_opportunities(opportunities: list[ArbitrageOpportunity], top_n: int = 10):
    """Pretty-print arbitrage opportunities."""
    if not opportunities:
        print("\nNo arbitrage opportunities found.")
        return

    print(f"\n{'='*70}")
    print(f"  ARBITRAGE OPPORTUNITIES — {len(opportunities)} found")
    print(f"{'='*70}")
    print("\nBuy both YES + NO when spread is positive to guarantee profit at resolution.")
    print("Minimum 2.5% spread recommended to overcome 2% fee + gas costs.\n")

    for i, opp in enumerate(opportunities[:top_n], 1):
        print(f"{i}. {opp.question[:80]}")
        print(f"   YES: ${opp.yes_price:.4f} | NO: ${opp.no_price:.4f}")
        print(f"   Spread: {opp.spread:+.4f} ({opp.spread*100:+.2f}%)")
        print(f"   Profit per $100: ${opp.profit_per_100:+.2f}")
        vol_str = f"${opp.volume_24h:,.0f}" if opp.volume_24h else "N/A"
        print(f"   Volume 24h: {vol_str}")
        print(f"   Market: https://polymarket.com/event/{opp.slug}")
        print()

    if len(opportunities) > top_n:
        print(f"... and {len(opportunities) - top_n} more\n")


def print_bonding_opportunities(opportunities: list[BondingOpportunity], top_n: int = 10):
    """Pretty-print bonding opportunities."""
    if not opportunities:
        print("\nNo bonding opportunities found.")
        return

    print(f"\n{'='*70}")
    print(f"  BONDING OPPORTUNITIES — {len(opportunities)} found")
    print(f"{'='*70}")
    print("\nNear-certain outcomes (>95% probability) with predictable returns.")
    print("Sorted by annualized return.\n")

    for i, opp in enumerate(opportunities[:top_n], 1):
        print(f"{i}. {opp.question[:80]}")
        print(f"   Side: {opp.near_certain_side} at ${opp.price:.4f}")
        print(f"   Return: {opp.return_pct:.2f}% over {opp.days_to_resolution} days")
        print(f"   Annualized: {opp.annualized_return:.1f}%")
        vol_str = f"${opp.volume_24h:,.0f}" if opp.volume_24h else "N/A"
        print(f"   Volume 24h: {vol_str}")
        print(f"   Resolves: {opp.end_date[:10]}")
        print(f"   Market: https://polymarket.com/event/{opp.slug}")
        print()

    if len(opportunities) > top_n:
        print(f"... and {len(opportunities) - top_n} more\n")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Market Scanner")

    # Scanner mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=["quick", "deep", "new"],
        help="Standard scan mode: quick (top 10), deep (all markets), new (24h listings)",
    )
    mode_group.add_argument("--arb", action="store_true", help="Arbitrage scanner mode")
    mode_group.add_argument("--bonds", action="store_true", help="Bonding strategy scanner mode")

    # Common options
    parser.add_argument("--top", type=int, default=10, help="Number of results to show")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Standard scan options
    parser.add_argument("--show-seen", action="store_true", help="Include previously seen markets (standard scan)")

    # Arbitrage options
    parser.add_argument("--threshold", type=float, default=0.025, help="Arbitrage spread threshold (default: 0.025 = 2.5%%)")

    # Bonding options
    parser.add_argument("--min-prob", type=float, default=0.95, help="Minimum probability for bonding (default: 0.95)")
    parser.add_argument("--max-days", type=int, default=90, help="Maximum days to resolution (default: 90)")

    args = parser.parse_args()

    try:
        if args.arb:
            # Arbitrage scanner
            opportunities = scan_arbitrage(threshold=args.threshold, limit=200, quiet=args.json)
            if args.json:
                print(json.dumps([asdict(o) for o in opportunities[:args.top]], indent=2))
            else:
                print_arbitrage_opportunities(opportunities, top_n=args.top)

        elif args.bonds:
            # Bonding scanner
            opportunities = scan_bonding(min_prob=args.min_prob, max_days=args.max_days, limit=200, quiet=args.json)
            if args.json:
                print(json.dumps([asdict(o) for o in opportunities[:args.top]], indent=2))
            else:
                print_bonding_opportunities(opportunities, top_n=args.top)

        else:
            # Standard scanner (default: quick mode)
            mode = args.mode or "quick"
            opportunities = scan(mode=mode, show_seen=args.show_seen)
            if args.json:
                print(json.dumps([asdict(o) for o in opportunities[:args.top]], indent=2))
            else:
                print_opportunities(opportunities, top_n=args.top)

    except httpx.HTTPError as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
