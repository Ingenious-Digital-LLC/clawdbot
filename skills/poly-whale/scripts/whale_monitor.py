#!/usr/bin/env python3
"""
Polymarket Whale Monitor - Track large trades and smart money.

Usage:
    python whale_monitor.py --threshold 10000        # Watch for $10K+ trades
    python whale_monitor.py --track 0xABC...         # Track specific wallet
    python whale_monitor.py --insider-scan            # Scan for suspicious patterns
    python whale_monitor.py --leaderboard             # Show top traders
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import httpx

# --- Config ---
DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
GOLDSKY_PNL = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/pnl-subgraph/0.0.14/gn"
GOLDSKY_POSITIONS = "https://api.goldsky.com/api/public/project_cl6mb8i9h0003e201j6li0diw/subgraphs/positions-subgraph/0.0.7/gn"

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
WHALE_FILE = DATA_DIR / "whale_history.json"

# Whale classification thresholds
WHALE_TIERS = {
    "mega_whale": 250_000,
    "whale": 50_000,
    "dolphin": 10_000,
    "fish": 1_000,
    "shrimp": 0,
}

WHALE_EMOJI = {
    "mega_whale": "🐳",
    "whale": "🐋",
    "dolphin": "🐬",
    "fish": "🐟",
    "shrimp": "🦐",
}

# Known whale wallets (ported from polyhuntr)
KNOWN_WHALES = {
    "0x1234...": {"alias": "domah", "notes": "Consistent 65% win rate on politics"},
    "0x5678...": {"alias": "50pence", "notes": "High-volume crypto market trader"},
}

# Categories to ignore (too efficient or too noisy)
IGNORE_KEYWORDS = [
    "bitcoin", "btc", "ethereum", "eth", "crypto", "solana",
    "nfl", "nba", "mlb", "nhl", "premier league", "champions league",
]


@dataclass
class WhaleTrade:
    wallet: str
    alias: Optional[str]
    market_question: str
    market_id: str
    side: str  # YES or NO
    price: float
    size_usd: float
    tier: str
    emoji: str
    timestamp: str
    wallet_age_days: Optional[int] = None
    previous_trades: Optional[int] = None
    win_rate: Optional[float] = None
    signal_strength: str = "MEDIUM"


def classify_whale(size_usd: float) -> tuple[str, str]:
    """Classify trader by trade size."""
    for tier, threshold in WHALE_TIERS.items():
        if size_usd >= threshold:
            return tier, WHALE_EMOJI[tier]
    return "shrimp", "🦐"


def should_ignore_market(question: str) -> bool:
    """Filter out markets that are too efficient or noisy."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in IGNORE_KEYWORDS)


def fetch_recent_trades(limit: int = 100) -> list[dict]:
    """Fetch recent trading activity from Data API."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{DATA_API}/activity", params={"limit": str(limit)})
        resp.raise_for_status()
        return resp.json()


def fetch_wallet_positions(address: str) -> list[dict]:
    """Fetch all positions for a wallet."""
    with httpx.Client(timeout=30) as client:
        resp = client.get(f"{DATA_API}/positions", params={"user": address})
        resp.raise_for_status()
        return resp.json()


def fetch_wallet_pnl(address: str) -> Optional[dict]:
    """Fetch PnL data from Goldsky subgraph."""
    query = """
    {
      users(where: {id: "%s"}) {
        id
        totalProfit
        totalLoss
        numTrades
        numWins
        numLosses
      }
    }
    """ % address.lower()

    with httpx.Client(timeout=15) as client:
        resp = client.post(GOLDSKY_PNL, json={"query": query})
        if resp.status_code == 200:
            data = resp.json()
            users = data.get("data", {}).get("users", [])
            return users[0] if users else None
    return None


def fetch_top_traders(limit: int = 20) -> list[dict]:
    """Fetch top traders from Goldsky PnL subgraph."""
    query = """
    {
      users(
        first: %d
        orderBy: totalProfit
        orderDirection: desc
        where: { numTrades_gt: 10 }
      ) {
        id
        totalProfit
        totalLoss
        numTrades
        numWins
        numLosses
      }
    }
    """ % limit

    with httpx.Client(timeout=15) as client:
        resp = client.post(GOLDSKY_PNL, json={"query": query})
        resp.raise_for_status()
        data = resp.json()
        return data.get("data", {}).get("users", [])


def calculate_signal_strength(
    tier: str,
    alias: Optional[str],
    win_rate: Optional[float],
    market_volume: float,
) -> str:
    """Determine signal strength based on whale quality indicators."""
    score = 0

    # Known profitable whale
    if alias and win_rate and win_rate > 0.55:
        score += 3

    # Large trade
    if tier in ("whale", "mega_whale"):
        score += 2
    elif tier == "dolphin":
        score += 1

    # Niche market (low volume = less efficient = more alpha)
    if market_volume < 50_000:
        score += 1

    if score >= 4:
        return "HIGH"
    elif score >= 2:
        return "MEDIUM"
    return "LOW"


def load_whale_data() -> dict:
    """Load whale tracking data."""
    if WHALE_FILE.exists():
        return json.loads(WHALE_FILE.read_text())
    return {"tracked_wallets": {}, "trade_history": [], "alert_history": []}


def save_whale_data(data: dict):
    """Save whale tracking data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WHALE_FILE.write_text(json.dumps(data, indent=2))


def scan_for_whales(threshold: float = 10_000) -> list[WhaleTrade]:
    """Scan recent trades for whale activity."""
    print(f"Scanning for trades >= ${threshold:,.0f}...")

    trades = fetch_recent_trades(limit=200)
    whale_trades = []

    for trade in trades:
        size_usd = float(trade.get("size", 0)) * float(trade.get("price", 0))
        if size_usd < threshold:
            continue

        question = trade.get("market", {}).get("question", "Unknown")
        if should_ignore_market(question):
            continue

        wallet = trade.get("maker_address", trade.get("user", "unknown"))
        alias = KNOWN_WHALES.get(wallet, {}).get("alias")
        tier, emoji = classify_whale(size_usd)

        # Try to get wallet stats
        win_rate = None
        pnl_data = fetch_wallet_pnl(wallet)
        if pnl_data:
            total = int(pnl_data.get("numTrades", 0))
            wins = int(pnl_data.get("numWins", 0))
            win_rate = wins / total if total > 0 else None

        signal = calculate_signal_strength(
            tier, alias, win_rate,
            float(trade.get("market", {}).get("volume", 0))
        )

        wt = WhaleTrade(
            wallet=wallet[:10] + "..." + wallet[-6:] if len(wallet) > 16 else wallet,
            alias=alias,
            market_question=question,
            market_id=trade.get("market", {}).get("conditionId", ""),
            side="YES" if trade.get("outcome", "") == "Yes" else "NO",
            price=float(trade.get("price", 0)),
            size_usd=round(size_usd, 2),
            tier=tier,
            emoji=emoji,
            timestamp=trade.get("timestamp", datetime.now(timezone.utc).isoformat()),
            win_rate=round(win_rate, 3) if win_rate else None,
            signal_strength=signal,
        )
        whale_trades.append(wt)
        time.sleep(0.5)  # Rate limiting for PnL lookups

    return whale_trades


def insider_scan() -> list[WhaleTrade]:
    """Scan for suspicious insider-like patterns."""
    print("Scanning for insider patterns...")
    print("  Criteria: Fresh wallet + Large bet + Niche market")

    trades = fetch_recent_trades(limit=200)
    suspicious = []

    for trade in trades:
        size_usd = float(trade.get("size", 0)) * float(trade.get("price", 0))
        if size_usd < 5_000:  # Lower threshold for insider scan
            continue

        question = trade.get("market", {}).get("question", "Unknown")
        if should_ignore_market(question):
            continue

        market_volume = float(trade.get("market", {}).get("volume", 0))
        wallet = trade.get("maker_address", trade.get("user", ""))

        # Check if niche market (low volume)
        if market_volume > 50_000:
            continue

        # Check wallet history
        pnl_data = fetch_wallet_pnl(wallet)
        num_trades = int(pnl_data.get("numTrades", 0)) if pnl_data else 0

        # Suspicious: few previous trades + large bet + niche market
        if num_trades < 5:
            tier, emoji = classify_whale(size_usd)
            wt = WhaleTrade(
                wallet=wallet[:10] + "..." + wallet[-6:] if len(wallet) > 16 else wallet,
                alias=None,
                market_question=question,
                market_id=trade.get("market", {}).get("conditionId", ""),
                side="YES" if trade.get("outcome", "") == "Yes" else "NO",
                price=float(trade.get("price", 0)),
                size_usd=round(size_usd, 2),
                tier=tier,
                emoji=emoji,
                timestamp=trade.get("timestamp", datetime.now(timezone.utc).isoformat()),
                previous_trades=num_trades,
                signal_strength="HIGH",
            )
            suspicious.append(wt)

        time.sleep(0.5)

    return suspicious


def print_whale_trades(trades: list[WhaleTrade]):
    """Pretty-print whale trades."""
    if not trades:
        print("\nNo whale trades detected.")
        return

    for wt in trades:
        print(f"\n{wt.emoji} WHALE ALERT")
        print(f"  Wallet: {wt.wallet}" + (f" ({wt.alias})" if wt.alias else " (Unknown)"))
        print(f"  Market: {wt.market_question}")
        print(f"  Side: {wt.side} @ ${wt.price:.2f}")
        print(f"  Size: ${wt.size_usd:,.2f} ({wt.tier})")
        if wt.win_rate:
            print(f"  Win Rate: {wt.win_rate:.1%}")
        if wt.previous_trades is not None:
            print(f"  Previous Trades: {wt.previous_trades}")
        print(f"  Signal Strength: {wt.signal_strength}")


def show_leaderboard():
    """Display top traders leaderboard."""
    print("Fetching leaderboard from Goldsky...")
    traders = fetch_top_traders(limit=20)

    if not traders:
        print("Could not fetch leaderboard data.")
        return

    print(f"\n{'='*60}")
    print(f"  TOP TRADERS LEADERBOARD")
    print(f"{'='*60}")

    for i, t in enumerate(traders, 1):
        profit = float(t.get("totalProfit", 0))
        loss = float(t.get("totalLoss", 0))
        trades = int(t.get("numTrades", 0))
        wins = int(t.get("numWins", 0))
        win_rate = wins / trades if trades > 0 else 0
        net = profit - loss

        wallet = t.get("id", "unknown")
        short_wallet = wallet[:10] + "..." + wallet[-6:] if len(wallet) > 16 else wallet
        alias = KNOWN_WHALES.get(wallet, {}).get("alias", "")

        print(f"\n{i:2d}. {short_wallet}" + (f" ({alias})" if alias else ""))
        print(f"    Net P&L: ${net:,.2f} | Win Rate: {win_rate:.1%} | Trades: {trades}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Whale Monitor")
    parser.add_argument("--threshold", type=float, default=10_000, help="Minimum trade size in USD")
    parser.add_argument("--track", type=str, help="Track a specific wallet address")
    parser.add_argument("--insider-scan", action="store_true", help="Scan for insider-like patterns")
    parser.add_argument("--leaderboard", action="store_true", help="Show top traders")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        if args.leaderboard:
            show_leaderboard()
        elif args.insider_scan:
            trades = insider_scan()
            if args.json:
                print(json.dumps([asdict(t) for t in trades], indent=2))
            else:
                print(f"\nFound {len(trades)} suspicious patterns:")
                print_whale_trades(trades)
        elif args.track:
            print(f"Fetching positions for {args.track}...")
            positions = fetch_wallet_positions(args.track)
            print(json.dumps(positions, indent=2))
        else:
            trades = scan_for_whales(threshold=args.threshold)
            if args.json:
                print(json.dumps([asdict(t) for t in trades], indent=2))
            else:
                print(f"\nFound {len(trades)} whale trades:")
                print_whale_trades(trades)

                # Save to history
                whale_data = load_whale_data()
                whale_data["trade_history"].extend([asdict(t) for t in trades])
                whale_data["trade_history"] = whale_data["trade_history"][-500:]  # Keep last 500
                save_whale_data(whale_data)

    except httpx.HTTPError as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
