#!/usr/bin/env python3
"""
Polymarket Watchlist Manager - Track markets with alerts.

Usage:
    python watchlist.py --status                      # Show all watched markets
    python watchlist.py --add "Will X happen?"        # Add market by question
    python watchlist.py --add-id 0xABC...             # Add market by condition ID
    python watchlist.py --remove "Will X happen?"     # Remove market
    python watchlist.py --alert 0xABC price_above 0.8 # Set price alert
    python watchlist.py --check-alerts                # Check all alert conditions
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# --- Config ---
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
WATCHLIST_FILE = DATA_DIR / "watchlist.json"


def load_watchlist() -> dict:
    """Load watchlist from disk."""
    if WATCHLIST_FILE.exists():
        return json.loads(WATCHLIST_FILE.read_text())
    return {"markets": []}


def save_watchlist(data: dict):
    """Save watchlist to disk."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    WATCHLIST_FILE.write_text(json.dumps(data, indent=2))


def fetch_market_by_id(condition_id: str) -> Optional[dict]:
    """Lookup market by condition ID or slug."""
    with httpx.Client(timeout=15) as client:
        if condition_id.startswith("0x"):
            # Use PLURAL param: condition_ids (not condition_id)
            resp = client.get(f"{GAMMA_API}/markets", params={"condition_ids": condition_id})
        else:
            resp = client.get(f"{GAMMA_API}/markets", params={"slug": condition_id})
        resp.raise_for_status()
        markets = resp.json()
        return markets[0] if markets else None


def search_market_by_question(question: str) -> Optional[dict]:
    """Search for a market by question text."""
    with httpx.Client(timeout=15) as client:
        resp = client.get(
            f"{GAMMA_API}/markets",
            params={"active": "true", "closed": "false", "limit": "20"},
        )
        resp.raise_for_status()
        markets = resp.json()

        q_lower = question.lower()
        for market in markets:
            if q_lower in market.get("question", "").lower():
                return market
    return None


def fetch_current_prices(market: dict) -> tuple[float, float]:
    """Get current YES/NO prices for a market."""
    prices_str = market.get("outcomePrices", "[]")
    try:
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        return float(prices[0]), float(prices[1])
    except (json.JSONDecodeError, IndexError, TypeError):
        pass

    # Fallback: try CLOB API
    token_ids_str = market.get("clobTokenIds", "[]")
    try:
        token_ids = json.loads(token_ids_str) if isinstance(token_ids_str, str) else token_ids_str
        if token_ids:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{CLOB_API}/prices",
                    params={"token_ids": ",".join(str(t) for t in token_ids)},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    prices_list = list(data.values())
                    if len(prices_list) >= 2:
                        return float(prices_list[0].get("price", 0.5)), float(prices_list[1].get("price", 0.5))
    except (json.JSONDecodeError, TypeError):
        pass

    return 0.5, 0.5


def add_market(identifier: str, is_id: bool = False, notes: str = "") -> bool:
    """Add a market to the watchlist."""
    if is_id:
        market = fetch_market_by_id(identifier)
    else:
        market = search_market_by_question(identifier)

    if not market:
        print(f"Market not found: {identifier}")
        return False

    watchlist = load_watchlist()

    # Check for duplicates
    condition_id = market.get("conditionId", market.get("id", ""))
    for m in watchlist["markets"]:
        if m.get("condition_id") == condition_id:
            print(f"Market already on watchlist: {market.get('question')}")
            return False

    entry = {
        "condition_id": condition_id,
        "question": market.get("question", "Unknown"),
        "category": market.get("category", "unknown"),
        "slug": market.get("slug", ""),
        "end_date": market.get("endDate", ""),
        "added_at": datetime.now(timezone.utc).isoformat(),
        "alerts": [],
        "notes": notes,
    }

    watchlist["markets"].append(entry)
    save_watchlist(watchlist)
    print(f"Added to watchlist: {entry['question']}")
    return True


def remove_market(identifier: str) -> bool:
    """Remove a market from the watchlist."""
    watchlist = load_watchlist()
    q_lower = identifier.lower()

    for i, m in enumerate(watchlist["markets"]):
        if q_lower in m.get("question", "").lower() or m.get("condition_id") == identifier:
            removed = watchlist["markets"].pop(i)
            save_watchlist(watchlist)
            print(f"Removed from watchlist: {removed['question']}")
            return True

    print(f"Market not found on watchlist: {identifier}")
    return False


def set_alert(condition_id: str, alert_type: str, threshold: float) -> bool:
    """Set an alert on a watched market."""
    valid_types = ["price_above", "price_below", "volume_spike", "resolution"]
    if alert_type not in valid_types:
        print(f"Invalid alert type. Must be one of: {valid_types}")
        return False

    watchlist = load_watchlist()

    for m in watchlist["markets"]:
        if m["condition_id"] == condition_id:
            alert = {
                "type": alert_type,
                "threshold": threshold,
                "triggered": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            m["alerts"].append(alert)
            save_watchlist(watchlist)
            print(f"Alert set: {alert_type} @ {threshold} for {m['question']}")
            return True

    print(f"Market {condition_id} not found on watchlist")
    return False


def check_alerts():
    """Check all alert conditions against current prices."""
    watchlist = load_watchlist()
    triggered = []

    for m in watchlist["markets"]:
        market_data = fetch_market_by_id(m["condition_id"])
        if not market_data:
            continue

        yes_price, no_price = fetch_current_prices(market_data)

        for alert in m["alerts"]:
            if alert["triggered"]:
                continue

            should_trigger = False

            if alert["type"] == "price_above" and yes_price > alert["threshold"]:
                should_trigger = True
            elif alert["type"] == "price_below" and yes_price < alert["threshold"]:
                should_trigger = True
            elif alert["type"] == "resolution":
                if market_data.get("closed", False):
                    should_trigger = True

            if should_trigger:
                alert["triggered"] = True
                alert["triggered_at"] = datetime.now(timezone.utc).isoformat()
                triggered.append({
                    "question": m["question"],
                    "alert_type": alert["type"],
                    "threshold": alert["threshold"],
                    "current_yes": yes_price,
                })

    save_watchlist(watchlist)

    if triggered:
        print(f"\n{'='*50}")
        print(f"  TRIGGERED ALERTS ({len(triggered)})")
        print(f"{'='*50}")
        for t in triggered:
            print(f"\n  {t['question']}")
            print(f"  Alert: {t['alert_type']} @ {t['threshold']}")
            print(f"  Current YES: ${t['current_yes']:.2f}")
    else:
        print("No alerts triggered.")


def show_status():
    """Display watchlist dashboard."""
    watchlist = load_watchlist()
    markets = watchlist.get("markets", [])

    if not markets:
        print("Watchlist is empty. Use --add to add markets.")
        return

    print(f"\n{'='*60}")
    print(f"  WATCHLIST ({len(markets)} markets)")
    print(f"{'='*60}")

    for i, m in enumerate(markets, 1):
        market_data = fetch_market_by_id(m["condition_id"])
        if market_data:
            yes_price, no_price = fetch_current_prices(market_data)
        else:
            yes_price, no_price = 0.0, 0.0

        print(f"\n{i}. \"{m['question']}\"")
        print(f"   YES: ${yes_price:.2f} | NO: ${no_price:.2f}")

        if market_data:
            vol = float(market_data.get("volume24hr", 0))
            print(f"   Vol 24h: ${vol:,.0f} | Category: {m.get('category', '?')}")

        # Show alerts
        for alert in m.get("alerts", []):
            status = "TRIGGERED" if alert["triggered"] else "pending"
            print(f"   Alert: {alert['type']} @ {alert['threshold']} [{status}]")

        if m.get("notes"):
            print(f"   Notes: {m['notes']}")

    print(f"\n{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Watchlist Manager")
    parser.add_argument("--status", action="store_true", help="Show watchlist dashboard")
    parser.add_argument("--add", type=str, help="Add market by question text")
    parser.add_argument("--add-id", type=str, help="Add market by condition ID")
    parser.add_argument("--remove", type=str, help="Remove market by question or ID")
    parser.add_argument("--alert", nargs=3, metavar=("ID", "TYPE", "THRESHOLD"),
                        help="Set alert: condition_id alert_type threshold")
    parser.add_argument("--check-alerts", action="store_true", help="Check alert conditions")
    parser.add_argument("--notes", type=str, default="", help="Notes when adding a market")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    try:
        if args.add:
            add_market(args.add, notes=args.notes)
        elif args.add_id:
            add_market(args.add_id, is_id=True, notes=args.notes)
        elif args.remove:
            remove_market(args.remove)
        elif args.alert:
            cid, atype, thresh = args.alert
            set_alert(cid, atype, float(thresh))
        elif args.check_alerts:
            check_alerts()
        elif args.json:
            print(json.dumps(load_watchlist(), indent=2))
        else:
            show_status()

    except httpx.HTTPError as e:
        print(f"API error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
