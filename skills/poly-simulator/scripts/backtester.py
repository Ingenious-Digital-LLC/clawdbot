#!/usr/bin/env python3
"""
PolyClaw Backtesting Framework — Event-Driven Historical Replay

Architecture:
    DataHandler → EventQueue → Strategy → Portfolio → ExecutionHandler → PerformanceAnalyzer

Components:
    1. DataHandler: Loads historical OHLCV candles, emits MarketEvent chronologically
    2. EventQueue: FIFO queue for events (MarketEvent, SignalEvent, OrderEvent, FillEvent)
    3. Strategy: Receives MarketEvents, generates SignalEvents based on strategy logic
    4. Portfolio: Tracks positions/cash, converts SignalEvents → OrderEvents
    5. ExecutionHandler: Simulates fills with slippage/fees, emits FillEvents
    6. PerformanceAnalyzer: Calculates Sharpe, Sortino, max drawdown, equity curve

Usage:
    # Run with sample data
    python backtester.py --strategy edge_hunter --data sample --output report.json

    # Run with historical CSV
    python backtester.py --strategy whale_follower --data /path/to/data.csv --output results.json

    # Quick test with synthetic data
    python backtester.py --strategy edge_hunter --data sample --verbose
"""

import argparse
import json
import math
import random
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List


# ============================================================================
# EVENT TYPES
# ============================================================================

class EventType(Enum):
    """Event types flowing through the system."""
    MARKET = "market"      # Price update from data feed
    SIGNAL = "signal"      # Strategy wants to trade
    ORDER = "order"        # Portfolio submits order
    FILL = "fill"          # Execution confirms fill


@dataclass
class MarketEvent:
    """Market price update event (OHLCV candle)."""
    type: EventType = EventType.MARKET
    timestamp: str = ""
    market_id: str = ""
    open: float = 0.0
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    volume_24h: float = 0.0  # Used for slippage calculation


@dataclass
class SignalEvent:
    """Trading signal from strategy."""
    type: EventType = EventType.SIGNAL
    timestamp: str = ""
    market_id: str = ""
    side: str = ""           # YES or NO
    confidence: float = 0.0  # 0.0-1.0
    edge_score: float = 0.0  # Strategy-specific score


@dataclass
class OrderEvent:
    """Order submitted to execution."""
    type: EventType = EventType.ORDER
    timestamp: str = ""
    market_id: str = ""
    side: str = ""
    size_usd: float = 0.0
    limit_price: float = 0.0  # Expected price


@dataclass
class FillEvent:
    """Order execution result."""
    type: EventType = EventType.FILL
    timestamp: str = ""
    market_id: str = ""
    side: str = ""
    size_usd: float = 0.0
    fill_price: float = 0.0
    slippage: float = 0.0
    fees: float = 0.0
    contracts: float = 0.0


# ============================================================================
# DATA HANDLER
# ============================================================================

class DataHandler:
    """
    Loads historical OHLCV data and replays it chronologically.

    Emits MarketEvents for each candle in the dataset.
    Supports CSV/JSON input and synthetic data generation.
    """

    def __init__(self, data_source: str):
        """
        Args:
            data_source: Path to CSV/JSON file, or 'sample' for synthetic data
        """
        self.data_source = data_source
        self.data: List[MarketEvent] = []
        self.index = 0

    def load_data(self):
        """Load data from source."""
        if self.data_source == "sample":
            self.data = generate_sample_data()
        elif self.data_source.endswith(".json"):
            self.data = self._load_json(self.data_source)
        elif self.data_source.endswith(".csv"):
            self.data = self._load_csv(self.data_source)
        else:
            raise ValueError(f"Unknown data source format: {self.data_source}")

        # Sort by timestamp
        self.data.sort(key=lambda x: x.timestamp)
        print(f"[DataHandler] Loaded {len(self.data)} candles from {self.data_source}")

    def _load_json(self, path: str) -> List[MarketEvent]:
        """Load from JSON file."""
        with open(path) as f:
            data = json.load(f)

        events = []
        for item in data:
            events.append(MarketEvent(
                timestamp=item["timestamp"],
                market_id=item["market_id"],
                open=item["open"],
                high=item["high"],
                low=item["low"],
                close=item["close"],
                volume_24h=item.get("volume_24h", 10000)
            ))
        return events

    def _load_csv(self, path: str) -> List[MarketEvent]:
        """Load from CSV file."""
        import csv
        events = []

        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                events.append(MarketEvent(
                    timestamp=row["timestamp"],
                    market_id=row["market_id"],
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume_24h=float(row.get("volume_24h", 10000))
                ))
        return events

    def get_next_event(self) -> Optional[MarketEvent]:
        """Get next chronological market event."""
        if self.index >= len(self.data):
            return None

        event = self.data[self.index]
        self.index += 1
        return event

    def has_more_data(self) -> bool:
        """Check if more data available."""
        return self.index < len(self.data)


# ============================================================================
# STRATEGY (Simplified for Backtesting)
# ============================================================================

class Strategy:
    """
    Base strategy class. Receives MarketEvents, generates SignalEvents.

    For backtesting, we use simplified logic based on STRATEGY_DEFAULTS from simulator.py.
    Real strategies would integrate with the research loop.
    """

    # Strategy parameter defaults (from simulator.py)
    DEFAULTS = {
        "edge_hunter": {
            "edge_threshold": 50,
            "position_size_pct": 0.05,
            "min_confidence": 0.60,
        },
        "whale_follower": {
            "edge_threshold": 30,
            "position_size_pct": 0.03,
            "min_confidence": 0.50,
        },
        "contrarian": {
            "edge_threshold": 40,
            "position_size_pct": 0.04,
            "min_confidence": 0.55,
        },
        "new_listing": {
            "edge_threshold": 30,
            "position_size_pct": 0.02,
            "min_confidence": 0.50,
        },
    }

    def __init__(self, strategy_name: str, params: Optional[Dict] = None):
        """
        Args:
            strategy_name: One of: edge_hunter, whale_follower, contrarian, new_listing
            params: Override default parameters
        """
        self.name = strategy_name
        self.params = self.DEFAULTS.get(strategy_name, self.DEFAULTS["edge_hunter"]).copy()
        if params:
            self.params.update(params)

        self.market_history: Dict[str, List[float]] = {}  # Track price history per market

    def on_market_event(self, event: MarketEvent) -> Optional[SignalEvent]:
        """
        Process market event and optionally generate signal.

        Simplified logic for backtesting:
        - Track price momentum
        - Generate signals based on simple rules (placeholder for real research loop)
        """
        market_id = event.market_id

        # Track history
        if market_id not in self.market_history:
            self.market_history[market_id] = []
        self.market_history[market_id].append(event.close)

        # Keep last 10 candles
        if len(self.market_history[market_id]) > 10:
            self.market_history[market_id].pop(0)

        # Need at least 5 candles for signal
        if len(self.market_history[market_id]) < 5:
            return None

        # Simplified signal generation (PLACEHOLDER — real strategy uses research loop)
        signal = self._generate_signal(market_id, event)

        return signal

    def _generate_signal(self, market_id: str, event: MarketEvent) -> Optional[SignalEvent]:
        """
        Generate trading signal based on strategy type.

        NOTE: This is a SIMPLIFIED placeholder. Real backtesting would:
        1. Use historical research agent outputs
        2. Apply full strategy logic from simulator.py
        3. Include whale tracking, sentiment analysis, etc.

        For MVP, we use simple momentum/mean-reversion rules.
        """
        prices = self.market_history[market_id]
        current_price = event.close

        # Calculate simple momentum
        if len(prices) >= 5:
            avg_5 = sum(prices[-5:]) / 5
            momentum = (current_price - avg_5) / avg_5 if avg_5 > 0 else 0
        else:
            return None

        # Generate signal based on strategy type
        if self.name == "edge_hunter":
            # Buy when price drops below moving average (assumes mispricing)
            if momentum < -0.05:  # 5% below average
                edge_score = abs(momentum) * 100
                if edge_score >= self.params["edge_threshold"]:
                    return SignalEvent(
                        timestamp=event.timestamp,
                        market_id=market_id,
                        side="YES",
                        confidence=min(0.5 + abs(momentum), 0.9),
                        edge_score=edge_score
                    )

        elif self.name == "contrarian":
            # Buy when price has moved a lot (mean reversion)
            if abs(momentum) > 0.10:  # 10% move
                edge_score = abs(momentum) * 100
                if edge_score >= self.params["edge_threshold"]:
                    side = "NO" if momentum > 0 else "YES"  # Fade the move
                    return SignalEvent(
                        timestamp=event.timestamp,
                        market_id=market_id,
                        side=side,
                        confidence=min(0.5 + abs(momentum) * 0.5, 0.9),
                        edge_score=edge_score
                    )

        elif self.name == "whale_follower":
            # Simplified: follow momentum (proxy for whale activity)
            if momentum > 0.03:
                edge_score = momentum * 100
                if edge_score >= self.params["edge_threshold"]:
                    return SignalEvent(
                        timestamp=event.timestamp,
                        market_id=market_id,
                        side="YES",
                        confidence=min(0.5 + momentum, 0.9),
                        edge_score=edge_score
                    )

        elif self.name == "new_listing":
            # Simplified: buy early dips (assumes initial mispricing)
            if momentum < -0.02 and current_price < 0.70:
                edge_score = abs(momentum) * 100 + (0.70 - current_price) * 50
                if edge_score >= self.params["edge_threshold"]:
                    return SignalEvent(
                        timestamp=event.timestamp,
                        market_id=market_id,
                        side="YES",
                        confidence=0.55,
                        edge_score=edge_score
                    )

        return None


# ============================================================================
# PORTFOLIO
# ============================================================================

@dataclass
class Position:
    """Open position."""
    market_id: str
    side: str
    entry_price: float
    size_usd: float
    contracts: float
    opened_at: str
    edge_score: float


class Portfolio:
    """
    Tracks cash, positions, and equity.
    Converts SignalEvents → OrderEvents using position sizing logic.
    Updates state from FillEvents.
    """

    def __init__(self, initial_capital: float, max_positions: int = 20):
        """
        Args:
            initial_capital: Starting bankroll
            max_positions: Maximum concurrent positions
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions

        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Dict] = []

        # Track current market prices for mark-to-market
        self.current_prices: Dict[str, float] = {}

    def update_market_price(self, market_id: str, price: float):
        """Update current market price (for mark-to-market)."""
        self.current_prices[market_id] = price

    def on_signal_event(self, signal: SignalEvent, strategy_params: Dict) -> Optional[OrderEvent]:
        """
        Convert signal to order.

        Returns:
            OrderEvent if position should be opened, None otherwise
        """
        # Check if we already have a position in this market
        if signal.market_id in self.positions:
            return None

        # Check max positions
        if len(self.positions) >= self.max_positions:
            return None

        # Check confidence threshold
        if signal.confidence < strategy_params.get("min_confidence", 0.60):
            return None

        # Calculate position size
        position_size_pct = strategy_params.get("position_size_pct", 0.05)
        equity = self.get_equity()
        size_usd = equity * position_size_pct

        # Minimum position check
        if size_usd < 10:
            return None

        # Don't exceed available cash
        size_usd = min(size_usd, self.cash * 0.95)  # Leave 5% buffer

        # Use current market price as limit price
        limit_price = self.current_prices.get(signal.market_id, 0.50)

        return OrderEvent(
            timestamp=signal.timestamp,
            market_id=signal.market_id,
            side=signal.side,
            size_usd=size_usd,
            limit_price=limit_price
        )

    def on_fill_event(self, fill: FillEvent):
        """Update portfolio from fill."""
        # Deduct cash (size + fees)
        total_cost = fill.size_usd + fill.fees
        self.cash -= total_cost

        # Open position
        self.positions[fill.market_id] = Position(
            market_id=fill.market_id,
            side=fill.side,
            entry_price=fill.fill_price,
            size_usd=fill.size_usd,
            contracts=fill.contracts,
            opened_at=fill.timestamp,
            edge_score=0  # Carried from signal, but not in fill event
        )

    def close_position(self, market_id: str, exit_price: float, timestamp: str) -> Dict:
        """
        Close a position and return trade result.

        Returns:
            Dict with trade details (pnl, hold_time, etc.)
        """
        if market_id not in self.positions:
            return {}

        pos = self.positions[market_id]

        # Calculate P&L
        # For YES positions: pnl = contracts * (exit_price - entry_price)
        # For NO positions: same logic (we bought NO at entry_price)
        pnl = pos.contracts * (exit_price - pos.entry_price)

        # Return cash
        exit_value = pos.contracts * exit_price
        self.cash += exit_value

        # Calculate hold time
        opened_dt = datetime.fromisoformat(pos.opened_at.replace("Z", "+00:00"))
        closed_dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        hold_days = (closed_dt - opened_dt).days

        # Remove position
        del self.positions[market_id]

        return {
            "market_id": market_id,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "size_usd": pos.size_usd,
            "contracts": pos.contracts,
            "pnl": round(pnl, 2),
            "opened_at": pos.opened_at,
            "closed_at": timestamp,
            "hold_days": hold_days,
            "edge_score": pos.edge_score,
        }

    def get_equity(self) -> float:
        """Calculate current equity (cash + position values)."""
        positions_value = 0
        for market_id, pos in self.positions.items():
            current_price = self.current_prices.get(market_id, pos.entry_price)
            positions_value += pos.contracts * current_price

        return self.cash + positions_value

    def record_equity(self, timestamp: str):
        """Record equity snapshot for equity curve."""
        positions_value = sum(
            pos.contracts * self.current_prices.get(market_id, pos.entry_price)
            for market_id, pos in self.positions.items()
        )

        self.equity_curve.append({
            "timestamp": timestamp,
            "equity": round(self.get_equity(), 2),
            "cash": round(self.cash, 2),
            "positions_value": round(positions_value, 2),
            "num_positions": len(self.positions)
        })


# ============================================================================
# EXECUTION HANDLER
# ============================================================================

class ExecutionHandler:
    """
    Simulates order execution with slippage and fees.

    Slippage Model (Level 1 — Fixed Spread):
    - Illiquid (< $5K volume): 5% spread
    - Medium ($5K-$50K): 2% spread
    - Liquid (> $50K): 1% spread

    Fee Model:
    - Standard markets: 0%
    - Crypto markets (15-min): 10 bps
    """

    def __init__(self, market_type: str = "standard"):
        """
        Args:
            market_type: 'standard' or 'crypto'
        """
        self.market_type = market_type

    def execute_order(self, order: OrderEvent, current_price: float, volume_24h: float) -> FillEvent:
        """
        Simulate order execution.

        Args:
            order: Order to execute
            current_price: Current market price
            volume_24h: 24h volume (for slippage calculation)

        Returns:
            FillEvent with actual fill price including slippage
        """
        # Calculate slippage
        slippage_pct = self._calculate_slippage(volume_24h, order.size_usd)
        slippage_usd = order.size_usd * slippage_pct

        # Fill price includes slippage (taker crosses spread)
        fill_price = current_price * (1 + slippage_pct)
        fill_price = max(0.01, min(fill_price, 0.99))  # Clamp to valid range

        # Calculate contracts
        contracts = order.size_usd / fill_price

        # Calculate fees
        fees = self._calculate_fees(order.size_usd)

        return FillEvent(
            timestamp=order.timestamp,
            market_id=order.market_id,
            side=order.side,
            size_usd=order.size_usd,
            fill_price=fill_price,
            slippage=slippage_usd,
            fees=fees,
            contracts=contracts
        )

    def _calculate_slippage(self, volume_24h: float, order_size: float) -> float:
        """
        Calculate slippage based on market liquidity.

        Level 1 model: Fixed spread based on volume tiers.
        """
        if volume_24h < 5000:
            spread_pct = 0.05  # 5%
        elif volume_24h < 50000:
            spread_pct = 0.02  # 2%
        else:
            spread_pct = 0.01  # 1%

        # Taker pays half the spread
        return spread_pct / 2

    def _calculate_fees(self, order_size: float) -> float:
        """Calculate trading fees."""
        if self.market_type == "crypto":
            return order_size * 0.001  # 10 bps
        else:
            return 0.0  # Standard markets have no fees


# ============================================================================
# PERFORMANCE ANALYZER
# ============================================================================

class PerformanceAnalyzer:
    """
    Calculates performance metrics:
    - Sharpe ratio (annualized)
    - Sortino ratio (downside deviation)
    - Max drawdown
    - Win rate, profit factor
    """

    def __init__(self):
        self.trades: List[Dict] = []

    def add_trade(self, trade: Dict):
        """Record closed trade."""
        self.trades.append(trade)

    def calculate_metrics(self, equity_curve: List[Dict], initial_capital: float) -> Dict:
        """
        Calculate all performance metrics.

        Returns:
            Dict with all metrics
        """
        if not self.trades:
            return {
                "total_trades": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "sortino_ratio": 0,
                "max_drawdown": 0,
                "total_pnl": 0,
                "total_return_pct": 0,
            }

        # Basic trade stats
        wins = [t for t in self.trades if t["pnl"] > 0]
        losses = [t for t in self.trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in self.trades)
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        # Profit factor
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio
        pnls = [t["pnl"] for t in self.trades]
        sharpe = self._calculate_sharpe(pnls)

        # Sortino ratio
        sortino = self._calculate_sortino(pnls)

        # Max drawdown from equity curve
        max_dd = self._calculate_max_drawdown(equity_curve)

        # Total return
        final_equity = equity_curve[-1]["equity"] if equity_curve else initial_capital
        total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100

        return {
            "total_trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
            "sharpe_ratio": round(sharpe, 4),
            "sortino_ratio": round(sortino, 4),
            "max_drawdown": round(max_dd, 4),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "avg_win": round(gross_profit / len(wins), 2) if wins else 0,
            "avg_loss": round(gross_loss / len(losses), 2) if losses else 0,
        }

    def _calculate_sharpe(self, pnls: List[float]) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(pnls) < 2:
            return 0.0

        mean_pnl = sum(pnls) / len(pnls)
        variance = sum((p - mean_pnl) ** 2 for p in pnls) / (len(pnls) - 1)
        std_pnl = math.sqrt(variance) if variance > 0 else 0.01

        # Annualize (assume ~1 trade per day → 252 trading days)
        sharpe = (mean_pnl / std_pnl) * math.sqrt(252)

        return sharpe

    def _calculate_sortino(self, pnls: List[float]) -> float:
        """Calculate Sortino ratio (only penalizes downside deviation)."""
        if len(pnls) < 2:
            return 0.0

        mean_pnl = sum(pnls) / len(pnls)

        # Downside deviation (only negative returns)
        downside_pnls = [p for p in pnls if p < 0]
        if not downside_pnls:
            return float('inf')

        downside_variance = sum(p ** 2 for p in downside_pnls) / len(downside_pnls)
        downside_std = math.sqrt(downside_variance)

        sortino = (mean_pnl / downside_std) * math.sqrt(252) if downside_std > 0 else 0

        return sortino

    def _calculate_max_drawdown(self, equity_curve: List[Dict]) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]["equity"]
        max_dd = 0.0

        for point in equity_curve:
            equity = point["equity"]
            if equity > peak:
                peak = equity

            drawdown = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)

        return max_dd


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class Backtester:
    """
    Main backtesting engine.

    Orchestrates all components:
    1. DataHandler loads historical data
    2. Events flow through queue
    3. Strategy generates signals
    4. Portfolio manages positions
    5. ExecutionHandler simulates fills
    6. PerformanceAnalyzer tracks results
    """

    def __init__(
        self,
        data_source: str,
        strategy_name: str,
        initial_capital: float = 10000,
        market_type: str = "standard",
        strategy_params: Optional[Dict] = None,
        verbose: bool = False
    ):
        """
        Args:
            data_source: Path to data or 'sample'
            strategy_name: Strategy to backtest
            initial_capital: Starting bankroll
            market_type: 'standard' or 'crypto'
            strategy_params: Override strategy parameters
            verbose: Print detailed logs
        """
        self.verbose = verbose

        # Initialize components
        self.data_handler = DataHandler(data_source)
        self.strategy = Strategy(strategy_name, strategy_params)
        self.portfolio = Portfolio(initial_capital)
        self.execution = ExecutionHandler(market_type)
        self.analyzer = PerformanceAnalyzer()

        # Event queue
        self.event_queue = deque()

        # Tracking
        self.current_time = ""
        self.events_processed = 0

    def run(self) -> Dict:
        """
        Run backtest.

        Returns:
            Dict with results (metrics, equity curve, trades)
        """
        print(f"\n{'='*60}")
        print(f"  BACKTESTING: {self.strategy.name}")
        print(f"{'='*60}")

        # Load data
        self.data_handler.load_data()

        # Main event loop
        while True:
            # Get next market event from data feed
            market_event = self.data_handler.get_next_event()

            if market_event is None:
                break

            self.event_queue.append(market_event)

            # Process all events in queue
            while self.event_queue:
                event = self.event_queue.popleft()
                self._process_event(event)
                self.events_processed += 1

            # Record equity snapshot after each candle
            if market_event:
                self.portfolio.record_equity(market_event.timestamp)

        # Close all remaining positions at final prices
        self._close_all_positions()

        # Calculate final metrics
        metrics = self.analyzer.calculate_metrics(
            self.portfolio.equity_curve,
            self.portfolio.initial_capital
        )

        # Print summary
        self._print_summary(metrics)

        return {
            "strategy": self.strategy.name,
            "metrics": metrics,
            "equity_curve": self.portfolio.equity_curve,
            "trades": self.analyzer.trades,
            "final_equity": self.portfolio.get_equity(),
            "events_processed": self.events_processed,
        }

    def _process_event(self, event):
        """Process single event based on type."""
        if isinstance(event, MarketEvent):
            self._handle_market_event(event)
        elif isinstance(event, SignalEvent):
            self._handle_signal_event(event)
        elif isinstance(event, OrderEvent):
            self._handle_order_event(event)
        elif isinstance(event, FillEvent):
            self._handle_fill_event(event)

    def _handle_market_event(self, event: MarketEvent):
        """Process market price update."""
        self.current_time = event.timestamp

        # Update portfolio's market prices (for mark-to-market)
        self.portfolio.update_market_price(event.market_id, event.close)

        # Strategy processes market event
        signal = self.strategy.on_market_event(event)

        if signal:
            self.event_queue.append(signal)
            if self.verbose:
                print(f"[{event.timestamp}] SIGNAL: {signal.side} {event.market_id} @ {event.close:.2f} (edge={signal.edge_score:.0f})")

    def _handle_signal_event(self, event: SignalEvent):
        """Process trading signal."""
        # Portfolio converts signal to order
        order = self.portfolio.on_signal_event(event, self.strategy.params)

        if order:
            self.event_queue.append(order)
            if self.verbose:
                print(f"[{event.timestamp}] ORDER: {order.side} ${order.size_usd:.0f} @ {order.limit_price:.2f}")

    def _handle_order_event(self, event: OrderEvent):
        """Process order submission."""
        # Get current market price and volume
        current_price = self.portfolio.current_prices.get(event.market_id, event.limit_price)

        # Get market event for volume (simplified — use last known volume)
        # In real backtesting, we'd track volume per market
        volume_24h = 10000  # Default assumption

        # Execute order
        fill = self.execution.execute_order(event, current_price, volume_24h)

        self.event_queue.append(fill)
        if self.verbose:
            print(f"[{event.timestamp}] FILL: {fill.contracts:.2f} contracts @ {fill.fill_price:.2f} (slippage=${fill.slippage:.2f}, fees=${fill.fees:.2f})")

    def _handle_fill_event(self, event: FillEvent):
        """Process fill confirmation."""
        # Update portfolio
        self.portfolio.on_fill_event(event)

    def _close_all_positions(self):
        """Close all open positions at final prices."""
        market_ids = list(self.portfolio.positions.keys())

        for market_id in market_ids:
            exit_price = self.portfolio.current_prices.get(market_id, 0.50)
            trade = self.portfolio.close_position(market_id, exit_price, self.current_time)

            if trade:
                self.analyzer.add_trade(trade)
                if self.verbose:
                    print(f"[{self.current_time}] CLOSE: {market_id} @ {exit_price:.2f} | P&L: ${trade['pnl']:.2f}")

    def _print_summary(self, metrics: Dict):
        """Print backtest summary."""
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"  Strategy: {self.strategy.name}")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Wins/Losses: {metrics['wins']}/{metrics['losses']}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Profit Factor: {metrics['profit_factor']}")
        print(f"\n  Performance:")
        print(f"    Total P&L: ${metrics['total_pnl']:.2f}")
        print(f"    Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"    Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"    Max Drawdown: {metrics['max_drawdown']:.1%}")
        print(f"\n  Trade Stats:")
        print(f"    Avg Win: ${metrics['avg_win']:.2f}")
        print(f"    Avg Loss: ${metrics['avg_loss']:.2f}")
        print(f"{'='*60}\n")


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_sample_data(
    num_candles: int = 200,
    num_markets: int = 5,
    start_date: str = "2025-01-01T00:00:00Z"
) -> List[MarketEvent]:
    """
    Generate synthetic OHLCV data for testing.

    Creates realistic prediction market price movements:
    - Bounded (0.01-0.99)
    - Trending + mean-reverting
    - Volume clustering
    - Multiple markets

    Args:
        num_candles: Number of 5-min candles per market
        num_markets: Number of markets to simulate
        start_date: Start timestamp

    Returns:
        List of MarketEvents
    """
    print(f"[DataGenerator] Generating {num_candles} candles for {num_markets} markets...")

    events = []
    start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))

    for market_idx in range(num_markets):
        market_id = f"market_{market_idx:03d}"

        # Random starting price (around 0.50)
        price = 0.50 + random.uniform(-0.20, 0.20)

        # Random trend direction
        trend = random.choice([-0.002, -0.001, 0, 0.001, 0.002])

        # Random volatility
        volatility = random.uniform(0.01, 0.05)

        for i in range(num_candles):
            # Timestamp
            timestamp = (start_dt + timedelta(minutes=5 * i)).isoformat().replace("+00:00", "Z")

            # Generate candle
            open_price = price

            # Random walk with trend
            price_change = trend + random.gauss(0, volatility)
            close_price = price + price_change

            # Clamp to valid range
            close_price = max(0.01, min(close_price, 0.99))

            # High/low
            high_price = max(open_price, close_price) + random.uniform(0, volatility * 0.5)
            low_price = min(open_price, close_price) - random.uniform(0, volatility * 0.5)

            high_price = max(0.01, min(high_price, 0.99))
            low_price = max(0.01, min(low_price, 0.99))

            # Volume (clustered)
            base_volume = random.uniform(5000, 50000)
            if random.random() < 0.1:  # 10% chance of volume spike
                base_volume *= random.uniform(2, 5)

            events.append(MarketEvent(
                timestamp=timestamp,
                market_id=market_id,
                open=round(open_price, 4),
                high=round(high_price, 4),
                low=round(low_price, 4),
                close=round(close_price, 4),
                volume_24h=round(base_volume, 2)
            ))

            # Update price for next candle
            price = close_price

            # Occasionally flip trend
            if random.random() < 0.05:
                trend = -trend

    return events


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PolyClaw Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with synthetic data
  python backtester.py --strategy edge_hunter --data sample --verbose

  # Run with CSV data
  python backtester.py --strategy whale_follower --data /path/to/data.csv --output results.json

  # Custom parameters
  python backtester.py --strategy contrarian --data sample --bankroll 50000 --market-type crypto
        """
    )

    parser.add_argument("--strategy", type=str, default="edge_hunter",
                        choices=["edge_hunter", "whale_follower", "contrarian", "new_listing"],
                        help="Trading strategy to backtest")
    parser.add_argument("--data", type=str, default="sample",
                        help="Data source: 'sample' or path to CSV/JSON")
    parser.add_argument("--bankroll", type=float, default=10000,
                        help="Starting capital (default: 10000)")
    parser.add_argument("--market-type", type=str, default="standard",
                        choices=["standard", "crypto"],
                        help="Market type for fee calculation")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed event logs")

    args = parser.parse_args()

    # Run backtest
    backtester = Backtester(
        data_source=args.data,
        strategy_name=args.strategy,
        initial_capital=args.bankroll,
        market_type=args.market_type,
        verbose=args.verbose
    )

    results = backtester.run()

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
