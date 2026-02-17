#!/usr/bin/env python3
"""
Polymarket Multi-Agent Research Engine - Intelligence layer.

Usage:
    python research.py --market "Will X happen?" --mode quick   # 3 agents, no web search
    python research.py --market "Will X happen?" --mode deep    # 5 agents + web search
    python research.py --from-scanner --top 10                  # Batch from scanner
    python research.py --show-weights                           # Show agent weights
"""

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

# --- Config ---
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"
RESEARCH_DIR = DATA_DIR / "research"
WEIGHTS_FILE = RESEARCH_DIR / "agent_weights.json"
HISTORY_FILE = RESEARCH_DIR / "analysis_history.json"

# LLM config — Multi-provider
MINIMAX_API_KEY = os.environ.get("MINIMAX_API_KEY", "")
MINIMAX_URL = os.environ.get("LLM_API_URL", "https://api.minimax.io/anthropic/v1/messages")
MINIMAX_MODEL = os.environ.get("LLM_MODEL", "MiniMax-M2.5")

ZHIPU_API_KEY = os.environ.get("ZHIPU_API_KEY", "")
GLM_URL = os.environ.get("GLM_API_URL", "https://api.z.ai/api/coding/paas/v4")
GLM_MODEL = "glm-4.7-flash"  # Free, no daily limit

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Agent → provider routing
AGENT_PROVIDERS = {
    "sentinel": "glm",       # Web search for news
    "oracle": "minimax",     # Deep Bayesian reasoning
    "maverick": "groq",      # Fast contrarian view
    "fundamental": "glm",    # Web search for macro data
    "technical": "minimax",  # Tool use for calculations
}

# Category efficiency multipliers (Becker research)
CATEGORY_EFFICIENCY = {
    "sports": 0.70,
    "entertainment": 0.75,
    "science": 0.80,
    "politics": 0.85,
    "pop-culture": 0.80,
    "business": 0.90,
    "crypto": 0.95,
    "finance": 1.20,
}

DEFAULT_EFFICIENCY = 0.85

# Agent timeout
AGENT_TIMEOUT = 60  # seconds


@dataclass
class AgentResult:
    agent_name: str
    probability: float
    confidence: float
    reasoning: str
    edge_direction: str  # YES or NO
    success: bool = True


@dataclass
class AnalysisReport:
    market_id: str
    question: str
    category: str
    yes_price: float
    no_price: float
    consensus_probability: float
    consensus_confidence: float
    consensus_direction: str
    edge_magnitude: float
    edge_score: int
    agent_results: list
    factors: dict
    timestamp: str


# --- Agent Definitions ---

AGENTS = {
    "sentinel": {
        "personality": "contrarian analyst who challenges consensus",
        "instruction": "Look for reasons the market is WRONG. Challenge popular opinion. What is the crowd missing?",
    },
    "oracle": {
        "personality": "neutral analyst who uses base rates and historical precedent",
        "instruction": "Use historical base rates and statistical analysis. What do similar past events suggest?",
    },
    "maverick": {
        "personality": "aggressive trend-spotter who identifies momentum shifts",
        "instruction": "Look for emerging trends, sentiment shifts, and early signals. What's changing right now?",
    },
    "fundamental": {
        "personality": "deep value analyst who examines underlying factors",
        "instruction": "Analyze the fundamental economics, incentives, and structural factors. What drives the outcome?",
    },
    "technical": {
        "personality": "pattern recognition specialist who reads market microstructure",
        "instruction": "Analyze volume patterns, price action, and orderbook dynamics. What does the market structure suggest?",
    },
}


def load_agent_weights() -> dict:
    """Load Bayesian-updated agent weights."""
    if WEIGHTS_FILE.exists():
        return json.loads(WEIGHTS_FILE.read_text())
    # Equal initial weights
    return {name: 1.0 / len(AGENTS) for name in AGENTS}


def save_agent_weights(weights: dict):
    """Save updated agent weights."""
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_FILE.write_text(json.dumps(weights, indent=2))


def load_analysis_history() -> list:
    """Load past analysis results."""
    if HISTORY_FILE.exists():
        return json.loads(HISTORY_FILE.read_text())
    return []


def save_analysis_history(history: list):
    """Save analysis history."""
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    # Keep last 200 analyses
    HISTORY_FILE.write_text(json.dumps(history[-200:], indent=2))


async def call_minimax(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call MiniMax M2.5 via Anthropic-compatible Messages API."""
    if not MINIMAX_API_KEY:
        return None
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            MINIMAX_URL,
            headers={
                "Authorization": f"Bearer {MINIMAX_API_KEY}",
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": MINIMAX_MODEL,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # MiniMax returns thinking + text blocks; find the text block
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"]
        return data["content"][0].get("text", str(data["content"][0]))


async def call_glm(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call GLM via OpenAI-compatible Chat Completions API."""
    if not ZHIPU_API_KEY:
        return await call_minimax(system_prompt, user_prompt)  # Fallback
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            f"{GLM_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {ZHIPU_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt + "\n\nIMPORTANT: Respond with ONLY the JSON object, no reasoning."},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
                "do_sample": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        # GLM reasoning models put analysis in reasoning_content, JSON in content
        content = msg.get("content") or ""
        if not content.strip():
            # Extract JSON from reasoning_content as fallback
            reasoning = msg.get("reasoning_content") or ""
            # Find JSON object in reasoning text
            start = reasoning.find("{")
            end = reasoning.rfind("}") + 1
            if start >= 0 and end > start:
                content = reasoning[start:end]
        return content if content.strip() else None


async def call_groq(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Call Groq via OpenAI-compatible Chat Completions API."""
    if not GROQ_API_KEY:
        return await call_minimax(system_prompt, user_prompt)  # Fallback
    async with httpx.AsyncClient(timeout=AGENT_TIMEOUT) as client:
        resp = await client.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def call_llm(system_prompt: str, user_prompt: str, provider: str = "minimax") -> Optional[str]:
    """Route to the correct LLM provider."""
    if provider == "glm":
        return await call_glm(system_prompt, user_prompt)
    elif provider == "groq":
        return await call_groq(system_prompt, user_prompt)
    else:
        return await call_minimax(system_prompt, user_prompt)


async def run_agent(
    agent_name: str,
    market_question: str,
    yes_price: float,
    volume_24h: float,
    category: str,
    days_to_resolution: Optional[int] = None,
) -> AgentResult:
    """Run a single AI agent analysis."""
    agent = AGENTS[agent_name]

    system_prompt = f"""You are {agent_name.title()}, a {agent['personality']}.
{agent['instruction']}

You MUST respond in valid JSON format:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "...", "edge_direction": "YES or NO"}}

probability: Your estimated probability of YES outcome (0.0-1.0)
confidence: How confident you are in your analysis (0.0-1.0)
reasoning: 2-3 sentences explaining your analysis
edge_direction: Which side has edge - "YES" or "NO"
"""

    user_prompt = f"""Analyze this Polymarket prediction market:

Market: {market_question}
Current YES Price: ${yes_price:.2f} (market's implied probability)
Category: {category}
Volume 24h: ${volume_24h:,.0f}
{"Days to Resolution: " + str(days_to_resolution) if days_to_resolution else ""}

What is the TRUE probability? Is the market mispriced?"""

    try:
        provider = AGENT_PROVIDERS.get(agent_name, "minimax")
        response = await call_llm(system_prompt, user_prompt, provider=provider)
        if response:
            # Extract JSON from response (handles markdown, extra text, etc.)
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            # Find first { ... } JSON block if there's extra text
            if not clean.startswith("{"):
                start = clean.find("{")
                end = clean.rfind("}") + 1
                if start >= 0 and end > start:
                    clean = clean[start:end]

            data = json.loads(clean)
            return AgentResult(
                agent_name=agent_name,
                probability=float(data.get("probability", 0.5)),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=str(data.get("reasoning", "")),
                edge_direction=str(data.get("edge_direction", "YES")).upper(),
            )
    except Exception as e:
        print(f"  Agent {agent_name} failed: {e}", file=sys.stderr)

    # Fallback signal: slight random offset from market price
    offset = random.uniform(-0.05, 0.05)
    return AgentResult(
        agent_name=agent_name,
        probability=max(0.01, min(0.99, yes_price + offset)),
        confidence=0.3,
        reasoning="Fallback signal (agent failed)",
        edge_direction="YES" if offset > 0 else "NO",
        success=False,
    )


def calculate_consensus(results: list[AgentResult], weights: dict) -> tuple[float, float, str]:
    """Calculate weighted consensus from agent results."""
    total_weight = 0
    weighted_prob = 0
    weighted_conf = 0

    for r in results:
        w = weights.get(r.agent_name, 0.2)
        weighted_prob += r.probability * w
        weighted_conf += r.confidence * w
        total_weight += w

    if total_weight > 0:
        consensus_prob = weighted_prob / total_weight
        consensus_conf = weighted_conf / total_weight
    else:
        consensus_prob = 0.5
        consensus_conf = 0.3

    direction = "YES" if consensus_prob > 0.5 else "NO"
    return round(consensus_prob, 4), round(consensus_conf, 4), direction


def calculate_edge_score(
    consensus_prob: float,
    yes_price: float,
    category: str,
    volume_24h: float,
    liquidity: float,
    agent_results: list[AgentResult],
    whale_agrees: bool = False,
) -> tuple[int, dict]:
    """Calculate 7-factor edge score (0-100)."""
    edge_magnitude = abs(consensus_prob - yes_price)

    # Base score (0-50)
    base_score = min(edge_magnitude * 500, 50)

    # Factor bonuses
    factors = {}

    # 1. Volume spike (simplified - would need historical data for real comparison)
    if volume_24h > 100_000:
        factors["volume_spike"] = 10
    elif volume_24h > 50_000:
        factors["volume_spike"] = 5

    # 2. Whale activity
    if whale_agrees:
        factors["whale_activity"] = 10

    # 3. Consensus strong (all agents agree on direction)
    directions = [r.edge_direction for r in agent_results if r.success]
    avg_confidence = sum(r.confidence for r in agent_results if r.success) / max(len([r for r in agent_results if r.success]), 1)
    if len(set(directions)) == 1 and avg_confidence > 0.7:
        factors["consensus_strong"] = 10
    elif len(set(directions)) == 1:
        factors["consensus_strong"] = 5

    # 4. Time pressure (would need end_date, simplified)
    # factors["time_pressure"] = 0

    # 5. Tail event
    if yes_price < 0.10 or yes_price > 0.90:
        factors["tail_event"] = 5

    # 6. Edge large
    if edge_magnitude > 0.15:
        factors["edge_large"] = 7
    elif edge_magnitude > 0.10:
        factors["edge_large"] = 4

    # 7. Liquidity depth
    if liquidity > 50_000:
        factors["liquidity_depth"] = 5
    elif liquidity > 10_000:
        factors["liquidity_depth"] = 3

    raw_score = base_score + sum(factors.values())

    # Category efficiency multiplier
    efficiency = CATEGORY_EFFICIENCY.get(category, DEFAULT_EFFICIENCY)
    multiplier = 1.0 / efficiency
    final_score = int(min(raw_score * multiplier, 100))

    factors["base_score"] = round(base_score, 1)
    factors["category_multiplier"] = round(multiplier, 2)

    return final_score, factors


async def analyze_market(
    market_question: str,
    mode: str = "quick",
    condition_id: Optional[str] = None,
) -> Optional[AnalysisReport]:
    """Run full multi-agent analysis on a market."""
    # Look up market data
    market_data = None
    with httpx.Client(timeout=15) as client:
        if condition_id:
            if condition_id.startswith("0x"):
                # Use PLURAL param: condition_ids (not condition_id)
                resp = client.get(
                    f"{GAMMA_API}/markets",
                    params={"condition_ids": condition_id},
                )
                resp.raise_for_status()
                markets = resp.json()
                market_data = markets[0] if markets else None
            else:
                # Treat as slug
                resp = client.get(f"{GAMMA_API}/markets", params={"slug": condition_id})
                resp.raise_for_status()
                slugged = resp.json()
                market_data = slugged[0] if slugged else None
        else:
            resp = client.get(
                f"{GAMMA_API}/markets",
                params={"active": "true", "closed": "false", "limit": "100"},
            )
            resp.raise_for_status()
            markets = resp.json()
            q_lower = market_question.lower()
            market_data = next(
                (m for m in markets if q_lower in m.get("question", "").lower()),
                None,
            )

    if not market_data:
        print(f"Market not found: {market_question}")
        return None

    # Extract market info
    question = market_data.get("question", market_question)
    cid = market_data.get("conditionId", condition_id or "")
    category = market_data.get("category", "unknown")
    volume_24h = float(market_data.get("volume24hr", 0))
    liquidity = float(market_data.get("liquidity", 0))

    prices_str = market_data.get("outcomePrices", "[]")
    try:
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        yes_price = float(prices[0])
        no_price = float(prices[1])
    except (json.JSONDecodeError, IndexError, TypeError):
        yes_price, no_price = 0.5, 0.5

    # Select agents based on mode
    if mode == "quick":
        agent_names = ["sentinel", "oracle", "maverick"]
    else:
        agent_names = list(AGENTS.keys())

    print(f"\nAnalyzing: {question}")
    print(f"  Mode: {mode} | Agents: {len(agent_names)} | Category: {category}")

    # Run agents: MiniMax/Groq in parallel, GLM sequentially (rate limit)
    glm_agents = [n for n in agent_names if AGENT_PROVIDERS.get(n) == "glm"]
    other_agents = [n for n in agent_names if AGENT_PROVIDERS.get(n) != "glm"]

    # Fire non-GLM agents in parallel
    parallel_tasks = [
        run_agent(name, question, yes_price, volume_24h, category)
        for name in other_agents
    ]
    parallel_results = await asyncio.gather(*parallel_tasks, return_exceptions=True)

    # Run GLM agents sequentially to avoid 429
    glm_results = []
    for name in glm_agents:
        r = await run_agent(name, question, yes_price, volume_24h, category)
        glm_results.append(r)

    results = list(parallel_results) + glm_results

    # Filter results
    agent_results = []
    for r in results:
        if isinstance(r, AgentResult):
            agent_results.append(r)
        else:
            print(f"  Agent error: {r}", file=sys.stderr)

    successful = [r for r in agent_results if r.success]
    if len(successful) < 2:
        print("  Insufficient agent responses for consensus")
        return None

    # Calculate consensus
    weights = load_agent_weights()
    consensus_prob, consensus_conf, direction = calculate_consensus(agent_results, weights)

    # Calculate edge score
    edge_magnitude = abs(consensus_prob - yes_price)
    edge_score, factors = calculate_edge_score(
        consensus_prob, yes_price, category, volume_24h, liquidity, agent_results
    )

    report = AnalysisReport(
        market_id=cid,
        question=question,
        category=category,
        yes_price=yes_price,
        no_price=no_price,
        consensus_probability=consensus_prob,
        consensus_confidence=consensus_conf,
        consensus_direction=direction,
        edge_magnitude=round(edge_magnitude, 4),
        edge_score=edge_score,
        agent_results=[asdict(r) for r in agent_results],
        factors=factors,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Save to history
    history = load_analysis_history()
    history.append(asdict(report))
    save_analysis_history(history)

    return report


def print_report(report: AnalysisReport):
    """Pretty-print an analysis report."""
    print(f"\n{'='*60}")
    print(f"  INTELLIGENCE REPORT")
    print(f"{'='*60}")
    print(f"  Market: {report.question}")
    print(f"  Current: YES ${report.yes_price:.2f} | NO ${report.no_price:.2f}")
    eff = CATEGORY_EFFICIENCY.get(report.category, DEFAULT_EFFICIENCY)
    print(f"  Category: {report.category} | Efficiency: {eff}")

    print(f"\n  AGENT ANALYSIS:")
    for r in report.agent_results:
        status = "" if r.get("success", True) else " [FALLBACK]"
        dir_arrow = "YES" if r["edge_direction"] == "YES" else "NO "
        print(
            f"    {r['agent_name']:15s}  P={r['probability']:.2f}  "
            f"C={r['confidence']:.2f}  -> {dir_arrow} edge{status}"
        )
        if r.get("reasoning") and "Fallback" not in r["reasoning"]:
            # Truncate reasoning for display
            reason = r["reasoning"][:80] + ("..." if len(r["reasoning"]) > 80 else "")
            print(f"      {reason}")

    print(f"\n  CONSENSUS: P={report.consensus_probability:.2f} (weighted)")
    print(f"  Direction: {report.consensus_direction}")
    print(f"  EDGE: {'+' if report.consensus_probability > report.yes_price else '-'}"
          f"{report.edge_magnitude:.2f} ({report.edge_magnitude*100:.0f} cents vs market)")

    print(f"\n  EDGE SCORE: {report.edge_score}/100")
    for factor, value in report.factors.items():
        if factor not in ("base_score", "category_multiplier"):
            print(f"    + {factor}: +{value}")
    print(f"    Base: {report.factors.get('base_score', 0)}/50")
    print(f"    x Category: {report.factors.get('category_multiplier', 1.0)}x")

    # Signal recommendation
    if report.edge_score >= 70:
        signal = "STRONG"
    elif report.edge_score >= 50:
        signal = "MODERATE"
    elif report.edge_score >= 30:
        signal = "WEAK"
    else:
        signal = "NO TRADE"

    action = f"BUY {report.consensus_direction}" if signal != "NO TRADE" else "PASS"
    print(f"\n  SIGNAL: {signal} — {action} @ ${report.yes_price:.2f}")
    print(f"{'='*60}")


def show_weights():
    """Display current agent weights."""
    weights = load_agent_weights()
    print("\nAgent Weights (Bayesian-updated):")
    print("-" * 40)
    for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
        bar = "#" * int(weight * 100)
        print(f"  {name:15s} {weight:.3f} {bar}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Multi-Agent Research")
    parser.add_argument("--market", type=str, help="Market question to analyze")
    parser.add_argument("--market-id", type=str, help="Market condition ID to analyze")
    parser.add_argument("--mode", choices=["quick", "deep"], default="quick")
    parser.add_argument("--from-scanner", action="store_true", help="Analyze top scanner results")
    parser.add_argument("--top", type=int, default=5, help="Number of scanner results to analyze")
    parser.add_argument("--show-weights", action="store_true", help="Show agent weights")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if args.show_weights:
        show_weights()
        return

    if args.market or args.market_id:
        report = asyncio.run(
            analyze_market(
                market_question=args.market or "",
                mode=args.mode,
                condition_id=args.market_id,
            )
        )
        if report:
            if args.json:
                print(json.dumps(asdict(report), indent=2))
            else:
                print_report(report)
    elif args.from_scanner:
        # Load scanner results
        scanner_file = DATA_DIR / "scanner_history.json"
        if not scanner_file.exists():
            print("No scanner data. Run poly-scanner first.")
            return

        # For now, just fetch top markets by volume
        print(f"Batch analyzing top {args.top} markets...")
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{GAMMA_API}/markets",
                params={"active": "true", "closed": "false", "order": "volume24hr", "ascending": "false", "limit": str(args.top)},
            )
            resp.raise_for_status()
            markets = resp.json()

        for m in markets:
            cid = m.get("conditionId", "")
            report = asyncio.run(analyze_market("", mode=args.mode, condition_id=cid))
            if report:
                if args.json:
                    print(json.dumps(asdict(report), indent=2))
                else:
                    print_report(report)
            time.sleep(2)  # Rate limiting between analyses
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
