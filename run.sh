#!/bin/bash
# Load env and run a skill script
# Usage: ./run.sh scanner --mode quick
#        ./run.sh research --market-id 0x...
#        ./run.sh whale --leaderboard
#        ./run.sh simulator --status
#        ./run.sh watchlist --status

set -e

# Load env vars (strips comments)
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep -v '^\s*$' | xargs)
fi

SKILL="$1"
shift

case "$SKILL" in
    scanner)
        python3 workspace/skills/poly-scanner/scripts/scan_markets.py "$@"
        ;;
    whale)
        python3 workspace/skills/poly-whale/scripts/whale_monitor.py "$@"
        ;;
    watchlist)
        python3 workspace/skills/poly-watchlist/scripts/watchlist.py "$@"
        ;;
    research)
        python3 workspace/skills/poly-research/scripts/research.py "$@"
        ;;
    simulator)
        python3 workspace/skills/poly-simulator/scripts/simulator.py "$@"
        ;;
    backtest)
        python3 workspace/skills/poly-simulator/scripts/backtester.py "$@"
        ;;
    *)
        echo "Usage: ./run.sh {scanner|whale|watchlist|research|simulator|backtest} [args...]"
        echo ""
        echo "Examples:"
        echo "  # Standard scanner"
        echo "  ./run.sh scanner --mode quick --top 5"
        echo "  ./run.sh scanner --mode deep"
        echo "  ./run.sh scanner --mode new"
        echo ""
        echo "  # Arbitrage scanner (YES + NO != \$1.00)"
        echo "  ./run.sh scanner --arb"
        echo "  ./run.sh scanner --arb --threshold 0.03 --top 20"
        echo ""
        echo "  # Bonding scanner (near-certain outcomes >95%)"
        echo "  ./run.sh scanner --bonds"
        echo "  ./run.sh scanner --bonds --min-prob 0.97 --max-days 30"
        echo ""
        echo "  # Other tools"
        echo "  ./run.sh whale --leaderboard"
        echo "  ./run.sh research --market-id 0x... --mode deep"
        echo "  ./run.sh simulator --start --strategy edge_hunter"
        echo "  ./run.sh backtest --strategy edge_hunter --data sample --verbose"
        echo "  ./run.sh watchlist --status"
        exit 1
        ;;
esac
