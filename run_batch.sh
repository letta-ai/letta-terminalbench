#!/bin/bash
# Convenience script for running specific batches

usage() {
    echo "Usage: $0 [batch_number|all] [options]"
    echo ""
    echo "Examples:"
    echo "  $0 1                    # Run batch 1"
    echo "  $0 001                  # Run batch 1 (with zero padding)"
    echo "  $0 all                  # Run all available batches"
    echo "  $0 1 --dry-run          # Show what would run for batch 1"
    echo "  $0 1 --agent my-agent   # Run batch 1 with custom agent"
    echo ""
    echo "Available batches:"
    ls debug/ | grep batch_ | sed 's/batch_/  /' | sort -n
    echo ""
    echo "Options:"
    echo "  --agent AGENT          Agent to use (default: letta-agent.letta_agent_v1:LettaAgent)"
    echo "  --model MODEL          Model to use (default: anthropic/claude-sonnet-4-20250514)"
    echo "  --dry-run              Show what would be executed without running"
    echo "  --help                 Show this help message"
}

# Check if no arguments provided
if [[ $# -eq 0 ]]; then
    usage
    exit 1
fi

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi

# Extract batch number from first argument
BATCH="$1"
shift

# Pass remaining arguments to run_dev.sh
exec bash run_dev.sh --batch "$BATCH" "$@"