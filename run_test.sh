#!/bin/bash
# Run Terminal-Bench on test split

AGENT=${1:-"letta-agent.letta_agent_v1:LettaAgent"}
MODEL=${2:-"anthropic/claude-sonnet-4-20250514"}

# Check for --dry-run flag
if [[ "$*" == *"--dry-run"* ]]; then
    DRY_RUN="--dry-run"
else
    DRY_RUN=""
fi

echo "Running test split with agent: $AGENT"
echo "Using model: $MODEL"

python run_split.py test --agent "$AGENT" --model "$MODEL" --n-concurrent 4 $DRY_RUN