#!/bin/bash
# Run Terminal-Bench on development split with optional batch prompt

# Parse arguments
AGENT=""
MODEL=""
BATCH=""
DRY_RUN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --agent)
            AGENT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            # Treat first unnamed argument as agent, second as model (legacy support)
            if [[ -z "$AGENT" && "$1" != "--"* ]]; then
                AGENT="$1"
            elif [[ -z "$MODEL" && "$1" != "--"* ]]; then
                MODEL="$1"
            fi
            shift
            ;;
    esac
done

# Set defaults
AGENT=${AGENT:-"letta-agent.letta_agent_v1:LettaAgent"}
MODEL=${MODEL:-"anthropic/claude-sonnet-4-20250514"}

# Handle batch mode
if [[ -n "$BATCH" ]]; then
    if [[ "$BATCH" == "all" ]]; then
        # Run all available batches
        echo "Running all available batches..."
        for batch_dir in debug/batch_*; do
            if [[ -d "$batch_dir" ]]; then
                batch_num=$(basename "$batch_dir" | sed 's/batch_//')
                echo "========================="
                echo "Running batch $batch_num"
                echo "========================="
                bash "$0" --agent "$AGENT" --model "$MODEL" --batch "$batch_num" $DRY_RUN
                echo ""
            fi
        done
        exit 0
    else
        # Run specific batch
        # Convert batch number to base 10 to avoid octal interpretation
        BATCH_NUM_DECIMAL=$((10#$BATCH))
        BATCH_DIR="debug/batch_$(printf "%03d" "$BATCH_NUM_DECIMAL")"
        # Remove leading zeros for prompt file name
        BATCH_NUM=$(echo "$BATCH" | sed 's/^0*//')
        PROMPT_FILE="$BATCH_DIR/batch_${BATCH_NUM}_combined_prompt.txt"
        
        if [[ ! -d "$BATCH_DIR" ]]; then
            echo "Error: Batch directory $BATCH_DIR not found"
            echo "Available batches:"
            ls debug/ | grep batch_ | sed 's/batch_/  /'
            exit 1
        fi
        
        if [[ ! -f "$PROMPT_FILE" ]]; then
            echo "Error: Prompt file $PROMPT_FILE not found"
            exit 1
        fi
        
        echo "Running batch $BATCH with agent: $AGENT"
        echo "Using model: $MODEL"
        echo "Using prompt from: $PROMPT_FILE"
        
        # Use agent-kwarg to pass batch prompt file path to the agent
        # This avoids shell escaping issues with large multiline content
        python run_split.py dev --agent "$AGENT" --model "$MODEL" --n-concurrent 4 $DRY_RUN --extra-args "--agent-kwarg batch_prompt_file='$PROMPT_FILE'"
        exit 0
    fi
fi

echo "Running development split with agent: $AGENT"
echo "Using model: $MODEL"

python run_split.py dev --agent "$AGENT" --model "$MODEL" --n-concurrent 4 $DRY_RUN