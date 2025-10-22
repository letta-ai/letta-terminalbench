# letta-terminalbench
Official code for Letta on [Terminal-Bench](https://www.tbench.ai/news/announcement). 

Check out our technical blog for more information: [Blog](https://www.letta.com/blog/terminal-bench)

## Available Agents

### LettaAgent (Basic)
The standard Letta terminal assistant with basic task execution capabilities.

```bash
tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path letta-agent.letta_agent_v1:LettaAgent \
    --n-concurrent 4 --model anthropic/claude-sonnet-4-20250514 
```

### LettaMetaAgent V1 (Task Management & Learning)
An enhanced version that can learn from previous tasks and apply that knowledge to new situations. Features include:

- **Task Switching**: Switch between different task contexts when needed
- **Learning Updates**: Capture and store new insights during task execution
- **High-Level Summaries**: Maintain comprehensive summaries of all tasks and learnings
- **Persistent Memory**: Learning survives across sessions and task executions

```bash
tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path letta-agent.letta_agent_meta_v1:LettaMetaAgent \
    --n-concurrent 4 --model anthropic/claude-sonnet-4-20250514 
```

For detailed information about the v1 meta-learning capabilities, see [META_LEARNING_README.md](letta-agent/META_LEARNING_README.md).

### LettaMetaAgent V2 (System Prompt Optimization)
A completely redesigned meta agent focused on learning optimal system prompts through parallel task execution and ground truth comparison. Features include:

- **Parallel Task Execution**: Runs K tasks in parallel to test different system prompts
- **System Prompt Variation**: Automatically generates multiple prompt variants for testing
- **Ground Truth Comparison**: Compares execution results with expected outcomes
- **Meta-Learning Analysis**: Identifies what makes certain prompts more successful
- **Final Optimized Output**: Produces one optimized system prompt incorporating all learnings
- **Terminal Benchmark Compatible**: Can execute individual tasks while collecting meta-learning data

```bash
tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path letta-agent.letta_agent_meta_v2:LettaMetaAgentV2 \
    --n-concurrent 4 --model anthropic/claude-sonnet-4-20250514
```

For detailed information about the v2 system prompt optimization capabilities, see [META_AGENT_V2_README.md](letta-agent/META_AGENT_V2_README.md).

## Train/Dev/Test Data Splits

This repository now supports splitting the Terminal-Bench dataset into train/dev/test sets for proper evaluation. The splitting is done deterministically with a fixed seed to ensure reproducibility.

### Creating Data Splits

Generate train/dev/test splits (70/15/15 by default):

```bash
python data_split.py --generate-commands
```

This creates a `splits/` directory with:
- `train.json`: 56 tasks for training/development
- `dev.json`: 12 tasks for validation/hyperparameter tuning  
- `test.json`: 12 tasks for final evaluation
- `metadata.json`: Split metadata and statistics

### Running on Specific Splits

Use the convenience scripts to run on specific data splits:

```bash
# Run on training split
./run_train.sh letta-agent.letta_agent_v1:LettaAgent

# Run on development split  
./run_dev.sh letta-agent.letta_agent_meta_v1:LettaMetaAgent

# Run on test split
./run_test.sh letta-agent.letta_agent_meta_v2:LettaMetaAgentV2
```

Or use the Python script directly:

```bash
# Run specific agent on dev split
python run_split.py dev --agent letta-agent.letta_agent_v1:LettaAgent

# Run with custom parameters
python run_split.py test --agent letta-agent.letta_agent_meta_v1:LettaMetaAgent \
    --model anthropic/claude-3-haiku-20240307 --n-concurrent 2
```

### Split Configuration

- **Training Set** (56 tasks): Use for agent development, prompt engineering, and iterative improvements
- **Development Set** (12 tasks): Use for hyperparameter tuning, model selection, and validation during development
- **Test Set** (12 tasks): Use only for final evaluation to avoid overfitting

The splits are created with `seed=42` for reproducibility and stratified to maintain task diversity across splits.

## Agent Comparison

| Feature | Basic Agent | MetaAgent V1 | MetaAgent V2 |
|---------|-------------|--------------|--------------|
| **Purpose** | Task execution | Task management & learning | System prompt optimization |
| **Execution** | Single task | Sequential with switching | Parallel with variants |
| **Learning** | None | Task-specific learnings | System prompt learnings |
| **Output** | Task completion | Task summaries | Optimized system prompt |
| **Focus** | Command execution | Task workflow | Prompt engineering |

## Example Usage

### Basic Agent
```bash
cd letta-agent
python example_usage.py
```

### MetaAgent V1 (Task Management)
```bash
cd letta-agent
python example_meta_usage.py
```

### MetaAgent V2 (System Prompt Optimization)
```bash
cd letta-agent
python example_meta_v2_usage.py
```

## Model Support

All agents support `--model anthropic/*`. However, one can modify `LlmConfig` to use other models.

## Key Features

- **Terminal Interaction**: Direct terminal command execution
- **Memory Management**: Persistent memory blocks for task context
- **Error Handling**: Robust error recovery and process management
- **Meta-Learning V1**: Continuous improvement through task experience
- **Meta-Learning V2**: System prompt optimization through parallel evaluation
- **State Persistence**: Learning state saved between sessions
- **Parallel Execution**: Efficient batch processing of multiple tasks (V2 only)
