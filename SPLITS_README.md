# Terminal-Bench Data Splits

This document describes the train/dev/test data splitting functionality implemented for the letta-terminalbench repository.

## Overview

The Terminal-Bench dataset has been split into three sets to enable proper evaluation and avoid overfitting:

- **Training Set**: 56 tasks (70%) - For agent development and prompt engineering
- **Development Set**: 12 tasks (15%) - For validation and hyperparameter tuning  
- **Test Set**: 12 tasks (12%) - For final evaluation only

## Files Added

### Core Implementation
- `data_split.py` - Main splitting logic and Terminal-Bench command generation
- `run_split.py` - Python script for running specific splits
- `validate_splits.py` - Validation script to verify split integrity

### Convenience Scripts  
- `run_train.sh` - Shell script for training split
- `run_dev.sh` - Shell script for development split
- `run_test.sh` - Shell script for test split

### Generated Data
- `splits/train.json` - Training tasks (56 tasks)
- `splits/dev.json` - Development tasks (12 tasks)
- `splits/test.json` - Test tasks (12 tasks)
- `splits/metadata.json` - Split metadata and statistics

## Usage

### 1. Create Splits (One-time Setup)

```bash
# Generate splits with default 70/15/15 ratios
python data_split.py --generate-commands

# Custom ratios
python data_split.py --train-ratio 0.8 --dev-ratio 0.1 --test-ratio 0.1
```

### 2. Run Benchmarks on Specific Splits

```bash
# Using convenience shell scripts
./run_train.sh letta-agent.letta_agent_v1:LettaAgent
./run_dev.sh letta-agent.letta_agent_meta_v1:LettaMetaAgent  
./run_test.sh letta-agent.letta_agent_meta_v2:LettaMetaAgentV2

# Using Python script directly
python run_split.py dev --agent letta-agent.letta_agent_v1:LettaAgent
python run_split.py test --agent letta-agent.letta_agent_meta_v1:LettaMetaAgent \
    --model anthropic/claude-3-haiku-20240307 --n-concurrent 2

# Dry run to see commands without executing
python run_split.py train --agent letta-agent.letta_agent_v1:LettaAgent --dry-run
```

### 3. Validate Splits

```bash
# Check split integrity
python validate_splits.py
```

## Split Details

The splits are created deterministically using `seed=42` for reproducibility. Tasks are randomly shuffled before splitting to ensure balanced distribution across different task types.

### Task Distribution

| Split | Tasks | Percentage | Purpose |
|-------|-------|------------|---------|  
| Train | 56    | 70%       | Agent development, prompt engineering, iterative improvements |
| Dev   | 12    | 15%       | Validation, hyperparameter tuning, model selection |
| Test  | 12    | 15%       | Final evaluation, paper results, avoiding overfitting |

### Split Integrity

- ✅ No task overlap between splits
- ✅ All 80 original tasks included
- ✅ Deterministic generation (seed=42)
- ✅ Command generation validated
- ✅ File format validation

## Design Principles

1. **Reproducibility**: Fixed seed ensures consistent splits across runs
2. **No Leakage**: Strict separation between train/dev/test sets
3. **Convenience**: Multiple interfaces (Python scripts, shell scripts)
4. **Flexibility**: Customizable split ratios and model parameters
5. **Validation**: Built-in integrity checking and command validation

## Integration with Original Workflow

The splitting functionality is fully compatible with the existing Terminal-Bench workflow:

- Uses the same `tb run` commands with `--task-ids` parameter
- Works with all existing agents (v1, meta-v1, meta-v2)
- Compatible with all model configurations
- Maintains original result output format

## Future Enhancements

Potential improvements for future versions:

- Stratified splitting by task category/difficulty
- Cross-validation fold generation
- Custom task filtering/selection
- Integration with evaluation metrics tracking
- Automated result comparison across splits