#!/usr/bin/env python3
"""
Validate the train/dev/test splits for Terminal-Bench.
"""

import json
from pathlib import Path
from data_split import TerminalBenchSplitter


def validate_splits():
    """Validate the integrity of the train/dev/test splits."""
    splits_dir = Path("splits")
    
    if not splits_dir.exists():
        print("❌ Splits directory not found. Run 'python data_split.py' first.")
        return False
        
    # Load splits
    try:
        with open(splits_dir / "train.json") as f:
            train_data = json.load(f)
        with open(splits_dir / "dev.json") as f:
            dev_data = json.load(f)  
        with open(splits_dir / "test.json") as f:
            test_data = json.load(f)
        with open(splits_dir / "metadata.json") as f:
            metadata = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Missing split file: {e}")
        return False
        
    train_tasks = train_data['tasks']
    dev_tasks = dev_data['tasks']
    test_tasks = test_data['tasks']
    
    # Validate counts
    print("📊 Split Statistics:")
    print(f"   Train: {len(train_tasks)} tasks")
    print(f"   Dev: {len(dev_tasks)} tasks")
    print(f"   Test: {len(test_tasks)} tasks")
    print(f"   Total: {len(train_tasks) + len(dev_tasks) + len(test_tasks)} tasks")
    
    # Check metadata consistency
    expected_counts = metadata['splits']
    if (len(train_tasks) != expected_counts['train'] or
        len(dev_tasks) != expected_counts['dev'] or  
        len(test_tasks) != expected_counts['test']):
        print("❌ Task counts don't match metadata")
        return False
    print("✅ Task counts match metadata")
    
    # Check for overlaps between splits
    all_tasks = train_tasks + dev_tasks + test_tasks
    unique_tasks = set(all_tasks)
    
    if len(unique_tasks) != len(all_tasks):
        print("❌ Found overlapping tasks between splits")
        return False
    print("✅ No overlapping tasks between splits")
    
    # Verify against full task list
    splitter = TerminalBenchSplitter(seed=metadata['seed'])
    if unique_tasks != set(splitter.all_tasks):
        missing = set(splitter.all_tasks) - unique_tasks
        extra = unique_tasks - set(splitter.all_tasks)
        if missing:
            print(f"❌ Missing tasks: {missing}")
        if extra:
            print(f"❌ Extra tasks: {extra}")
        return False
    print("✅ All tasks from original dataset included")
    
    # Test command generation
    print("\n🧪 Testing command generation:")
    for split_name in ['train', 'dev', 'test']:
        cmd = splitter.get_tb_command(split_name, "test-agent", splits_dir="splits")
        if "Error:" in cmd:
            print(f"❌ Command generation failed for {split_name}: {cmd}")
            return False
        print(f"✅ {split_name} command generation works")
    
    print(f"\n🎉 All validations passed! Splits created with seed={metadata['seed']}")
    return True


if __name__ == "__main__":
    validate_splits()