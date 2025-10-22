#!/usr/bin/env python3
"""
Data splitting utility for Terminal-Bench tasks.

This script splits the terminal-bench-core dataset tasks into train/dev/test sets
and provides utilities to run benchmarks on specific splits.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import argparse


class TerminalBenchSplitter:
    def __init__(self, seed: int = 42):
        """Initialize the data splitter with a random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        
        # Full list of tasks from terminal-bench-core v0.1.1
        # This is based on the task_ids from the run metadata
        self.all_tasks = [
            "super-benchmark-upet",
            "build-linux-kernel-qemu", 
            "play-zork",
            "eval-mteb",
            "eval-mteb.hard",
            "conda-env-conflict-resolution",
            "swe-bench-astropy-2",
            "swe-bench-fsspec",
            "run-pdp11-code",
            "swe-bench-astropy-1",
            "blind-maze-explorer-algorithm.easy",
            "blind-maze-explorer-5x5",
            "blind-maze-explorer-algorithm.hard",
            "blind-maze-explorer-algorithm",
            "train-fasttext",
            "incompatible-python-fasttext.base_with_hint",
            "incompatible-python-fasttext",
            "grid-pattern-transform",
            "count-dataset-tokens",
            "qemu-startup",
            "qemu-alpine-ssh",
            "download-youtube",
            "git-multibranch",
            "pytorch-model-cli.hard",
            "pytorch-model-cli.easy",
            "pytorch-model-cli",
            "cron-broken-network",
            "crack-7z-hash.easy",
            "path-tracing",
            "get-bitcoin-nodes",
            "build-initramfs-qemu",
            "polyglot-rust-c",
            "hf-model-inference",
            "raman-fitting",
            "cartpole-rl-training",
            "heterogeneous-dates",
            "fix-pandas-version",
            "reshard-c4-data",
            "intrusion-detection",
            "tmux-advanced-workflow",
            "prove-plus-comm",
            "swe-bench-langcodes",
            "simple-web-scraper",
            "decommissioning-service-with-sensitive-data",
            "solana-data",
            "oom",
            "build-tcc-qemu",
            "jupyter-notebook-server",
            "configure-git-webserver",
            "security-vulhub-minio",
            "sanitize-git-repo.hard",
            "openssl-selfsigned-cert",
            "git-workflow-hack",
            "write-compressor",
            "create-bucket",
            "simple-sheets-put",
            "polyglot-c-py",
            "sanitize-git-repo",
            "raman-fitting.easy",
            "nginx-request-logging",
            "sqlite-db-truncate",
            "new-encrypt-command",
            "modernize-fortran-build",
            "fix-permissions",
            "processing-pipeline",
            "hello-world",
            "extract-moves-from-video",
            "crack-7z-hash.hard",
            "organization-json-generator",
            "crack-7z-hash",
            "gpt2-codegolf",
            "csv-to-parquet",
            "sqlite-with-gcov",
            "chess-best-move",
            "password-recovery",
            "fibonacci-server",
            "fix-git",
            "path-tracing-reverse",
            "extract-safely",
            "vim-terminal-task"
        ]
        
    def create_splits(self, train_ratio: float = 0.7, dev_ratio: float = 0.15, 
                      test_ratio: float = 0.15) -> Dict[str, List[str]]:
        """
        Create train/dev/test splits from the full task list.
        
        Args:
            train_ratio: Proportion of tasks for training set
            dev_ratio: Proportion of tasks for development set  
            test_ratio: Proportion of tasks for test set
            
        Returns:
            Dictionary with 'train', 'dev', 'test' keys containing task lists
        """
        if abs((train_ratio + dev_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
            
        tasks = self.all_tasks.copy()
        random.shuffle(tasks)
        
        n_tasks = len(tasks)
        n_train = int(n_tasks * train_ratio)
        n_dev = int(n_tasks * dev_ratio)
        n_test = n_tasks - n_train - n_dev  # Ensure all tasks are included
        
        splits = {
            'train': tasks[:n_train],
            'dev': tasks[n_train:n_train + n_dev],
            'test': tasks[n_train + n_dev:n_train + n_dev + n_test]
        }
        
        return splits
        
    def save_splits(self, splits: Dict[str, List[str]], output_dir: str = "splits"):
        """Save splits to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for split_name, task_list in splits.items():
            split_file = output_path / f"{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump({
                    'tasks': task_list,
                    'count': len(task_list),
                    'seed': self.seed,
                    'split': split_name
                }, f, indent=2)
            print(f"Saved {len(task_list)} tasks to {split_file}")
            
        # Also save a combined metadata file
        metadata = {
            'seed': self.seed,
            'total_tasks': len(self.all_tasks),
            'splits': {name: len(tasks) for name, tasks in splits.items()},
            'created_at': Path(__file__).stat().st_mtime
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved split metadata to {output_path / 'metadata.json'}")
        
    def load_split(self, split_name: str, splits_dir: str = "splits") -> List[str]:
        """Load a specific split from saved files."""
        split_file = Path(splits_dir) / f"{split_name}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file {split_file} not found")
            
        with open(split_file, 'r') as f:
            data = json.load(f)
            return data['tasks']
            
    def get_tb_command(self, split_name: str, agent_path: str, 
                      model: str = "anthropic/claude-sonnet-4-20250514",
                      n_concurrent: int = 4, splits_dir: str = "splits") -> str:
        """Generate terminal-bench command for a specific split."""
        try:
            tasks = self.load_split(split_name, splits_dir)
        except FileNotFoundError:
            return f"Error: Split '{split_name}' not found. Run data_split.py first."
            
        # Create multiple --task-id flags for tb command
        task_flags = " \\\n".join([f"    --task-id {task}" for task in tasks])
        
        cmd = (
            f"tb run \\\n"
            f"    --dataset-name terminal-bench-core \\\n" 
            f"    --dataset-version 0.1.1 \\\n"
            f"    --agent-import-path {agent_path} \\\n"
            f"    --n-concurrent {n_concurrent} \\\n"
            f"    --model {model} \\\n"
            f"{task_flags}"
        )
        
        return cmd


def main():
    parser = argparse.ArgumentParser(description="Split Terminal-Bench tasks into train/dev/test sets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Training set ratio")
    parser.add_argument("--dev-ratio", type=float, default=0.15, help="Development set ratio") 
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test set ratio")
    parser.add_argument("--output-dir", default="splits", help="Output directory for split files")
    parser.add_argument("--generate-commands", action="store_true", 
                       help="Generate sample tb commands for each split")
    
    args = parser.parse_args()
    
    splitter = TerminalBenchSplitter(seed=args.seed)
    
    print(f"Creating train/dev/test splits with ratios {args.train_ratio}/{args.dev_ratio}/{args.test_ratio}")
    print(f"Using seed: {args.seed}")
    print(f"Total tasks: {len(splitter.all_tasks)}")
    
    splits = splitter.create_splits(args.train_ratio, args.dev_ratio, args.test_ratio)
    
    print("\nSplit sizes:")
    for split_name, tasks in splits.items():
        print(f"  {split_name}: {len(tasks)} tasks")
    
    splitter.save_splits(splits, args.output_dir)
    
    if args.generate_commands:
        print("\n" + "="*60)
        print("Sample Terminal-Bench commands for each split:")
        print("="*60)
        
        agents = [
            "letta-agent.letta_agent_v1:LettaAgent",
            "letta-agent.letta_agent_meta_v1:LettaMetaAgent", 
            "letta-agent.letta_agent_meta_v2:LettaMetaAgentV2"
        ]
        
        for split_name in ['train', 'dev', 'test']:
            print(f"\n### {split_name.upper()} SET ###")
            cmd = splitter.get_tb_command(split_name, agents[0], splits_dir=args.output_dir)
            print(cmd)


if __name__ == "__main__":
    main()