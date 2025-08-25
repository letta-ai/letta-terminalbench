#!/usr/bin/env python3
"""
Convenience script for running Terminal-Bench on train/dev/test splits.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from data_split import TerminalBenchSplitter


def run_tb_command(cmd: str, dry_run: bool = False):
    """Execute a terminal-bench command."""
    if dry_run:
        print("DRY RUN - would execute:")
        print(cmd)
        return
        
    print(f"Executing: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"Command completed with return code: {result.returncode}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code: {e.returncode}")
        sys.exit(e.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run Terminal-Bench on specific data splits")
    parser.add_argument("split", choices=["train", "dev", "test"], 
                       help="Data split to run on")
    parser.add_argument("--agent", required=True, 
                       help="Agent import path (e.g., letta-agent.letta_agent_v1:LettaAgent)")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-20250514",
                       help="Model to use")
    parser.add_argument("--n-concurrent", type=int, default=4,
                       help="Number of concurrent trials")
    parser.add_argument("--splits-dir", default="splits",
                       help="Directory containing split files")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show command without executing")
    parser.add_argument("--extra-args", default="",
                       help="Additional arguments to pass to tb run")
    
    args = parser.parse_args()
    
    # Check if splits exist
    splits_dir = Path(args.splits_dir)
    if not splits_dir.exists():
        print(f"Error: Splits directory {splits_dir} not found.")
        print("Run 'python data_split.py' first to create splits.")
        sys.exit(1)
        
    split_file = splits_dir / f"{args.split}.json"
    if not split_file.exists():
        print(f"Error: Split file {split_file} not found.")
        print("Run 'python data_split.py' first to create splits.")
        sys.exit(1)
    
    # Generate and execute command
    splitter = TerminalBenchSplitter()
    cmd = splitter.get_tb_command(
        args.split, 
        args.agent, 
        args.model, 
        args.n_concurrent, 
        args.splits_dir
    )
    
    if args.extra_args:
        cmd += f" {args.extra_args}"
    
    print(f"Running {args.split} split with {args.agent}")
    run_tb_command(cmd, args.dry_run)


if __name__ == "__main__":
    main()