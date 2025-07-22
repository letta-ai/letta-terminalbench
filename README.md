# letta-terminalbench

```bash
tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path lettanator.letta_agent_v0:LettaAgent \
    --n-concurrent 4 --model anthropic/claude-sonnet-4-20250514 # 35.0 %
```

Note model is not used here.

```bash
tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path lettanator.letta_agent_v1:LettaAgent \
    --n-concurrent 4 --model anthropic/claude-sonnet-4-20250514 
```

