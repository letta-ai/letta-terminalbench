# letta-terminalbench
Official code for Letta on [Terminal-Bench](https://www.tbench.ai/news/announcement). 

Check out our technical blog for more information: [Blog](https://www.letta.com/blog/terminal-bench)

```bash
tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path letta-agent.letta_agent_v1:LettaAgent \
    --n-concurrent 4 --model anthropic/claude-sonnet-4-20250514 
```
