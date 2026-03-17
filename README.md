# letta-terminalbench
Official code for Letta on [Terminal-Bench](https://www.tbench.ai/news/announcement). 

Check out our technical blog for more information: [Blog](https://www.letta.com/blog/terminal-bench)

## Quick Start
First, install the dependencies:
```bash
pip install -r requirements.txt
```

Second, create a `.env` file and add your Anthropic API key:
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
```

Then, start the Letta server with Docker:
```bash
docker run \
  -v ~/.letta/.persist/pgdata:/var/lib/postgresql/data \
  -p 8283:8283 \
  --env-file .env \
  letta/letta:latest
```
More detailed instructions for running the Letta server are available [here](https://docs.letta.com/guides/docker/).


Finally, run the following command to start the evaluation:

```bash
tb run \
    --dataset-name terminal-bench-core \
    --dataset-version 0.1.1 \
    --agent-import-path letta-agent.letta_agent_v1:LettaAgent \
    --n-concurrent 4 --model anthropic/claude-sonnet-4-20250514 

# note this script supports --model anthropic/*. However, one can modify LlmConfig to use other models.
```
