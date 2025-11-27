# ShadowBench

Exposing reward-hacking in AI coding agents through hidden truth tests.

The benchmark aims to answer the question: Do LLMs implement the spec or the tests ?

It uses a coding task benchmark with visible and hidden tests. There are 4 scenarios:
- Baseline: The agent is allowed to modify the test file.
- No tests provided: The agent has to implement the function without any tests.
- No edit: The agent is given the visible tests but is not allowed to modify the test file.
- Anti-gaming: The agent is given the visible tests but is not allowed to delete or weaken existing tests.

## Installation

```bash
uv sync
```

Build the sandbox container

```bash
docker build -f Dockerfile.sandbox -t reward-hacking-sandbox .
```

Environment variables setup

```bash
cat << EOF > .env
OPENAI_API_KEY=your-api-key
ANTHROPIC_API_KEY=your-api-key
GOOGLE_API_KEY=your-api-key

INSPECT_LOG_DIR=./logs
INSPECT_LOG_LEVEL=info

export INSPECT_MODEL=google/gemini-2.5-flash
export INSPECT_SCENARIO=baseline
export INSPECT_MAX_MESSAGES=5
EOF
```

## Run the evaluation

Start the inspect web app

```bash
uv run inspect view
```

Run the evaluation

```bash
uv run inspect eval inspect_tasks/reward_hacking.py
```