# ShadowBench

Exposing reward-hacking in AI coding agents through hidden truth tests.

## Installation

```bash
uv sync
```

install SWE benchmark

```bash
git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench
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