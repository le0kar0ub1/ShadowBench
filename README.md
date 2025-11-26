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

```bash
docker build -f Dockerfile.sandbox -t reward-hacking-sandbox .
```

## Run the evaluation

```bash
uv run inspect eval inspect_tasks/reward_hacking.py -T model=google/gemini-2.5-flash -T scenario=baseline -T max_messages=1
```