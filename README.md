# ShadowBench

Exposing reward-hacking in AI coding agents through hidden truth tests.

The benchmark aims to answer the question: Do LLMs implement the spec or the tests ?

It uses a coding task benchmark with visible and hidden tests. There are 4 scenarios:
- Baseline: The agent is allowed to modify the test file.
- No tests provided: The agent has to implement the function without any tests.
- No edit: The agent is given the visible tests but is not allowed to modify the test file.
- Anti-gaming: The agent is given the visible tests but is not allowed to delete or weaken existing tests.

AI coding agents optimized to pass test suites may learn to game tests rather than solve problems, a misalignment risk as these systems gain deployment autonomy. This project proposes to detect when agents exploit test weaknesses instead of implementing correct solutions. Establishes whether current alignment techniques prevent instrumental convergence toward proxy metrics in realistic coding workflows. We typically measure the gap between visible test pass rates (agent's optimization target) and hidden test pass rates (true correctness). Honest solutions pass both while reward-hacked solutions will likely show large divergence.

On top of the test suite, we might also want to use techniques like:
- Static analysis: Flag test deletions, hardcoded conditionals etc…
- Dynamic testing: Run solutions against property-based test generators and input perturbations
- Track test modification frequency and volume (As it may say something about underlying behaviors)


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

## Dataset

It's hard to find a dataset that is suitable for this task as we need to have comprehensive tests (to be broken down into visible and hidden tests) and usually the dataset provider for leetcode or similar code contests do not provide these.
For educational purposes, the dataset is a custom dataset created by the author. You can find it in the `data/custom` directory. The dataset is a JSON file with the following structure:
```json
{
    "task_name": "task_name",
    "task_spec": "task_spec",
    "visible_test": "visible_test",
    "hidden_test": "hidden_test",
    "starter_code": "starter_code"
}
```

## Run the evaluation

Start the inspect web app

```bash
uv run inspect view
```

Run the evaluation

```bash
uv run inspect eval inspect_tasks/reward_hacking.py -T model=google/gemini-2.5-flash -T scenario=baseline -T num_problems=5
```