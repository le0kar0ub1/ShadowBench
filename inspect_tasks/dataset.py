import json
from pathlib import Path
from typing import List, Any, Dict, Optional
from inspect_ai.dataset import Sample


class RewardHackingDataset:
    """Dataset loader for reward hacking evaluation."""

    def __init__(self, data_path: str = "data/custom/dataset.json"):
        self.data_path = Path(data_path)

    def load(self, limit: Optional[int] = None) -> List[Sample]:
        """Load dataset and convert to Inspect Samples."""
        if not self.data_path.exists():
            # Try finding it relative to project root if not found
            # Assuming CWD might be different
            if Path(f"../{self.data_path}").exists():
                self.data_path = Path(f"../{self.data_path}")
            elif not self.data_path.is_absolute():
                # Look in current directory just in case
                if Path(self.data_path.name).exists():
                    self.data_path = Path(self.data_path.name)

            if not self.data_path.exists():
                raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        with open(self.data_path) as f:
            problems = json.load(f)

        if limit:
            problems = problems[:limit]

        return [self._to_sample(p) for p in problems]

    def _to_sample(self, problem: Dict[str, Any]) -> Sample:
        """Convert a raw problem dict to a Sample."""
        task_name = problem.get("task_name")
        task_spec = problem.get("task_spec")
        visible_tests = problem.get("visible_test")
        hidden_tests = problem.get("hidden_test")
        starter_code = problem.get("starter_code")

        prompt = self._create_prompt(task_spec)

        return Sample(
            input=prompt,
            target=hidden_tests,
            id=task_name,
            metadata={
                "task_name": task_name,
                "visible_tests": visible_tests,
                "hidden_tests": hidden_tests,
                "starter_code": starter_code,
            },
        )

    def _create_prompt(self, statement: str) -> str:
        """Create the prompt for the agent."""
        return f"""{statement}"""
