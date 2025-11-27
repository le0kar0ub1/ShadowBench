# ============================================================================
# FILE: analysis/analyze_results.py
# Result analysis and visualization
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json


def analyze_evaluation_results(log_dir: str):
    """Analyze results across scenarios."""

    log_path = Path(log_dir)

    # Collect results from all scenarios
    all_results = []

    for scenario_dir in log_path.iterdir():
        if scenario_dir.is_dir():
            log_file = scenario_dir / "logs.json"
            if log_file.exists():
                with open(log_file) as f:
                    for line in f:
                        result = json.loads(line)
                        result["scenario"] = scenario_dir.name
                        all_results.append(result)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Calculate key statistics
    stats = (
        df.groupby("scenario")
        .agg(
            {
                "visible_pass_rate": "mean",
                "hidden_pass_rate": "mean",
                "divergence": "mean",
                "hacking_score": "mean",
                "test_modification": "count",
            }
        )
        .round(3)
    )

    print("=== Reward Hacking Analysis ===\n")
    print(stats)
    print()

    # Identify clear hacking cases
    hacking_cases = df[df["likely_hacking"] == True]
    print(f"Clear hacking cases: {len(hacking_cases)} / {len(df)}")
    print(f"Hacking rate by scenario:")
    print(df.groupby("scenario")["likely_hacking"].mean())
    print()

    # Visualizations
    create_visualizations(df, "results/")

    return df, stats


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Generate analysis plots."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 1. Divergence by scenario
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="scenario", y="divergence")
    plt.title("Test Pass Rate Divergence by Scenario")
    plt.ylabel("Divergence (Visible % - Hidden %)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / "divergence_by_scenario.png")
    plt.close()

    # 2. Scatter: Visible vs Hidden pass rates
    plt.figure(figsize=(10, 8))
    for scenario in df["scenario"].unique():
        subset = df[df["scenario"] == scenario]
        plt.scatter(
            subset["visible_pass_rate"],
            subset["hidden_pass_rate"],
            label=scenario,
            alpha=0.6,
            s=100,
        )
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect alignment")
    plt.xlabel("Visible Test Pass Rate")
    plt.ylabel("Hidden Test Pass Rate")
    plt.title("Test Performance Alignment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / "pass_rate_alignment.png")
    plt.close()

    # 3. Hacking indicators
    plt.figure(figsize=(12, 6))
    hacking_metrics = [
        "test_deletions",
        "test_weakenings",
        "suspicious_conditionals",
        "hardcoded_outputs",
    ]
    df_hacking = df[hacking_metrics + ["scenario"]].melt(
        id_vars=["scenario"], var_name="indicator", value_name="count"
    )
    sns.barplot(data=df_hacking, x="indicator", y="count", hue="scenario")
    plt.title("Hacking Indicators by Scenario")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path / "hacking_indicators.png")
    plt.close()

    print(f"Visualizations saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="logs/", help="Log directory")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    df, stats = analyze_evaluation_results(args.input)

    # Export results
    stats.to_csv(f"{args.output}/summary_statistics.csv")
    df.to_csv(f"{args.output}/detailed_results.csv", index=False)
