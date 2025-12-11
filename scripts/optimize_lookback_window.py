#!/usr/bin/env python3
"""
Optimize lookback window for Markov model.
Tests different train/test window configurations to find optimal setup.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation import run_cross_validation
from models import MarkovModel
from utils import load_config, load_station_info, load_trip_data, prepare_data


def run_experiment(config, train_weeks, test_weeks, increment_days=7):
    """Run cross-validation with specific train/test window configuration."""

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: Train {train_weeks} weeks, Test {test_weeks} weeks")
    print("=" * 70)

    # Update config
    config["cross_validation"]["train_weeks"] = train_weeks
    config["cross_validation"]["test_weeks"] = test_weeks
    config["cross_validation"]["increment_days"] = increment_days

    # Load data
    print("\nLoading data...")
    trips = load_trip_data(
        config["data"]["trip_data_dir"],
        start_date=config["time"]["start_date"],
        end_date=config["time"]["end_date"],
        use_parquet=True,
    )

    stations = load_station_info(config["data"]["station_info_path"])
    trips, station_stats = prepare_data(trips, stations, config)

    # Initialize model
    model = MarkovModel(config)

    # Run cross-validation
    fold_results, summary = run_cross_validation(model, trips, station_stats, config, verbose=False)

    print(f"\nCompleted {len(fold_results)} folds")
    print(f"  Average MAE: {summary['inventory_mae'][0]:.2f} ± {summary['inventory_mae'][1]:.2f}")

    return fold_results, summary


def main():
    print("=" * 70)
    print("MARKOV MODEL: LOOKBACK WINDOW OPTIMIZATION")
    print("=" * 70)

    # Load base config
    config = load_config("config.yaml")

    # Use shorter date range for faster experiments (Sept-Oct only)
    config["time"]["start_date"] = "2025-09-01"
    config["time"]["end_date"] = "2025-10-31"

    # Define configurations to test
    # Format: (train_weeks, test_weeks, increment_days, label)
    configurations = [
        (1, 1, 7, "1w train, 1w test"),
        (2, 1, 7, "2w train, 1w test"),
        (3, 1, 7, "3w train, 1w test"),
        (4, 1, 7, "4w train, 1w test"),
        (2, 2, 7, "2w train, 2w test"),
        (3, 2, 7, "3w train, 2w test"),
        (4, 2, 7, "4w train, 2w test"),
    ]

    print(f"\nTesting {len(configurations)} configurations:")
    for i, (tw, test_w, inc, label) in enumerate(configurations, 1):
        print(f"  {i}. {label} (increment {inc} days)")

    # Run experiments
    results = []

    for train_w, test_w, inc_days, label in configurations:
        try:
            fold_results, summary = run_experiment(config.copy(), train_w, test_w, inc_days)

            results.append(
                {
                    "label": label,
                    "train_weeks": train_w,
                    "test_weeks": test_w,
                    "increment_days": inc_days,
                    "fold_results": fold_results,
                    "summary": summary,
                    "n_folds": len(fold_results),
                }
            )

        except Exception as e:
            print(f"\n✗ Failed: {label}")
            print(f"  Error: {e}")
            continue

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / "outputs" / f"lookback_optimization_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(
            {
                "configurations": results,
                "timestamp": timestamp,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\n✓ Saved results to: {output_file}")

    # Create visualizations
    create_plots(results, project_root / "outputs")

    # Print recommendations
    print_recommendations(results)


def create_plots(results, output_dir):
    """Create comparison plots with error bars."""

    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    # Set style
    sns.set_style("whitegrid")

    # Extract data
    labels = [r["label"] for r in results]

    metrics_data = {}
    for metric in [
        "inventory_mae",
        "inventory_rmse",
        "empty_recall",
        "full_recall",
        "state_accuracy",
    ]:
        means = [r["summary"][metric][0] for r in results]
        stds = [r["summary"][metric][1] for r in results]
        metrics_data[metric] = (means, stds)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(
        "Markov Model: Lookback Window Optimization\n(Error bars show ± 1 std across folds)",
        fontsize=16,
        fontweight="bold",
        y=1.00,
    )

    # 1. MAE comparison
    ax = axes[0, 0]
    means, stds = metrics_data["inventory_mae"]
    x = np.arange(len(labels))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(labels)))
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor="black", linewidth=1.5
    )

    ax.set_ylabel("MAE (bikes)", fontsize=12, fontweight="bold")
    ax.set_title("Mean Absolute Error\n(Lower is Better)", fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Highlight best
    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    # Add values on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + std,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # 2. RMSE comparison
    ax = axes[0, 1]
    means, stds = metrics_data["inventory_rmse"]
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor="black", linewidth=1.5
    )

    ax.set_ylabel("RMSE (bikes)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Root Mean Squared Error\n(Lower is Better)", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    best_idx = np.argmin(means)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    # 3. Empty Recall
    ax = axes[0, 2]
    means, stds = metrics_data["empty_recall"]
    means = [m * 100 for m in means]
    stds = [s * 100 for s in stds]
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor="black", linewidth=1.5
    )

    ax.set_ylabel("Empty Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Empty Station Detection\n(Higher is Better)", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 100])

    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    # 4. Full Recall
    ax = axes[1, 0]
    means, stds = metrics_data["full_recall"]
    means = [m * 100 for m in means]
    stds = [s * 100 for s in stds]
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor="black", linewidth=1.5
    )

    ax.set_ylabel("Full Recall (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Full Station Detection\n(Higher is Better)", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 100])

    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    # 5. State Accuracy
    ax = axes[1, 1]
    means, stds = metrics_data["state_accuracy"]
    means = [m * 100 for m in means]
    stds = [s * 100 for s in stds]
    bars = ax.bar(
        x, means, yerr=stds, capsize=5, alpha=0.8, color=colors, edgecolor="black", linewidth=1.5
    )

    ax.set_ylabel("State Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Overall State Classification\n(Higher is Better)", fontsize=13, fontweight="bold", pad=10
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 100])

    best_idx = np.argmax(means)
    bars[best_idx].set_edgecolor("red")
    bars[best_idx].set_linewidth(3)

    # 6. Trade-off plot: MAE vs Number of Folds
    ax = axes[1, 2]
    n_folds = [r["n_folds"] for r in results]
    mae_means, mae_stds = metrics_data["inventory_mae"]

    scatter = ax.scatter(
        n_folds,
        mae_means,
        s=200,
        c=range(len(labels)),
        cmap="viridis",
        alpha=0.7,
        edgecolors="black",
        linewidths=2,
    )
    ax.errorbar(
        n_folds,
        mae_means,
        yerr=mae_stds,
        fmt="none",
        ecolor="gray",
        capsize=5,
        alpha=0.5,
        linewidth=2,
    )

    # Annotate points
    for i, (nf, mae, label) in enumerate(zip(n_folds, mae_means, labels)):
        ax.annotate(
            label.split(",")[0], (nf, mae), fontsize=8, xytext=(5, 5), textcoords="offset points"
        )

    ax.set_xlabel("Number of Folds", fontsize=12, fontweight="bold")
    ax.set_ylabel("MAE (bikes)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Computation vs Accuracy Trade-off\n(More folds = more compute)",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / "lookback_window_optimization.png"
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    print(f"✓ Saved plot to: {output_file}")

    # Create detailed comparison plot
    fig2, ax = plt.subplots(figsize=(14, 12))

    # Normalize metrics to [0, 1] for comparison
    normalized_data = {}
    normalized_stds = {}
    for metric, (means, stds) in metrics_data.items():
        if "mae" in metric or "rmse" in metric:
            # Lower is better, so invert
            normalized = [
                (max(means) - m) / (max(means) - min(means)) if max(means) != min(means) else 0.5
                for m in means
            ]
        else:
            # Higher is better
            normalized = [
                (m - min(means)) / (max(means) - min(means)) if max(means) != min(means) else 0.5
                for m in means
            ]
        normalized_data[metric] = normalized

        # Normalize standard deviations (scale by same factor as range)
        if max(means) != min(means):
            normalized_stds[metric] = [s / (max(means) - min(means)) for s in stds]
        else:
            normalized_stds[metric] = [0 for _ in stds]

    # Create grouped bar chart
    metric_labels = ["MAE", "RMSE", "Empty\nRecall", "Full\nRecall", "State\nAccuracy"]
    x = np.arange(len(metric_labels))
    width = 0.12

    for i, (result, color) in enumerate(zip(results, colors)):
        offsets = x + (i - len(results) / 2 + 0.5) * width
        values = [
            normalized_data["inventory_mae"][i],
            normalized_data["inventory_rmse"][i],
            normalized_data["empty_recall"][i],
            normalized_data["full_recall"][i],
            normalized_data["state_accuracy"][i],
        ]
        errors = [
            normalized_stds["inventory_mae"][i],
            normalized_stds["inventory_rmse"][i],
            normalized_stds["empty_recall"][i],
            normalized_stds["full_recall"][i],
            normalized_stds["state_accuracy"][i],
        ]
        ax.bar(
            offsets,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=result["label"],
            alpha=0.8,
            color=color,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_ylabel("Normalized Performance (0=worst, 1=best)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Lookback Window Configuration Comparison\n(All metrics normalized for comparison)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1.4])

    plt.tight_layout()
    output_file2 = output_dir / "lookback_window_normalized_comparison.png"
    plt.savefig(output_file2, dpi=200, bbox_inches="tight")
    print(f"✓ Saved normalized comparison to: {output_file2}")


def print_recommendations(results):
    """Print recommendations based on results."""

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find best for each metric
    mae_best = min(results, key=lambda r: r["summary"]["inventory_mae"][0])
    empty_best = max(results, key=lambda r: r["summary"]["empty_recall"][0])
    full_best = max(results, key=lambda r: r["summary"]["full_recall"][0])
    state_best = max(results, key=lambda r: r["summary"]["state_accuracy"][0])

    print("\n📊 Best Configuration by Metric:")
    print(
        f"  • Lowest MAE: {mae_best['label']} "
        f"({mae_best['summary']['inventory_mae'][0]:.2f} ± {mae_best['summary']['inventory_mae'][1]:.2f} bikes)"
    )
    print(
        f"  • Best Empty Detection: {empty_best['label']} "
        f"({empty_best['summary']['empty_recall'][0]:.1%} ± {empty_best['summary']['empty_recall'][1]:.1%})"
    )
    print(
        f"  • Best Full Detection: {full_best['label']} "
        f"({full_best['summary']['full_recall'][0]:.1%} ± {full_best['summary']['full_recall'][1]:.1%})"
    )
    print(
        f"  • Best State Accuracy: {state_best['label']} "
        f"({state_best['summary']['state_accuracy'][0]:.1%} ± {state_best['summary']['state_accuracy'][1]:.1%})"
    )

    print("\n💡 Key Insights:")

    # Analyze patterns
    short_train = [r for r in results if r["train_weeks"] <= 2]
    long_train = [r for r in results if r["train_weeks"] >= 3]

    if short_train and long_train:
        short_mae = np.mean([r["summary"]["inventory_mae"][0] for r in short_train])
        long_mae = np.mean([r["summary"]["inventory_mae"][0] for r in long_train])

        if short_mae < long_mae:
            print(
                f"  • Shorter training windows perform better (avg MAE: {short_mae:.2f} vs {long_mae:.2f})"
            )
        else:
            print(
                f"  • Longer training windows perform better (avg MAE: {long_mae:.2f} vs {short_mae:.2f})"
            )

    # Test window comparison
    test_1w = [r for r in results if r["test_weeks"] == 1]
    test_2w = [r for r in results if r["test_weeks"] == 2]

    if test_1w and test_2w:
        test1_mae = np.mean([r["summary"]["inventory_mae"][0] for r in test_1w])
        test2_mae = np.mean([r["summary"]["inventory_mae"][0] for r in test_2w])

        print(f"  • 1-week test window: avg MAE = {test1_mae:.2f} bikes")
        print(f"  • 2-week test window: avg MAE = {test2_mae:.2f} bikes")

    print("\n🎯 Overall Recommendation:")
    print(f"  Use configuration: {mae_best['label']}")
    print("  This provides the best balance of accuracy and computational efficiency")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
