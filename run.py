#!/usr/bin/env python3
"""
Main script to run the CitiBike inventory prediction pipeline.

Usage:
    python run.py                    # Run with default config
    python run.py --config custom.yaml  # Run with custom config
    python run.py --model baseline   # Override model choice
    python run.py --seed 12345       # Set random seed for reproducibility
    python run.py --compare          # Compare against cached results
"""

import argparse
import json
import logging
from datetime import datetime
from glob import glob
from pathlib import Path

from citibike.evaluation import run_cross_validation
from citibike.models import get_model
from citibike.utils import load_config, load_station_info, load_trip_data, prepare_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CitiBike Inventory Prediction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation (just fit model)",
    )
    parser.add_argument(
        "--compare",
        type=str,
        default=None,
        help="Compare against cached results for specified model (e.g., --compare baseline)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (overrides config)",
    )
    return parser.parse_args()


def load_cached_results(output_dir: Path, model_name: str) -> dict:
    """Load the most recent cached results for a model."""
    pattern = output_dir / f"cv_results_{model_name}_*.json"
    files = sorted(glob(str(pattern)))

    if not files:
        return None

    latest_file = files[-1]
    with open(latest_file) as f:
        results = json.load(f)

    print(f"Loaded cached {model_name} results from: {latest_file}")
    return results


def compare_results(current: dict, other: dict):
    """Print comparison between current and other model results."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    current_summary = current.get("summary", {})
    other_summary = other.get("summary", {})

    current_name = current.get("model", "Current")
    other_name = other.get("model", "Other")

    print(f"\n{'Metric':<25} {other_name:>15} {current_name:>15} {'Δ':>12} {'Better?':>10}")
    print("-" * 70)

    # Key metrics to compare (lower is better for errors, higher for recall/accuracy)
    key_metrics = [
        ("inventory_mae", "lower"),
        ("inventory_rmse", "lower"),
        ("correlation", "higher"),
        ("empty_recall", "higher"),
        ("empty_precision", "higher"),
        ("full_recall", "higher"),
        ("full_precision", "higher"),
        ("state_accuracy", "higher"),
    ]

    improvements = 0

    for metric, better_direction in key_metrics:
        if metric in current_summary and metric in other_summary:
            curr_mean = current_summary[metric]["mean"]
            other_mean = other_summary[metric]["mean"]

            delta = curr_mean - other_mean

            if better_direction == "lower":
                is_better = delta < 0
            else:
                is_better = delta > 0

            if is_better:
                improvements += 1

            delta_str = f"{delta:+.4f}"
            better_str = "✅ Yes" if is_better else "❌ No"

            # Format based on metric type
            if "recall" in metric or "precision" in metric or "accuracy" in metric:
                print(
                    f"{metric:<25} {other_mean:>14.1%} {curr_mean:>14.1%} {delta_str:>12} {better_str:>10}"
                )
            elif "correlation" in metric:
                print(
                    f"{metric:<25} {other_mean:>15.3f} {curr_mean:>15.3f} {delta_str:>12} {better_str:>10}"
                )
            else:
                print(
                    f"{metric:<25} {other_mean:>15.2f} {curr_mean:>15.2f} {delta_str:>12} {better_str:>10}"
                )

    print("-" * 70)
    print(f"\nImproved on {improvements}/{len(key_metrics)} metrics")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    print("=" * 60)
    print("CitiBike Inventory Prediction Pipeline")
    print("=" * 60)

    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")

    # Override config with command line args
    if args.model:
        config["model"]["name"] = args.model
    if args.output_dir:
        config["data"]["output_dir"] = args.output_dir
    if args.seed is not None:
        # Override random seed in model configs
        if "markov" not in config.get("model", {}):
            config["model"]["markov"] = {}
        config["model"]["markov"]["random_seed"] = args.seed

    # Log the random seed being used
    random_seed = config.get("model", {}).get("markov", {}).get("random_seed", None)
    logging.info(f"Random seed for this run: {random_seed}")

    # Create output directory
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\n" + "-" * 40)
    print("Loading Data")
    print("-" * 40)

    trips = load_trip_data(
        config["data"]["trip_data_dir"],
        start_date=config["time"]["start_date"],
        end_date=config["time"]["end_date"],
    )

    stations = load_station_info(config["data"]["station_info_path"])

    # Prepare data
    print("\n" + "-" * 40)
    print("Preparing Data")
    print("-" * 40)

    trips, station_stats = prepare_data(trips, stations, config)

    # Initialize model
    print("\n" + "-" * 40)
    print("Initializing Model")
    print("-" * 40)

    model_name = config["model"]["name"]
    print(f"Model: {model_name}")

    model = get_model(model_name, config)

    # Run cross-validation
    if not args.no_cv:
        print("\n" + "-" * 40)
        print("Running Cross-Validation")
        print("-" * 40)

        fold_results, summary = run_cross_validation(
            model,
            trips,
            station_stats,
            config,
            verbose=True,
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"cv_results_{model_name}_{timestamp}.json"

        # Get the random seed used (for logging)
        random_seed = config.get("model", {}).get("markov", {}).get("random_seed", None)

        results = {
            "model": model_name,
            "config": config,
            "fold_results": fold_results,
            "summary": {k: {"mean": v[0], "std": v[1]} for k, v in summary.items()},
            "timestamp": timestamp,
            "random_seed": random_seed,
        }

        logging.info(f"Run completed with random_seed={random_seed}")

        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {results_file}")

        # Compare against other model if requested
        if args.compare:
            other_results = load_cached_results(output_dir, args.compare)
            if other_results:
                compare_results(results, other_results)
            else:
                print(f"\nNo cached results found for model: {args.compare}")

    else:
        # Just fit on all data
        print("\n" + "-" * 40)
        print("Fitting Model on All Data")
        print("-" * 40)

        model.fit(trips, station_stats)
        print(f"Model fitted: {model.get_params()}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
