#!/usr/bin/env python3
"""
Main script to run the CitiBike flow prediction pipeline.

Usage:
    python run.py                    # Run with default config
    python run.py --config custom.yaml  # Run with custom config
    python run.py --model baseline   # Override model choice
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

from utils import load_config, load_trip_data, load_station_info, prepare_data
from models import get_model
from evaluation import run_cross_validation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CitiBike Flow Prediction Pipeline"
    )
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
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load configuration
    print("=" * 60)
    print("CitiBike Flow Prediction Pipeline")
    print("=" * 60)
    
    config = load_config(args.config)
    print(f"\nLoaded config from: {args.config}")
    
    # Override config with command line args
    if args.model:
        config["model"]["name"] = args.model
    if args.output_dir:
        config["data"]["output_dir"] = args.output_dir
    
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
        
        results = {
            "model": model_name,
            "config": config,
            "fold_results": fold_results,
            "summary": {k: {"mean": v[0], "std": v[1]} for k, v in summary.items()},
            "timestamp": timestamp,
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {results_file}")
    
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

