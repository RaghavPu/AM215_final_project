#!/usr/bin/env python3
"""Regenerate lookback window plots from existing results."""

import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the plotting function
from optimize_lookback_window import create_plots

# Load existing results
output_dir = project_root / "outputs"
results_file = output_dir / "lookback_optimization_20251211_112211.json"

with open(results_file, 'r') as f:
    data = json.load(f)
    results = data["configurations"]

# Regenerate plots with error bars
create_plots(results, output_dir)

print("\n✓ Plots regenerated successfully!")
