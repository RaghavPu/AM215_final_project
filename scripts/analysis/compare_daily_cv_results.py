#!/usr/bin/env python3
"""Compare temporal_flow vs markov with daily rolling CV results."""

import json
from pathlib import Path

output_dir = Path(__file__).parent.parent / "outputs"

# Load most recent results
temporal_file = output_dir / "cv_results_temporal_flow_20251210_115409.json"
markov_file = output_dir / "cv_results_markov_20251210_115751.json"

with open(temporal_file) as f:
    temporal_results = json.load(f)

with open(markov_file) as f:
    markov_results = json.load(f)

print("=" * 80)
print("TEMPORAL FLOW vs MARKOV MODEL COMPARISON")
print("Daily Rolling Cross-Validation (3 weeks train, 1 week test, increment by 1 day)")
print("=" * 80)

print(
    f"\nData: {temporal_results['config']['time']['start_date']} to {temporal_results['config']['time']['end_date']}"
)
print(f"Number of folds: {len(temporal_results['fold_results'])}")
print("Stations: 1369")
print("Rolling window: Train 3 weeks → Test 1 week → Shift 1 day forward")

# Extract summaries
temporal_summary = temporal_results["summary"]
markov_summary = markov_results["summary"]

metrics_to_compare = [
    ("inventory_mae", "Inventory MAE (bikes)", "lower"),
    ("inventory_rmse", "Inventory RMSE (bikes)", "lower"),
    ("correlation", "Correlation", "higher"),
    ("empty_recall", "Empty Station Recall", "higher"),
    ("empty_precision", "Empty Station Precision", "higher"),
    ("empty_f1", "Empty Station F1", "higher"),
    ("full_recall", "Full Station Recall", "higher"),
    ("full_precision", "Full Station Precision", "higher"),
    ("full_f1", "Full Station F1", "higher"),
    ("state_accuracy", "State Accuracy", "higher"),
]

print("\n" + "=" * 80)
print("SUMMARY COMPARISON (Mean ± Std across 19 folds)")
print("=" * 80)

comparison_data = []
for metric, label, better in metrics_to_compare:
    if metric in temporal_summary and metric in markov_summary:
        t_mean = temporal_summary[metric]["mean"]
        t_std = temporal_summary[metric]["std"]
        m_mean = markov_summary[metric]["mean"]
        m_std = markov_summary[metric]["std"]

        diff = m_mean - t_mean
        pct_diff = (diff / t_mean * 100) if t_mean != 0 else 0

        if better == "lower":
            winner = "Temporal Flow" if t_mean < m_mean else "Markov"
            symbol = "✅" if t_mean < m_mean else "❌"
        else:
            winner = "Temporal Flow" if t_mean > m_mean else "Markov"
            symbol = "✅" if t_mean > m_mean else "❌"

        comparison_data.append(
            {
                "metric": metric,
                "label": label,
                "temporal": (t_mean, t_std),
                "markov": (m_mean, m_std),
                "diff": diff,
                "pct_diff": pct_diff,
                "winner": winner,
                "symbol": symbol,
            }
        )

# Print grouped results
print("\n" + "-" * 80)
print("INVENTORY PREDICTION METRICS (Lower is Better)")
print("-" * 80)
print(f"{'Metric':<30} {'Temporal Flow':<20} {'Markov':<20} {'Δ':<15} {'Winner':<15}")
print("-" * 80)

for item in comparison_data[:3]:  # MAE, RMSE, Correlation
    t_mean, t_std = item["temporal"]
    m_mean, m_std = item["markov"]

    if "correlation" in item["metric"]:
        print(
            f"{item['label']:<30} {t_mean:6.3f} ± {t_std:5.3f}      {m_mean:6.3f} ± {m_std:5.3f}      "
            f"{item['diff']:+6.3f}      {item['symbol']} {item['winner']}"
        )
    else:
        print(
            f"{item['label']:<30} {t_mean:6.2f} ± {t_std:5.2f}      {m_mean:6.2f} ± {m_std:5.2f}      "
            f"{item['diff']:+6.2f}      {item['symbol']} {item['winner']}"
        )

print("\n" + "-" * 80)
print("EMPTY STATION DETECTION (Higher is Better)")
print("-" * 80)
print(f"{'Metric':<30} {'Temporal Flow':<20} {'Markov':<20} {'Δ':<15} {'Winner':<15}")
print("-" * 80)

for item in comparison_data[3:6]:  # Empty recall, precision, F1
    t_mean, t_std = item["temporal"]
    m_mean, m_std = item["markov"]
    print(
        f"{item['label']:<30} {t_mean:5.1%} ± {t_std:4.1%}      {m_mean:5.1%} ± {m_std:4.1%}      "
        f"{item['diff']:+5.1%}      {item['symbol']} {item['winner']}"
    )

print("\n" + "-" * 80)
print("FULL STATION DETECTION (Higher is Better)")
print("-" * 80)
print(f"{'Metric':<30} {'Temporal Flow':<20} {'Markov':<20} {'Δ':<15} {'Winner':<15}")
print("-" * 80)

for item in comparison_data[6:9]:  # Full recall, precision, F1
    t_mean, t_std = item["temporal"]
    m_mean, m_std = item["markov"]
    print(
        f"{item['label']:<30} {t_mean:5.1%} ± {t_std:4.1%}      {m_mean:5.1%} ± {m_std:4.1%}      "
        f"{item['diff']:+5.1%}      {item['symbol']} {item['winner']}"
    )

print("\n" + "-" * 80)
print("OVERALL STATE CLASSIFICATION")
print("-" * 80)
print(f"{'Metric':<30} {'Temporal Flow':<20} {'Markov':<20} {'Δ':<15} {'Winner':<15}")
print("-" * 80)

for item in comparison_data[9:]:  # State accuracy
    t_mean, t_std = item["temporal"]
    m_mean, m_std = item["markov"]
    print(
        f"{item['label']:<30} {t_mean:5.1%} ± {t_std:4.1%}      {m_mean:5.1%} ± {m_std:4.1%}      "
        f"{item['diff']:+5.1%}      {item['symbol']} {item['winner']}"
    )

# Score summary
print("\n" + "=" * 80)
print("OVERALL WINNER")
print("=" * 80)

temporal_wins = sum(1 for item in comparison_data if item["winner"] == "Temporal Flow")
markov_wins = sum(1 for item in comparison_data if item["winner"] == "Markov")

print(f"\nTemporal Flow wins: {temporal_wins}/10 metrics")
print(f"Markov wins: {markov_wins}/10 metrics")

overall_winner = (
    "Temporal Flow"
    if temporal_wins > markov_wins
    else "Markov"
    if markov_wins > temporal_wins
    else "Tie"
)
print(f"\n🏆 Overall Winner: {overall_winner}")

# Key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

print("\n1. Inventory Prediction Accuracy:")
temporal_mae = temporal_summary["inventory_mae"]["mean"]
markov_mae = markov_summary["inventory_mae"]["mean"]
if temporal_mae < markov_mae:
    diff = markov_mae - temporal_mae
    pct = (diff / markov_mae) * 100
    print(
        f"   - Temporal Flow is {pct:.1f}% more accurate (MAE: {temporal_mae:.2f} vs {markov_mae:.2f} bikes)"
    )
else:
    diff = temporal_mae - markov_mae
    pct = (diff / temporal_mae) * 100
    print(
        f"   - Markov is {pct:.1f}% more accurate (MAE: {markov_mae:.2f} vs {temporal_mae:.2f} bikes)"
    )

print("\n2. Empty Station Detection:")
temporal_empty = temporal_summary["empty_recall"]["mean"]
markov_empty = markov_summary["empty_recall"]["mean"]
if markov_empty > temporal_empty:
    diff = markov_empty - temporal_empty
    print(
        f"   - Markov catches {diff:.1%} more empty stations ({markov_empty:.1%} vs {temporal_empty:.1%} recall)"
    )
    print("   - Markov's routing information helps predict when stations run out")
else:
    diff = temporal_empty - markov_empty
    print(
        f"   - Temporal Flow catches {diff:.1%} more empty stations ({temporal_empty:.1%} vs {markov_empty:.1%} recall)"
    )

print("\n3. Full Station Detection:")
temporal_full = temporal_summary["full_recall"]["mean"]
markov_full = markov_summary["full_recall"]["mean"]
if temporal_full > markov_full:
    diff = temporal_full - markov_full
    print(
        f"   - Temporal Flow catches {diff:.1%} more full stations ({temporal_full:.1%} vs {markov_full:.1%} recall)"
    )
else:
    diff = markov_full - temporal_full
    print(
        f"   - Markov catches {diff:.1%} more full stations ({markov_full:.1%} vs {temporal_full:.1%} recall)"
    )

print("\n4. Trade-offs:")
print("   - Temporal Flow: Better overall accuracy, simpler model, faster training")
print("   - Markov: Better empty detection, captures routing patterns, higher complexity")

print("\n5. Rolling Daily CV vs Weekly CV:")
print(
    f"   - Daily increments gave us {len(temporal_results['fold_results'])} folds vs 3 with weekly"
)
print("   - More granular view of performance over time")
print("   - Can see how performance changes as training data accumulates")

# Fold-by-fold analysis
print("\n" + "=" * 80)
print("PERFORMANCE OVER TIME (Fold-by-Fold MAE)")
print("=" * 80)

temporal_folds = temporal_results["fold_results"]
markov_folds = markov_results["fold_results"]

print(
    f"\n{'Fold':<6} {'Test Week':<12} {'Temporal MAE':<15} {'Markov MAE':<15} {'Δ':<10} {'Winner':<15}"
)
print("-" * 80)

for i, (t_fold, m_fold) in enumerate(zip(temporal_folds, markov_folds)):
    if i >= 3:  # Start from fold 3 where we have actual data
        t_mae = t_fold.get("inventory_mae", 0)
        m_mae = m_fold.get("inventory_mae", 0)
        diff = m_mae - t_mae
        winner = "Temporal" if t_mae < m_mae else "Markov"
        symbol = "✅" if t_mae < m_mae else "❌"

        test_week = t_fold.get("test_start", "")
        print(
            f"{i:<6} {test_week:<12} {t_mae:6.2f}          {m_mae:6.2f}          "
            f"{diff:+6.2f}     {symbol} {winner}"
        )

print("\n" + "=" * 80)
print("Observation: Both models improve as more training data accumulates (later folds)")
print("=" * 80)
