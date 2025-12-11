#!/usr/bin/env python3
"""Create comprehensive visualizations comparing Temporal Flow vs Markov models."""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'

output_dir = Path(__file__).parent.parent / "outputs"

# Load results
temporal_file = output_dir / "cv_results_temporal_flow_20251210_115409.json"
markov_file = output_dir / "cv_results_markov_20251210_115751.json"

with open(temporal_file, 'r') as f:
    temporal_results = json.load(f)

with open(markov_file, 'r') as f:
    markov_results = json.load(f)

temporal_summary = temporal_results["summary"]
markov_summary = markov_results["summary"]
temporal_folds = temporal_results["fold_results"]
markov_folds = markov_results["fold_results"]

# Create figure with multiple subplots
fig = plt.figure(figsize=(20, 12))

# 1. Bar chart: Summary comparison of key metrics
ax1 = plt.subplot(2, 3, 1)
metrics = ['inventory_mae', 'inventory_rmse', 'correlation']
labels = ['MAE\n(bikes)', 'RMSE\n(bikes)', 'Correlation']
temporal_vals = [temporal_summary[m]["mean"] for m in metrics]
markov_vals = [markov_summary[m]["mean"] for m in metrics]

x = np.arange(len(labels))
width = 0.35

bars1 = ax1.bar(x - width/2, temporal_vals, width, label='Temporal Flow', color='#2ecc71', alpha=0.8)
bars2 = ax1.bar(x + width/2, markov_vals, width, label='Markov', color='#e74c3c', alpha=0.8)

ax1.set_ylabel('Value', fontsize=11, fontweight='bold')
ax1.set_title('Inventory Prediction Metrics\n(Lower MAE/RMSE, Higher Correlation is Better)',
              fontsize=12, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

# 2. Bar chart: State detection metrics
ax2 = plt.subplot(2, 3, 2)
metrics = ['empty_recall', 'empty_f1', 'full_recall', 'full_f1', 'state_accuracy']
labels = ['Empty\nRecall', 'Empty\nF1', 'Full\nRecall', 'Full\nF1', 'State\nAccuracy']
temporal_vals = [temporal_summary[m]["mean"] * 100 for m in metrics]
markov_vals = [markov_summary[m]["mean"] * 100 for m in metrics]

x = np.arange(len(labels))
width = 0.35

bars1 = ax2.bar(x - width/2, temporal_vals, width, label='Temporal Flow', color='#2ecc71', alpha=0.8)
bars2 = ax2.bar(x + width/2, markov_vals, width, label='Markov', color='#e74c3c', alpha=0.8)

ax2.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
ax2.set_title('State Detection Performance\n(Higher is Better)',
              fontsize=12, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=9)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0, 100])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=8)

# 3. Performance over time (MAE)
ax3 = plt.subplot(2, 3, 3)
temporal_mae = [f.get("inventory_mae", np.nan) for f in temporal_folds[3:]]
markov_mae = [f.get("inventory_mae", np.nan) for f in markov_folds[3:]]
fold_ids = list(range(3, len(temporal_folds)))

ax3.plot(fold_ids, temporal_mae, marker='o', linewidth=2, markersize=6,
         label='Temporal Flow', color='#2ecc71')
ax3.plot(fold_ids, markov_mae, marker='s', linewidth=2, markersize=6,
         label='Markov', color='#e74c3c')
ax3.set_xlabel('Fold Number (Daily Increment)', fontsize=11, fontweight='bold')
ax3.set_ylabel('MAE (bikes)', fontsize=11, fontweight='bold')
ax3.set_title('MAE Over Time\n(Both Models Improve with More Data)',
              fontsize=12, fontweight='bold', pad=15)
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)

# 4. Performance over time (Empty Recall)
ax4 = plt.subplot(2, 3, 4)
temporal_empty = [f.get("empty_recall", np.nan) * 100 for f in temporal_folds[3:]]
markov_empty = [f.get("empty_recall", np.nan) * 100 for f in markov_folds[3:]]

ax4.plot(fold_ids, temporal_empty, marker='o', linewidth=2, markersize=6,
         label='Temporal Flow', color='#2ecc71')
ax4.plot(fold_ids, markov_empty, marker='s', linewidth=2, markersize=6,
         label='Markov', color='#e74c3c')
ax4.set_xlabel('Fold Number (Daily Increment)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Empty Station Recall (%)', fontsize=11, fontweight='bold')
ax4.set_title('Empty Detection Over Time\n(Markov Consistently Better)',
              fontsize=12, fontweight='bold', pad=15)
ax4.legend(fontsize=10)
ax4.grid(alpha=0.3)
ax4.set_ylim([0, 105])

# 5. Performance over time (Full Recall)
ax5 = plt.subplot(2, 3, 5)
temporal_full = [f.get("full_recall", np.nan) * 100 for f in temporal_folds[3:]]
markov_full = [f.get("full_recall", np.nan) * 100 for f in markov_folds[3:]]

ax5.plot(fold_ids, temporal_full, marker='o', linewidth=2, markersize=6,
         label='Temporal Flow', color='#2ecc71')
ax5.plot(fold_ids, markov_full, marker='s', linewidth=2, markersize=6,
         label='Markov', color='#e74c3c')
ax5.set_xlabel('Fold Number (Daily Increment)', fontsize=11, fontweight='bold')
ax5.set_ylabel('Full Station Recall (%)', fontsize=11, fontweight='bold')
ax5.set_title('Full Detection Over Time\n(Temporal Flow Consistently Better)',
              fontsize=12, fontweight='bold', pad=15)
ax5.legend(fontsize=10)
ax5.grid(alpha=0.3)
ax5.set_ylim([0, 105])

# 6. Difference plot (Temporal - Markov for each metric)
ax6 = plt.subplot(2, 3, 6)
metrics = ['inventory_mae', 'inventory_rmse', 'empty_recall', 'empty_f1',
           'full_recall', 'full_f1', 'state_accuracy']
labels = ['MAE', 'RMSE', 'Empty\nRecall', 'Empty\nF1',
          'Full\nRecall', 'Full\nF1', 'State\nAccuracy']

# Calculate differences (negative = Temporal better, positive = Markov better)
differences = []
colors = []
for m in metrics:
    t_val = temporal_summary[m]["mean"]
    m_val = markov_summary[m]["mean"]

    # For MAE/RMSE, lower is better, so flip sign
    if 'mae' in m or 'rmse' in m:
        diff = m_val - t_val  # positive = Temporal better
    else:
        diff = t_val - m_val  # positive = Temporal better

    differences.append(diff)
    colors.append('#2ecc71' if diff > 0 else '#e74c3c')

x = np.arange(len(labels))
bars = ax6.barh(x, differences, color=colors, alpha=0.8)

ax6.set_yticks(x)
ax6.set_yticklabels(labels, fontsize=10)
ax6.set_xlabel('Advantage (units vary)', fontsize=11, fontweight='bold')
ax6.set_title('Temporal Flow Advantage\n(Green = Temporal Better, Red = Markov Better)',
              fontsize=12, fontweight='bold', pad=15)
ax6.axvline(x=0, color='black', linewidth=1.5, linestyle='--')
ax6.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, differences)):
    if abs(val) > 0.1:  # Only show if significant
        ax6.text(val, i, f'{val:+.1f}', ha='left' if val > 0 else 'right',
                va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
output_file = output_dir / "model_comparison_dashboard.png"
plt.savefig(output_file, dpi=200, bbox_inches='tight')
print(f"✓ Saved comparison dashboard to: {output_file}")

# Create second figure: detailed fold-by-fold comparison
fig2, axes = plt.subplots(2, 2, figsize=(16, 10))

# Subplot 1: State Accuracy over time
ax = axes[0, 0]
temporal_state = [f.get("state_accuracy", np.nan) * 100 for f in temporal_folds[3:]]
markov_state = [f.get("state_accuracy", np.nan) * 100 for f in markov_folds[3:]]

ax.plot(fold_ids, temporal_state, marker='o', linewidth=2.5, markersize=7,
        label='Temporal Flow', color='#2ecc71', alpha=0.8)
ax.plot(fold_ids, markov_state, marker='s', linewidth=2.5, markersize=7,
        label='Markov', color='#e74c3c', alpha=0.8)
ax.fill_between(fold_ids, temporal_state, alpha=0.2, color='#2ecc71')
ax.fill_between(fold_ids, markov_state, alpha=0.2, color='#e74c3c')
ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
ax.set_ylabel('State Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Overall State Classification Accuracy', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)

# Subplot 2: Correlation over time
ax = axes[0, 1]
temporal_corr = [f.get("correlation", np.nan) for f in temporal_folds[3:]]
markov_corr = [f.get("correlation", np.nan) for f in markov_folds[3:]]

ax.plot(fold_ids, temporal_corr, marker='o', linewidth=2.5, markersize=7,
        label='Temporal Flow', color='#2ecc71', alpha=0.8)
ax.plot(fold_ids, markov_corr, marker='s', linewidth=2.5, markersize=7,
        label='Markov', color='#e74c3c', alpha=0.8)
ax.fill_between(fold_ids, temporal_corr, alpha=0.2, color='#2ecc71')
ax.fill_between(fold_ids, markov_corr, alpha=0.2, color='#e74c3c')
ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Correlation', fontsize=12, fontweight='bold')
ax.set_title('Prediction-Actual Correlation', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)

# Subplot 3: Precision comparison (Empty vs Full)
ax = axes[1, 0]
temporal_empty_prec = [f.get("empty_precision", np.nan) * 100 for f in temporal_folds[3:]]
temporal_full_prec = [f.get("full_precision", np.nan) * 100 for f in temporal_folds[3:]]
markov_empty_prec = [f.get("empty_precision", np.nan) * 100 for f in markov_folds[3:]]
markov_full_prec = [f.get("full_precision", np.nan) * 100 for f in markov_folds[3:]]

ax.plot(fold_ids, temporal_empty_prec, marker='o', linewidth=2, markersize=6,
        label='Temporal: Empty', color='#2ecc71', linestyle='-')
ax.plot(fold_ids, temporal_full_prec, marker='o', linewidth=2, markersize=6,
        label='Temporal: Full', color='#27ae60', linestyle='--')
ax.plot(fold_ids, markov_empty_prec, marker='s', linewidth=2, markersize=6,
        label='Markov: Empty', color='#e74c3c', linestyle='-')
ax.plot(fold_ids, markov_full_prec, marker='s', linewidth=2, markersize=6,
        label='Markov: Full', color='#c0392b', linestyle='--')
ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
ax.set_ylabel('Precision (%)', fontsize=12, fontweight='bold')
ax.set_title('Empty vs Full Station Precision', fontsize=13, fontweight='bold', pad=15)
ax.legend(fontsize=9, loc='best', ncol=2)
ax.grid(alpha=0.3)

# Subplot 4: Difference in MAE over time
ax = axes[1, 1]
mae_diff = [m - t for m, t in zip(markov_mae, temporal_mae)]

colors = ['#2ecc71' if d > 0 else '#e74c3c' for d in mae_diff]
bars = ax.bar(fold_ids, mae_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linewidth=2, linestyle='-')
ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE Advantage (bikes)', fontsize=12, fontweight='bold')
ax.set_title('Temporal Flow MAE Advantage by Fold\n(Positive = Temporal Better)',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(axis='y', alpha=0.3)

# Add mean line
mean_diff = np.mean(mae_diff)
ax.axhline(y=mean_diff, color='blue', linewidth=2, linestyle='--',
           label=f'Mean: {mean_diff:+.2f} bikes')
ax.legend(fontsize=10)

plt.tight_layout()
output_file2 = output_dir / "model_comparison_timeseries.png"
plt.savefig(output_file2, dpi=200, bbox_inches='tight')
print(f"✓ Saved time series comparison to: {output_file2}")

print("\n" + "=" * 70)
print("Visualizations created successfully!")
print("=" * 70)
print("\nGenerated files:")
print(f"  1. {output_file}")
print(f"  2. {output_file2}")
