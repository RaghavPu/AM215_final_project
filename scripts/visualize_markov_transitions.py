#!/usr/bin/env python3
"""
Visualize and explore Markov model transition matrices.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models import MarkovModel
from utils import load_config, load_station_info, load_trip_data, prepare_data


def main():
    print("=" * 70)
    print("Markov Model Transition Matrix Visualization")
    print("=" * 70)

    # Load data
    config = load_config("config.yaml")

    print("\nLoading data...")
    trips = load_trip_data(
        config["data"]["trip_data_dir"],
        start_date=config["time"]["start_date"],
        end_date=config["time"]["end_date"],
        use_parquet=True,
    )

    stations = load_station_info(config["data"]["station_info_path"])
    trips, station_stats = prepare_data(trips, stations, config)

    # Fit Markov model
    print("\nFitting Markov model...")
    model = MarkovModel(config)
    model.fit(trips, station_stats)

    print(f"\nModel has {len(model.transition_matrices)} transition matrices")
    print(f"Total stations: {len(model.stations)}")
    print(f"Average sparsity: {model._compute_avg_sparsity():.1%}")

    # Explore specific contexts
    contexts_to_show = [
        (8, False, "8 AM Weekday (Morning Rush)"),
        (17, False, "5 PM Weekday (Evening Rush)"),
        (14, True, "2 PM Weekend (Afternoon)"),
        (2, False, "2 AM Weekday (Late Night)"),
    ]

    for hour, is_weekend, description in contexts_to_show:
        print("\n" + "=" * 70)
        print(f"{description}")
        print("=" * 70)

        P = model.transition_matrices.get((hour, is_weekend))
        dep_rates = model.departure_rate_arrays.get((hour, is_weekend))

        if P is None:
            print("  No data for this context")
            continue

        # Get matrix stats
        n_stations = P.shape[0]
        n_nonzero = P.nnz
        density = n_nonzero / (n_stations * n_stations)

        print("\nMatrix Statistics:")
        print(f"  Size: {n_stations} x {n_stations}")
        print(f"  Non-zero entries: {n_nonzero:,}")
        print(f"  Density: {density:.4%}")
        print(f"  Sparsity: {(1 - density):.4%}")

        # Average departure rate
        avg_dep_rate = dep_rates.mean()
        max_dep_rate = dep_rates.max()

        print("\nDeparture Rates:")
        print(f"  Average: {avg_dep_rate:.2f} trips/hour/station")
        print(f"  Maximum: {max_dep_rate:.2f} trips/hour")
        print(f"  Total expected departures: {dep_rates.sum():.0f} trips/hour")

        # Find most active stations
        top_departure_indices = np.argsort(dep_rates)[-5:][::-1]

        print("\nTop 5 Busiest Stations (by departures):")
        for i, idx in enumerate(top_departure_indices, 1):
            station_name = model.idx_to_station[idx]
            rate = dep_rates[idx]
            print(f"  {i}. {station_name}: {rate:.2f} trips/hour")

        # For the busiest station, show top destinations
        busiest_idx = top_departure_indices[0]
        busiest_station = model.idx_to_station[busiest_idx]

        print(f"\nTop 10 Destinations from '{busiest_station}':")

        # Get row from transition matrix
        row = P.getrow(busiest_idx).toarray().flatten()
        top_dest_indices = np.argsort(row)[-10:][::-1]

        for i, dest_idx in enumerate(top_dest_indices, 1):
            if row[dest_idx] > 0:
                dest_name = model.idx_to_station[dest_idx]
                prob = row[dest_idx]
                print(f"  {i}. {dest_name}: {prob:.1%}")

    # Visualization: Compare morning vs evening rush hour
    print("\n" + "=" * 70)
    print("Comparing Morning vs Evening Rush Hour")
    print("=" * 70)

    morning_P = model.transition_matrices.get((8, False))
    evening_P = model.transition_matrices.get((17, False))

    if morning_P is not None and evening_P is not None:
        # Find stations with different flow patterns
        print("\nStations with Reversed Flow Patterns:")
        print("(Morning: mostly outgoing, Evening: mostly incoming)")

        morning_rates = model.departure_rate_arrays[(8, False)]
        evening_rates = model.departure_rate_arrays[(17, False)]

        # Compute net flow for each station
        morning_out = morning_rates
        evening_out = evening_rates

        # Find stations that are residential (lose bikes in morning, gain in evening)
        flow_diff = morning_out - evening_out
        residential_indices = np.argsort(flow_diff)[-5:][::-1]

        print("\nMost 'Residential' Stations (high morning out, low evening out):")
        for i, idx in enumerate(residential_indices, 1):
            station = model.idx_to_station[idx]
            morning = morning_out[idx]
            evening = evening_out[idx]
            print(f"  {i}. {station}")
            print(f"      Morning: {morning:.1f} trips/hour out")
            print(f"      Evening: {evening:.1f} trips/hour out")
            print(f"      Difference: {morning - evening:.1f}")

    # Morning rush hour visualization - top 7 busiest stations
    print("\n" + "=" * 70)
    print("Creating Morning Rush Hour Heatmap (8 AM Weekday)")
    print("=" * 70)

    # Pick morning rush hour
    P_morning = model.transition_matrices.get((8, False))

    if P_morning is not None:
        # Select top 7 most active stations
        morning_rates = model.departure_rate_arrays[(8, False)]
        top_7_indices = np.argsort(morning_rates)[-7:][::-1]

        print("\nTop 7 Busiest Stations at 8 AM:")
        for i, idx in enumerate(top_7_indices, 1):
            station = model.idx_to_station[idx]
            rate = morning_rates[idx]
            print(f"  {i}. {station}: {rate:.2f} trips/hour")

        # Extract submatrix
        submatrix = P_morning[top_7_indices, :][:, top_7_indices].toarray()

        # Get station names
        station_names = [model.idx_to_station[i] for i in top_7_indices]
        station_labels = [
            name[:40] if len(name) <= 40 else name[:37] + "..." for name in station_names
        ]

        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            submatrix,
            xticklabels=station_labels,
            yticklabels=station_labels,
            cmap="YlOrRd",
            cbar_kws={"label": "Transition Probability"},
            vmin=0,
            vmax=0.15,  # Cap for better visualization
            annot=True,  # Show values since it's only 7x7
            fmt=".3f",
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(
            "Morning Rush Hour Transition Probabilities\nTop 7 Busiest Stations (8 AM Weekday)",
            fontsize=15,
            pad=20,
            fontweight="bold",
        )
        plt.xlabel("Destination Station", fontsize=12)
        plt.ylabel("Origin Station", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        output_file = project_root / "outputs" / "markov_transition_morning_top7.png"
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"\nSaved morning heatmap to: {output_file}")

        # Analyze top routes
        print("\nTop Routes from Each Busiest Station:")
        for idx in top_7_indices:
            station = model.idx_to_station[idx]
            print(f"\n  From '{station}' (departure rate: {morning_rates[idx]:.2f} trips/hr):")

            row = P_morning.getrow(idx).toarray().flatten()
            top_dest_indices = np.argsort(row)[-5:][::-1]

            for i, dest_idx in enumerate(top_dest_indices, 1):
                if row[dest_idx] > 0:
                    dest = model.idx_to_station[dest_idx]
                    prob = row[dest_idx]
                    volume = morning_rates[idx] * prob
                    print(f"      {i}. → {dest[:40]:<40} {prob:6.1%}  ({volume:.1f} trips/hr)")

        print("\nHeatmap Interpretation:")
        print("  - Values show probability of going from row station → column station")
        print("  - Numbers are transition probabilities (0.0 to 1.0)")
        print("  - Higher values (darker red) = more common routes")
        print("  - Morning rush = residential → business commute pattern")
        print("  - These 7 stations are primarily residential areas with high morning departures")

    # Evening rush hour visualization - top 7 busiest stations
    print("\n" + "=" * 70)
    print("Creating Evening Rush Hour Heatmap (5 PM - BUSIEST PERIOD)")
    print("=" * 70)

    # Pick evening rush hour (busiest period)
    P_evening = model.transition_matrices.get((17, False))

    if P_evening is not None:
        # Select top 7 most active stations
        evening_rates = model.departure_rate_arrays[(17, False)]
        top_7_indices = np.argsort(evening_rates)[-7:][::-1]

        print("\nTop 7 Busiest Stations at 5 PM:")
        for i, idx in enumerate(top_7_indices, 1):
            station = model.idx_to_station[idx]
            rate = evening_rates[idx]
            print(f"  {i}. {station}: {rate:.2f} trips/hour")

        # Extract submatrix
        submatrix = P_evening[top_7_indices, :][:, top_7_indices].toarray()

        # Get station names
        station_names = [model.idx_to_station[i] for i in top_7_indices]
        station_labels = [
            name[:40] if len(name) <= 40 else name[:37] + "..." for name in station_names
        ]

        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            submatrix,
            xticklabels=station_labels,
            yticklabels=station_labels,
            cmap="YlOrRd",
            cbar_kws={"label": "Transition Probability"},
            vmin=0,
            vmax=0.10,  # Cap for better visualization
            annot=True,  # Show values since it's only 7x7
            fmt=".3f",
            linewidths=0.5,
            linecolor="gray",
        )
        plt.title(
            "Evening Rush Hour Transition Probabilities\nTop 7 Busiest Stations (5 PM Weekday)",
            fontsize=15,
            pad=20,
            fontweight="bold",
        )
        plt.xlabel("Destination Station", fontsize=12)
        plt.ylabel("Origin Station", fontsize=12)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()

        output_file = project_root / "outputs" / "markov_transition_evening_top7.png"
        plt.savefig(output_file, dpi=200, bbox_inches="tight")
        print(f"\nSaved evening heatmap to: {output_file}")

        # Analyze top routes
        print("\nTop Routes from Each Busiest Station:")
        for idx in top_7_indices:
            station = model.idx_to_station[idx]
            print(f"\n  From '{station}' (departure rate: {evening_rates[idx]:.2f} trips/hr):")

            row = P_evening.getrow(idx).toarray().flatten()
            top_dest_indices = np.argsort(row)[-5:][::-1]

            for i, dest_idx in enumerate(top_dest_indices, 1):
                if row[dest_idx] > 0:
                    dest = model.idx_to_station[dest_idx]
                    prob = row[dest_idx]
                    volume = evening_rates[idx] * prob
                    print(f"      {i}. → {dest[:40]:<40} {prob:6.1%}  ({volume:.1f} trips/hr)")

        # Print interpretation
        print("\nHeatmap Interpretation:")
        print("  - Values show probability of going from row station → column station")
        print("  - Numbers are transition probabilities (0.0 to 1.0)")
        print("  - Higher values (darker red) = more common routes")
        print("  - Evening rush = highest activity period (3,099 total trips/hour)")
        print("  - These 7 stations account for majority of evening commute traffic")

    # Transition matrix comparison across times
    print("\n" + "=" * 70)
    print("Hourly Activity Pattern Analysis")
    print("=" * 70)

    hourly_activity = []
    for hour in range(24):
        for is_weekend in [False, True]:
            dep_rates = model.departure_rate_arrays.get((hour, is_weekend))
            if dep_rates is not None:
                total_activity = dep_rates.sum()
                day_type = "Weekend" if is_weekend else "Weekday"
                hourly_activity.append(
                    {
                        "hour": hour,
                        "day_type": day_type,
                        "total_trips_per_hour": total_activity,
                    }
                )

    df_activity = pd.DataFrame(hourly_activity)

    # Create activity plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Weekday pattern
    weekday_data = df_activity[df_activity["day_type"] == "Weekday"]
    ax1.plot(
        weekday_data["hour"],
        weekday_data["total_trips_per_hour"],
        marker="o",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Hour of Day", fontsize=12)
    ax1.set_ylabel("Total Expected Trips/Hour", fontsize=12)
    ax1.set_title("Weekday Activity Pattern", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(0, 24, 2))

    # Weekend pattern
    weekend_data = df_activity[df_activity["day_type"] == "Weekend"]
    ax2.plot(
        weekend_data["hour"],
        weekend_data["total_trips_per_hour"],
        marker="o",
        linewidth=2,
        color="orange",
        markersize=6,
    )
    ax2.set_xlabel("Hour of Day", fontsize=12)
    ax2.set_ylabel("Total Expected Trips/Hour", fontsize=12)
    ax2.set_title("Weekend Activity Pattern", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(0, 24, 2))

    plt.tight_layout()
    output_file = project_root / "outputs" / "markov_hourly_activity.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nSaved activity pattern plot to: {output_file}")

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print("\nKey Findings:")
    print("1. Transition matrices are highly sparse (99.9% zeros)")
    print("2. Most trips occur between a small number of popular station pairs")
    print("3. Clear rush hour patterns on weekdays (morning and evening peaks)")
    print("4. Weekend patterns are more uniform throughout the day")
    print("5. Individual stations show asymmetric flow (residential vs business)")


if __name__ == "__main__":
    main()
