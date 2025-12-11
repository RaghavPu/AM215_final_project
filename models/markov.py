"""Markov Chain model for bike inventory prediction using Monte Carlo simulation."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from .base import BaseModel


class MarkovModel(BaseModel):
    """Markov Chain model for predicting bike station inventory.

    This model:
    1. Builds time-dependent transition matrices P[i→j | hour, is_weekend]
    2. Learns departure rates per station per time context
    3. Simulates multiple random walks and averages predictions

    Key features:
    - State-dependent: Departures depend on current inventory
    - Capacity-aware: Arrivals capped at station capacity
    - Monte Carlo: Average over N simulations for robust predictions
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Model parameters
        model_config = config.get("model", {}).get("markov", {})
        self.smoothing_alpha = model_config.get("smoothing_alpha", 0.0)
        self.min_transitions = model_config.get("min_transitions", 1)
        self.n_simulations = model_config.get("n_simulations", 50)
        self.random_seed = model_config.get("random_seed", None)  # None = use system entropy

        # Learned parameters
        self.stations = []
        self.station_to_idx = {}
        self.idx_to_station = {}

        # Transition matrices: (hour, is_weekend) -> sparse matrix
        self.transition_matrices = {}

        # Departure rates: (hour, is_weekend) -> array of rates per station
        self.departure_rate_arrays = {}

        # Fallback for missing contexts
        self.global_transition_matrix = None
        self.global_departure_rates = None

    def fit(
        self,
        trips: pd.DataFrame,
        station_stats: pd.DataFrame,
    ) -> "BaseModel":
        """Build time-dependent transition matrices from trip data.

        Uses vectorized pandas operations for speed.
        """
        print(f"Fitting {self.get_name()} on {len(trips):,} trips...")

        # Get list of stations and create index mappings
        self.stations = station_stats.index.tolist()
        self.station_capacities = station_stats["capacity"].to_dict()
        self.station_to_idx = {s: i for i, s in enumerate(self.stations)}
        self.idx_to_station = dict(enumerate(self.stations))
        n_stations = len(self.stations)

        # Build capacity array for vectorized operations
        self.capacity_array = np.array(
            [self.station_capacities.get(s, 30) for s in self.stations], dtype=float
        )

        print(f"  Building transition matrices for {n_stations} stations...")

        # Prepare data
        trips = trips.copy()
        if "hour" not in trips.columns:
            trips["hour"] = trips["started_at"].dt.hour
        if "is_weekend" not in trips.columns:
            trips["is_weekend"] = trips["started_at"].dt.dayofweek.isin([5, 6])

        # Filter to known stations
        trips = trips[
            trips["start_station_name"].isin(self.station_to_idx)
            & trips["end_station_name"].isin(self.station_to_idx)
        ].copy()

        # Map station names to indices (vectorized)
        trips["from_idx"] = trips["start_station_name"].map(self.station_to_idx)
        trips["to_idx"] = trips["end_station_name"].map(self.station_to_idx)

        # Count days per context for rate normalization
        trips["date"] = trips["started_at"].dt.date
        days_per_context = trips.groupby(["hour", "is_weekend"])["date"].nunique()

        # Build all transition counts at once using groupby (FAST!)
        print("  Counting transitions (vectorized)...")
        transition_counts = (
            trips.groupby(["hour", "is_weekend", "from_idx", "to_idx"])
            .size()
            .reset_index(name="count")
        )

        departure_counts = (
            trips.groupby(["hour", "is_weekend", "from_idx"]).size().reset_index(name="departures")
        )

        # Build transition matrices for each context
        print("  Building sparse matrices...")
        contexts = [(h, w) for h in range(24) for w in [False, True]]

        for hour, is_weekend in contexts:
            # Filter to this context
            ctx_trans = transition_counts[
                (transition_counts["hour"] == hour)
                & (transition_counts["is_weekend"] == is_weekend)
            ]
            ctx_deps = departure_counts[
                (departure_counts["hour"] == hour) & (departure_counts["is_weekend"] == is_weekend)
            ]

            if len(ctx_trans) < self.min_transitions:
                continue

            # Get total departures per origin
            dep_totals = ctx_deps.set_index("from_idx")["departures"]

            # Compute probabilities
            ctx_trans = ctx_trans.copy()
            dep_totals_local = dep_totals  # Capture for lambda
            ctx_trans["prob"] = ctx_trans.apply(
                lambda row, dt=dep_totals_local: row["count"] / dt.get(row["from_idx"], 1),
                axis=1,
            )

            # Create sparse matrix
            P = sparse.csr_matrix(
                (
                    ctx_trans["prob"].values,
                    (
                        ctx_trans["from_idx"].values.astype(int),
                        ctx_trans["to_idx"].values.astype(int),
                    ),
                ),
                shape=(n_stations, n_stations),
            )

            self.transition_matrices[(hour, is_weekend)] = P

            # Compute departure rates as array (vectorized for simulation)
            n_days = days_per_context.get((hour, is_weekend), 1)
            dep_rate_array = np.zeros(n_stations)
            for _, row in ctx_deps.iterrows():
                idx = int(row["from_idx"])
                dep_rate_array[idx] = row["departures"] / n_days

            self.departure_rate_arrays[(hour, is_weekend)] = dep_rate_array

        # Build global fallback
        self._build_global_fallback(trips, n_stations)

        self.is_fitted = True
        print(f"  Built {len(self.transition_matrices)} transition matrices")
        print(f"  Sparsity: {self._compute_avg_sparsity():.1%}")
        print(f"  Simulations per prediction: {self.n_simulations}")

        return self

    def _build_global_fallback(self, trips: pd.DataFrame, n_stations: int):
        """Build a global transition matrix as fallback."""
        # Count all transitions (vectorized)
        trans_counts = trips.groupby(["from_idx", "to_idx"]).size().reset_index(name="count")
        dep_counts = trips.groupby("from_idx").size()

        # Compute probabilities
        trans_counts["prob"] = trans_counts.apply(
            lambda row: row["count"] / dep_counts.get(row["from_idx"], 1), axis=1
        )

        # Build sparse matrix
        self.global_transition_matrix = sparse.csr_matrix(
            (
                trans_counts["prob"].values,
                (
                    trans_counts["from_idx"].values.astype(int),
                    trans_counts["to_idx"].values.astype(int),
                ),
            ),
            shape=(n_stations, n_stations),
        )

        # Global departure rates as array
        n_hours = trips["started_at"].dt.floor("h").nunique()
        self.global_departure_rates = np.zeros(n_stations)
        for idx, count in dep_counts.items():
            self.global_departure_rates[int(idx)] = count / max(n_hours, 1)

    def _compute_avg_sparsity(self) -> float:
        """Compute average sparsity of transition matrices."""
        if not self.transition_matrices:
            return 1.0

        n_stations = len(self.stations)
        total_possible = n_stations * n_stations

        sparsities = []
        for P in self.transition_matrices.values():
            n_nonzero = P.nnz
            sparsity = 1 - (n_nonzero / total_possible)
            sparsities.append(sparsity)

        return np.mean(sparsities)

    def _simulate_one_walk(
        self,
        initial_inventory: np.ndarray,
        times: pd.DatetimeIndex,
        rng: np.random.Generator,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Simulate a single random walk.

        Args:
            initial_inventory: Starting bike counts (array)
            times: Timestamps to simulate
            rng: Random number generator
            deterministic: If True, use expected values instead of sampling

        Returns:
            Array of shape (n_stations, n_times) with inventory trajectory
        """
        n_stations = len(self.stations)
        n_times = len(times)

        # Initialize trajectory
        trajectory = np.zeros((n_stations, n_times))
        trajectory[:, 0] = initial_inventory

        current = initial_inventory.copy()

        for t_idx in range(1, n_times):
            t = times[t_idx]
            hour = t.hour
            is_weekend = t.dayofweek in [5, 6]
            context = (hour, is_weekend)

            # Get transition matrix and departure rates
            P = self.transition_matrices.get(context, self.global_transition_matrix)
            dep_rates = self.departure_rate_arrays.get(context, self.global_departure_rates)

            if P is None:
                trajectory[:, t_idx] = current
                continue

            if deterministic:
                # --- DETERMINISTIC: Use expected values ---
                # Departures = rate, capped at available
                departures = np.minimum(dep_rates, current)

                # Arrivals = P.T @ departures (expected arrivals)
                arrivals = P.T @ departures

            else:
                # --- STOCHASTIC: Sample from distributions ---
                # Expected departures scaled by availability
                availability_factor = current / np.maximum(self.capacity_array, 1)
                expected_departures = dep_rates * availability_factor

                # Sample actual departures (Poisson)
                departures = np.minimum(
                    rng.poisson(expected_departures), current.astype(int)
                ).astype(float)

                # Route departures through transition matrix
                arrivals = np.zeros(n_stations)
                for i in range(n_stations):
                    n_depart = int(departures[i])
                    if n_depart > 0:
                        probs = P.getrow(i).toarray().flatten()
                        prob_sum = probs.sum()
                        if prob_sum > 0:
                            probs = probs / prob_sum
                            destinations = rng.choice(n_stations, size=n_depart, p=probs)
                            for dest in destinations:
                                arrivals[dest] += 1

            # --- UPDATE INVENTORY ---
            new_inventory = current - departures + arrivals

            # Clip to [0, capacity]
            new_inventory = np.clip(new_inventory, 0, self.capacity_array)

            current = new_inventory
            trajectory[:, t_idx] = current

        return trajectory

    def predict_inventory(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict inventory using Monte Carlo random walks.

        Runs N simulations and averages the results.

        Args:
            initial_inventory: Starting bike count per station
            start_time: Start time for prediction
            end_time: End time for prediction
            freq: Time frequency

        Returns:
            DataFrame with predicted inventory (mean over simulations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Generate time periods
        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        n_times = len(times)
        n_stations = len(self.stations)

        # Convert initial inventory to array (aligned with self.stations)
        init_array = np.array([initial_inventory.get(s, 0) for s in self.stations], dtype=float)

        # Run simulations
        all_trajectories = np.zeros((self.n_simulations, n_stations, n_times))
        rng = np.random.default_rng(seed=self.random_seed)

        # Use deterministic mode for single simulation
        deterministic = self.n_simulations == 1

        for sim in range(self.n_simulations):
            all_trajectories[sim] = self._simulate_one_walk(init_array, times, rng, deterministic)

        # Average over simulations
        mean_trajectory = all_trajectories.mean(axis=0)

        # Convert to DataFrame
        predictions = pd.DataFrame(
            mean_trajectory,
            index=self.stations,
            columns=times,
        )

        return predictions

    def predict_with_uncertainty(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Predict inventory with uncertainty bounds.

        Returns mean, lower (5th percentile), and upper (95th percentile).
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        n_times = len(times)
        n_stations = len(self.stations)

        init_array = np.array([initial_inventory.get(s, 0) for s in self.stations], dtype=float)

        all_trajectories = np.zeros((self.n_simulations, n_stations, n_times))
        rng = np.random.default_rng(seed=self.random_seed)

        for sim in range(self.n_simulations):
            all_trajectories[sim] = self._simulate_one_walk(init_array, times, rng)

        mean_traj = all_trajectories.mean(axis=0)
        lower_traj = np.percentile(all_trajectories, 5, axis=0)
        upper_traj = np.percentile(all_trajectories, 95, axis=0)

        mean_df = pd.DataFrame(mean_traj, index=self.stations, columns=times)
        lower_df = pd.DataFrame(lower_traj, index=self.stations, columns=times)
        upper_df = pd.DataFrame(upper_traj, index=self.stations, columns=times)

        return mean_df, lower_df, upper_df

    def predict_flow(
        self,
        stations: list,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict net flow (legacy method, kept for compatibility)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        known_stations = [s for s in stations if s in self.station_to_idx]

        predictions = pd.DataFrame(0.0, index=stations, columns=times)

        for t in times:
            hour = t.hour
            is_weekend = t.dayofweek in [5, 6]
            context = (hour, is_weekend)

            P = self.transition_matrices.get(context, self.global_transition_matrix)
            dep_rates = self.departure_rate_arrays.get(context, self.global_departure_rates)

            if P is None:
                continue

            arrivals_vector = P.T @ dep_rates

            for station in known_stations:
                idx = self.station_to_idx[station]
                departures = dep_rates[idx]
                arrivals = arrivals_vector[idx]
                predictions.loc[station, t] = arrivals - departures

        return predictions

    def get_transition_matrix(self, hour: int, is_weekend: bool) -> tuple[sparse.csr_matrix, list]:
        """Get transition matrix for a specific context."""
        context = (hour, is_weekend)
        P = self.transition_matrices.get(context, self.global_transition_matrix)
        return P, self.stations

    def get_top_destinations(
        self,
        station: str,
        hour: int,
        is_weekend: bool,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Get top destinations from a station for a given context."""
        if station not in self.station_to_idx:
            return pd.DataFrame()

        context = (hour, is_weekend)
        P = self.transition_matrices.get(context, self.global_transition_matrix)

        if P is None:
            return pd.DataFrame()

        station_idx = self.station_to_idx[station]
        row = P.getrow(station_idx).toarray().flatten()

        top_indices = np.argsort(row)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if row[idx] > 0:
                results.append(
                    {
                        "destination": self.idx_to_station[idx],
                        "probability": row[idx],
                    }
                )

        return pd.DataFrame(results)

    def get_params(self) -> dict[str, Any]:
        """Return model parameters."""
        return {
            "name": self.get_name(),
            "n_stations": len(self.stations),
            "n_transition_matrices": len(self.transition_matrices),
            "avg_sparsity": self._compute_avg_sparsity() if self.transition_matrices else None,
            "smoothing_alpha": self.smoothing_alpha,
            "n_simulations": self.n_simulations,
            "random_seed": self.random_seed,
        }
