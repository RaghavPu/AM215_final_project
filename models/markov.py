"""Markov Chain model for bike flow prediction."""

import pandas as pd
import numpy as np
from scipy import sparse
from typing import Dict, Any, Tuple
from .base import BaseModel


class MarkovModel(BaseModel):
    """Markov Chain model for predicting bike station flows.
    
    This model:
    1. Builds time-dependent transition matrices P[i→j | hour, is_weekend]
    2. Estimates departure rates per station per time context
    3. Predicts flow by routing departures through the transition matrix
    
    Key insight: Models WHERE bikes go, not just how many leave.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Model parameters
        model_config = config.get("model", {}).get("markov", {})
        self.smoothing_alpha = model_config.get("smoothing_alpha", 0.0)
        self.min_transitions = model_config.get("min_transitions", 1)
        
        # Learned parameters
        self.stations = []
        self.station_to_idx = {}
        self.idx_to_station = {}
        
        # Transition matrices: (hour, is_weekend) -> sparse matrix
        self.transition_matrices = {}
        
        # Departure rates: (hour, is_weekend) -> {station: rate}
        self.departure_rates = {}
        
        # Fallback for missing contexts
        self.global_transition_matrix = None
        self.global_departure_rates = {}
        
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
        self.idx_to_station = {i: s for i, s in enumerate(self.stations)}
        n_stations = len(self.stations)
        
        print(f"  Building transition matrices for {n_stations} stations...")
        
        # Prepare data
        trips = trips.copy()
        if "hour" not in trips.columns:
            trips["hour"] = trips["started_at"].dt.hour
        if "is_weekend" not in trips.columns:
            trips["is_weekend"] = trips["started_at"].dt.dayofweek.isin([5, 6])
        
        # Filter to known stations
        trips = trips[
            trips["start_station_name"].isin(self.station_to_idx) &
            trips["end_station_name"].isin(self.station_to_idx)
        ].copy()
        
        # Map station names to indices (vectorized)
        trips["from_idx"] = trips["start_station_name"].map(self.station_to_idx)
        trips["to_idx"] = trips["end_station_name"].map(self.station_to_idx)
        
        # Count days per context for rate normalization
        trips["date"] = trips["started_at"].dt.date
        days_per_context = trips.groupby(["hour", "is_weekend"])["date"].nunique()
        
        # Build all transition counts at once using groupby (FAST!)
        print("  Counting transitions (vectorized)...")
        transition_counts = trips.groupby(
            ["hour", "is_weekend", "from_idx", "to_idx"]
        ).size().reset_index(name="count")
        
        departure_counts = trips.groupby(
            ["hour", "is_weekend", "from_idx"]
        ).size().reset_index(name="departures")
        
        # Build transition matrices for each context
        print("  Building sparse matrices...")
        contexts = [(h, w) for h in range(24) for w in [False, True]]
        
        for hour, is_weekend in contexts:
            # Filter to this context
            ctx_trans = transition_counts[
                (transition_counts["hour"] == hour) & 
                (transition_counts["is_weekend"] == is_weekend)
            ]
            ctx_deps = departure_counts[
                (departure_counts["hour"] == hour) & 
                (departure_counts["is_weekend"] == is_weekend)
            ]
            
            if len(ctx_trans) < self.min_transitions:
                continue
            
            # Get total departures per origin
            dep_totals = ctx_deps.set_index("from_idx")["departures"]
            
            # Compute probabilities
            ctx_trans = ctx_trans.copy()
            ctx_trans["prob"] = ctx_trans.apply(
                lambda row: row["count"] / dep_totals.get(row["from_idx"], 1),
                axis=1
            )
            
            # Create sparse matrix
            P = sparse.csr_matrix(
                (ctx_trans["prob"].values, 
                 (ctx_trans["from_idx"].values, ctx_trans["to_idx"].values)),
                shape=(n_stations, n_stations)
            )
            
            self.transition_matrices[(hour, is_weekend)] = P
            
            # Compute departure rates
            n_days = days_per_context.get((hour, is_weekend), 1)
            self.departure_rates[(hour, is_weekend)] = {
                self.idx_to_station[int(row["from_idx"])]: row["departures"] / n_days
                for _, row in ctx_deps.iterrows()
            }
        
        # Build global fallback
        self._build_global_fallback(trips, n_stations)
        
        self.is_fitted = True
        print(f"  Built {len(self.transition_matrices)} transition matrices")
        print(f"  Sparsity: {self._compute_avg_sparsity():.1%}")
        
        return self
    
    def _build_global_fallback(self, trips: pd.DataFrame, n_stations: int):
        """Build a global transition matrix as fallback."""
        # Count all transitions (vectorized)
        trans_counts = trips.groupby(["from_idx", "to_idx"]).size().reset_index(name="count")
        dep_counts = trips.groupby("from_idx").size()
        
        # Compute probabilities
        trans_counts["prob"] = trans_counts.apply(
            lambda row: row["count"] / dep_counts.get(row["from_idx"], 1),
            axis=1
        )
        
        # Build sparse matrix
        self.global_transition_matrix = sparse.csr_matrix(
            (trans_counts["prob"].values,
             (trans_counts["from_idx"].values, trans_counts["to_idx"].values)),
            shape=(n_stations, n_stations)
        )
        
        # Global departure rates
        n_hours = trips["started_at"].dt.floor("h").nunique()
        self.global_departure_rates = {
            self.idx_to_station[idx]: count / max(n_hours, 1)
            for idx, count in dep_counts.items()
        }
    
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
    
    def predict_flow(
        self,
        stations: list,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict net flow using Markov transition matrices.
        
        For each station i:
        - departures[i] = historical departure rate
        - arrivals[i] = Σ_j (departures[j] × P[j→i])
        - net_flow[i] = arrivals[i] - departures[i]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate time periods
        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        
        # Filter to known stations
        known_stations = [s for s in stations if s in self.station_to_idx]
        station_indices = [self.station_to_idx[s] for s in known_stations]
        
        # Initialize predictions
        predictions = pd.DataFrame(
            0.0,
            index=stations,
            columns=times,
        )
        
        # Predict for each time period
        for t in times:
            hour = t.hour
            is_weekend = t.dayofweek in [5, 6]
            context = (hour, is_weekend)
            
            # Get transition matrix and departure rates
            P = self.transition_matrices.get(context, self.global_transition_matrix)
            dep_rates = self.departure_rates.get(context, self.global_departure_rates)
            
            if P is None:
                continue
            
            # Build departure vector for all stations
            dep_vector = np.zeros(len(self.stations))
            for station, rate in dep_rates.items():
                if station in self.station_to_idx:
                    dep_vector[self.station_to_idx[station]] = rate
            
            # Compute arrivals: arrivals = P.T @ departures
            # (each column of P.T tells us where arrivals come from)
            arrivals_vector = P.T @ dep_vector
            
            # Net flow for each station
            for station in known_stations:
                idx = self.station_to_idx[station]
                departures = dep_vector[idx]
                arrivals = arrivals_vector[idx]
                predictions.loc[station, t] = arrivals - departures
        
        return predictions
    
    def get_transition_matrix(
        self, 
        hour: int, 
        is_weekend: bool
    ) -> Tuple[sparse.csr_matrix, list]:
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
        
        # Get top destinations
        top_indices = np.argsort(row)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if row[idx] > 0:
                results.append({
                    "destination": self.idx_to_station[idx],
                    "probability": row[idx],
                })
        
        return pd.DataFrame(results)
    
    def predict_inventory(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict inventory using Markov model.
        
        TODO: Implement state-dependent bike simulation.
        For now, uses a simple flow-based approach.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Get stations from initial inventory
        stations = initial_inventory.index.tolist()
        
        # Generate time periods
        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        
        # Initialize predictions
        predictions = pd.DataFrame(
            index=stations,
            columns=times,
            dtype=float
        )
        
        # Set initial state
        predictions[times[0]] = initial_inventory
        
        # Get flow predictions
        flow_pred = self.predict_flow(stations, start_time, end_time, freq)
        
        # Simulate forward
        current_inventory = initial_inventory.copy()
        
        for i, t in enumerate(times[1:], 1):
            # Apply predicted flow
            new_inventory = current_inventory.copy()
            
            for station in stations:
                net_flow = flow_pred.loc[station, t] if station in flow_pred.index and t in flow_pred.columns else 0
                
                # Update inventory
                new_bikes = current_inventory.get(station, 0) + net_flow
                
                # Clamp to valid range [0, capacity]
                capacity = self.station_capacities.get(station, 30)
                new_inventory[station] = np.clip(new_bikes, 0, capacity)
            
            predictions[t] = new_inventory
            current_inventory = new_inventory
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            "name": self.get_name(),
            "n_stations": len(self.stations),
            "n_transition_matrices": len(self.transition_matrices),
            "avg_sparsity": self._compute_avg_sparsity() if self.transition_matrices else None,
            "smoothing_alpha": self.smoothing_alpha,
        }
