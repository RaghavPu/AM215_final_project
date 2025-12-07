"""Baseline model using historical average flow to predict inventory."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseModel


class BaselineModel(BaseModel):
    """Simple baseline model using historical average net flow.
    
    This model:
    1. Computes average hourly net flow (arrivals - departures) per station
    2. Predicts future inventory by applying average flow to initial state:
       inventory[t+1] = inventory[t] + avg_net_flow
    
    This serves as a baseline to compare more sophisticated models against.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.hourly_net_flow = {}  # (station, hour, is_weekend) -> avg net flow
        self.global_avg_flow = 0.0
        self.stations = []
        
    def fit(
        self,
        trips: pd.DataFrame,
        station_stats: pd.DataFrame,
    ) -> "BaseModel":
        """Compute historical average hourly net flow per station.
        
        Args:
            trips: Trip data with start/end stations and timestamps
            station_stats: Station info with capacity
        """
        print(f"Fitting {self.get_name()} on {len(trips):,} trips...")
        
        # Store stations and capacities
        self.stations = station_stats.index.tolist()
        self.station_capacities = station_stats["capacity"].to_dict()
        
        # Ensure time features exist
        trips = trips.copy()
        if "hour" not in trips.columns:
            trips["hour"] = trips["started_at"].dt.hour
        if "is_weekend" not in trips.columns:
            trips["is_weekend"] = trips["started_at"].dt.dayofweek.isin([5, 6])
        
        # Compute departures per (station, hour, is_weekend)
        departures = (
            trips.groupby(["start_station_name", "hour", "is_weekend"])
            .size()
            .reset_index(name="departures")
        )
        departures.columns = ["station", "hour", "is_weekend", "departures"]
        
        # Compute arrivals per (station, hour, is_weekend)
        arrivals = (
            trips.groupby(["end_station_name", "hour", "is_weekend"])
            .size()
            .reset_index(name="arrivals")
        )
        arrivals.columns = ["station", "hour", "is_weekend", "arrivals"]
        
        # Merge to get net flow
        flow_df = departures.merge(
            arrivals,
            on=["station", "hour", "is_weekend"],
            how="outer"
        ).fillna(0)
        
        flow_df["net_flow"] = flow_df["arrivals"] - flow_df["departures"]
        
        # Count number of each type of day to get average per hour
        date_info = trips.groupby(trips["started_at"].dt.date).agg({
            "is_weekend": "first"
        })
        n_weekdays = (~date_info["is_weekend"]).sum()
        n_weekend_days = date_info["is_weekend"].sum()
        
        # Normalize by number of days to get average flow per hour
        def normalize_flow(row):
            n_days = n_weekend_days if row["is_weekend"] else n_weekdays
            return row["net_flow"] / max(n_days, 1)
        
        flow_df["avg_net_flow"] = flow_df.apply(normalize_flow, axis=1)
        
        # Store as lookup dictionary
        self.hourly_net_flow = {}
        for _, row in flow_df.iterrows():
            key = (row["station"], int(row["hour"]), bool(row["is_weekend"]))
            self.hourly_net_flow[key] = row["avg_net_flow"]
        
        # Compute global average as fallback
        self.global_avg_flow = flow_df["avg_net_flow"].mean()
        
        self.is_fitted = True
        print(f"Fitted model with {len(self.hourly_net_flow)} (station, hour, weekend) combinations")
        
        return self
    
    def predict_inventory(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict inventory by applying average hourly net flow.
        
        Args:
            initial_inventory: Starting bike count per station
            start_time: Start time for prediction
            end_time: End time for prediction
            freq: Time frequency
            
        Returns:
            DataFrame with predicted inventory at each hour
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate time periods
        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        
        # Get stations from initial inventory
        stations = initial_inventory.index.tolist()
        
        # Initialize predictions
        predictions = pd.DataFrame(
            index=stations,
            columns=times,
            dtype=float
        )
        
        # Set initial state
        predictions[times[0]] = initial_inventory
        
        # Simulate forward
        current_inventory = initial_inventory.copy()
        
        for i, t in enumerate(times[1:], 1):
            hour = t.hour
            is_weekend = t.dayofweek in [5, 6]
            
            # Apply net flow for each station
            new_inventory = current_inventory.copy()
            
            for station in stations:
                key = (station, hour, is_weekend)
                net_flow = self.hourly_net_flow.get(key, self.global_avg_flow)
                
                # Update inventory
                new_bikes = current_inventory[station] + net_flow
                
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
            "n_flow_combinations": len(self.hourly_net_flow) if self.hourly_net_flow else 0,
            "global_avg_flow": self.global_avg_flow,
        }
