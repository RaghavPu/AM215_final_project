"""Baseline model using historical average flow."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseModel


class BaselineModel(BaseModel):
    """Simple baseline model using historical average net flow.
    
    This model:
    1. Computes average hourly net flow (arrivals - departures) per station
    2. Predicts future flow by returning the historical average for that 
       (station, hour, is_weekend) combination
    
    This serves as a baseline to compare more sophisticated models against.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.hourly_net_flow = None  # Average net flow by (station, hour, is_weekend)
        self.global_avg_flow = 0.0  # Fallback for missing combinations
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
        
        # Store list of stations
        self.stations = station_stats.index.tolist()
        
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
        
        # Store as lookup dictionary: (station, hour, is_weekend) -> avg_net_flow
        self.hourly_net_flow = {}
        for _, row in flow_df.iterrows():
            key = (row["station"], int(row["hour"]), bool(row["is_weekend"]))
            self.hourly_net_flow[key] = row["avg_net_flow"]
        
        # Compute global average as fallback (should be ~0 for balanced system)
        self.global_avg_flow = flow_df["avg_net_flow"].mean()
        
        self.is_fitted = True
        print(f"Fitted model with {len(self.hourly_net_flow)} (station, hour, weekend) combinations")
        
        return self
    
    def predict_flow(
        self,
        stations: list,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict net flow by returning historical averages.
        
        Args:
            stations: List of station names
            start_time: Start of prediction period
            end_time: End of prediction period
            freq: Time frequency
            
        Returns:
            DataFrame with predicted net flow per station per time period
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate time periods
        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        
        # Initialize predictions
        predictions = pd.DataFrame(
            index=stations,
            columns=times,
            dtype=float
        )
        
        # Fill in predictions using historical averages
        for t in times:
            hour = t.hour
            is_weekend = t.dayofweek in [5, 6]
            
            for station in stations:
                key = (station, hour, is_weekend)
                predictions.loc[station, t] = self.hourly_net_flow.get(
                    key, self.global_avg_flow
                )
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return {
            "name": self.get_name(),
            "n_flow_combinations": len(self.hourly_net_flow) if self.hourly_net_flow else 0,
            "global_avg_flow": self.global_avg_flow,
        }
