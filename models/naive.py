"""Simple naive baseline models for comparison."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from .base import BaseModel


class PersistenceModel(BaseModel):
    """Persistence (Naive) baseline - predicts inventory stays constant.
    
    inventory[t+1] = inventory[t] = initial_inventory
    
    This is the simplest possible baseline. Any useful model should beat this.
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def fit(
        self,
        trips: pd.DataFrame,
        station_stats: pd.DataFrame,
    ) -> "BaseModel":
        """No training needed - just store station info."""
        print(f"Fitting {self.get_name()} (no-op)...")
        
        self.stations = station_stats.index.tolist()
        self.station_capacities = station_stats["capacity"].to_dict()
        self.is_fitted = True
        
        print(f"  Ready to predict for {len(self.stations)} stations")
        return self
    
    def predict_inventory(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict inventory stays constant at initial state."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        stations = initial_inventory.index.tolist()
        
        # All time periods have the same inventory as initial
        predictions = pd.DataFrame(
            index=stations,
            columns=times,
            dtype=float
        )
        
        for t in times:
            predictions[t] = initial_inventory
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        return {"name": self.get_name()}


class StationAverageModel(BaseModel):
    """Station-only average baseline - ignores temporal patterns.
    
    Learns average net flow per station (across all hours/days),
    then applies it uniformly.
    
    inventory[t+1] = inventory[t] + station_avg_flow
    
    This shows the value of temporal conditioning (hour, weekend).
    """
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.station_avg_flow = {}  # station -> average hourly net flow
        self.global_avg_flow = 0.0
        
    def fit(
        self,
        trips: pd.DataFrame,
        station_stats: pd.DataFrame,
    ) -> "BaseModel":
        """Compute average net flow per station (ignoring time)."""
        print(f"Fitting {self.get_name()} on {len(trips):,} trips...")
        
        self.stations = station_stats.index.tolist()
        self.station_capacities = station_stats["capacity"].to_dict()
        
        # Count total departures per station
        departures = trips.groupby("start_station_name").size()
        
        # Count total arrivals per station
        arrivals = trips.groupby("end_station_name").size()
        
        # Net flow per station (total over training period)
        net_flow = arrivals.subtract(departures, fill_value=0)
        
        # Count number of hours in training period
        trips["hour_bucket"] = trips["started_at"].dt.floor("1h")
        n_hours = trips["hour_bucket"].nunique()
        
        # Average hourly net flow per station
        self.station_avg_flow = (net_flow / max(n_hours, 1)).to_dict()
        
        # Global fallback
        self.global_avg_flow = net_flow.mean() / max(n_hours, 1)
        
        self.is_fitted = True
        print(f"  Computed avg flow for {len(self.station_avg_flow)} stations")
        print(f"  Global avg flow: {self.global_avg_flow:.4f} bikes/hour")
        
        return self
    
    def predict_inventory(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict inventory using station average flow (no time patterns)."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        times = pd.date_range(start=start_time, end=end_time, freq=freq, inclusive="left")
        stations = initial_inventory.index.tolist()
        
        predictions = pd.DataFrame(
            index=stations,
            columns=times,
            dtype=float
        )
        
        # Set initial state
        predictions[times[0]] = initial_inventory
        
        # Simulate forward with constant flow per station
        current_inventory = initial_inventory.copy()
        
        for i, t in enumerate(times[1:], 1):
            new_inventory = current_inventory.copy()
            
            for station in stations:
                avg_flow = self.station_avg_flow.get(station, self.global_avg_flow)
                
                # Update inventory
                new_bikes = current_inventory[station] + avg_flow
                
                # Clamp to valid range
                capacity = self.station_capacities.get(station, 30)
                new_inventory[station] = np.clip(new_bikes, 0, capacity)
            
            predictions[t] = new_inventory
            current_inventory = new_inventory
        
        return predictions
    
    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.get_name(),
            "n_stations": len(self.station_avg_flow),
            "global_avg_flow": self.global_avg_flow,
        }

