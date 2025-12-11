"""Base model class defining the interface for inventory prediction."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any


class BaseModel(ABC):
    """Abstract base class for all bike inventory prediction models.
    
    All models must implement:
    - fit(): Train the model on historical data
    - predict_inventory(): Predict future bike counts per station
    """
    
    def __init__(self, config: dict):
        """Initialize model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_fitted = False
        self.station_capacities = {}
    
    @abstractmethod
    def fit(
        self,
        trips: pd.DataFrame,
        station_stats: pd.DataFrame,
    ) -> "BaseModel":
        """Train the model on historical trip data.
        
        Args:
            trips: DataFrame with columns [started_at, ended_at, 
                   start_station_name, end_station_name, ...]
            station_stats: DataFrame indexed by station_name with capacity
            
        Returns:
            self (for method chaining)
        """
        pass
    
    @abstractmethod
    def predict_inventory(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict bike inventory at each station over time.
        
        Args:
            initial_inventory: Series indexed by station_name with starting bike counts
            start_time: Start of prediction period
            end_time: End of prediction period
            freq: Time frequency (e.g., "1h" for hourly)
            
        Returns:
            DataFrame with predictions:
                - index: station_name
                - columns: timestamps
                - values: predicted bike counts
        """
        pass
    
    def predict_states(
        self,
        initial_inventory: pd.Series,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict station states (empty/normal/full) over time.
        
        Args:
            initial_inventory: Starting bike counts per station
            start_time: Start of prediction period
            end_time: End of prediction period
            freq: Time frequency
            
        Returns:
            DataFrame with state predictions ("empty", "normal", "full")
        """
        # Get inventory predictions
        inventory = self.predict_inventory(initial_inventory, start_time, end_time, freq)
        
        # Get thresholds
        thresholds = self.config.get("thresholds", {"empty": 0.1, "full": 0.9})
        
        # Convert to states
        states = pd.DataFrame(index=inventory.index, columns=inventory.columns, dtype=str)
        
        for station in inventory.index:
            capacity = self.station_capacities.get(station, 30)
            empty_threshold = capacity * thresholds["empty"]
            full_threshold = capacity * thresholds["full"]
            
            for col in inventory.columns:
                bikes = inventory.loc[station, col]
                if bikes <= empty_threshold:
                    states.loc[station, col] = "empty"
                elif bikes >= full_threshold:
                    states.loc[station, col] = "full"
                else:
                    states.loc[station, col] = "normal"
        
        return states
    
    def get_name(self) -> str:
        """Return the model name."""
        return self.__class__.__name__
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for logging."""
        return {"name": self.get_name()}
