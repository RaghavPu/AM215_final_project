"""Base model class defining the interface for flow prediction."""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any


class BaseModel(ABC):
    """Abstract base class for all flow prediction models.
    
    All models must implement:
    - fit(): Train the model on historical data
    - predict_flow(): Predict future net flow per station
    """
    
    def __init__(self, config: dict):
        """Initialize model with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_fitted = False
    
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
    def predict_flow(
        self,
        stations: list,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        freq: str = "1h",
    ) -> pd.DataFrame:
        """Predict net flow (arrivals - departures) per station per time period.
        
        Args:
            stations: List of station names to predict for
            start_time: Start of prediction period
            end_time: End of prediction period
            freq: Time frequency (e.g., "1h" for hourly)
            
        Returns:
            DataFrame with predictions:
                - index: station_name
                - columns: time periods
                - values: predicted net flow (positive = net arrivals)
        """
        pass
    
    def get_name(self) -> str:
        """Return the model name."""
        return self.__class__.__name__
    
    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for logging."""
        return {"name": self.get_name()}
