"""Models for CitiBike inventory prediction."""

from .base import BaseModel
from .baseline import TemporalFlowModel
from .naive import PersistenceModel, StationAverageModel
from .markov import MarkovModel

# Alias for backwards compatibility
BaselineModel = TemporalFlowModel

__all__ = [
    "BaseModel",
    "TemporalFlowModel",
    "BaselineModel",  # Alias
    "PersistenceModel",
    "StationAverageModel",
    "MarkovModel",
]


def get_model(name: str, config: dict) -> BaseModel:
    """Factory function to get model by name.
    
    Args:
        name: Model name
        config: Configuration dictionary
        
    Returns:
        Model instance
    
    Available models:
        - "persistence": Predicts inventory stays constant
        - "station_avg": Uses station-only average (no time patterns)
        - "temporal_flow" / "baseline": Time-conditioned flow model
        - "markov": Markov chain with transition matrices
    """
    models = {
        # Simple baselines
        "persistence": PersistenceModel,
        "station_avg": StationAverageModel,
        
        # Main models
        "temporal_flow": TemporalFlowModel,
        "baseline": TemporalFlowModel,  # Alias for backwards compat
        "markov": MarkovModel,
    }
    
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name](config)
