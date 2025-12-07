"""Models for CitiBike flow prediction."""

from .base import BaseModel
from .baseline import BaselineModel
from .markov import MarkovModel

__all__ = [
    "BaseModel",
    "BaselineModel",
    "MarkovModel",
]


def get_model(name: str, config: dict) -> BaseModel:
    """Factory function to get model by name.
    
    Args:
        name: Model name ("baseline", "markov", etc.)
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    models = {
        "baseline": BaselineModel,
        "markov": MarkovModel,
    }
    
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    
    return models[name](config)

