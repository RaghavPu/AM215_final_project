"""Helper utilities."""

import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent

