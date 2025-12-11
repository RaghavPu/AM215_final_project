"""Helper utilities."""

from pathlib import Path

import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent
