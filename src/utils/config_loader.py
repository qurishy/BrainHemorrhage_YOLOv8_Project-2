"""
Configuration loader utility.
Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration manager with dot notation access."""

    def __init__(self, config_dict: Dict[str, Any]):
        if not isinstance(config_dict, dict):
            raise TypeError(f"Config must be initialized with a dict, got {type(config_dict)}")
        self._config = config_dict

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            return object.__getattribute__(self, name)

        if name not in self._config:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'. "
                f"Available keys: {list(self._config.keys())}"
            )

        value = self._config[name]

        # If value is a dict, wrap it in Config for dot notation
        if isinstance(value, dict):
            return Config(value)

        return value

    def __getitem__(self, key: str) -> Any:
        if key not in self._config:
            raise KeyError(f"'{key}' not found. Available keys: {list(self._config.keys())}")

        value = self._config[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._config:
            return default

        value = self._config[key]
        if isinstance(value, dict):
            return Config(value)
        return value

    def items(self):
        """Return dict items for iteration."""
        return self._config.items()

    def keys(self):
        """Return dict keys."""
        return self._config.keys()

    def values(self):
        """Return dict values."""
        return self._config.values()

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to plain dictionary."""
        return self._config

    def __repr__(self):
        return f"Config({self._config})"

    def __contains__(self, key):
        return key in self._config


def load_config(config_path: str = "configs/config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with dot notation access
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(config_dict)}")

    return Config(config_dict)


def save_config(config_dict: Dict[str, Any], save_path: str):
    """Save configuration dictionary to YAML file."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
