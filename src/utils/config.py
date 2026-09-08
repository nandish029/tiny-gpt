import yaml
import os

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
        
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    if config is None:
        return {}
        
    return config
