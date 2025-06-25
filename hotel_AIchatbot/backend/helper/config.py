import json
import os

def load_config():
    """Load configuration from config.json file."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Error loading config: {e}")