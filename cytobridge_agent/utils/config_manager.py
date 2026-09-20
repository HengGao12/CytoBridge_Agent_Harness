import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from .skill_store import ensure_cellcompass_skills_migrated

# Path to the global config file for cellcompass
CONFIG_DIR = Path.home() / ".cellcompass"
CONFIG_FILE = CONFIG_DIR / "config.json"

def get_saved_config() -> dict:
    """Read the saved configuration from ~/.cellcompass/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_cellcompass_skills_migrated()
    if not CONFIG_FILE.exists():
        return {}
        
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        import logging
        logging.getLogger("cytobridge_agent.config_manager").warning(
            f"Failed to read config file {CONFIG_FILE}: {e}"
        )
        return {}

def save_config(new_config: dict) -> None:
    """Save the configuration to ~/.cellcompass/config.json."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    ensure_cellcompass_skills_migrated()
    
    # Read existing to merge, avoiding overwriting other keys if any
    current_config = get_saved_config()
    current_config.update(new_config)
    
    try:
        with NamedTemporaryFile("w", dir=str(CONFIG_DIR), delete=False) as tmp:
            json.dump(current_config, tmp, indent=4)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, CONFIG_FILE)
        os.chmod(CONFIG_FILE, 0o600)
    except Exception as e:
        import logging
        logging.getLogger("cytobridge_agent.config_manager").error(
            f"Failed to write config file {CONFIG_FILE}: {e}"
        )
