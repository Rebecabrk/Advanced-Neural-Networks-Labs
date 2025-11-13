import yaml
import sys

from utils.constants import DEFAULT_CONFIG_PATH

def load_config(config_path):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration successfully loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        print("Please ensure your file is in the correct location.")
        return None
    except yaml.YAMLError:
        print(f"Error: Invalid YAML format in {config_path}.")
        return None
    except Exception as e:
        print(f"Error: Could not load configuration from {config_path} — {e}")
        print("Please check the file path, permissions, and that the file is a readable YAML file.")
        return None

def get_argfile_config(args):
    if args.config:
        print(f"Attempting to load custom config from: {args.config}")
        config = load_config(args.config)
        if config:
            print("Loaded custom configuration.")
            return config
        else:
            print("Failed to load custom configuration.")
            sys.exit(1)
    else:
        print("\n*****************************************************************")
        print("  PyTorch Training Pipeline - Author: Rebeca")
        print("  No config file provided. Attempting to load default config.")
        print("*****************************************************************")

        config = load_config(DEFAULT_CONFIG_PATH)

        if config:
            print(f"Using default configuration loaded from {DEFAULT_CONFIG_PATH}.")
            return config
        else:
            # Fallback if the default file is missing/broken (Highly discouraged in production)
            print("WARNING: Could not load default config file.")
            sys.exit(1)

def save_config_to_yaml(config, save_path):
    try:
        with open(save_path, 'w') as f:
            yaml.dump(config, f)
        print(f"Configuration successfully saved to {save_path}")
    except Exception as e:
        print(f"Error: Could not save configuration to {save_path} — {e}")
        raise