import argparse
import sys
import yaml
from pathlib import Path

from rich import print as rprint
from rich.pretty import Pretty

from utils.custom_config import get_custom_config
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

def get_config(args):
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

def main():
    parser = argparse.ArgumentParser(
        description="""
        PyTorch Training Pipeline. Author: Rebeca
        """
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a custom YAML configuration file to load settings."
    )
    args = parser.parse_args()
    config = get_config(args)

    print("\n--- Final Configuration Used ---")
    rprint(Pretty(config))
    
    if not args.config:
        choice = input("\nDo you want to use this [D]efault configuration or set a [C]ustom configuration interactively? (D/C): ").strip().lower()
        
        if choice == 'c':
            config = get_custom_config(config)
            
        elif choice == 'd':
            print("\n Starting pipeline with loaded DEFAULT configuration...")
            
        else:
            print("Invalid choice. Exiting pipeline.")
            sys.exit(1)


    print("\n(Pipeline execution finished. Training would start with the above configuration.)")


if __name__ == "__main__":
    main()