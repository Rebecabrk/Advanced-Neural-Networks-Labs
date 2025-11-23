import os
import sys
import argparse
from pathlib import Path
from rich import print as rprint
from rich.pretty import Pretty

from utils.custom_config import get_commandline_config
from utils.config_loader import get_argfile_config
from pipeline import run_pipeline
from utils.config_loader import save_config_to_yaml

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
    config = get_argfile_config(args)

    if not args.config:
        print("\n--- Default Configuration  ---")
        rprint(Pretty(config))
        choice = input("\nDo you want to use this [D]efault configuration or set a [C]ustom configuration interactively? (D/C): ").strip().lower()
        
        if choice == 'c':
            config = get_commandline_config(config)
            
        elif choice == 'd':
            print("\n Starting pipeline with loaded DEFAULT configuration...")
            
        else:
            print("Invalid choice. Exiting pipeline.")
            sys.exit(1)
    else:
        print("\n--- Final Configuration ---")
        rprint(Pretty(config))
        start_choice = input("\n Would you like to start training with the above configuration? (y/n): ").strip().lower()
        if start_choice == 'n':
            print("Training aborted by user.")
            sys.exit(0)

    if not args.config:
        save_choice = input("\nDo you want to save this configuration to a YAML file? (y/n): ").strip().lower()
        if save_choice == 'y':
            filename = input("Enter name/small description for the configuration: ").strip()
            save_path = input("Enter the file path to save the configuration (Default: /saved-configurations)): ").strip()
            if not save_path:
                root = os.getcwd()
                save_path = os.path.join(root, "saved-configurations", "{}_config.yaml".format(filename.replace(" ", "_")))
            try:
                save_config_to_yaml(config, save_path)
                print(f"Configuration saved to {save_path}")
            except Exception as e:
                print(f"Failed to save configuration: {e}")

    try:
        run_pipeline(config)
    except Exception as e:
        print(f"Pipeline setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
    # activate virtual environment
    # source venv/bin/activate