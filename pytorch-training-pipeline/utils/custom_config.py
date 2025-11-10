import yaml
from rich import print as rprint
from rich.pretty import Pretty

from utils.validation import get_validated_input
from utils.constants import (
    VALID_DATASETS,
    VALID_MODELS,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    OPTIMIZER_PARAMS
)

def get_optimizer_params_interactively(optimizer_name, custom_cfg):
    """Prompts the user for specific parameters of the chosen optimizer."""
    
    print(f"\n  -- Specific {optimizer_name} Parameters --")
    optimizer_specific_params = OPTIMIZER_PARAMS.get(optimizer_name, {})
    
    # Initialize a new dictionary for specific params
    custom_cfg['optimizer'][f'{optimizer_name.lower()}_params'] = {}
    
    for param_name, param_details in optimizer_specific_params.items():
        
        # Determine the user-friendly prompt and type
        prompt = f"    Enter value for {param_name}"
        validator_type = param_details['validator']
        default_val = param_details['default']
        constraints = {}
        
        # Set constraints based on validator type
        if 'range' in param_details:
            constraints['range'] = param_details['range']
        elif 'valid_list' in param_details:
            constraints['valid_list'] = param_details['valid_list']

        # Get the validated input
        value = get_validated_input(
            prompt,
            validator_type,
            f"{param_name} validation failed.",
            default=default_val,
            constraints=constraints
        )
        
        # Store the result in the custom config
        custom_cfg['optimizer'][f'{optimizer_name.lower()}_params'][param_name] = value

    return custom_cfg

def get_custom_config(current_config):
    print("\n--- Starting Custom Configuration ---")
    custom_cfg = current_config.copy()

    # --- 1. DATASET CONFIGURATION ---
    print("\n[DATASET CONFIGURATION]")
    custom_cfg['data']['dataset_name'] = get_validated_input(
        f"Choose dataset {VALID_DATASETS}",
        'choice',
        "Invalid dataset choice",
        default=current_config['data']['dataset_name'],
        constraints={'valid_list': VALID_DATASETS}
    )
    custom_cfg['data']['data_augmentation'] = get_validated_input(
        "Enable Data Augmentation (True/False)",
        'boolean',
        "Invalid boolean value",
        default=current_config['data']['data_augmentation']
    )

    # --- 2. MODEL CONFIGURATION ---
    print("\n[MODEL CONFIGURATION]")
    custom_cfg['model']['name'] = get_validated_input(
        f"Choose model {VALID_MODELS}",
        'choice',
        "Invalid model choice",
        default=current_config['model']['name'],
        constraints={'valid_list': VALID_MODELS}
    )
    custom_cfg['model']['pretrained'] = get_validated_input(
        "Use pretrained weights (True/False)",
        'boolean',
        "Invalid boolean value",
        default=current_config['model']['pretrained']
    )

    # MLP Specific Parameters
    if custom_cfg['model']['name'] == 'MLP':
        print("\n  -- MLP Parameters (Simplified) --")
        custom_cfg['model']['mlp_params']['input_size'] = get_validated_input(
            "Enter MLP input size",
            'positive_int',
            "Input size must be a positive integer",
            default=current_config.get('model', {}).get('mlp_params', {}).get('input_size', 3072)
        )

    # --- 3. OPTIMIZER CONFIGURATION ---
    print("\n[OPTIMIZER CONFIGURATION]")
    custom_cfg['optimizer']['name'] = get_validated_input(
        f"Choose optimizer {VALID_OPTIMIZERS}",
        'choice',
        "Invalid optimizer choice",
        default=current_config['optimizer']['name'],
        constraints={'valid_list': VALID_OPTIMIZERS}
    )
    custom_cfg['optimizer']['lr'] = get_validated_input(
        "Enter initial learning rate (0.0 to 1.0)",
        'float_range',
        "Learning rate must be a float between 0 and 1.0",
        default=current_config['optimizer']['lr'],
        constraints={'range': [0.0, 1.0]}
    )
    custom_cfg['optimizer']['weight_decay'] = get_validated_input(
        "Enter weight decay (0.0 to 1.0)",
        'float_range',
        "Weight decay must be a float between 0 and 1.0",
        default=0.0001, # Standard default not in the minimal config
        constraints={'range': [0.0, 1.0]}
    )
    
    # CALL THE FUNCTION TO GET OPTIMIZER-SPECIFIC PARAMETERS
    custom_cfg = get_optimizer_params_interactively(custom_cfg['optimizer']['name'], custom_cfg)
    
    # --- 4. SCHEDULER CONFIGURATION ---
    print("\n[SCHEDULER CONFIGURATION]")
    custom_cfg['lr_scheduler']['name'] = get_validated_input(
        f"Choose LR Scheduler {VALID_SCHEDULERS}",
        'choice',
        "Invalid scheduler choice",
        default=current_config['lr_scheduler']['name'],
        constraints={'valid_list': VALID_SCHEDULERS}
    )
    # NOTE: Add scheduler-specific parameters (step_size, patience, etc.) here.

    # --- 5. CONFIRMATION ---
    print("\n--- Custom Configuration Complete ---")
    print("Final Configuration (Rich pretty print):")
    rprint(Pretty(custom_cfg))

    return custom_cfg