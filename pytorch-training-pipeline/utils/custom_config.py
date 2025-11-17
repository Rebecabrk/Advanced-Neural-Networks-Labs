from rich import print as rprint
from rich.pretty import Pretty

from utils.validation import get_validated_input
from utils.constants import (
    VALID_DATASETS,
    VALID_MODELS,
    VALID_OPTIMIZERS,
    VALID_SCHEDULERS,
    OPTIMIZER_PARAMS,
    TRANSFORM_PRESETS,
    MLP_PARAMS,
    SCHEDULER_PARAMS
)

def get_optimizer_params(optimizer_name, custom_cfg):
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

def get_transforms(custom_cfg):
    enabled_transforms = {}
    print("\n  -- Data Augmentation Transforms --")
    for transform_name, transform_details in TRANSFORM_PRESETS.items():
        enabled = get_validated_input(
            f"    Enable {transform_name} (True/False)",
            'boolean',
            "Invalid boolean value",
            default=transform_details['enabled']
        )
        custom_cfg.setdefault('data', {}).setdefault('transforms', {})[transform_name] = {'enabled': enabled}
        if enabled:
            params = {'enabled': True}
            for param, val in transform_details.items():
                if param != 'enabled':
                    param_value = get_validated_input(
                        f"      Set value for {param}",
                        val['validator'],
                        f"Invalid value for {param}",
                        default=val['default'],
                        constraints={k: v for k, v in val.items() if k not in ['default', 'validator']}
                    )
                    params[param] = param_value
            enabled_transforms[transform_name] = params
    custom_cfg['data']['transforms'] = enabled_transforms
    return custom_cfg

def get_lr_scheduler_params(custom_cfg):
    scheduler_name = custom_cfg['lr_scheduler']['name']
    print(f"\n  -- Specific {scheduler_name} Parameters --")
    scheduler_specific_params = SCHEDULER_PARAMS.get(scheduler_name, {})
    
    # Initialize a new dictionary for specific params
    custom_cfg['lr_scheduler'][f'{scheduler_name.lower()}_params'] = {}
    
    for param_name, param_details in scheduler_specific_params.items():
        
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
        custom_cfg['lr_scheduler'][f'{scheduler_name.lower()}_params'][param_name] = value

    return custom_cfg

def get_commandline_config(current_config):
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
    if custom_cfg['data']['data_augmentation']:
        custom_cfg = get_transforms(custom_cfg)

    # --- 2. MODEL CONFIGURATION ---
    print("\n[MODEL CONFIGURATION]")
    custom_cfg['model']['name'] = get_validated_input(
        f"Choose model {VALID_MODELS}",
        'choice',
        "Invalid model choice",
        default=current_config['model']['name'],
        constraints={'valid_list': VALID_MODELS}
    )

    # MLP Specific Parameters
    if custom_cfg['model']['name'] == 'MLP':
        custom_cfg['model']['pretrained'] = False  # MLP does not use pretrained weights
        custom_cfg['model']['mlp_params'] = {}

        print("\n  -- MLP Parameters --")
        custom_cfg['model']['mlp_params']['hidden_layers'] = get_validated_input(
            "Enter hidden layers (comma-separated)",
            'list_float',
            "Hidden layers must be a comma-separated list of integers",
            default=MLP_PARAMS['hidden_layers']['default']
        )
        custom_cfg['model']['mlp_params']['activations'] = get_validated_input(
            "Enter activation functions (comma-separated)",
            'list_choice',
            "Invalid activation function",
            default=MLP_PARAMS['activations']['default'],
            constraints={'valid_list': MLP_PARAMS['activations']['valid_list']}
        )
        custom_cfg['model']['mlp_params']['dropout'] = get_validated_input(
            "Enter dropout rate (0.0 to 1.0)",
            'float_range',
            "Dropout must be a float between 0 and 1.0",
            default=MLP_PARAMS['dropout']['default'],
            constraints={'range': MLP_PARAMS['dropout']['range']}
        )
    else:
        custom_cfg['model']['pretrained'] = get_validated_input(
            "Use pretrained weights (True/False)",
            'boolean',
            "Invalid boolean value",
            default=current_config['model']['pretrained']
        )
    # --- 3. TRAINING CONFIGURATION ---
    print("\n[TRAINING CONFIGURATION]")
    custom_cfg['training']['epochs'] = get_validated_input(
        "Enter number of epochs (positive integer)",
        'positive_int',
        "Number of epochs must be a positive integer",
        default=current_config['training']['epochs']
    )
    custom_cfg['training']['early_stopping']['enabled'] = get_validated_input(
        "Enable Early Stopping (True/False)",
        'boolean',
        "Invalid boolean value",
        default=current_config['training']['early_stopping']['enabled']
    )
    if custom_cfg['training']['early_stopping']['enabled']:
        custom_cfg['training']['early_stopping']['patience'] = get_validated_input(
            f"   Enter Early Stopping Patience (positive integer)",
            'positive_int',
            "Patience must be a positive integer",
            default=current_config['training']['early_stopping']['patience']
        )
    custom_cfg['training']['batch_scheduler'] = get_validated_input(
        "Enable Batch Size Scheduler (True/False)",
        'boolean',
        "Invalid boolean value",
        default=current_config['training'].get('batch_scheduler', {}).get('enabled', False)
    )
    if custom_cfg['training']['batch_scheduler']:
        custom_cfg['training']['batch_scheduler']['schedule_epochs'] = get_validated_input(
            "Enter schedule epochs (comma-separated positive integers)",
            'list_float',
            "Schedule epochs must be a comma-separated list of positive integers",
            default=current_config['training'].get('batch_scheduler', {}).get('schedule_epochs', [10, 25, 40])
        )
        custom_cfg['training']['batch_scheduler']['batch_size_increments'] = get_validated_input(
            "Enter batch size increments (comma-separated positive integers)",
            'list_float',
            "Batch size increments must be a comma-separated list of positive integers",
            default=current_config['training'].get('batch_scheduler', {}).get('batch_size_increments', [128, 256, 512])
        )
    else:
        custom_cfg['training']['batch_size'] = get_validated_input(
            "Enter batch size (positive integer)",
            'positive_int',
            "Batch size must be a positive integer",
            default=current_config['training']['batch_size']
        )

    # --- 4. OPTIMIZER CONFIGURATION ---
    print("\n[OPTIMIZER CONFIGURATION]")
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
    custom_cfg['optimizer']['name'] = get_validated_input(
        f"Choose optimizer {VALID_OPTIMIZERS}",
        'choice',
        "Invalid optimizer choice",
        default=current_config['optimizer']['name'],
        constraints={'valid_list': VALID_OPTIMIZERS}
    )
    
    custom_cfg = get_optimizer_params(custom_cfg['optimizer']['name'], custom_cfg)
    
    # --- 5. SCHEDULER CONFIGURATION ---
    print("\n[SCHEDULER CONFIGURATION]")
    custom_cfg['lr_scheduler']['enabled'] = get_validated_input(
        "Enable Learning Rate Scheduler (True/False)",
        'boolean',
        "Invalid boolean value",
        default=current_config['lr_scheduler']['enabled']
    )
    if custom_cfg['lr_scheduler']['enabled']:
        custom_cfg['lr_scheduler']['name'] = get_validated_input(
            f"  Choose LR Scheduler {VALID_SCHEDULERS}",
            'choice',
            "Invalid scheduler choice",
            default=current_config['lr_scheduler']['name'],
            constraints={'valid_list': VALID_SCHEDULERS}
        )
        custom_cfg = get_lr_scheduler_params(custom_cfg)
    custom_cfg['logging']['run_name'] = f'Custom_' + custom_cfg['model']['name'] + \
                                            '_' + custom_cfg['optimizer']['name'] 

    return custom_cfg