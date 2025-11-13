from utils.constants import VALID_OPTIMIZERS


def validate_choice(value, valid_list):
    choice = str(value).strip().upper()
    valid_upper = [item.upper() for item in valid_list]
    if choice not in valid_upper:
        raise ValueError(f"Must be one of: {valid_list}")
    return valid_list[valid_upper.index(choice)]

def validate_list_choice(value, valid_list):
    items = [v.strip() for v in str(value).split(',')]
    valid_upper = [item.upper() for item in valid_list]
    result = []
    for item in items:
        if item.upper() not in valid_upper:
            raise ValueError(f"Each activation must be one of: {valid_list}")
        result.append(valid_list[valid_upper.index(item.upper())])
    return result

def validate_positive_int(value):
    i = int(value)
    if i <= 0:
        raise ValueError("Must be a positive integer (> 0).")
    return i

def validate_float_range(value, min_val=0.0, max_val=1.0):
    f = float(value)
    if not (min_val <= f <= max_val):
        raise ValueError(f"Must be a float between {min_val} and {max_val} (inclusive).")
    return f

def validate_int_range(value, min_val=0, max_val=100):
    i = int(value)
    if not (min_val <= i <= max_val):
        raise ValueError(f"Must be an integer between {min_val} and {max_val} (inclusive).")
    return i

def validate_boolean(value):
    v = str(value).strip().lower()
    if v in ('yes', 'y', 'true', 't', '1'):
        return True
    elif v in ('no', 'n', 'false', 'f', '0'):
        return False
    else:
        raise ValueError("Must be 'True' or 'False' (e.g., 'y' or 'n').")

def validate_list_float(value):
    try:
        return [float(x.strip()) for x in str(value).split(',')]
    except ValueError:
        raise ValueError("Must be a comma-separated list of floating point numbers (e.g., 0.9,0.999).")

def get_validated_input(prompt, validator_type, error_msg, default=None, constraints=None):
    validators = {
        'choice': lambda x: validate_choice(x, constraints['valid_list']),
        'list_choice': lambda x: validate_list_choice(x, constraints['valid_list']),
        'positive_int': validate_positive_int,
        'int_range': lambda x: validate_int_range(x, constraints['range'][0], constraints['range'][1]),
        'float_range': lambda x: validate_float_range(x, constraints['range'][0], constraints['range'][1]),
        'boolean': validate_boolean,
        'list_float': validate_list_float,
        'optimizer_choice': lambda x: validate_choice(x, VALID_OPTIMIZERS)
    }

    validator_func = validators.get(validator_type, str)

    while True:
        try:
            default_display = default
            if isinstance(default, list):
                default_display = ','.join(map(str, default))

            user_input = input(f"{prompt} (Default: {default_display}): ")

            if not user_input and default is not None:
                if validator_type == 'list_float' and isinstance(default, str):
                    return validate_list_float(default)
                if validator_type == 'list_choice' and isinstance(default, str):
                    return validate_list_choice(default, constraints['valid_list'])
                return default

            return validator_func(user_input)

        except ValueError as e:
            print(f"Invalid input: {error_msg}. {e}")
        except TypeError:
            print("A validation type error occurred. Check constraints.")

