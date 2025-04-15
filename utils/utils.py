import importlib
import random
import numpy as np
import os
import yaml

def encode_action(direction, step_size, num_directions=4) -> int:
    step_size_index = step_size - 1
    return step_size_index * num_directions + direction


def decode_actions(action, num_directions = 4) -> tuple[int, int]:
    direction = action % num_directions
    step_size = (action - direction) // num_directions + 1
    return direction, step_size


def load_class(class_path):
    """Loads a class dynamically from its module path string."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import class '{class_path}': {e}")

def setup_seed(seed):
    """Sets random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        # Add seeding for torch/tf here if used
        print(f"Set master random seed to {seed}")
        return True
    else:
        print("No master seed specified.")
        return False

def get_frozenlake_terminal_states(env):
    """Helper to find terminal states for FrozenLake."""
    terminal_states = set()
    unwrapped_env = env.unwrapped
    if hasattr(unwrapped_env, 'desc'):
        desc = unwrapped_env.desc.flatten()
        for i, char in enumerate(desc):
            try:
                char_str = char.decode('utf-8')
            except (AttributeError, UnicodeDecodeError):
                char_str = str(char)
            if char_str in ('H', 'G'):
                terminal_states.add(i)
    else:
        print("Warning: Could not access env.unwrapped.desc for FrozenLake.")
    return terminal_states


def deep_merge(dict1, dict2):
    """
    Deep merge two dictionaries. dict2 values will override dict1 values for the same keys.
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path):
    """Load and merge configuration files."""

    # Add a constructor for scientific notation
    def scientific_constructor(loader, node):
        value = loader.construct_scalar(node)
        try:
            return float(value)
        except ValueError:
            return value

    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", scientific_constructor)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader) or {}

    # Get the directory of the main config file
    config_dir = os.path.dirname(config_path)

    # Load and merge imported configs
    if "imports" in config:
        for import_path in config["imports"]:
            import_path = os.path.join(config_dir, import_path)
            try:
                with open(import_path, "r") as f:
                    imported_config = yaml.load(f, Loader=yaml.SafeLoader) or {}
                    config = deep_merge(config, imported_config)
            except FileNotFoundError:
                print(f"Warning: Could not find config file {import_path}")
                continue

        del config["imports"]

    return config


