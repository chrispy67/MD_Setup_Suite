"""Simulation type mappings for consistent key resolution across the codebase."""

from typing import Dict, List, Any, Optional, Tuple
from omegaconf import DictConfig
from src.parameter_groups import (
    create_em_parameter_group,
    create_nvt_parameter_group,
    create_npt_parameter_group,
    create_production_parameter_group
)

# Mapping from canonical keys to their factory functions, group names, and display names
# This allows flexible YAML keys while maintaining consistency across the codebase
SIMULATION_GROUP_MAPPING: Dict[str, Dict[str, Any]] = {
    "em": {
        "factory": create_em_parameter_group,
        "group_name": ["energy_minimization", "em_ensemble", "minimization", "em"],
        "display_name": "EM"
    },
    "NVT_ensemble": {
        "factory": create_nvt_parameter_group,
        "group_name": ["nvt_ensemble", "NVT_ensemble", "NVT", "nvt_equilibration", "nvt"],
        "display_name": "NVT"
    },
    "NPT_ensemble": {
        "factory": create_npt_parameter_group,
        "group_name": ["npt_ensemble", "NPT_ensemble", "NPT", "npt_equilibration"],
        "display_name": "NPT"
    },
    "production": {
        "factory": create_production_parameter_group,
        "group_name": ["production_ensemble", "production", "prod"],
        "display_name": "Production"
    }
}


def get_group_name_mapping() -> Dict[str, List[str]]:
    """Get a simplified mapping of canonical keys to group name arrays.
    
    This is useful for code that only needs the group name aliases without
    the factory functions or display names.
    
    Returns:
        Dictionary mapping canonical keys to lists of group name aliases
    """
    return {
        canonical_key: mapping["group_name"]
        for canonical_key, mapping in SIMULATION_GROUP_MAPPING.items()
    }


def get_primary_group_name(group_name: Any) -> str:
    """Get the primary group name from either a string or list.
    
    Args:
        group_name: Either a string or a list of strings
        
    Returns:
        The primary group name (first element if list, otherwise the string itself)
    """
    if isinstance(group_name, list):
        return group_name[0] if group_name else ""
    return str(group_name)


def find_canonical_key(yaml_key: str) -> Optional[str]:
    """Find the canonical key by checking if yaml_key is in any group_name array.
    
    Args:
        yaml_key: The YAML key from the configuration file
        
    Returns:
        The canonical key if found, None otherwise
    """
    # First check if it's a direct match (canonical key itself)
    if yaml_key in SIMULATION_GROUP_MAPPING:
        return yaml_key
    
    # Check if yaml_key is in any group_name array
    for canonical_key, mapping in SIMULATION_GROUP_MAPPING.items():
        group_names = mapping.get("group_name", [])
        if isinstance(group_names, list):
            if yaml_key in group_names:
                return canonical_key
        elif yaml_key == group_names:
            return canonical_key
    
    return None


def get_simulation_order(cfg: DictConfig) -> Tuple[List[str], Dict[str, str]]:
    """Extract the order of simulations from the YAML configuration.
    
    Matches YAML keys against group_name arrays in SIMULATION_GROUP_MAPPING.
    The order is determined by the order they appear in the simulations section.
    
    Raises:
        ValueError: If a YAML key in simulations doesn't match any known simulation type
    
    Args:
        cfg: The configuration object
        
    Returns:
        Tuple of (list of canonical keys in order, mapping from original YAML keys to canonical keys)
    """
    if not hasattr(cfg, 'simulations') or cfg.simulations is None:
        return [], {}
    
    order = []
    unknown_keys = []
    yaml_to_canonical = {}  # Map original YAML key to canonical key
    
    for yaml_key in cfg.simulations.keys():
        canonical_key = find_canonical_key(yaml_key)
        if canonical_key:
            order.append(canonical_key)
            yaml_to_canonical[yaml_key] = canonical_key
        else:
            unknown_keys.append(yaml_key)
    
    # Raise exception if unknown keys are found
    if unknown_keys:
        # Build helpful error message showing all valid group_name options
        all_options = []
        for canonical_key, mapping in SIMULATION_GROUP_MAPPING.items():
            group_names = mapping.get("group_name", [])
            if isinstance(group_names, list):
                options = ", ".join(group_names)
            else:
                options = str(group_names)
            all_options.append(f"  {canonical_key}: [{options}]")
        
        raise ValueError(
            f"Unknown simulation type(s) in YAML: {', '.join(unknown_keys)}\n"
            f"Valid options for each simulation type:\n" + "\n".join(all_options) + "\n"
            f"Please check your simulation_config.yaml file."
        )
    
    return order, yaml_to_canonical

