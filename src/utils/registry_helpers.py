"""Helper functions for working with ParameterRegistry."""

import logging
from typing import Dict, List, Any
from src.models import ParameterRegistry
from src.utils.simulation_mappings import (
    SIMULATION_GROUP_MAPPING,
    get_primary_group_name
)

logger = logging.getLogger(__name__)


def register_simulation_groups(
    registry: ParameterRegistry,
    simulation_order: List[str]
) -> Dict[str, Any]:
    """Dynamically register parameter groups based on YAML configuration.
    
    Args:
        registry: The ParameterRegistry instance
        simulation_order: List of simulation YAML keys in order
        
    Returns:
        Dictionary mapping primary group names to their ParameterGroup instances
    """
    registered_groups = {}
    
    for yaml_key in simulation_order:
        if yaml_key not in SIMULATION_GROUP_MAPPING:
            # This should not happen if get_simulation_order() is called first,
            # but handle it gracefully
            logger.warning(f"Unknown simulation type: {yaml_key}, skipping...")
            continue
        
        mapping = SIMULATION_GROUP_MAPPING[yaml_key]
        factory = mapping["factory"]
        group_name_list = mapping["group_name"]
        
        # Get primary group name (first in list, or the string itself)
        primary_group_name = get_primary_group_name(group_name_list)
        
        # Create and register the group
        group = factory()
        registry.add_group(group)
        registered_groups[primary_group_name] = group
        print(f"Registered group: {primary_group_name}")
    
    return registered_groups

