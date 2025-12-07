"""Utility functions for displaying parameter summaries in CLI."""

from typing import Dict, Any, Optional, List
from src.models.group import ParameterGroup
from src.models.parameter import AmberParameter
from src.enums import ParameterCategory, ParameterType


def format_parameter_value(param: AmberParameter, value: Any) -> str:
    """Format a parameter value for display.
    
    Args:
        param: The parameter object
        value: The parameter value
        
    Returns:
        Formatted string representation of the value
    """
    if value is None:
        return "(not set)"
    
    if param.param_type == ParameterType.BOOLEAN:
        return "Enabled" if value else "Disabled"
    elif param.param_type == ParameterType.FLOAT:
        # Format floats with appropriate precision
        if isinstance(value, float):
            # Show 1 decimal place for small values, 0 for large integers
            if value < 10:
                return f"{value:.1f}"
            elif value.is_integer():
                return f"{int(value)}"
            else:
                return f"{value:.2f}"
    return str(value)


def get_parameter_display_value(param: AmberParameter, config: Dict[str, Any]) -> Optional[Any]:
    """Get the display value for a parameter from config or default.
    
    Args:
        param: The parameter object
        config: Configuration dictionary
        
    Returns:
        Parameter value if set (in config or default), None otherwise
    """
    # Check if parameter is in config
    if param.yaml_key in config:
        return config[param.yaml_key]
    
    # Use default if available
    if param.default_value is not None:
        return param.default_value
    
    return None


def format_parameter_line(param: AmberParameter, value: Any, indent: str = "   ") -> str:
    """Format a single parameter as a bullet point line.
    
    Args:
        param: The parameter object
        value: The parameter value
        indent: Indentation string
        
    Returns:
        Formatted string line
    """
    formatted_value = format_parameter_value(param, value)
    line = f"{indent}• {param.description}: {formatted_value}"
    
    # Add notes if available
    if param.notes:
        line += f"\n{indent}    └─ {param.notes}"
    
    return line


def display_parameter_summary(
    group: ParameterGroup,
    config: Dict[str, Any],
    show_only_set: bool = True,
    group_by_category: bool = False,
    title: Optional[str] = None
) -> None:
    """Display a formatted summary of parameters from a ParameterGroup.
    
    This function provides a modular, extensible framework for displaying
    parameter summaries in a consistent CLI format. It handles optional
    parameters, defaults, and notes automatically.
    
    Args:
        group: The ParameterGroup to display
        config: Configuration dictionary with parameter values
        show_only_set: If True, only show parameters that are set (in config or have defaults)
        group_by_category: If True, group parameters by category
        title: Optional custom title (defaults to group description)
    """
    # Get title
    display_title = title or group.description
    
    # Collect parameters with values
    params_with_values: List[tuple[AmberParameter, Any]] = []
    
    for param in group.parameters:
        value = get_parameter_display_value(param, config)
        if value is not None or not show_only_set:
            params_with_values.append((param, value))
    
    # If no parameters to display, return early
    if not params_with_values:
        return
    
    # Print header
    print("\n" + "="*70)
    print(f"{display_title}")
    print("="*70)
    
    if group_by_category:
        # Group by category
        categories: Dict[ParameterCategory, List[tuple[AmberParameter, Any]]] = {}
        
        for param, value in params_with_values:
            cat = param.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((param, value))
        
        # Display by category
        for category in sorted(categories.keys(), key=lambda c: c.value):
            category_params = categories[category]
            if category_params:
                print(f"\n  {category.value.upper().replace('_', ' ')}:")
                for param, value in category_params:
                    if value is not None:
                        print(format_parameter_line(param, value))
                    else:
                        print(f"   • {param.description}: (not set)")
    else:
        # Display all parameters in order
        for param, value in params_with_values:
            if value is not None:
                print(format_parameter_line(param, value))
            else:
                print(f"   • {param.description}: (not set)")
    
    print("="*70)

