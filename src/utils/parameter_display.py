"""Utility functions for displaying parameter summaries in CLI."""

import re
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
            # For very small values (< 1), use more decimal places to show precision
            if 0 < abs(value) < 1:
                # Use enough decimal places to show at least 2 significant digits
                # For 0.004, we want to show 0.004 (3 decimal places)
                # For 0.1, we want to show 0.1 (1 decimal place)
                if value < 0.01:
                    return f"{value:.4f}".rstrip('0').rstrip('.')
                elif value < 0.1:
                    return f"{value:.3f}".rstrip('0').rstrip('.')
                else:
                    return f"{value:.2f}".rstrip('0').rstrip('.')
            # Show 1 decimal place for small values (1-10)
            elif value < 10:
                return f"{value:.1f}"
            # For large integers, show as integer
            elif value.is_integer():
                return f"{int(value)}"
            # For larger floats, show 2 decimal places
            else:
                return f"{value:.2f}"
    elif param.param_type == ParameterType.RESTRAINT_STRING_ARRAY:
        # Format array of restraint strings for display
        if isinstance(value, list):
            if len(value) == 0:
                return "(empty)"
            if len(value) == 1:
                return value[0]
            # For multiple items, show as a formatted list
            return f"[{', '.join(value)}]"
        return str(value)
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


def format_parameter_description(param: AmberParameter, value: Any) -> tuple[str, bool]:
    """Format a parameter description, replacing placeholders with actual values.
    
    Supports placeholders:
    - {value}: The formatted parameter value
    - {yaml_key}: The YAML key name
    - {amber_flag}: The AMBER flag name
    - Any placeholder matching yaml_key or amber_flag (case-insensitive)
    
    Args:
        param: The parameter object
        value: The parameter value
        
    Returns:
        Tuple of (formatted description string, has_placeholders)
    """
    description = param.description
    formatted_value = format_parameter_value(param, value)
    has_placeholders = '{' in description and '}' in description
    
    if not has_placeholders:
        return description, False
    
    # Build replacement dictionary
    replacements = {
        'value': formatted_value,
        'yaml_key': param.yaml_key,
    }
    
    # Add amber_flag if available
    if param.amber_flag:
        replacements['amber_flag'] = param.amber_flag
    
    # Find all placeholders in the description
    placeholders = re.findall(r'\{(\w+)\}', description)
    
    # Build custom replacements for parameter name placeholders
    custom_replacements = {}
    for placeholder in placeholders:
        placeholder_lower = placeholder.lower()
        # Check if placeholder matches yaml_key or amber_flag (case-insensitive)
        if placeholder_lower == param.yaml_key.lower():
            custom_replacements[placeholder] = formatted_value
        elif param.amber_flag and placeholder_lower == param.amber_flag.lower():
            custom_replacements[placeholder] = formatted_value
        # If it's a standard placeholder, use the replacement dict
        elif placeholder_lower in replacements:
            custom_replacements[placeholder] = replacements[placeholder_lower]
    
    # Perform replacements (case-insensitive for custom placeholders)
    for placeholder, replacement in custom_replacements.items():
        # Replace both {placeholder} and {PLACEHOLDER} variations
        description = re.sub(
            r'\{' + re.escape(placeholder) + r'\}',
            replacement,
            description,
            flags=re.IGNORECASE
        )
    
    return description, True


def format_parameter_line(param: AmberParameter, value: Any, indent: str = "   ") -> str:
    """Format a single parameter as a bullet point line.
    
    Args:
        param: The parameter object
        value: The parameter value
        indent: Indentation string
        
    Returns:
        Formatted string line
    """
    formatted_description, has_placeholders = format_parameter_description(param, value)
    
    # If description had placeholders, use it as-is (value already embedded)
    # Otherwise, use traditional format: description: value
    if has_placeholders:
        line = f"{indent}• {formatted_description}"
    else:
        formatted_value = format_parameter_value(param, value)
        line = f"{indent}• {formatted_description}: {formatted_value}"
    
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
                        # For None values, check if description has placeholders
                        formatted_desc, has_placeholders = format_parameter_description(param, None)
                        if has_placeholders:
                            # Replace placeholders with "(not set)"
                            formatted_desc = re.sub(r'\{[^}]+\}', '(not set)', formatted_desc)
                            print(f"   • {formatted_desc}")
                        else:
                            print(f"   • {param.description}: (not set)")
    else:
        # Display all parameters in order
        for param, value in params_with_values:
            if value is not None:
                print(format_parameter_line(param, value))
            else:
                # For None values, check if description has placeholders
                formatted_desc, has_placeholders = format_parameter_description(param, None)
                if has_placeholders:
                    # Replace placeholders with "(not set)"
                    formatted_desc = re.sub(r'\{[^}]+\}', '(not set)', formatted_desc)
                    print(f"   • {formatted_desc}")
                else:
                    print(f"   • {param.description}: (not set)")
    
    print("="*70)

