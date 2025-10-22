"""
Pydantic-based parameter models with automatic validation and serialization.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any, Union, Literal
from enum import Enum
import json


# for type hints in IDE!
class ParameterType(str, Enum):
    """Supported parameter types."""
    INT = "int"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "bool"
    LIST = "list"
    DICT = "dict"


# How to group these?
class ParameterCategory(str, Enum):
    """Parameter categories for organization."""
    CONTROL = "control"
    GENERAL = "general"

    THERMOSTAT = "thermostat"
    BAROSTAT = "barostat"
    RESTRAINT = "restraint"
    OUTPUT = "output"


class ParameterValidation(BaseModel):
    """Validation rules for a parameter."""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    valid_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class ParameterDependency(BaseModel):
    """Dependency information for a parameter."""
    parameter: str = Field(..., description="Name of the parameter this depends on")
    condition: str = Field(..., description="Condition that must be met")
    required_value: Any = Field(..., description="Required value for the dependency")


class AmberParameter(BaseModel):
    """AMBER parameter definition with validation."""
    yaml_key: str = Field(..., description="YAML configuration key")
    amber_flag: str = Field(..., description="AMBER input file flag")
    description: str = Field(..., description="Human-readable description")
    param_type: ParameterType = Field(..., description="Parameter data type")
    category: ParameterCategory = Field(default=ParameterCategory.GENERAL, description="Parameter category")
    
    # Optional fields
    default_value: Optional[Any] = Field(default=None, description="Default value")
    validation: Optional[ParameterValidation] = Field(default=None, description="Validation rules")
    dependencies: Optional[List[ParameterDependency]] = Field(default=None, description="Parameter dependencies")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    
    # Metadata
    is_required: bool = Field(default=False, description="Whether parameter is required")
    is_deprecated: bool = Field(default=False, description="Whether parameter is deprecated")
    version_added: Optional[str] = Field(default=None, description="Version when parameter was added")
    version_deprecated: Optional[str] = Field(default=None, description="Version when parameter was deprecated")
    
    @field_validator('yaml_key')
    @classmethod
    def validate_yaml_key(cls, v):
        """Validate YAML key format."""
        if not v or not isinstance(v, str):
            raise ValueError("YAML key must be a non-empty string")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("YAML key must contain only alphanumeric characters, underscores, and hyphens")
        return v
    
    @field_validator('amber_flag')
    @classmethod
    def validate_amber_flag(cls, v):
        """Validate AMBER flag format."""
        if not v or not isinstance(v, str):
            raise ValueError("AMBER flag must be a non-empty string")
        return v
    
    def validate_value(self, value: Any) -> tuple[bool, str]:
        """Validate a parameter value."""
        # Type checking
        try:
            converted_value = self._convert_value(value)
        except (ValueError, TypeError) as e:
            return False, f"Type conversion error: {e}"
        
        # Custom validation rules
        if self.validation:
            validation_result = self._validate_with_rules(converted_value)
            if not validation_result[0]:
                return validation_result
        
        return True, "Valid"
    
    def _convert_value(self, value: Any) -> Any:
        """Convert value to the expected type."""
        if self.param_type == ParameterType.BOOLEAN:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            raise ValueError(f"Cannot convert {value} to boolean")
        
        elif self.param_type == ParameterType.INT:
            return int(value)
        elif self.param_type == ParameterType.FLOAT:
            return float(value)
        elif self.param_type == ParameterType.STRING:
            return str(value)
        elif self.param_type == ParameterType.LIST:
            if isinstance(value, list):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return [value]
            return [value]
        elif self.param_type == ParameterType.DICT:
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return {"value": value}
            return {"value": value}
        
        return value
    
    def _validate_with_rules(self, value: Any) -> tuple[bool, str]:
        """Validate value against custom rules."""
        if not self.validation:
            return True, "Valid"
        
        # Range validation
        if self.validation.min_value is not None and value < self.validation.min_value:
            return False, f"Value {value} is below minimum {self.validation.min_value}"
        
        if self.validation.max_value is not None and value > self.validation.max_value:
            return False, f"Value {value} is above maximum {self.validation.max_value}"
        
        # Valid values validation
        if self.validation.valid_values is not None and value not in self.validation.valid_values:
            return False, f"Value {value} not in valid values: {self.validation.valid_values}"
        
        # String length validation
        if isinstance(value, str):
            if self.validation.min_length is not None and len(value) < self.validation.min_length:
                return False, f"String length {len(value)} is below minimum {self.validation.min_length}"
            
            if self.validation.max_length is not None and len(value) > self.validation.max_length:
                return False, f"String length {len(value)} is above maximum {self.validation.max_length}"
        
        return True, "Valid"
    
    def get_amber_value(self, value: Any) -> Any:
        """Get the value formatted for AMBER input."""
        converted_value = self._convert_value(value)
        
        # Special AMBER formatting
        if self.param_type == ParameterType.BOOLEAN:
            return 1 if converted_value else 0
        
        return converted_value


class ParameterGroup(BaseModel):
    """Group of related AMBER parameters."""
    name: str = Field(..., description="Group name")
    description: str = Field(..., description="Group description")
    parameters: List[AmberParameter] = Field(default_factory=list, description="Parameters in this group")
    version: str = Field(default="1.0.0", description="Group version")
    
    def add_parameter(self, parameter: AmberParameter):
        """Add a parameter to this group."""
        self.parameters.append(parameter)
    
    def get_parameter(self, yaml_key: str) -> Optional[AmberParameter]:
        """Get parameter by YAML key."""
        for param in self.parameters:
            if param.yaml_key == yaml_key:
                return param
        return None
    
    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a configuration against this group."""
        errors = []
        
        for param in self.parameters:
            if param.yaml_key in config:
                is_valid, error_msg = param.validate_value(config[param.yaml_key])
                if not is_valid:
                    errors.append(f"{param.yaml_key}: {error_msg}")
            elif param.is_required:
                errors.append(f"{param.yaml_key}: Required parameter missing")
        
        return len(errors) == 0, errors
    
    def generate_amber_mapping(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AMBER input mapping from configuration."""
        amber_mapping = {}
        
        for param in self.parameters:
            if param.yaml_key in config:
                value = config[param.yaml_key]
                amber_value = param.get_amber_value(value)
                amber_mapping[param.amber_flag] = amber_value
        
        return amber_mapping


class ParameterRegistry(BaseModel):
    """Registry for all AMBER parameter groups."""
    groups: Dict[str, ParameterGroup] = Field(default_factory=dict, description="Parameter groups")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Registry metadata")
    
    def add_group(self, group: ParameterGroup):
        """Add a parameter group to the registry."""
        self.groups[group.name] = group
    
    def get_group(self, name: str) -> Optional[ParameterGroup]:
        """Get a parameter group by name."""
        return self.groups.get(name)
    
    def get_parameter(self, yaml_key: str, group_name: Optional[str] = None) -> Optional[AmberParameter]:
        """Get a parameter by YAML key."""
        if group_name:
            group = self.get_group(group_name)
            return group.get_parameter(yaml_key) if group else None
        
        # Search all groups
        for group in self.groups.values():
            param = group.get_parameter(yaml_key)
            if param:
                return param
        return None
    
    def search_parameters(self, query: str) -> List[AmberParameter]:
        """Search parameters by description or key."""
        results = []
        query_lower = query.lower()
        
        for group in self.groups.values():
            for param in group.parameters:
                if (query_lower in param.yaml_key.lower() or 
                    query_lower in param.description.lower() or
                    query_lower in param.amber_flag.lower()):
                    results.append(param)
        
        return results
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export registry to dictionary format."""
        return {
            "groups": {name: group.dict() for name, group in self.groups.items()},
            "metadata": self.metadata
        }
    
    def export_to_yaml(self, file_path: str):
        """Export registry to YAML file."""
        import yaml
        
        with open(file_path, 'w') as f:
            yaml.dump(self.export_to_dict(), f, default_flow_style=False, indent=2)


# Example usage and factory functions
def create_em_parameter_group() -> ParameterGroup:

    """Create energy minimization parameter group."""
    group = ParameterGroup(
        name="energy_minimization", # Grouped parameters
        description="Parameters for energy minimization protocol for simulations"
    )
    
    # Add parameters
    group.add_parameter(AmberParameter(
        yaml_key="method",
        amber_flag="ntmin",
        description="Minimization method",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 5, 6, 7]),
        notes="0=MD, 1=SD, 5=CG, 6=SD+CG, 7=SD+CG+MD"
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="maxcyc",
        description="Maximum number of minimization cycles",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(min_value=1, max_value=10000000),
        default_value=1000
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="restraint",
        amber_flag="ntr",
        description="Enable positional restraints",
        param_type=ParameterType.BOOLEAN,
        category=ParameterCategory.RESTRAINT,
        default_value=False
    ))

    group.add_parameter(AmberParameter(
        yaml_key="max_force",
        amber_flag="restraint_wt",
        description="Max force of restraint applied to indicated atoms (kcal/mol)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory
    ))
    return group


def create_nvt_parameter_group() -> ParameterGroup:
    """Create NVT ensemble parameter group."""
    group = ParameterGroup(
        name="nvt_ensemble",
        description="Parameters for NVT ensemble simulations"
    )
    
    group.add_parameter(AmberParameter(
        yaml_key="thermostat",
        amber_flag="ntt",
        description="Temperature control method",
        param_type=ParameterType.INT,
        category=ParameterCategory.THERMOSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3]),
        notes="0=none, 1=Berendsen, 2=Andersen, 3=Hoover"
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="temperature",
        amber_flag="temp0",
        description="Target temperature (K)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.THERMOSTAT,
        validation=ParameterValidation(min_value=0.0, max_value=1000.0),
        default_value=300.0
    ))
    
    return group


# Example usage
if __name__ == "__main__":
    # Create registry
    registry = ParameterRegistry()
    
    # Add groups
    registry.add_group(create_em_parameter_group())
    registry.add_group(create_nvt_parameter_group())
    
    # Test parameter retrieval
    param = registry.get_parameter("method", "energy_minimization")
    print(f"Found parameter: {param}")
    
    # Test validation
    em_config = {
        "method": 1,
        "steps": 5000,
        "restraint": True
    }
    
    em_group = registry.get_group("energy_minimization")
    is_valid, errors = em_group.validate_config(em_config)
    print(f"Config valid: {is_valid}, Errors: {errors}")
    
    # Test mapping generation
    amber_mapping = em_group.generate_amber_mapping(em_config)
    print(f"AMBER mapping: {amber_mapping}")
    
    # Test search
    results = registry.search_parameters("temperature")
    print(f"Search results: {len(results)} parameters found")
