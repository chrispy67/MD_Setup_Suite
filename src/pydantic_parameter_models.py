"""
Pydantic-based parameter models with automatic validation and serialization.
"""

from pickle import FLOAT
from unicodedata import category
from pydantic import BaseModel, Field, field_validator, model_validator
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
    CONTROL = "control" # control for individual simulations (no consistencies necessary)
    GENERAL = "general" # CRITICAL parameters that must be consistent through workup

    THERMOSTAT = "thermostat"
    BAROSTAT = "barostat"
    RESTRAINT = "restraint"
    OUTPUT = "output"
    
    # Workflow control categories
    WORKFLOW = "workflow"


class ParameterValidation(BaseModel):
    """Validation rules for a parameter."""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    valid_values: Optional[List[Any]] = None
    pattern: Optional[str] = None  # Regex pattern for strings
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class ParameterDependency(BaseModel):
    """Defines a dependency rule between parameters."""
    condition_param: str = Field(..., description="YAML key of the parameter that triggers the dependency")
    condition_value: Any = Field(..., description="Value that must match for dependency to be active")
    required_param: Optional[str] = Field(default=None, description="YAML key of the parameter that must be set/valid (single param)")
    required_params: Optional[Dict[str, Any]] = Field(default=None, description="Dict of {param_key: expected_value} for multiple params")
    required_condition: str = Field(..., description="Condition: 'required', 'required_gt_zero', 'must_be_zero', 'must_equal', etc.")
    error_message: str = Field(..., description="Error message if dependency is violated")
    
    @model_validator(mode='after')
    def validate_required_fields(self):
        """Ensure either required_param or required_params is set, but not both."""
        if self.required_param is None and self.required_params is None:
            raise ValueError("Either 'required_param' or 'required_params' must be provided")
        if self.required_param is not None and self.required_params is not None:
            raise ValueError("Cannot specify both 'required_param' and 'required_params'")
        return self


class AmberParameter(BaseModel):
    """AMBER parameter definition with validation."""
    yaml_key: str = Field(..., description="YAML configuration key")
    amber_flag: Optional[str] = Field(default=None, description="AMBER input file flag (None for workflow parameters)")
    description: str = Field(..., description="Human-readable description")
    param_type: ParameterType = Field(..., description="Parameter data type")
    category: ParameterCategory = Field(default=ParameterCategory.GENERAL, description="Parameter category")
    
    # Optional fields
    default_value: Optional[Any] = Field(default=None, description="Default value")
    validation: Optional[ParameterValidation] = Field(default=None, description="Validation rules")
    notes: Optional[str] = Field(default=None, description="Additional notes")
    
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
        if v is not None and not isinstance(v, str):
            raise ValueError("AMBER flag must be a string or None")
        
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
        # Here I want to check consistencies across simulation groups. 
        if not self.validation:
            return True, "Valid"
        
        ## Simple error handling that AMBER will catch, but should we stop the user from entering non-sensical values?
        # Range validation
        if self.validation.min_value is not None and value < self.validation.min_value:
            return False, f"Value {value} is below minimum {self.validation.min_value}"
        
        if self.validation.max_value is not None and value > self.validation.max_value:
            return False, f"Value {value} is above maximum {self.validation.max_value}"
        
        # Valid values validation
        if self.validation.valid_values is not None and value not in self.validation.valid_values:
            return False, f"Value {value} not in valid values: {self.validation.valid_values}"
        
        return True, "Valid"
    
    def get_amber_value(self, value: Any) -> Any:
        """Get the value formatted for AMBER input."""
        if self.amber_flag is None:
            # This is a workflow parameter, return the value as-is
            return self._convert_value(value)
            
        converted_value = self._convert_value(value)
        
        # Special AMBER formatting
        if self.param_type == ParameterType.BOOLEAN:
            return 1 if converted_value else 0
        
        return converted_value
    
    def is_workflow_parameter(self) -> bool:
        """Check if this is a workflow parameter (no amber_flag)."""
        return self.amber_flag is None


class ParameterGroup(BaseModel):
    """Group of related AMBER parameters."""
    name: str = Field(..., description="Group name")
    description: str = Field(..., description="Group description")
    parameters: List[AmberParameter] = Field(default_factory=list, description="Parameters in this group")
    dependencies: List[ParameterDependency] = Field(default_factory=list, description="Cross-parameter dependency rules")
    
    def add_parameter(self, parameter: AmberParameter):
        """Add a parameter to this group."""
        self.parameters.append(parameter)
    
    # These are going to be consistencies WITHIN 
    def add_dependency(self, dependency: ParameterDependency):
        """Add a dependency rule to this group."""
        self.dependencies.append(dependency)
    
    def get_parameter(self, yaml_key: str) -> Optional[AmberParameter]:
        """Get parameter by YAML key."""
        for param in self.parameters:
            if param.yaml_key == yaml_key:
                return param
        return None
    
    # For error checking by ParameterCategory 
    def get_parameters_by_category(self, category: ParameterCategory) -> List[AmberParameter]:
        """Get all parameters in a specific category."""
        return [param for param in self.parameters if param.category == category]


    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Check for basic consistencies like typing, ranges, valid values, etc."""
        errors = []
        
        # First, validate individual parameters
        for param in self.parameters:
            if param.yaml_key in config:
                is_valid, error_msg = param.validate_value(config[param.yaml_key])
                if not is_valid:
                    errors.append(f"{param.yaml_key}: {error_msg}")
        
        # Then, validate cross-parameter dependencies
        dependency_errors = self.validate_dependencies(config)
        errors.extend(dependency_errors)
        
        return len(errors) == 0, errors
    
    def validate_dependencies(self, config: Dict[str, Any]) -> List[str]:
        """Validate cross-parameter dependencies based on already chosen parameters."""
        errors = []
        
        for dependency in self.dependencies:
            # Check if condition parameter exists and matches condition value
            condition_value = config.get(dependency.condition_param)
            condition_param_obj = self.get_parameter(dependency.condition_param)
            
            # Apply default if condition parameter not set
            if condition_value is None and condition_param_obj and condition_param_obj.default_value is not None:
                condition_value = condition_param_obj.default_value
            
            if condition_value == dependency.condition_value:
                # Condition is met, check required parameter(s)
                
                # Handle single parameter dependency (backward compatible)
                if dependency.required_param:
                    errors.extend(self._validate_single_param_dependency(
                        dependency, config, dependency.required_param
                    ))
                
                # Handle multiple parameter dependencies
                elif dependency.required_params:
                    errors.extend(self._validate_multi_param_dependency(
                        dependency, config, dependency.required_params
                    ))
        
        return errors
    
    def _validate_single_param_dependency(
        self, 
        dependency: ParameterDependency, 
        config: Dict[str, Any], 
        param_key: str
    ) -> List[str]:
        """Validate a single parameter dependency."""
        errors = []
        required_value = config.get(param_key)
        required_param_obj = self.get_parameter(param_key)
        
        if required_param_obj is None:
            errors.append(f"Dependency error: Required parameter '{param_key}' not found in group")
            return errors
        
        # Apply default value if parameter not in config
        if required_value is None and required_param_obj.default_value is not None:
            required_value = required_param_obj.default_value
        
        # Check dependency condition
        if dependency.required_condition == "required":
            if required_value is None:
                errors.append(dependency.error_message)
        elif dependency.required_condition == "required_gt_zero":
            if required_value is None or (isinstance(required_value, (int, float)) and required_value <= 0):
                errors.append(dependency.error_message)
        elif dependency.required_condition == "must_be_zero":
            if required_value is not None and isinstance(required_value, (int, float)) and required_value != 0:
                errors.append(dependency.error_message)
        elif dependency.required_condition == "must_equal":
            # For must_equal, the expected value should be in the error message or we need to extend the model
            # For now, we'll use a simple approach where required_params dict contains expected values
            pass  # Handled in multi-param validation
        
        return errors
    
    def _validate_multi_param_dependency(
        self,
        dependency: ParameterDependency,
        config: Dict[str, Any],
        required_params: Dict[str, Any]
    ) -> List[str]:
        """Validate multiple parameter dependencies."""
        errors = []
        
        for param_key, expected_value in required_params.items():
            param_value = config.get(param_key)
            param_obj = self.get_parameter(param_key)
            
            if param_obj is None:
                errors.append(f"Dependency error: Required parameter '{param_key}' not found in group")
                continue
            
            # Apply default value if parameter not in config
            if param_value is None and param_obj.default_value is not None:
                param_value = param_obj.default_value
            
            # Check condition based on required_condition type
            if dependency.required_condition == "must_equal":
                if param_value != expected_value:
                    errors.append(f"{dependency.error_message} (Expected {param_key}={expected_value}, got {param_value})")
            elif dependency.required_condition == "required":
                if param_value is None:
                    errors.append(f"{dependency.error_message} (Missing {param_key})")
            elif dependency.required_condition == "required_gt_zero":
                if param_value is None or (isinstance(param_value, (int, float)) and param_value <= 0):
                    errors.append(f"{dependency.error_message} (Parameter {param_key} must be > 0)")
        
        return errors
    
    def validate_category_dependencies(self, config: Dict[str, Any], category: ParameterCategory) -> List[str]:
        """
        Validate dependencies for parameters in a specific category.
        
        Filters dependencies where either the condition_param or required_param belongs to the specified category.
        """
        errors = []
        
        # Filter dependencies by category
        for dependency in self.dependencies:
            condition_param_obj = self.get_parameter(dependency.condition_param)
            required_param_obj = None
            
            # Get required param object - handle both single and multiple params
            if dependency.required_param: #singular
                required_param_obj = self.get_parameter(dependency.required_param)
            elif dependency.required_params: #plural
                # For multiple params, check if any belong to the category
                for param_key in dependency.required_params.keys():
                    param_obj = self.get_parameter(param_key)
                    if param_obj and param_obj.category == category:
                        required_param_obj = param_obj
                        break
            
            # Check if either parameter belongs to the specified category
            condition_in_category = condition_param_obj and condition_param_obj.category == category
            required_in_category = required_param_obj and required_param_obj.category == category
            
            if condition_in_category or required_in_category:
                # This dependency is relevant to this category, validate it
                condition_value = config.get(dependency.condition_param)
                
                # Apply default if condition parameter not set
                if condition_value is None and condition_param_obj and condition_param_obj.default_value is not None:
                    condition_value = condition_param_obj.default_value
                
                if condition_value == dependency.condition_value:
                    # Condition is met, check required parameter(s)
                    
                    # Handle single parameter dependency
                    if dependency.required_param:
                        required_value = config.get(dependency.required_param)
                        
                        if required_param_obj is None:
                            errors.append(f"Dependency error: Required parameter '{dependency.required_param}' not found in group")
                            continue
                        
                        # Apply default value if parameter not in config
                        if required_value is None and required_param_obj.default_value is not None:
                            required_value = required_param_obj.default_value
                        
                        # Check dependency condition
                        if dependency.required_condition == "required":
                            if required_value is None:
                                errors.append(dependency.error_message)
                        elif dependency.required_condition == "required_gt_zero":
                            if required_value is None or (isinstance(required_value, (int, float)) and required_value <= 0):
                                errors.append(dependency.error_message)
                        elif dependency.required_condition == "must_be_zero":
                            if required_value is not None and isinstance(required_value, (int, float)) and required_value != 0:
                                errors.append(dependency.error_message)
                    
                    # Handle multiple parameter dependencies
                    elif dependency.required_params:
                        for param_key, expected_value in dependency.required_params.items():
                            param_obj = self.get_parameter(param_key)
                            if param_obj and param_obj.category == category:
                                param_value = config.get(param_key)
                                
                                if param_obj.default_value is not None and param_value is None:
                                    param_value = param_obj.default_value
                                
                                # Check condition based on required_condition type
                                if dependency.required_condition == "must_equal":
                                    if param_value != expected_value:
                                        errors.append(f"{dependency.error_message} (Expected {param_key}={expected_value}, got {param_value})")
                                elif dependency.required_condition == "required":
                                    if param_value is None:
                                        errors.append(f"{dependency.error_message} (Missing {param_key})")
                                elif dependency.required_condition == "required_gt_zero":
                                    if param_value is None or (isinstance(param_value, (int, float)) and param_value <= 0):
                                        errors.append(f"{dependency.error_message} (Parameter {param_key} must be > 0)")
        
        return errors


class ParameterRegistry(BaseModel):
    """Registry for all AMBER parameter groups."""
    groups: Dict[str, ParameterGroup] = Field(default_factory=dict, description="Parameter groups")
    
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
                if ((param.yaml_key and query_lower in param.yaml_key.lower()) or 
                    (param.description and query_lower in param.description.lower()) or
                    (param.amber_flag and query_lower in param.amber_flag.lower())):
                    results.append(param)
        
        return results

    def add_cross_group_dependency(
        self,
        condition_group: str,
        condition_param: str,
        condition_value: Any,
        target_group: str,
        required_params: Dict[str, Any],
        error_message: str
    ):
        """Declaratively add cross-group dependency rules."""
        # Store in a list for validation
        if not hasattr(self, '_cross_group_dependencies'):
            self._cross_group_dependencies = []
        
        self._cross_group_dependencies.append({
            'condition_group': condition_group,
            'condition_param': condition_param,
            'condition_value': condition_value,
            'target_group': target_group,
            'required_params': required_params,
            'error_message': error_message
        })

    def validate_cross_group_dependencies(self, configs: Dict[str, Dict[str, Any]]) -> List[str]:
        """Validate all registered cross-group dependencies."""
        errors = []
        
        if not hasattr(self, '_cross_group_dependencies'):
            return errors
        
        for dep in self._cross_group_dependencies:
            condition_config = configs.get(dep['condition_group'], {})
            condition_value = condition_config.get(dep['condition_param'])
            
            if condition_value == dep['condition_value']:
                target_config = configs.get(dep['target_group'], {})
                target_group = self.get_group(dep['target_group'])
                
                for param_key, expected_value in dep['required_params'].items():
                    param_value = target_config.get(param_key)
                    param_obj = target_group.get_parameter(param_key) if target_group else None
                    
                    if param_obj and param_obj.default_value is not None and param_value is None:
                        param_value = param_obj.default_value
                    
                    if param_value != expected_value:
                        errors.append(
                            f"{dep['error_message']} "
                            f"(Expected {param_key}={expected_value}, got {param_value})"
                        )
        
        return errors


def create_workflow_parameter_group() -> ParameterGroup:
    """Create workflow control parameter group."""
    group = ParameterGroup(
        name="workflow_control",
        description="Parameters for workflow control and directory management"
    )

    # Global workflow parameters
    # Heating control parameters
    group.add_parameter(AmberParameter(
        yaml_key="ramped_heating",
        amber_flag=None,  # Workflow parameter
        description="Enable ramped heating protocol",
        param_type=ParameterType.BOOLEAN,
        category=ParameterCategory.WORKFLOW,
        default_value=False
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="ramps",
        amber_flag=None,  # Workflow parameter
        description="Number of heating ramps",
        param_type=ParameterType.INT,
        category=ParameterCategory.WORKFLOW,
        default_value=0
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="windows",
        amber_flag=None,  # Workflow parameter
        description="Number of umbrella sampling windows",
        param_type=ParameterType.INT,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(min_value=1, max_value=100),
        default_value=10
    ))


    # Universal system settings that should NOT change
    group.add_parameter(AmberParameter(
        yaml_key="force_field",
        amber_flag=None,  # Workflow parameter
        description="Force field name",
        param_type=ParameterType.STRING,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(valid_values=["amber", "charmm", "gromos"]),
        default_value="amber"
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="water_model",
        amber_flag=None,  # Workflow parameter
        description="Water model",
        param_type=ParameterType.STRING,
        category=ParameterCategory.WORKFLOW,
        validation=ParameterValidation(valid_values=["TIP3P", "TIP4P", "SPCE"]),
        default_value="tip3p"
    ))
    
    return group

# Example usage and factory functions
def create_em_parameter_group() -> ParameterGroup:

    """Create energy minimization parameter group."""
    group = ParameterGroup(
        name="energy_minimization", # Grouped parameters
        description="Parameters for energy minimization protocol for simulations"
    )
    
    # Add parameters
    group.add_parameter(AmberParameter(
        yaml_key="min_method",
        amber_flag="ntmin",
        description="Minimization method",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3, 4, 5]),
        # notes="0=Molecular Dynamics, 1=Energy Minimization, 5=CG, 6=SD+CG, 7=SD+CG+MD"
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
        default_value=True
    ))

    group.add_parameter(AmberParameter(
        yaml_key="max_force",
        amber_flag="restraint_wt",
        description="Max force of restraint applied to indicated atoms (kcal/mol)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.RESTRAINT,
        default_value=10.0,
    ))

    group.add_parameter(AmberParameter(
        yaml_key="output_frequency",
        amber_flag="ntpr",
        description="Output frequency in simulation steps",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=50
    ))

    # This is one of the things that should stay consistent through each simulation?
    group.add_parameter(AmberParameter(
        yaml_key="nonbonded_cut",
        amber_flag="cut",
        description="Nonbonded Cutoff off for VdW interactions (Å)",
        param_type=ParameterType.FLOAT, 
        category=ParameterCategory.GENERAL,
        default_value=10.0
    ))

    return group


def create_nvt_parameter_group() -> ParameterGroup:
    """Create NVT ensemble parameter group."""
    group = ParameterGroup(
        name="nvt_ensemble",
        description="Parameters for NVT ensemble equilibrations"
    )

    group.add_parameter(AmberParameter(
        yaml_key="MD_method",
        amber_flag="imin",
        description="MD Method ",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(valid_values=[0, 1, 5, 6, 7]),
        notes="0=Molecular Dynamics, 1=Energy Minimization, 5=CG, 6=SD+CG, 7=SD+CG+MD"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="PBC_treatment",
        amber_flag="ntb",
        description="Periodic boundary condition",
        param_type=ParameterType.INT,
        category=ParameterCategory.GENERAL,
        notes="0=No periodicity, 1=Constant Volume, 2=Constant Pressure",
        default_value=1
    ))

    group.add_parameter(AmberParameter(
        yaml_key="timestep",
        amber_flag="dt",
        description="Timestep, in ps, of simulation",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.GENERAL, # NEEDS TO BE CONSISTENT THROUGHOUT SIMULATION ENSEMBLE
        default_value=0.002 # BUT NOT FOR HMASS REPARTITIONING
    ))

    group.add_parameter(AmberParameter(
        yaml_key="Force_calculation",
        amber_flag="ntf",
        description="Which forces to calculate?", # this should change between equil -> prod
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        notes="1=all interactions calculated, 2=bond interactions including H omitted(NTC=2), 3=all bond interactions are omitted (NTC=3), 4=Angles involving H-atom and all bonds omitted, 5=Bond and Angle interactions omitted, 6=Dihedrals involving H-atoms omitted, 7=Bond, Angle and Dihedral interactions omitted, 8=Bond, Angle, Dihedral, AND nonbonded interactions ommitted"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="SHAKE_param",
        amber_flag="ntc",
        description="SHAKE constraints for equilibrations",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL, # Generally turned off during production?
        default_value=2, # Is this default for heating sims??
        notes="1=No SHAKE constraints, 2=Hydrogen bonds constrained, 3=All bonds constrainted"
    
    ))

    group.add_parameter(AmberParameter(
        yaml_key="nonbonded_cut",
        amber_flag="cut",
        description="Nonbonded Cutoff off for VdW interactions (Å)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.CONTROL, # THIS MUST BE SET IN EM
        default_value=10.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="thermostat",
        amber_flag="ntt",
        description="Temperature control method",
        param_type=ParameterType.INT,
        category=ParameterCategory.THERMOSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3]),
        notes="0=Constant energy classical dynamics, 1=Constant temperature (weak coupling), 2=Andersen, 3=Langevian, 9=Optimized Isokinetic Nose-Hoover chain ensemble (OIN), 10=Stochastic Isokinetic Nose-Hoover RESPA integrator, 11=Stochastic Berendsen (Bussi) "
    ))
    
    group.add_parameter(AmberParameter(
        yaml_key="temperature",
        amber_flag="temp0",
        description="Target temperature (K)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.GENERAL, # heating_window[-1] == prod target temperature!
        validation=ParameterValidation(min_value=0.0, max_value=1000.0),
        default_value=300.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="steps",
        amber_flag="nstlim",
        description="Simulation steps",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        validation=ParameterValidation(min_value=0),
        default_value=100
    ))

    group.add_parameter(AmberParameter(
        yaml_key="restraint",
        amber_flag="ntr",
        description="Enable positional restraints",
        param_type=ParameterType.BOOLEAN,
        category=ParameterCategory.RESTRAINT,
        default_value=True
    ))

    group.add_parameter(AmberParameter(
        yaml_key="max_force",
        amber_flag="restraint_wt",
        description="Max force of restraint applied to indicated atoms (kcal/mol)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.RESTRAINT,
        default_value=10.0,
    ))

    group.add_parameter(AmberParameter(
        yaml_key="Collision_frequency",
        amber_flag="gamma_ln",
        description="Collision frequency in ps^-1 (Required with Langevian thermostat! ntt=3)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.THERMOSTAT,
        default_value=0.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="heat_bath_coupling_constant",
        amber_flag="tautp",
        description="Time constant, in ps, for heat bath coupling for the system (Required with Consntant Temperaure, weak coupling! ntt=1)",
        param_type=ParameterType.FLOAT,
        category=ParameterCategory.THERMOSTAT,
        default_value=0.0
    ))

    group.add_parameter(AmberParameter(
        yaml_key="read_prev_coordinates",
        amber_flag="ntx",
        description="Read in previous coordinates from input file",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=1,
        notes="1=Read coordinates, but not velocities, 5= Read coordinates AND velocities"
    ))

    group.add_parameter(AmberParameter(
        yaml_key="restart_sim",
        amber_flag="irest",
        description="Restart Simulation from provided input file?",
        param_type=ParameterType.INT,
        category=ParameterCategory.CONTROL,
        default_value=0,
        notes="0=Do not Restart simulation, 1=read coordinates AND velocities to continue simulation"
    
    ))

    # Add THERMOSTAT category dependency rules using ParameterDependency
    # Declarative approach: Easy to add, modify, or remove dependencies
    # These are for parameteres WITHIN THE SAME GROUP THAT MUST BE CONSISTENT, EASY!
    group.add_dependency(ParameterDependency(
        condition_param="thermostat",
        condition_value=3,  # Langevin thermostat
        required_param="Collision_frequency",
        required_condition="required_gt_zero",
        error_message="THERMOSTAT dependency: 'Collision_frequency' (gamma_ln) must be > 0 when using Langevin thermostat (thermostat=3)"
    ))
    
    group.add_dependency(ParameterDependency(
        condition_param="thermostat",
        condition_value=1,  # Weak coupling thermostat
        required_param="heat_bath_coupling_constant",
        required_condition="required_gt_zero",
        error_message="THERMOSTAT dependency: 'heat_bath_coupling_constant' (tautp) must be > 0 when using weak coupling thermostat (thermostat=1)"
    ))

    return group


# Example usage
if __name__ == "__main__":
    # Create registry
    registry = ParameterRegistry()
    
    # Add groups
    registry.add_group(create_em_parameter_group())
    registry.add_group(create_nvt_parameter_group())
    registry.add_group(create_workflow_parameter_group())
    
    # Test parameter retrieval
    param = registry.get_parameter("method", "energy_minimization")
    print(f"Found AMBER parameter: {param}")

    # Test validation
    em_config = {
        "method": 1,
        "steps": 5000,
        "restraint": True
    }
    
    em_group = registry.get_group("energy_minimization")
    is_valid, errors = em_group.validate_config(em_config)
    print(f"Config valid: {is_valid}, Errors: {errors}")

    print(em_group.get_parameter("steps"))
    

    shake_config = {
        "Force_calculation": 2,
        "SHAKE_param": 2,
        "water_model": "SPCE"
    }

    # Example of cross-group dependency 
    # Set up cross-group dependency: TIP3P water model requires NTF=2 and NTC=2

    # this is something that I would need to do when checking the consistency of parameters BETWEEN GROUPS
    # such as time step for NVT/NPT MUST EQUAL TIMESTEP FOR PROD!
    registry.add_cross_group_dependency(
        condition_group="workflow_control",
        condition_param="water_model",
        condition_value="tip3p",  # Note: lowercase to match validation
        target_group="nvt_ensemble",
        required_params={"Force_calculation": 2, "SHAKE_param": 2},
        error_message="TIP3P Water Model requires NTF=NTC=2!"
    )

    # Add cross-group dependency tests after Example 5 (around line 924):
    print("\n" + "="*60)
    print("CROSS-GROUP Dependency Validation Examples (TIP3P)")
    print("="*60)

    # Example 6: Valid - TIP3P with correct NTF=2 and NTC=2
    print("\n6. Valid - TIP3P water model with NTF=2 and NTC=2:")
    workflow_config_valid = {
        "water_model": "tip3p",
        "force_field": "amber",
        "windows": 10
    }
    nvt_config_valid_tip3p = {
        "Force_calculation": 2,  # NTF=2 ✓
        "SHAKE_param": 2,  # NTC=2 ✓
        "thermostat": 3,
        "temperature": 300.0
    }
    all_configs_valid = {
        "workflow_control": workflow_config_valid,
        "nvt_ensemble": nvt_config_valid_tip3p
    }
    cross_errors = registry.validate_cross_group_dependencies(all_configs_valid)
    print(f"   Valid: {len(cross_errors) == 0}")
    if cross_errors:
        print(f"   Errors: {cross_errors}")

    # Example 7: Invalid - TIP3P with wrong NTF (should be 2, got 1)
    print("\n7. Invalid - TIP3P water model with NTF=1 (should be 2):")
    workflow_config_tip3p = {
        "water_model": "tip3p",
        "force_field": "amber"
    }
    nvt_config_invalid_ntf = {
        "Force_calculation": 1,  # NTF=1 ✗ (should be 2)
        "SHAKE_param": 2,  # NTC=2 ✓
        "thermostat": 3,
        "temperature": 300.0
    }
    all_configs_invalid_ntf = {
        "workflow_control": workflow_config_tip3p,
        "nvt_ensemble": nvt_config_invalid_ntf
    }
    cross_errors = registry.validate_cross_group_dependencies(all_configs_invalid_ntf)
    print(f"   Valid: {len(cross_errors) == 0}")
    if cross_errors:
        print(f"   Errors: {cross_errors}")

    # Example 8: Invalid - TIP3P with wrong NTC (should be 2, got 1)
    print("\n8. Invalid - TIP3P water model with NTC=1 (should be 2):")
    nvt_config_invalid_ntc = {
        "Force_calculation": 2,  # NTF=2 ✓
        "SHAKE_param": 1,  # NTC=1 ✗ (should be 2)
        "thermostat": 3,
        "temperature": 300.0
    }
    all_configs_invalid_ntc = {
        "workflow_control": workflow_config_tip3p,
        "nvt_ensemble": nvt_config_invalid_ntc
    }
    cross_errors = registry.validate_cross_group_dependencies(all_configs_invalid_ntc)
    print(f"   Valid: {len(cross_errors) == 0}")
    if cross_errors:
        print(f"   Errors: {cross_errors}")

    # Example 9: Valid - SPCE water model (no restrictions)
    print("\n9. Valid - SPCE water model (no TIP3P restrictions):")
    workflow_config_spce = {
        "water_model": "spce",
        "force_field": "amber"
    }
    nvt_config_spce = {
        "Force_calculation": 1,  # Can be any value for SPCE
        "SHAKE_param": 1,  # Can be any value for SPCE
        "thermostat": 3,
        "temperature": 300.0
    }
    all_configs_spce = {
        "workflow_control": workflow_config_spce,
        "nvt_ensemble": nvt_config_spce
    }
    cross_errors = registry.validate_cross_group_dependencies(all_configs_spce)
    print(f"   Valid: {len(cross_errors) == 0}")
    if cross_errors:
        print(f"   Errors: {cross_errors}")