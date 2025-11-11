"""AMBER parameter model with validation."""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Any
import json

from src.enums import ParameterType, ParameterCategory
from src.models.validation import ParameterValidation


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
        # All this does is enforce basic rules that AMBER will generally catch, but MDSS should intervene
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
