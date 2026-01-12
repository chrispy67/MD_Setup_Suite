> **Under Construction**

This project is actively being developed! Features, documentation, and interfaces are subject to update as we work toward a stable release.


# Introduction

Welcome to the **Molecular Dynamics Setup Suite**!  
This project aims to provide a **robust, reproducible, and extensible system** for running molecular dynamics (MD) simulations with enhanced sampling protocols.  
This modular, declarative approach allows users and contributors to **easily add, modify, or extend simulation parameters and workflows in AMBER**.

---

# Installation

1. **Clone the repository**
    ```bash
    git clone https://github.com/chrispy67/MD_Setup_Suite
    ```


2. **Install dependencies**
    ```bash
    pip install -r environment.yml
    ```

---

## Directory Structure

Below is a high-level overview of the project structure and schematics:

<details>
<summary><strong>Project Hierarchy</strong> (click to expand)</summary>

```
MD_Setup_Suite/
│
├── simulation_setup.py        # Entrypoint: high-level workflow orchestration (WIP)
├── requirements.txt
├── README.md
│
├── config/
│   ├── min_BLANK.in           # simple template files defining basic MD ensembles
│   ├── heat_BLANK.in
│   ├── equil_BLANK.in
│   ├── simulation_config.yaml # config that is parsed and takes runtime user values.  
├── src/
│   ├── models/                # Core objects (Parameter, ParameterGroup, Dependency, simple validations, etc)
│   │   ├── parameter.py
│   │   ├── group.py
│   │   ├── dependency.py
│   │   └── ...
│   ├── enums.py                 # Enumerations for types and choices
│   ├── utils/                 # Utility functions for parsing, display, validation
│   │   ├──parameter_display.py
│   │   ├──simulation_mappings.py # User flexibility for YAML key and simulation order
│   │   ├──registry_helpers.py    # Helper functions 
│   └── parameter_groups/      # Parameter groups that define/are necessary for common MD ensembles  
│       ├── energy_minimization.py
│       ├── nvt_ensemble.py
│       ├── npt_ensemble.py
│       ├── production.py
│       └── (add more as needed)
```
</details>

---

## Adding or Editing Parameter Groups

This is the format that is expected when adding an AMBER parameter to a simulation group:
```python
class AmberParameter(BaseModel):
    yaml_key: str = Field(...) # "human-readable" key in YAML
    amber_flag: Optional[str] = Field(default=None) # AMBER flag equivalent to yaml_key, but usable by AMBER MD engine
    description: str = Field(..., description="Human-readable description") # used to generate summary of ParameterGroup
    param_type: ParameterType = Field(..., description="Parameter data type")
    category: ParameterCategory = Field(default=ParameterCategory.GENERAL, description="Parameter category")
    
    # Optional fields
    default_value: Optional[Any] = Field(default=None, description="Default value") # default values can be set in src/parameter_groups
    validation: Optional[ParameterValidation] = Field(default=None, description="Validation rules") # SIMPLE validation rules
    notes: Optional[str] = Field(default=None, description="Additional notes") # This 
```

You can add custom parameters and mappings to ensembles, create your own simulation ensembles, and create your own validation rules with the `add_parameter()` function. 

For example,here you can control the barostat that is used for you NPT equilibration run(s). Notice a default value is set to define this ensemble. Furthermore, the organization of the `notes` field in this example allows the program to display and summarize the selected `AmberParameter`. This is important for `parameter_display.py`

```python
    group.add_parameter(AmberParameter(
        yaml_key="pressure_control",
        amber_flag="ntp",
        description="Pressure control method (NOT THE BARSOSTAT): {value}",
        param_type=ParameterType.INT,
        category=ParameterCategory.BAROSTAT,
        validation=ParameterValidation(valid_values=[0, 1, 2, 3, 4])default_value=1, # Isotropic is the default for NPT ensemble
        notes="""0=No pressure control, 
        1=Isotropic 
        2=Anisotropic 
        3=Semi-isotropic
        4=MD to target volume (REMD)"""
    ))
```

Within one ensemble, you can enforce simple validation rules based on dependencies indicated by AMBER. While AMBER will likely catch issues with required parameters based on a certain thermostat, barostat, PBC treatment, etc. While this isn't novel, catching these issues prior to setup/runtime is important and an opprotunity to learn the finer points of running an MD simulation. This can be as simple as making sure users select an AMBER flag that actually exists, or making sure we pick parameters that are necessary to a method. 

For example, you need to provide (or at least consider) the collision frequency parameter when using a Langevian thermostat. 
```python
    group.add_dependency(ParameterDependency(
        condition_param="thermostat",
        condition_value=3,  # Langevin thermostat
        required_param="Collision_frequency",
        required_condition="required_gt_zero",
        error_message="THERMOSTAT dependency: 'Collision_frequency' (gamma_ln) must be > 0 when using Langevin thermostat (thermostat=3)"
    ))
```

> [!TIP] 
> The system automatically detects and registers new parameter groups as long as you follow the convention shown above. There are plenty of examples and various use cases found in `src/parameter_groups/`*.

The main goal of this project is to make MD simulations more accessible, reproducible, and less prone to errors when dealing with complex directory hirearchies. Most MD engines are completely complacent letting you do bad science such as:
- Setting a temperature outside physical plausibility (e.g., 10,000 K or -100 K)
- Using a timestep that's too large for hydrogen integration, leading to instability
- Omitting SHAKE/constraints where bonds to hydrogens demand it, resulting in unphysical motions
- Forgetting to set positional restraints during minimization or equilibration, causing the structure to drift or distort
- Setting non-bonded cutoffs inconsistently, causing artifacts at interfaces or periodic boundaries


this can be avoided with `add_cross_group_depdendency()`. There are limited examples currently, but here is how the `timestep` is checked for consistency across all parts of a simulation:

```python
    for i in range(len(selected_simulation_ensemble) - 1):
        src = selected_simulation_ensemble[i]
        tgt = selected_simulation_ensemble[i + 1]

        tgt_config = group_configs.get(tgt, {})
        src_config = group_configs.get(src, {})
        timestep = src_config.get("timestep")

        if timestep is not None:
            registry.add_cross_group_dependency(
                condition_group=src,
                condition_param="timestep",
                condition_value=src_config.get("timestep"),
                target_group=tgt,
                required_params={"timestep": src_config.get("timestep")},
                error_message=f"All simulations must use the same timestep (ps)! |  {src} is {src_config.get('timestep')}ps and {tgt} is {tgt_config.get('timestep')}ps"
            )
```

## Summarizing Simulation 

# Examples

***WIP***
Here I want to have a variety of YAML files that can create ready-to-go simulation ensembles or directory hirearchies.

## Contributing

This project started as a collaboration started with [UW CTMR](https://ctmr.washington.edu/) to support their MD simulations with AMBER. As of writing, this program only supports the AMBER engine. However, the format of this project is meant to accomodate other engines. For example, I would like to suppport GROMACS parameters with a `GromacsParameter()` class. 

---

**Ready to build upon! To add or subtract from the simulation parameterization, edit or add Python files in [`src/parameter_groups/`](src/parameter_groups/).**


