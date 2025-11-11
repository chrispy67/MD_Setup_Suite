"""Simulation directory setup and management."""

import os
from pathlib import Path
from typing import Dict, Any, List, Union
from omegaconf import DictConfig
import logging

# Set up logging
logger = logging.getLogger(__name__)


class SimulationSetup:
    """Class to handle simulation directory setup and file generation/modification."""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
    
    def build_directories(self, system_name: str, window_num: int = None, optional: str = "") -> Union[Path, List[Path]]:
        """Build directories for a single window or all windows if window_num is None."""
        
        # Get base path and naming pattern from config
        base_path = Path(self.cfg.directories.base_path)
        naming_pattern = self.cfg.directories.naming_pattern
        subdirectories = self.cfg.directories.subdirectories
        
        created_dirs = []
        
        # If window_num is None, create directories for all windows
        if window_num is None:
            num_windows = self.cfg["global"]["windows"]
            for i in range(0, num_windows):
                dir_path = self._create_single_window_directories(
                    system_name, i, optional, base_path, naming_pattern, subdirectories
                )
                created_dirs.append(dir_path)
        else:
            # Create directory for a specific window
            dir_path = self._create_single_window_directories(
                system_name, window_num, optional, base_path, naming_pattern, subdirectories
            )
            created_dirs.append(dir_path)
        
        return created_dirs if len(created_dirs) > 1 else created_dirs[0]
    
    def _create_single_window_directories(self, system_name: str, window_num: int, optional: str,
                                        base_path: Path, naming_pattern: str, subdirectories: list):
        
        # Build main directory name using the pattern
        # Handle optional underscore: only add it if optional string is not empty
        if optional:
            main_dir_name = naming_pattern.format(
                system_name=system_name,
                window_num=f"window_{window_num}",
                optional=optional
            )

        else:
            # Create pattern without optional part and its underscore
            pattern_without_optional = naming_pattern.replace("_{optional}", "")
            main_dir_name = pattern_without_optional.format(
                system_name=system_name,
                window_num=f"window_{window_num}"
            )
        
        # Create main directory path
        main_dir = base_path / main_dir_name
        
        # Create the main directory | ADD LOGIC FOR RESTARTING/REBUILDING/OVERWRITING SIMULATION DIRECTORY?
        try:
            main_dir.mkdir(parents=True, exist_ok=False)

        except FileExistsError:
            print(f"Error! {main_dir} already exists!")
        
        # Create subdirectories for each simulation type
        for sim_type, sim_config in self.cfg.simulations.items():
            # Get the subdirectory path from config
            
            sim_subdir = main_dir / Path(sim_config.subdirectory).name
            
            # Create simulation-specific directory
            try:
                sim_subdir.mkdir(parents=True, exist_ok=False)
            
            except FileExistsError:
                print(f"Error! {sim_subdir} already exists!")
            
            # Create standard subdirectories within each simulation window
            for subdir in subdirectories:

                try:
                    (sim_subdir / subdir).mkdir(exist_ok=False)
                except FileExistsError:
                    print(f"Error! {subdir} already exists!")
        return main_dir

    def _distribute_input_cards(self, base_path: Path):
        import shutil

        # Source directory containing prepared input files
        input_files_dir = Path(__file__).parent / "input_files"

        if not input_files_dir.exists():
            logger.error(f"Input files directory not found: {input_files_dir}")
            return

        # Define simple distribution rules based on filename prefixes
        # - min_*.in   -> em/
        # - heat_*.in  -> NVT/
        # - equil_*.in -> NPT/
        # - prod_*.in -> prod/ (often in NVT ensemble)
        distribution_rules = [
            ("min_", "em"),
            ("heat_", "NVT"),
            ("equil_", "NPT"),
            ("prod_", "prod")
        ]

        # Iterate over each window directory inside base_path
        for window_dir in base_path.iterdir():
            if not window_dir.is_dir():
                continue

            for prefix, target_subdir_name in distribution_rules:
                destination_dir = window_dir / target_subdir_name  # simulations/my_protein_window0/em/min_*...
                if not destination_dir.exists():
                    # Skip silently if the target subdir is not present in this window
                    continue

                for src_file in input_files_dir.glob(f"{prefix}*.in"):
                    dest_file = destination_dir / src_file.name
                    try:
                        shutil.copy2(src_file, dest_file)
                        # logger.info(f"Copied {src_file.name} -> {destination_dir}")
                    except Exception as exc:
                        logger.error(
                            f"Failed to copy {src_file} to {dest_file}: {exc}"
                        )
        return

