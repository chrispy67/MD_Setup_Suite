"""Simulation directory setup and management."""

import os
import shutil
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
    
    def _prompt_user_for_overwrite(self, existing_dirs: List[Path], missing_dirs: List[Path] = None) -> str:
        """Prompt user with a menu to choose how to handle existing directories.
        
        Args:
            existing_dirs: List of existing directory paths
            missing_dirs: List of missing directory paths that could be created (optional)
            
        Returns:
            str: "overwrite", "skip", or "add_missing"
        """
        print("\n" + "="*70)
        print("⚠️  WARNING: The following directories already exist:")
        for dir_path in existing_dirs:
            print(f"   - {dir_path}")
        print("="*70)
        
        if missing_dirs:
            print(f"\n📁 Additionally, {len(missing_dirs)} window(s) need to be created:")
            for dir_path in missing_dirs:
                print(f"   - {dir_path}")
        
        print("\nWhat would you like to do?")
        print("  1) Overwrite existing directories (delete all data and recreate from scratch)")
        print("  2) Skip (do not create or overwrite)")
        if missing_dirs:
            print(f"  3) Keep existing directories and add missing windows ({len(missing_dirs)} window(s))")
        print()
        
        while True:
            try:
                max_choice = "3" if missing_dirs else "2"
                choice = input(f"Enter your choice (1-{max_choice}): ").strip()
                if choice == "1":
                    return "overwrite"
                elif choice == "2":
                    return "skip"
                elif choice == "3" and missing_dirs:
                    return "add_missing"
                else:
                    print(f"❌ Invalid choice. Please enter 1-{max_choice}.")
            except (EOFError, KeyboardInterrupt):
                print("\n\n⚠️  Interrupted by user. Skipping directory creation.")
                return "skip"
    
    def build_directories(self, system_name: str, window_num: int = None, optional: str = "") -> Union[Path, List[Path]]:
        """Build directories for a single window or all windows if window_num is None.
        
        If directories already exist, the user will be prompted interactively to choose
        whether to overwrite them, skip creation, or add missing windows.
        
        Args:
            system_name: Name of the system
            window_num: Window number (None for all windows)
            optional: Optional string to append to directory name
        
        Returns:
            Path or List[Path]: Created directory path(s), or None if skipped
        """
        
        # Get base path and naming pattern from config
        base_path = Path(self.cfg.directories.base_path)
        naming_pattern = self.cfg.directories.naming_pattern
        subdirectories = self.cfg.directories.subdirectories
        
        # Check for existing and missing directories first
        existing_dirs = []
        missing_dirs = []
        dirs_to_create = []
        
        if window_num is None:
            num_windows = self.cfg["global"]["windows"]
            for i in range(0, num_windows):
                dir_path = self._get_window_directory_path(
                    system_name, i, optional, base_path, naming_pattern
                )
                dirs_to_create.append((i, dir_path))
                if dir_path.exists():
                    existing_dirs.append(dir_path)
                else:
                    missing_dirs.append(dir_path)
        else:
            dir_path = self._get_window_directory_path(
                system_name, window_num, optional, base_path, naming_pattern
            )
            dirs_to_create.append((window_num, dir_path))
            if dir_path.exists():
                existing_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        # If any directories exist, prompt the user
        action = "create"  # Default: create all as this is default behavior (no existing dirs)
        if existing_dirs:
            action = self._prompt_user_for_overwrite(existing_dirs, missing_dirs if missing_dirs else None)
            if action == "skip":
                logger.info("User chose to skip directory creation.")
                return None if len(dirs_to_create) == 1 else []
        
        # Determine which directories to create based on user's choice
        if action == "add_missing":
            # Only create missing directories, keep existing ones
            dirs_to_create = [(w, d) for w, d in dirs_to_create if d not in existing_dirs]
            overwrite = False
        elif action == "overwrite":
            # Create all directories, overwriting existing ones
            overwrite = True
        else:  # action == "create" (no existing dirs)
            # Create all directories normally
            overwrite = False
        
        # Create directories
        created_dirs = []
        for window_num, dir_path in dirs_to_create:
            result = self._create_single_window_directories(
                system_name, window_num, optional, base_path, naming_pattern, subdirectories, overwrite
            )
            if result is not None:
                created_dirs.append(result)
        
        return created_dirs if len(created_dirs) > 1 else created_dirs[0] if created_dirs else None
    
    def _get_window_directory_path(self, system_name: str, window_num: int, optional: str,
                                   base_path: Path, naming_pattern: str) -> Path:
        """Get the directory path for a window without creating it."""
        if optional:
            main_dir_name = naming_pattern.format(
                system_name=system_name,
                window_num=f"window_{window_num}",
                optional=optional
            )
        else:
            pattern_without_optional = naming_pattern.replace("_{optional}", "")
            main_dir_name = pattern_without_optional.format(
                system_name=system_name,
                window_num=f"window_{window_num}"
            )
        return base_path / main_dir_name
    
    def _create_single_window_directories(self, system_name: str, window_num: int, optional: str,
                                        base_path: Path, naming_pattern: str, subdirectories: list, overwrite: bool = False):
        """Create directories for a single window.
        
        Args:
            system_name: Name of the system
            window_num: Window number
            optional: Optional string to append to directory name
            base_path: Base path for directories
            naming_pattern: Pattern for directory naming
            subdirectories: List of subdirectories to create
            overwrite: If True, overwrite existing directory
        
        Returns:
            Path: Created directory path, or None if skipped
        """
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
        
        # Handle overwrite option
        if main_dir.exists():
            if overwrite:
                logger.info(f"Overwriting existing directory: {main_dir}")
                shutil.rmtree(main_dir)
            else:
                logger.info(f"Directory {main_dir} already exists. Skipping creation.")
                return None
        
        # Create the main directory
        main_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for each simulation type
        for sim_type, sim_config in self.cfg.simulations.items():
            # Get the subdirectory path from config
            sim_subdir = main_dir / Path(sim_config.subdirectory).name
            
            # Create simulation-specific directory
            sim_subdir.mkdir(parents=True, exist_ok=True)
            
            # Create standard subdirectories within each simulation window
            for subdir in subdirectories:
                (sim_subdir / subdir).mkdir(exist_ok=True)
        
        return main_dir

    def _distribute_input_cards(self, base_path: Path):
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

