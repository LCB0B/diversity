"""
Path configuration for Fashion Model Directory (FMD) data analysis project.

This module centralizes all data file paths to ensure consistency across
analysis scripts and make path management easier.
"""

import os
from pathlib import Path

# Base project directory (parent of src/)add this st
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
MODELS_HTML_DIR = PROJECT_ROOT / "models_html"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_ARXIV_DIR = PROJECT_ROOT / "figures_arxiv"

# Core data files
MODELS_SHOWS_ALL = DATA_DIR / "models_shows_all.json"
MODELS_MEASURE = DATA_DIR / "models_measure.csv"
MODELS_NATIONALITY = DATA_DIR / "models_nationality.csv"
MODELS_NATIONALITY_JSON = DATA_DIR / "models_nationality.json"

# Gender detection files
GENDER_DICT = DATA_DIR / "gender_dict.json"
NAME_GENDER_DICT = DATA_DIR / "name_gender_dataset.csv"
NAME_GENDER_DICT_LARGE = DATA_DIR / "wgnd_2_0_name-gender_nocode.csv"
US_CENSUS_GENDER_DICT = DATA_DIR / "us_census_gender_dict_90.json"
US_CENSUS_GENDER_COMPREHENSIVE = DATA_DIR / "us_census_gender_comprehensive.csv"


def ensure_directories_exist():
    """Create necessary directories if they don't exist."""
    directories = [DATA_DIR, FIGURES_DIR, FIGURES_ARXIV_DIR, MODELS_HTML_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_file_path(filename, base_dir=DATA_DIR):
    """
    Get full path for a file in the specified base directory.
    
    Args:
        filename (str): Name of the file
        base_dir (Path): Base directory (default: DATA_DIR)
    
    Returns:
        Path: Full path to the file
    """
    return base_dir / filename

def check_file_exists(file_path):
    """
    Check if a file exists and return boolean result.
    
    Args:
        file_path (Path or str): Path to check
    
    Returns:
        bool: True if file exists, False otherwise
    """
    return Path(file_path).exists()

def get_required_files():
    """
    Return list of required files for basic analysis.
    
    Returns:
        list: List of Path objects for required files
    """
    return [
        MODELS_SHOWS_ALL,
        MODELS_MEASURE,
        MODELS_NATIONALITY,
        GENDER_DICT
    ]

def validate_data_files():
    """
    Validate that required data files exist.
    
    Returns:
        dict: Dictionary with file paths as keys and existence status as values
    """
    required_files = get_required_files()
    validation_results = {}
    
    for file_path in required_files:
        validation_results[str(file_path)] = check_file_exists(file_path)
    
    return validation_results

def print_data_status():
    """Print status of all important data files."""
    print("=" * 60)
    print("FMD DATA FILES STATUS")
    print("=" * 60)
    
    # Core data files
    print("\nCORE DATA FILES:")
    core_files = [
        MODELS_SHOWS_ALL,
        MODELS_MEASURE, 
        MODELS_NATIONALITY,
        GENDER_DICT
    ]
    
    for file_path in core_files:
        status = "✓ EXISTS" if check_file_exists(file_path) else "✗ MISSING"
        size = ""
        if check_file_exists(file_path):
            size_mb = file_path.stat().st_size / (1024 * 1024)
            size = f" ({size_mb:.1f} MB)"
        print(f"  {status:<10} {file_path.name}{size}")
    
    # Analysis results
    print("\nANALYSIS RESULTS:")
    analysis_files = [
        TEMPORAL_DOUBLE_DIVERSITY_FEMALE_QUANTILE,
        CENTRALITY_SUMMARY,
        INDIVIDUAL_BRAND_DIVERSITY_RESULTS
    ]
    
    for file_path in analysis_files:
        status = "✓ EXISTS" if check_file_exists(file_path) else "✗ MISSING"
        print(f"  {status:<10} {file_path.name}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    # When run directly, print data status
    print_data_status()