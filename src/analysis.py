"""
Comprehensive analysis of Fashion Model Directory data.

This script provides a complete analysis of the fashion model dataset including:
- Model measurements and trends over time
- Gender distribution and analysis
- Nationality diversity and geographic patterns
- Fashion show participation and brand networks
- Body measurement evolution and standardization

Uses centralized path management and loads all core datasets.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
from scipy.stats import entropy
import os
from tqdm import tqdm  # Not available in current environment
warnings.filterwarnings('ignore')

# Import centralized path management
from src.paths import (
    MODELS_MEASURE, 
    MODELS_SHOWS_ALL, 
    MODELS_NATIONALITY, 
    DATA_DIR,
    FIGURES_DIR,
    ensure_directories_exist
)

def load_core_datasets():
    """
    Load all core datasets for comprehensive analysis.
    
    Returns:
        dict: Dictionary containing loaded dataframes
    """
    print("Loading core datasets...")
    
    datasets = {}
    
    # Load model measurements
    print("  Loading model measurements...")
    try:
        measurements_df = pd.read_csv(MODELS_MEASURE)
        datasets['measurements'] = measurements_df
        print(f"    Loaded {len(measurements_df):,} measurement records")
    except FileNotFoundError:
        print(f"    Error: {MODELS_MEASURE} not found")
        datasets['measurements'] = pd.DataFrame()
    
    # Load fashion shows data
    print("  Loading fashion shows data...")
    try:
        with open(MODELS_SHOWS_ALL, 'r', encoding='utf-8') as f:
            shows_data = json.load(f)
        
        # Convert to DataFrame
        shows_records = []
        for model_data in shows_data:
            model_id = model_data.get('model_id', '')
            model_name = model_data.get('model_name', '')
            shows = model_data.get('shows', [])
            
            for show in shows:
                # Convert year to integer, handle string years
                year = show.get('year', 0)
                try:
                    year = int(year) if year else 0
                except (ValueError, TypeError):
                    year = 0
                
                record = {
                    'model_id': model_id,
                    'model_name': model_name,
                    'brand': show.get('brand', ''),
                    'season': show.get('season', ''),
                    'year': year,
                    'city': show.get('location', ''),  # Note: using 'location' from the JSON
                    'show_type': show.get('category', '')  # Note: using 'category' from the JSON
                }
                shows_records.append(record)
        
        shows_df = pd.DataFrame(shows_records)
        datasets['shows'] = shows_df
        print(f"    Loaded {len(shows_df):,} fashion show records")
        
    except FileNotFoundError:
        print(f"    Error: {MODELS_SHOWS_ALL} not found")
        datasets['shows'] = pd.DataFrame()
    except json.JSONDecodeError:
        print(f"    Error: Invalid JSON in {MODELS_SHOWS_ALL}")
        datasets['shows'] = pd.DataFrame()
    
    # Load nationality data
    print("  Loading nationality data...")
    try:
        nationality_df = pd.read_csv(MODELS_NATIONALITY)
        datasets['nationality'] = nationality_df
        print(f"    Loaded {len(nationality_df):,} nationality records")
    except FileNotFoundError:
        print(f"    Error: {MODELS_NATIONALITY} not found")
        datasets['nationality'] = pd.DataFrame()
    
    # Load gender data (if available)
    print("  Loading gender data...")
    gender_file = DATA_DIR / "models_gender.csv"
    try:
        gender_df = pd.read_csv(gender_file)
        datasets['gender'] = gender_df
        print(f"    Loaded {len(gender_df):,} gender records")
    except FileNotFoundError:
        print(f"    Gender data not found at {gender_file}")
        print("    Run src/get_gender.py first to generate gender classifications")
        datasets['gender'] = pd.DataFrame()
    
    return datasets

def create_master_dataset(datasets):
    """
    Create a comprehensive master dataset by merging all data sources.
    
    Args:
        datasets (dict): Dictionary of loaded dataframes
        
    Returns:
        pd.DataFrame: Master dataset with all information
    """
    print("\nCreating master dataset...")
    
    # Start with measurements as the base (most complete dataset)
    if datasets['measurements'].empty:
        print("  Error: No measurements data available")
        return pd.DataFrame()
    
    master_df = datasets['measurements'].copy()
    print(f"  Starting with {len(master_df):,} measurement records")
    
    # Add gender information
    if not datasets['gender'].empty:
        # Merge on model_id and name for better matching
        gender_cols = ['model_id', 'name', 'gender_consensus', 'first_name']
        available_gender_cols = [col for col in gender_cols if col in datasets['gender'].columns]
        
        master_df = master_df.merge(
            datasets['gender'][available_gender_cols],
            on=['model_id', 'name'],
            how='left'
        )
        print(f"  Added gender information")
        
        # Check gender coverage
        gender_coverage = master_df['gender_consensus'].notna().sum()
        print(f"    Gender coverage: {gender_coverage:,}/{len(master_df):,} ({gender_coverage/len(master_df)*100:.1f}%)")
    
    # Add nationality information
    if not datasets['nationality'].empty:
        nationality_cols = ['model_id', 'name', 'nationality', 'birth_place', 'birth_year']
        available_nat_cols = [col for col in nationality_cols if col in datasets['nationality'].columns]
        
        master_df = master_df.merge(
            datasets['nationality'][available_nat_cols],
            on=['model_id', 'name'],
            how='left'
        )
        print(f"  Added nationality information")
        
        # Check nationality coverage
        nat_coverage = master_df['nationality'].notna().sum()
        print(f"    Nationality coverage: {nat_coverage:,}/{len(master_df):,} ({nat_coverage/len(master_df)*100:.1f}%)")
    
    # Skip show participation statistics calculation as requested
    
    print(f"\n  Master dataset created with {len(master_df):,} records and {len(master_df.columns)} columns")
    
    return master_df

def analyze_data_quality(master_df):
    """
    Analyze data quality and completeness across all datasets.
    
    Args:
        master_df (pd.DataFrame): Master dataset
    """
    print("\nDATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    print(f"Total records: {len(master_df):,}")
    print(f"Total columns: {len(master_df.columns)}")
    print(f"Memory usage: {master_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Column completeness analysis
    print(f"\nColumn Completeness:")
    print("-" * 40)
    
    completeness = []
    for col in master_df.columns:
        non_null = master_df[col].notna().sum()
        completeness.append({
            'column': col,
            'non_null_count': non_null,
            'completeness_pct': non_null / len(master_df) * 100,
            'data_type': str(master_df[col].dtype)
        })
    
    completeness_df = pd.DataFrame(completeness).sort_values('completeness_pct', ascending=False)
    
    # Show top complete and incomplete columns
    print("Most complete columns:")
    for _, row in completeness_df.head(10).iterrows():
        print(f"  {row['column']:<25} {row['completeness_pct']:>6.1f}% ({row['non_null_count']:,})")
    
    print("\nLeast complete columns:")
    for _, row in completeness_df.tail(10).iterrows():
        print(f"  {row['column']:<25} {row['completeness_pct']:>6.1f}% ({row['non_null_count']:,})")
    
    # Key measurement completeness
    measurement_cols = ['height', 'bust', 'waist', 'hips', 'dress_size', 'shoe_size']
    available_measurement_cols = [col for col in measurement_cols if col in master_df.columns]
    
    if available_measurement_cols:
        print(f"\nKey Measurements Completeness:")
        print("-" * 40)
        for col in available_measurement_cols:
            if col in master_df.columns:
                count = master_df[col].notna().sum()
                pct = count / len(master_df) * 100
                print(f"  {col:<15} {pct:>6.1f}% ({count:,})")
    
    # Gender distribution
    if 'gender_consensus' in master_df.columns:
        print(f"\nGender Distribution:")
        print("-" * 40)
        gender_counts = master_df['gender_consensus'].value_counts(dropna=False)
        for gender, count in gender_counts.items():
            pct = count / len(master_df) * 100
            print(f"  {str(gender):<15} {pct:>6.1f}% ({count:,})")
    
    # Nationality diversity
    if 'nationality' in master_df.columns:
        print(f"\nNationality Diversity:")
        print("-" * 40)
        nat_counts = master_df['nationality'].value_counts()
        print(f"  Total countries: {len(nat_counts)}")
        print(f"  Top 5 countries:")
        for country, count in nat_counts.head(5).items():
            pct = count / master_df['nationality'].notna().sum() * 100
            print(f"    {country:<20} {pct:>6.1f}% ({count:,})")
    
    # Show participation statistics removed as requested

def clean_measurements(df):
    """
    Clean and convert measurement columns to numeric values.
    """
    df = df.copy()
    
    print("Cleaning measurements...")
    # Convert height-metric to numeric (remove any non-numeric characters)
    df['height_cm'] = pd.to_numeric(df['height-metric'], errors='coerce')
    
    # Convert US measurements to numeric
    for col in tqdm(['bust-us', 'waist-us', 'hips-us'], desc="Processing US measurements"):
        # Remove any non-numeric characters and convert to float
        df[col + '_clean'] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col + '_clean'] = pd.to_numeric(df[col + '_clean'], errors='coerce')
    
    # Convert EU measurements to numeric
    for col in tqdm(['bust-eu', 'waist-eu', 'hips-eu'], desc="Processing EU measurements"):
        df[col + '_clean'] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df[col + '_clean'] = pd.to_numeric(df[col + '_clean'], errors='coerce')
    
    # Convert dress sizes to numeric
    df['dress-us_clean'] = pd.to_numeric(df['dress-us'], errors='coerce')
    df['dress-eu_clean'] = pd.to_numeric(df['dress-eu'], errors='coerce')
    
    return df

def remove_spines(ax):
    """
    Remove top and right spines from plot.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_yearly_counts_by_gender(merged_df, start_year=2000):
    """
    Plot the number of data points (models) per year, split by gender.
    """
    merged_df = merged_df[merged_df['year'] >= start_year]
    fig, ax = plt.subplots(figsize=(15, 6))
    
    # Get yearly counts by gender
    yearly_gender = merged_df.groupby(['year', 'gender_consensus']).size().unstack(fill_value=0)
    
    # Create stacked bar chart
    yearly_gender.plot(kind='bar', stacked=True, ax=ax, alpha=0.7, 
                      color=['lightcoral', 'lightblue', 'lightgray'])
    
    ax.set_title('Number of Fashion Show Records per Year by Gender', fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of Records')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Gender')
    remove_spines(ax)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'yearly_counts_by_gender.png', dpi=300, bbox_inches='tight')
    ax.set_yscale('log')  # Use log scale for better visibility of trends 
    plt.savefig(FIGURES_DIR / 'yearly_counts_by_gender_log.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_measurements_evolution_by_gender(merged_df, gender_filter='female', start_year=2000, measurement_system='us'):
    """
    Plot the evolution of measurements over time for a specific gender.
    """
    # Filter by gender and start year
    df_filtered = merged_df[
        (merged_df['gender_consensus'] == gender_filter) & 
        (merged_df['year'] >= start_year)
    ]
    
    if len(df_filtered) == 0:
        print(f"No data available for gender: {gender_filter} from year {start_year}")
        return
    
    # Define measurements based on system
    if measurement_system.lower() == 'us':
        measurements = [
            ('height_cm', 'Height (cm)'),
            ('bust-us_clean', 'Bust (inches)'),
            ('waist-us_clean', 'Waist (inches)'),
            ('hips-us_clean', 'Hips (inches)')
        ]
        system_label = 'US'
    elif measurement_system.lower() == 'eu':
        measurements = [
            ('height_cm', 'Height (cm)'),
            ('bust-eu_clean', 'Bust (cm)'),
            ('waist-eu_clean', 'Waist (cm)'),
            ('hips-eu_clean', 'Hips (cm)')
        ]
        system_label = 'EU'
    else:
        raise ValueError("measurement_system must be 'us' or 'eu'")
    
    # Create subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{gender_filter.title()} Model Measurements Evolution ({system_label} System) - From {start_year}', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    color = 'lightcoral' if gender_filter == 'female' else 'lightblue'
    
    for idx, (measurement_col, title) in enumerate(measurements):
        ax = axes_flat[idx]
        
        # Filter data for this measurement
        measurement_data = df_filtered.dropna(subset=[measurement_col])
        
        if len(measurement_data) == 0:
            ax.text(0.5, 0.5, f'No data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue
        
        # Calculate yearly statistics
        yearly_stats = measurement_data.groupby('year')[measurement_col].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        yearly_stats.columns = ['year', 'mean', 'std', 'count']
        
        # Filter years with sufficient data (at least 5 data points)
        yearly_stats = yearly_stats[yearly_stats['count'] >= 5]
        
        if len(yearly_stats) == 0:
            ax.text(0.5, 0.5, f'Insufficient data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            continue
        
        # Plot mean as line with std as error bars
        ax.plot(yearly_stats['year'], yearly_stats['mean'],
                 color='black', linewidth=1, markersize=4, label='Mean')
        ax.errorbar(yearly_stats['year'], yearly_stats['mean'],
                    yerr=yearly_stats['std'], fmt='none', 
                    ecolor='black', elinewidth=1, capsize=3, label='Std Dev')
         # Add linear fit
        z = np.polyfit(yearly_stats['year'], yearly_stats['mean'], 1)
        p = np.poly1d(z)
        ax.plot(yearly_stats['year'], p(yearly_stats['year']), 
                color='firebrick', linestyle='--', linewidth=2, label='Linear Fit') 
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        remove_spines(ax)
        
        # Add sample size annotations for some years
        for i, row in yearly_stats.iterrows():
            if i % max(1, len(yearly_stats) // 5) == 0:  # Show every 5th annotation
                ax.annotate(f'n={int(row["count"])}', 
                           xy=(row['year'], row['mean'] + row['std']), 
                           xytext=(0, 5), textcoords='offset points',
                           ha='center', fontsize=8, alpha=0.7)
        
        # Set x-axis to show reasonable number of ticks
        years = yearly_stats['year'].values
        if len(years) > 10:
            step = max(1, len(years) // 10)
            ax.set_xticks(years[::step])
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    filename = FIGURES_DIR / f'evolution_{gender_filter}_{measurement_system}_from_{start_year}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def plot_hair_color_evolution_by_gender(merged_df, start_year=2000):
    """
    Plot the evolution of hair colors over time by gender with two subplots.
    """
    df_filtered = merged_df[merged_df['year'] >= start_year]
    
    # Clean hair color data
    df_filtered = df_filtered.dropna(subset=['hair'])
    if len(df_filtered) == 0:
        print("No hair color data available")
        return
    
    # Define colors that match hair colors
    hair_color_map = {
        'black': '#2F2F2F',
        'dark brown': '#654321',
        'brown': '#8B4513',
        'chestnut': '#954535',
        'red / brown': '#8B4513',
        'brown / red': '#A0522D',
        'auburn': '#A0522D',
        'red': '#B22222',
        'grey': '#808080',
        'dark blonde': '#B8860B',
        'light brown': '#D2B48C',
        'red / blonde': '#CD853F',
        'red blonde': '#DAA520',
        'blonde / red': '#FFB347',
        'light red': '#FA8072',
        'blonde': '#F4E28C',
        'light blonde': '#FFF8DC',
        'bald': '#E6E6FA',
        'white': '#F5F5F5',
    }
    
    # Create 1x2 subplot for gender comparison
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('Hair Color Evolution by Gender (Proportions)', fontsize=16, fontweight='bold')
    
    genders = ['female', 'male']
    
    for i, gender in enumerate(genders):
        ax = axes[i]
        gender_data = df_filtered[df_filtered['gender_consensus'] == gender]
        
        if len(gender_data) == 0:
            ax.text(0.5, 0.5, f'No hair data for {gender}', transform=ax.transAxes,
                   ha='center', va='center')
            continue
        
        # Get hair color counts by year for this gender
        hair_by_year = gender_data.groupby(['year', 'hair']).size().unstack(fill_value=0)
        hair_proportions = hair_by_year.div(hair_by_year.sum(axis=1), axis=0)
        yearly_totals = hair_by_year.sum(axis=1)
        
        # Get all years
        all_years = hair_proportions.index
        
        # Get hair types ordered by appearance in map (case-insensitive matching)
        hair_types_ordered = []
        for map_key in hair_color_map.keys():
            for col in hair_proportions.columns:
                if col.lower() == map_key.lower():
                    hair_types_ordered.append(col)
                    break
        
        # Create bars
        bottoms = np.zeros(len(all_years))
        
        for hair_type in hair_types_ordered:
            color = hair_color_map.get(hair_type.lower(), '#CCCCCC')
            
            bar_heights = []
            for year in all_years:
                if yearly_totals[year] >= 50:  # Lower threshold for gender-specific analysis
                    bar_heights.append(hair_proportions.loc[year, hair_type])
                else:
                    bar_heights.append(0)
            
            ax.bar(range(len(all_years)), bar_heights, width=0.9, 
                   bottom=bottoms, color=color, alpha=0.8, 
                   label=hair_type, edgecolor='none')
            
            for j, year in enumerate(all_years):
                if yearly_totals[year] >= 50:
                    bottoms[j] += hair_proportions.loc[year, hair_type]
        
        # Add sample size annotations
        for j, (year, total) in enumerate(yearly_totals.items()):
            text_color = 'red' if total < 50 else 'black'
            weight = 'bold' if total < 50 else 'normal'
            ax.text(j, 1.02, f'n={int(total)}', ha='center', va='bottom',
                   fontsize=8, alpha=0.7, color=text_color, weight=weight, rotation=45)
        
        ax.set_xticks(range(len(all_years)))
        ax.set_xticklabels([str(year) for year in all_years], rotation=45)
        ax.set_title(f'{gender.title()} Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Proportion of Models')
        ax.set_ylim(0, 1.1)
        ax.legend(title='Hair Color', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        remove_spines(ax)
    
    # Add note about missing bars
    fig.text(0.02, 0.02, 'No bars shown when n < 50 (red labels)', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hair_color_evolution_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_eye_color_evolution_by_gender(merged_df, start_year=2000):
    """
    Plot the evolution of eye colors over time by gender with two subplots.
    """
    df_filtered = merged_df[merged_df['year'] >= start_year]
    
    # Clean eye color data
    df_filtered = df_filtered.dropna(subset=['eyes'])
    if len(df_filtered) == 0:
        print("No eye color data available")
        return
    
    # Define colors that match eye colors (ordered by brightness)
    eye_color_map = {
        'black': '#2F2F2F',
        'dark brown': '#654321',
        'blue / brown': '#483D8B',
        'brown': '#8B4513',
        'brown / hazel': '#8B7355',
        'light brown': '#D2B48C',
        'hazel': '#8E7618',
        'green / grey': '#2F4F4F',
        'green / brown': '#556B2F',
        'brown / green': '#556B2F',
        'green / hazel': '#6B8E23',
        'green': '#228B22',
        'blue / green': '#008B8B',
        'blue': '#4169E1',
        'blue / grey': '#6495ED',
        'grey': '#708090',
    }
    
    # Create 1x2 subplot for gender comparison
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('Eye Color Evolution by Gender (Proportions)', fontsize=16, fontweight='bold')
    
    genders = ['female', 'male']
    
    for i, gender in enumerate(genders):
        ax = axes[i]
        gender_data = df_filtered[df_filtered['gender_consensus'] == gender]
        
        if len(gender_data) == 0:
            ax.text(0.5, 0.5, f'No eye data for {gender}', transform=ax.transAxes,
                   ha='center', va='center')
            continue
        
        # Get eye color counts by year for this gender
        eyes_by_year = gender_data.groupby(['year', 'eyes']).size().unstack(fill_value=0)
        eyes_proportions = eyes_by_year.div(eyes_by_year.sum(axis=1), axis=0)
        yearly_totals = eyes_by_year.sum(axis=1)
        
        # Get all years
        all_years = eyes_proportions.index
        
        # Get eye types ordered by appearance in map
        # Get eye types ordered by appearance in map (case-insensitive matching)
        eye_types_ordered = []
        for map_key in eye_color_map.keys():
            for col in eyes_proportions.columns:
                if col.lower() == map_key.lower():
                    eye_types_ordered.append(col)
                    break
        
        # Create bars
        bottoms = np.zeros(len(all_years))
        
        for eye_type in eye_types_ordered:
            color = eye_color_map.get(eye_type.lower(), '#CCCCCC')
            
            bar_heights = []
            for year in all_years:
                if yearly_totals[year] >= 50:  # Lower threshold for gender-specific analysis
                    bar_heights.append(eyes_proportions.loc[year, eye_type])
                else:
                    bar_heights.append(0)
            
            ax.bar(range(len(all_years)), bar_heights, width=0.9,  
                   bottom=bottoms, color=color, alpha=0.8, 
                   label=eye_type, edgecolor='none')
            
            for j, year in enumerate(all_years):
                if yearly_totals[year] >= 50:
                    bottoms[j] += eyes_proportions.loc[year, eye_type]
        
        # Add sample size annotations
        for j, (year, total) in enumerate(yearly_totals.items()):
            text_color = 'red' if total < 50 else 'black'
            weight = 'bold' if total < 50 else 'normal'
            ax.text(j, 1.02, f'n={int(total)}', ha='center', va='bottom', 
                   fontsize=8, alpha=0.7, color=text_color, weight=weight, rotation=45)
        
        ax.set_xticks(range(len(all_years)))
        ax.set_xticklabels([str(year) for year in all_years], rotation=45)
        ax.set_title(f'{gender.title()} Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Proportion of Models')
        ax.set_ylim(0, 1.1)
        ax.legend(title='Eye Color', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        remove_spines(ax)
    
    # Add note about missing bars
    fig.text(0.02, 0.02, 'No bars shown when n < 50 (red labels)', 
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eye_color_evolution_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

def prepare_shows_and_measurements_data(datasets):
    """
    Prepare merged shows and measurements data for analysis.
    """
    print("\nPreparing shows and measurements data for analysis...")
    
    # Clean measurements data
    models_clean = clean_measurements(datasets['measurements'])
    
    # Filter out data quality issues if the column exists
    if 'data_quality_issues' in models_clean.columns:
        print(f"Initial rows in models_clean: {len(models_clean)}")
        models_clean = models_clean[models_clean['data_quality_issues'].fillna('').eq('')]
        print(f"Rows after filtering data quality issues: {len(models_clean)}")
    
    # Convert shows data to DataFrame if it's not already
    shows_df = datasets['shows']
    
    # Merge shows with measurements
    merged_df = shows_df.merge(models_clean, on='model_id', how='left')
    
    # Filter out records without valid year or measurements
    if 'year' in merged_df.columns:
        merged_df = merged_df.dropna(subset=['year'])
        # Ensure 'year' is integer type
        if not pd.api.types.is_integer_dtype(merged_df['year']):
            try:
                merged_df['year'] = merged_df['year'].astype('Int64')
            except (ValueError, TypeError):
                print("Warning: Could not convert 'year' to integer.")
                merged_df['year'] = pd.to_numeric(merged_df['year'], errors='coerce').astype('Int64')
    
    # Filter out records without measurements
    measurement_cols = ['height_cm', 'bust-us_clean', 'waist-us_clean', 'hips-us_clean']
    existing_measurement_cols = [col for col in measurement_cols if col in merged_df.columns]
    if existing_measurement_cols:
        merged_df = merged_df.dropna(subset=existing_measurement_cols, how='all')
    
    # Add gender information from gender dataset
    if not datasets['gender'].empty:
        gender_cols = ['model_id', 'name', 'gender_consensus']
        available_gender_cols = [col for col in gender_cols if col in datasets['gender'].columns]
        
        merged_df = merged_df.merge(
            datasets['gender'][available_gender_cols],
            on=['model_id', 'name'],
            how='left'
        )
        print(f"Added gender information to merged dataset")
    
    print(f"Final merged dataset: {len(merged_df):,} records")
    return merged_df

def plot_measurement_std_evolution_by_gender(merged_df, gender_filter='female', start_year=2000, measurement_system='us'):
    """
    Plot the evolution of measurement standard deviations over time for a specific gender.
    This shows how body type diversity changes over time.
    """
    # Filter by gender and start year
    df_filtered = merged_df[
        (merged_df['gender_consensus'] == gender_filter) & 
        (merged_df['year'] >= start_year)
    ]
    
    if len(df_filtered) == 0:
        print(f"No data available for gender: {gender_filter} from year {start_year}")
        return
    
    # Define measurements based on system
    if measurement_system.lower() == 'us':
        measurements = [
            ('height_cm', 'Height (cm)'),
            ('bust-us_clean', 'Bust (inches)'),
            ('waist-us_clean', 'Waist (inches)'),
            ('hips-us_clean', 'Hips (inches)')
        ]
        system_label = 'US'
    elif measurement_system.lower() == 'eu':
        measurements = [
            ('height_cm', 'Height (cm)'),
            ('bust-eu_clean', 'Bust (cm)'),
            ('waist-eu_clean', 'Waist (cm)'),
            ('hips-eu_clean', 'Hips (cm)')
        ]
        system_label = 'EU'
    else:
        raise ValueError("measurement_system must be 'us' or 'eu'")
    
    # Create subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{gender_filter.title()} Model Measurement Standard Deviation Evolution ({system_label} System) - From {start_year}', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    color = 'lightcoral' if gender_filter == 'female' else 'lightblue'
    
    for idx, (measurement_col, title) in enumerate(measurements):
        ax = axes_flat[idx]
        
        # Filter data for this measurement
        measurement_data = df_filtered.dropna(subset=[measurement_col])
        
        if len(measurement_data) == 0:
            ax.text(0.5, 0.5, f'No data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} - Standard Deviation', fontsize=14, fontweight='bold')
            continue
        
        # Calculate yearly statistics
        yearly_stats = measurement_data.groupby('year')[measurement_col].agg([
            'std', 'count'
        ]).reset_index()
        
        yearly_stats.columns = ['year', 'std', 'count']
        
        # Filter years with sufficient data (at least 10 data points for reliable std)
        yearly_stats = yearly_stats[yearly_stats['count'] >= 10]
        
        if len(yearly_stats) == 0:
            ax.text(0.5, 0.5, f'Insufficient data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} - Standard Deviation', fontsize=14, fontweight='bold')
            continue
        
        # Plot standard deviation as line with markers
        ax.plot(yearly_stats['year'], yearly_stats['std'],
                color='black', linewidth=2, marker='o', markersize=4, 
                label='Standard Deviation')
        
        # Add linear fit to show trend
        if len(yearly_stats) > 2:
            z = np.polyfit(yearly_stats['year'], yearly_stats['std'], 1)
            p = np.poly1d(z)
            ax.plot(yearly_stats['year'], p(yearly_stats['year']), 
                    color='firebrick', linestyle='--', linewidth=2, 
                    label=f'Trend (slope: {z[0]:.3f})')
        
        # Customize plot
        ax.set_title(f'{title} - Standard Deviation', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'Standard Deviation ({title.split("(")[1].replace(")", "") if "(" in title else "units"})', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        remove_spines(ax)
        
        # Add sample size annotations for some years
        for i, row in yearly_stats.iterrows():
            if i % max(1, len(yearly_stats) // 5) == 0:  # Show every 5th annotation
                ax.annotate(f'n={int(row["count"])}', 
                           xy=(row['year'], row['std']), 
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=8, alpha=0.7)
        
        # Set x-axis to show reasonable number of ticks
        years = yearly_stats['year'].values
        if len(years) > 10:
            step = max(1, len(years) // 10)
            ax.set_xticks(years[::step])
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    filename = FIGURES_DIR / f'std_evolution_{gender_filter}_{measurement_system}_from_{start_year}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_measurement_iqr_evolution_by_gender(merged_df, gender_filter='female', start_year=2000, measurement_system='us'):
    """
    Plot the evolution of measurement Interquartile Range (IQR) over time for a specific gender.
    This shows how the spread of the middle 50% of body measurements changes over time.
    """
    # Filter by gender and start year
    df_filtered = merged_df[
        (merged_df['gender_consensus'] == gender_filter) & 
        (merged_df['year'] >= start_year)
    ]
    
    if len(df_filtered) == 0:
        print(f"No data available for gender: {gender_filter} from year {start_year}")
        return
    
    # Define measurements based on system
    if measurement_system.lower() == 'us':
        measurements = [
            ('height_cm', 'Height (cm)'),
            ('bust-us_clean', 'Bust (inches)'),
            ('waist-us_clean', 'Waist (inches)'),
            ('hips-us_clean', 'Hips (inches)')
        ]
        system_label = 'US'
    elif measurement_system.lower() == 'eu':
        measurements = [
            ('height_cm', 'Height (cm)'),
            ('bust-eu_clean', 'Bust (cm)'),
            ('waist-eu_clean', 'Waist (cm)'),
            ('hips-eu_clean', 'Hips (cm)')
        ]
        system_label = 'EU'
    else:
        raise ValueError("measurement_system must be 'us' or 'eu'")
    
    # Create subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{gender_filter.title()} Model Measurement IQR Evolution ({system_label} System) - From {start_year}', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    color = 'lightcoral' if gender_filter == 'female' else 'lightblue'
    
    for idx, (measurement_col, title) in enumerate(measurements):
        ax = axes_flat[idx]
        
        # Filter data for this measurement
        measurement_data = df_filtered.dropna(subset=[measurement_col])
        
        if len(measurement_data) == 0:
            ax.text(0.5, 0.5, f'No data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} - IQR', fontsize=14, fontweight='bold')
            continue
        
        # Calculate yearly statistics including quartiles
        yearly_stats = measurement_data.groupby('year')[measurement_col].agg([
            lambda x: x.quantile(0.25),  # Q1
            lambda x: x.quantile(0.75),  # Q3
            'count'
        ]).reset_index()
        
        yearly_stats.columns = ['year', 'q1', 'q3', 'count']
        yearly_stats['iqr'] = yearly_stats['q3'] - yearly_stats['q1']
        
        # Filter years with sufficient data (at least 10 data points for reliable IQR)
        yearly_stats = yearly_stats[yearly_stats['count'] >= 10]
        
        if len(yearly_stats) == 0:
            ax.text(0.5, 0.5, f'Insufficient data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} - IQR', fontsize=14, fontweight='bold')
            continue
        
        # Plot IQR as line with markers
        ax.plot(yearly_stats['year'], yearly_stats['iqr'],
                color='black', linewidth=2, marker='o', markersize=4, 
                label='IQR (Q3 - Q1)')
        
        # Fill area between Q1 and Q3 for a few representative years to show what IQR represents
        if len(yearly_stats) >= 3:
            sample_years = yearly_stats['year'].iloc[::max(1, len(yearly_stats)//3)]
            for sample_year in sample_years:
                year_row = yearly_stats[yearly_stats['year'] == sample_year].iloc[0]
                ax.axvspan(sample_year - 0.2, sample_year + 0.2, 
                          ymin=(year_row['q1'] - yearly_stats['iqr'].min()) / (yearly_stats['iqr'].max() - yearly_stats['iqr'].min()),
                          ymax=(year_row['q3'] - yearly_stats['iqr'].min()) / (yearly_stats['iqr'].max() - yearly_stats['iqr'].min()),
                          alpha=0.2, color=color)
        
        # Add linear fit to show trend
        if len(yearly_stats) > 2:
            z = np.polyfit(yearly_stats['year'], yearly_stats['iqr'], 1)
            p = np.poly1d(z)
            ax.plot(yearly_stats['year'], p(yearly_stats['year']), 
                    color='firebrick', linestyle='--', linewidth=2, 
                    label=f'Trend (slope: {z[0]:.3f})')
            
            # Add trend direction indicator
            direction = "↗️" if z[0] > 0 else "↘️" if z[0] < 0 else "→"
            ax.text(0.02, 0.98, f'Trend: {direction} {z[0]:.3f}/year', 
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Customize plot
        ax.set_title(f'{title} - IQR', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel(f'IQR ({title.split("(")[1].replace(")", "") if "(" in title else "units"})', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        remove_spines(ax)
        
        # Add sample size annotations for some years
        for i, row in yearly_stats.iterrows():
            if i % max(1, len(yearly_stats) // 5) == 0:  # Show every 5th annotation
                ax.annotate(f'n={int(row["count"])}', 
                           xy=(row['year'], row['iqr']), 
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=8, alpha=0.7)
        
        # Set x-axis to show reasonable number of ticks
        years = yearly_stats['year'].values
        if len(years) > 10:
            step = max(1, len(years) // 10)
            ax.set_xticks(years[::step])
        
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    filename = FIGURES_DIR / f'iqr_evolution_{gender_filter}_{measurement_system}_from_{start_year}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_hair_eye_color_by_gender(merged_df, start_year=2000):
    """
    Plot hair and eye color distribution by gender over time.
    """
    df_filtered = merged_df[merged_df['year'] >= start_year]
    
    # Create 2x2 subplot for hair/eye by gender
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('Hair and Eye Color Evolution by Gender (Proportions)', fontsize=16, fontweight='bold')
    
    # Hair color maps ordered by brightness (dark to light)
    hair_color_map = {
        'black': '#2F2F2F',
        'dark brown': '#654321',
        'brown': '#8B4513',
        'chestnut': '#954535',
        'red / brown': '#8B4513',
        'brown / red': '#A0522D',
        'auburn': '#A0522D',
        'red': '#B22222',
        'grey': '#808080',
        'dark blonde': '#B8860B',
        'light brown': '#D2B48C',
        'red / blonde': '#CD853F',
        'red blonde': '#DAA520',
        'blonde / red': '#FFB347',
        'light red': '#FA8072',
        'blonde': '#F4E28C',
        'light blonde': '#FFF8DC',
        'bald': '#E6E6FA',
        'white': '#F5F5F5',
    }
    
    # Eye color maps ordered by brightness (dark to light)
    eye_color_map = {
        'black': '#2F2F2F',
        'dark brown': '#654321',
        'blue / brown': '#483D8B',
        'brown': '#8B4513',
        'brown / hazel': '#8B7355',
        'light brown': '#D2B48C',
        'hazel': '#8E7618',
        'green / grey': '#2F4F4F',
        'green / brown': '#556B2F',
        'brown / green': '#556B2F',
        'green / hazel': '#6B8E23',
        'green': '#228B22',
        'blue / green': '#008B8B',
        'blue': '#4169E1',
        'blue / grey': '#6495ED',
        'grey': '#708090',
    }

    genders = ['female', 'male']

    for i, gender in enumerate(genders):
        gender_data = df_filtered[df_filtered['gender_consensus'] == gender]
        
        # Hair colors for this gender
        ax_hair = axes[i, 0]
        hair_data = gender_data.dropna(subset=['hair'])
        if len(hair_data) > 0:
            hair_by_year = hair_data.groupby(['year', 'hair']).size().unstack(fill_value=0)
            hair_proportions = hair_by_year.div(hair_by_year.sum(axis=1), axis=0)
            yearly_totals_hair = hair_by_year.sum(axis=1)
            
            # Get all years
            all_years_hair = hair_proportions.index
            
            # Get hair types ordered by appearance in map (case-insensitive matching)
            hair_types_ordered = []
            for map_key in hair_color_map.keys():
                for col in hair_proportions.columns:
                    if col.lower() == map_key.lower():
                        hair_types_ordered.append(col)
                        break
            
            # Create bars
            bottoms_hair = np.zeros(len(all_years_hair))
            
            for hair_type in hair_types_ordered:
                color = hair_color_map.get(hair_type.lower(), '#CCCCCC')
                
                bar_heights = []
                for year in all_years_hair:
                    if yearly_totals_hair[year] >= 100:
                        bar_heights.append(hair_proportions.loc[year, hair_type])
                    else:
                        bar_heights.append(0)
                
                ax_hair.bar(range(len(all_years_hair)), bar_heights, width=0.9, 
                            bottom=bottoms_hair, color=color, alpha=0.8, 
                            label=hair_type, edgecolor='none')
                
                for j, year in enumerate(all_years_hair):
                    if yearly_totals_hair[year] >= 100:
                        bottoms_hair[j] += hair_proportions.loc[year, hair_type]
            
            # Add sample size annotations
            for j, (year, total) in enumerate(yearly_totals_hair.items()):
                text_color = 'red' if total < 100 else 'black'
                weight = 'bold' if total < 100 else 'normal'
                ax_hair.text(j, 1.02, f'n={int(total)}', ha='center', va='bottom',
                            fontsize=8, alpha=0.7, color=text_color, weight=weight, rotation=45)
            
            ax_hair.set_xticks(range(len(all_years_hair)))
            ax_hair.set_xticklabels([str(year) for year in all_years_hair], rotation=45)
            ax_hair.set_title(f'{gender.title()} Hair Colors', fontsize=14, fontweight='bold')
            ax_hair.set_xlabel('Year')
            ax_hair.set_ylabel('Proportion of Models')
            ax_hair.set_ylim(0, 1.1)
            ax_hair.legend(title='Hair Color', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Eye colors for this gender
        ax_eyes = axes[i, 1]
        eye_data = gender_data.dropna(subset=['eyes'])
        if len(eye_data) > 0:
            eyes_by_year = eye_data.groupby(['year', 'eyes']).size().unstack(fill_value=0)
            eyes_proportions = eyes_by_year.div(eyes_by_year.sum(axis=1), axis=0)
            yearly_totals_eyes = eyes_by_year.sum(axis=1)
            
            all_years_eyes = eyes_proportions.index
            # Get eye types ordered by appearance in map (case-insensitive matching)
        eye_types_ordered = []
        for map_key in eye_color_map.keys():
            for col in eyes_proportions.columns:
                if col.lower() == map_key.lower():
                    eye_types_ordered.append(col)
                    break
            
            bottoms_eyes = np.zeros(len(all_years_eyes))
            
            for eye_type in eye_types_ordered:
                color = eye_color_map.get(eye_type.lower(), '#CCCCCC')
                
                bar_heights = []
                for year in all_years_eyes:
                    if yearly_totals_eyes[year] >= 100:
                        bar_heights.append(eyes_proportions.loc[year, eye_type])
                    else:
                        bar_heights.append(0)
                
                ax_eyes.bar(range(len(all_years_eyes)), bar_heights, width=0.9,  
                            bottom=bottoms_eyes, color=color, alpha=0.8, 
                            label=eye_type, edgecolor='none')
                
                for j, year in enumerate(all_years_eyes):
                    if yearly_totals_eyes[year] >= 100:
                        bottoms_eyes[j] += eyes_proportions.loc[year, eye_type]
            
            # Add sample size annotations
            for j, (year, total) in enumerate(yearly_totals_eyes.items()):
                text_color = 'red' if total < 100 else 'black'
                weight = 'bold' if total < 100 else 'normal'
                ax_eyes.text(j, 1.02, f'n={int(total)}', ha='center', va='bottom', 
                            fontsize=8, alpha=0.7, color=text_color, weight=weight, rotation=45)
            
            ax_eyes.set_xticks(range(len(all_years_eyes)))
            ax_eyes.set_xticklabels([str(year) for year in all_years_eyes], rotation=45)
            ax_eyes.set_title(f'{gender.title()} Eye Colors', fontsize=14, fontweight='bold')
            ax_eyes.set_xlabel('Year')
            ax_eyes.set_ylabel('Proportion of Models')
            ax_eyes.set_ylim(0, 1.1)
            ax_eyes.legend(title='Eye Color', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove spines
        remove_spines(ax_hair)
        remove_spines(ax_eyes)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hair_eye_color_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_nationality_to_region_mapping():
    """
    Create a mapping from nationalities to world regions.
    """
    nationality_to_region = {
        # North America
        'American': 'North America',
        'Canadian': 'North America',
        'Mexican': 'North America',
        
        # Caribbean/Central America
        'Dominican': 'Caribbean/Central America',
        'Cuban': 'Caribbean/Central America',
        'Honduran': 'Caribbean/Central America',
        'Jamaican': 'Caribbean/Central America',
        'Bahamian': 'Caribbean/Central America',
        'Puerto Rican': 'Caribbean/Central America',
        'Haitian': 'Caribbean/Central America',
        'Guatemalan': 'Caribbean/Central America',
        'Costa Rican': 'Caribbean/Central America',
        'Panamanian': 'Caribbean/Central America',
        'Trinidadian': 'Caribbean/Central America',
        'Martinican': 'Caribbean/Central America',
        'Aruban': 'Caribbean/Central America',
        'Caymanian': 'Caribbean/Central America',
        'Curaçaoan': 'Caribbean/Central America',
        'Antiguan': 'Caribbean/Central America',
        'Bermudian': 'Caribbean/Central America',
        
        # South America
        'Brazilian': 'South America',
        'Argentinian': 'South America',
        'Colombian': 'South America',
        'Chilean': 'South America',
        'Peruvian': 'South America',
        'Venezuelan': 'South America',
        'Uruguayan': 'South America',
        'Guyanese': 'South America',
        
        # Western Europe
        'British': 'Western Europe',
        'British (English)': 'Western Europe',
        'British (Northern Irish)': 'Western Europe',
        'British (Scottish)': 'Western Europe',
        'British (Welsh)': 'Western Europe',
        'German': 'Western Europe',
        'French': 'Western Europe',
        'Dutch': 'Western Europe',
        'Belgian': 'Western Europe',
        'Swiss': 'Western Europe',
        'Austrian': 'Western Europe',
        'Irish': 'Western Europe',
        'Luxembourger': 'Western Europe',
        
        # Southern Europe
        'Spanish': 'Southern Europe',
        'Italian': 'Southern Europe',
        'Portuguese': 'Southern Europe',
        'Greek': 'Southern Europe',
        'Maltese': 'Southern Europe',
        'Cypriot': 'Southern Europe',
        
        # Northern Europe
        'Swedish': 'Northern Europe',
        'Danish': 'Northern Europe',
        'Norwegian': 'Northern Europe',
        'Finnish': 'Northern Europe',
        'Icelandic': 'Northern Europe',
        
        # Eastern Europe
        'Russian': 'Eastern Europe',
        'Polish': 'Eastern Europe',
        'Czech': 'Eastern Europe',
        'Slovakian': 'Eastern Europe',
        'Ukrainian': 'Eastern Europe',
        'Romanian': 'Eastern Europe',
        'Bulgarian': 'Eastern Europe',
        'Hungarian': 'Eastern Europe',
        'Croatian': 'Eastern Europe',
        'Serbian': 'Eastern Europe',
        'Slovenian': 'Eastern Europe',
        'Estonian': 'Eastern Europe',
        'Latvian': 'Eastern Europe',
        'Lithuanian': 'Eastern Europe',
        'Belarusian': 'Eastern Europe',
        'Montenegrin': 'Eastern Europe',
        'Moldovan': 'Eastern Europe',
        'Bosnian': 'Eastern Europe',
        'Albanian': 'Eastern Europe',
        'Macedonian': 'Eastern Europe',
        'Kosovar': 'Eastern Europe',
        
        # East Asia
        'Chinese': 'East Asia',
        'Japanese': 'East Asia',
        'South Korean': 'East Asia',
        'Taiwanese': 'East Asia',
        'Mongolian': 'East Asia',
        
        # Southeast Asia
        'Thai': 'Southeast Asia',
        'Indonesian': 'Southeast Asia',
        'Malaysian': 'Southeast Asia',
        'Singaporean': 'Southeast Asia',
        'Vietnamese': 'Southeast Asia',
        'Filipino': 'Southeast Asia',
        'Burmese': 'Southeast Asia',
        
        # South Asia
        'Indian': 'South Asia',
        'Sri Lankan': 'South Asia',
        'Nepali': 'South Asia',
        'Afghan': 'South Asia',
        
        # Middle East
        'Israeli': 'Middle East',
        'Lebanese': 'Middle East',
        'Turkish': 'Middle East',
        'Syrian': 'Middle East',
        'Armenian': 'Middle East',
        
        # Central Asia
        'Uzbekistani': 'Central Asia',
        'Kazakhstani': 'Central Asia',
        
        # Sub-Saharan Africa
        'Nigerian': 'Sub-Saharan Africa',
        'South African': 'Sub-Saharan Africa',
        'Senegalese': 'Sub-Saharan Africa',
        'Ghanaian': 'Sub-Saharan Africa',
        'Kenyan': 'Sub-Saharan Africa',
        'Somalian': 'Sub-Saharan Africa',
        'South Sudanese': 'Sub-Saharan Africa',
        'Sudanese': 'Sub-Saharan Africa',
        'Angolan': 'Sub-Saharan Africa',
        'Burundian': 'Sub-Saharan Africa',
        'Ivorian': 'Sub-Saharan Africa',
        'Malian': 'Sub-Saharan Africa',
        'Rwandan': 'Sub-Saharan Africa',
        'Mozambican': 'Sub-Saharan Africa',
        'Cape Verdean': 'Sub-Saharan Africa',
        'Ethiopian': 'Sub-Saharan Africa',
        'Ugandan': 'Sub-Saharan Africa',
        'Guinean': 'Sub-Saharan Africa',
        'Gambian': 'Sub-Saharan Africa',
        'Sierra Leonean': 'Sub-Saharan Africa',
        'Beninese': 'Sub-Saharan Africa',
        'Tanzanian': 'Sub-Saharan Africa',
        'Burkinabe': 'Sub-Saharan Africa',
        'Congolese': 'Sub-Saharan Africa',
        'Namibian': 'Sub-Saharan Africa',
        
        # North Africa
        'Moroccan': 'North Africa',
        'Algerian': 'North Africa',
        'Tunisian': 'North Africa',
        'Egyptian': 'North Africa',
        
        # Caucasus
        'Georgian': 'Caucasus',
        
        # Oceania
        'Australian': 'Oceania',
        'New Zealander': 'Oceania',
        'French Polynesian': 'Oceania',
    }
    
    return nationality_to_region

def plot_nationality_evolution_by_region_and_gender(merged_df, start_year=2000):
    """
    Plot the evolution of model nationalities by world region and gender over time.
    """
    # Load nationality data from file
    try:
        nationality_df = pd.read_csv(MODELS_NATIONALITY)
        print(f"Loaded nationality data for {len(nationality_df)} models")
    except FileNotFoundError:
        print("Nationality data file not found.")
        return
    
    # Filter years
    df_filtered = merged_df[merged_df['year'] >= start_year]
    
    # Merge nationality data
    nationality_merged = df_filtered.merge(
        nationality_df[['model_id', 'nationality']], 
        on='model_id', 
        how='left'
    )
    
    # Filter to records with nationality data
    nationality_merged = nationality_merged.dropna(subset=['nationality'])
    
    if len(nationality_merged) == 0:
        print("No nationality data available")
        return
    
    # Create region mapping
    nationality_to_region = create_nationality_to_region_mapping()
    nationality_merged['region'] = nationality_merged['nationality'].map(nationality_to_region)
    nationality_merged = nationality_merged.dropna(subset=['region'])
    
    # Define colors for regions
    region_color_map = {
        'North America': '#FF6B6B',
        'South America': '#4ECDC4',
        'Caribbean/Central America': '#45B7D1',
        'Western Europe': '#96CEB4',
        'Southern Europe': '#FECA57',
        'Northern Europe': '#FF9FF3',
        'Eastern Europe': '#54A0FF',
        'East Asia': '#5F27CD',
        'Southeast Asia': '#00D2D3',
        'South Asia': '#FF9F43',
        'Middle East': '#EE5A24',
        'Central Asia': '#C44569',
        'Sub-Saharan Africa': '#6C5CE7',
        'North Africa': '#A55EEA',
        'Caucasus': '#26DE81',
        'Oceania': '#FD79A8',
    }
    
    # Get overall region ordering for consistency
    overall_regions_by_year = nationality_merged.groupby(['year', 'region']).size().unstack(fill_value=0)
    overall_region_totals = overall_regions_by_year.sum().sort_values(ascending=False)
    regions_sorted_global = overall_region_totals.index.tolist()
    
    # Create 1x2 subplot for gender comparison
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    fig.suptitle('Model Nationality Evolution by World Region and Gender (Proportions)', 
                 fontsize=16, fontweight='bold')
    
    genders = ['female', 'male']
    
    for i, gender in enumerate(genders):
        ax = axes[i]
        gender_data = nationality_merged[nationality_merged['gender_consensus'] == gender]
        
        if len(gender_data) == 0:
            ax.text(0.5, 0.5, f'No nationality data for {gender}', 
                   transform=ax.transAxes, ha='center', va='center')
            continue
        
        # Get region counts by year for this gender
        regions_by_year = gender_data.groupby(['year', 'region']).size().unstack(fill_value=0)
        regions_proportions = regions_by_year.div(regions_by_year.sum(axis=1), axis=0)
        yearly_totals = regions_by_year.sum(axis=1)
        
        all_years = regions_proportions.index
        regions_sorted = [region for region in regions_sorted_global 
                         if region in regions_proportions.columns]
        region_totals_this_gender = regions_by_year.sum()
        
        bottoms = np.zeros(len(all_years))
        
        for region in regions_sorted:
            color = region_color_map.get(region, '#CCCCCC')
            
            bar_heights = []
            for year in all_years:
                if yearly_totals[year] >= 15:
                    bar_heights.append(regions_proportions.loc[year, region])
                else:
                    bar_heights.append(0)
            
            count_for_legend = region_totals_this_gender.get(region, 0)
            ax.bar(range(len(all_years)), bar_heights, width=0.9,
                   bottom=bottoms, color=color, alpha=0.8, 
                   label=f'{region} (n={count_for_legend})', 
                   edgecolor='none')
            
            for j, year in enumerate(all_years):
                if yearly_totals[year] >= 15:
                    bottoms[j] += regions_proportions.loc[year, region]
        
        # Add sample size annotations
        for j, (year, total) in enumerate(yearly_totals.items()):
            text_color = 'red' if total < 15 else 'black'
            weight = 'bold' if total < 15 else 'normal'
            ax.text(j, 1.02, f'n={int(total)}', ha='center', va='bottom', 
                   fontsize=8, alpha=0.7, color=text_color, weight=weight, rotation=45)
        
        ax.set_title(f'{gender.title()} Models', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Proportion of Models', fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_xticks(range(len(all_years)))
        ax.set_xticklabels([str(year) for year in all_years], rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title='World Region', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        remove_spines(ax)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nationality_evolution_by_region_and_gender.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_measurement_entropy_evolution_by_gender(merged_df, gender_filter='female', start_year=2000):
    """
    Plot the evolution of measurement entropy over time for a specific gender.
    Uses binning to compute entropy for continuous measurements.
    """
    # Filter by gender and start year
    df_filtered = merged_df[
        (merged_df['gender_consensus'] == gender_filter) & 
        (merged_df['year'] >= start_year)
    ]
    
    if len(df_filtered) == 0:
        print(f"No data available for gender: {gender_filter} from year {start_year}")
        return
    
    # Define measurements to analyze
    measurements = [
        ('height_cm', 'Height (cm)'),
        ('bust-us_clean', 'Bust (inches)'),
        ('waist-us_clean', 'Waist (inches)'),
        ('hips-us_clean', 'Hips (inches)')
    ]
    
    # Create subplots: 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'{gender_filter.title()} Model Measurement Entropy Evolution (Normalized 0-1)', 
                 fontsize=16, fontweight='bold')
    
    axes_flat = axes.flatten()
    color = 'lightcoral' if gender_filter == 'female' else 'lightblue'
    
    for idx, (measurement_col, title) in enumerate(measurements):
        ax = axes_flat[idx]
        
        # Filter data for this measurement
        measurement_data = df_filtered.dropna(subset=[measurement_col])
        
        if len(measurement_data) == 0:
            ax.text(0.5, 0.5, f'No data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} - Entropy', fontsize=14, fontweight='bold')
            continue
        
        # Get overall min/max for this measurement across all years to create consistent bins
        overall_min = measurement_data[measurement_col].min()
        overall_max = measurement_data[measurement_col].max()
        
        # Create bins every cm/inch
        bin_edges = np.arange(np.floor(overall_min), np.ceil(overall_max) + 1, 1)
        max_entropy = np.log(len(bin_edges) - 1)  # Maximum possible entropy for normalization
        
        yearly_entropy = []
        yearly_counts = []
        valid_years = []
        
        for year in sorted(measurement_data['year'].unique()):
            year_data = measurement_data[measurement_data['year'] == year]
            
            if len(year_data) >= 20:  # Minimum for reliable entropy
                # Create histogram bins
                counts, _ = np.histogram(year_data[measurement_col], bins=bin_edges)
                
                # Remove empty bins for entropy calculation
                non_zero_counts = counts[counts > 0]
                
                if len(non_zero_counts) > 1:  # Need at least 2 non-empty bins
                    probabilities = non_zero_counts / non_zero_counts.sum()
                    year_entropy = entropy(probabilities)
                    
                    # Normalize entropy to 0-1 scale
                    normalized_entropy = year_entropy / max_entropy if max_entropy > 0 else 0
                    
                    yearly_entropy.append(normalized_entropy)
                    yearly_counts.append(len(year_data))
                    valid_years.append(year)
        
        if len(yearly_entropy) == 0:
            ax.text(0.5, 0.5, f'Insufficient data for {title}', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
            ax.set_title(f'{title} - Entropy', fontsize=14, fontweight='bold')
            continue
        
        # Plot entropy evolution
        ax.plot(valid_years, yearly_entropy,
                color='black', linewidth=2, marker='o', markersize=6,
                markerfacecolor=color, markeredgecolor='black',
                label='Normalized Entropy')
        
        # Add linear fit to show trend
        if len(valid_years) > 2:
            z = np.polyfit(valid_years, yearly_entropy, 1)
            p = np.poly1d(z)
            ax.plot(valid_years, p(valid_years), 
                    color='firebrick', linestyle='--', linewidth=2, 
                    label=f'Trend (slope: {z[0]:.4f})')
        
        # Customize plot
        ax.set_title(f'{title} - Entropy', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Normalized Entropy (0-1)', fontsize=12)
        #ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        remove_spines(ax)
        
        # Add sample size annotations for some years
        for i in range(0, len(valid_years), max(1, len(valid_years) // 5)):
            year = valid_years[i]
            entropy_val = yearly_entropy[i]
            count = yearly_counts[i]
            ax.annotate(f'n={count}', 
                       xy=(year, entropy_val), 
                       xytext=(0, 10), textcoords='offset points',
                       ha='center', fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    filename = FIGURES_DIR / f'measurement_entropy_evolution_{gender_filter}_from_{start_year}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def plot_hair_color_entropy_evolution_by_gender(merged_df, start_year=2000):
    """
    Plot the evolution of hair color entropy over time by gender.
    """
    df_filtered = merged_df[merged_df['year'] >= start_year]
    df_filtered = df_filtered.dropna(subset=['hair'])
    
    if len(df_filtered) == 0:
        print("No hair color data available")
        return
    
    # Get all unique hair colors for max entropy calculation
    all_hair_colors = df_filtered['hair'].unique()
    max_entropy = np.log(len(all_hair_colors))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    genders = ['female', 'male']
    colors = ['lightcoral', 'lightblue']
    
    for gender, color in zip(genders, colors):
        gender_data = df_filtered[df_filtered['gender_consensus'] == gender]
        
        if len(gender_data) == 0:
            continue
        
        yearly_entropy = []
        yearly_counts = []
        valid_years = []
        
        for year in sorted(gender_data['year'].unique()):
            year_data = gender_data[gender_data['year'] == year]
            
            if len(year_data) >= 30:  # Minimum for reliable entropy
                hair_counts = year_data['hair'].value_counts()
                probabilities = hair_counts.values / hair_counts.sum()
                year_entropy = entropy(probabilities)
                
                # Normalize entropy to 0-1 scale
                normalized_entropy = year_entropy / max_entropy if max_entropy > 0 else 0
                
                yearly_entropy.append(normalized_entropy)
                yearly_counts.append(len(year_data))
                valid_years.append(year)
        
        if len(yearly_entropy) > 0:
            ax.plot(valid_years, yearly_entropy, 
                    color='black', linewidth=2, marker='o', markersize=6,
                    markerfacecolor=color, markeredgecolor='black',
                    label=f'{gender.title()} Models')
            
            # Add trend line
            if len(valid_years) > 2:
                z = np.polyfit(valid_years, yearly_entropy, 1)
                p = np.poly1d(z)
                trend_direction = "↗️" if z[0] > 0 else "↘️"
                ax.plot(valid_years, p(valid_years), 
                        color=color, linestyle='--', linewidth=2, alpha=0.8,
                        label=f'{gender.title()} Trend ({z[0]*10:.4f}/decade) {trend_direction}')
    
    ax.set_title('Hair Color Diversity Evolution by Gender (Normalized Entropy)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Normalized Entropy (0-1)', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    remove_spines(ax)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'hair_color_entropy_evolution_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_eye_color_entropy_evolution_by_gender(merged_df, start_year=2000):
    """
    Plot the evolution of eye color entropy over time by gender.
    """
    df_filtered = merged_df[merged_df['year'] >= start_year]
    df_filtered = df_filtered.dropna(subset=['eyes'])
    
    if len(df_filtered) == 0:
        print("No eye color data available")
        return
    
    # Get all unique eye colors for max entropy calculation
    all_eye_colors = df_filtered['eyes'].unique()
    max_entropy = np.log(len(all_eye_colors))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    genders = ['female', 'male']
    colors = ['lightcoral', 'lightblue']
    
    for gender, color in zip(genders, colors):
        gender_data = df_filtered[df_filtered['gender_consensus'] == gender]
        
        if len(gender_data) == 0:
            continue
        
        yearly_entropy = []
        yearly_counts = []
        valid_years = []
        
        for year in sorted(gender_data['year'].unique()):
            year_data = gender_data[gender_data['year'] == year]
            
            if len(year_data) >= 30:  # Minimum for reliable entropy
                eye_counts = year_data['eyes'].value_counts()
                probabilities = eye_counts.values / eye_counts.sum()
                year_entropy = entropy(probabilities)
                
                # Normalize entropy to 0-1 scale
                normalized_entropy = year_entropy / max_entropy if max_entropy > 0 else 0
                
                yearly_entropy.append(normalized_entropy)
                yearly_counts.append(len(year_data))
                valid_years.append(year)
        
        if len(yearly_entropy) > 0:
            ax.plot(valid_years, yearly_entropy, 
                    color='black', linewidth=2, marker='o', markersize=6,
                    markerfacecolor=color, markeredgecolor='black',
                    label=f'{gender.title()} Models')
            
            # Add trend line
            if len(valid_years) > 2:
                z = np.polyfit(valid_years, yearly_entropy, 1)
                p = np.poly1d(z)
                trend_direction = "↗️" if z[0] > 0 else "↘️"
                ax.plot(valid_years, p(valid_years), 
                        color=color, linestyle='--', linewidth=2, alpha=0.8,
                        label=f'{gender.title()} Trend ({z[0]*10:.4f}/decade) {trend_direction}')
    
    ax.set_title('Eye Color Diversity Evolution by Gender (Normalized Entropy)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Normalized Entropy (0-1)', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    remove_spines(ax)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'eye_color_entropy_evolution_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_nationality_entropy_evolution_by_gender(merged_df, start_year=2000):
    """
    Plot the evolution of nationality entropy over time by gender.
    """
    # Load nationality data from file
    try:
        nationality_df = pd.read_csv(MODELS_NATIONALITY)
    except FileNotFoundError:
        print("Nationality data file not found.")
        return
    
    # Filter years and merge nationality data
    df_filtered = merged_df[merged_df['year'] >= start_year]
    nationality_merged = df_filtered.merge(
        nationality_df[['model_id', 'nationality']], 
        on='model_id', 
        how='left'
    )
    nationality_merged = nationality_merged.dropna(subset=['nationality'])
    
    if len(nationality_merged) == 0:
        print("No nationality data available after merging")
        return
    
    # Create region mapping
    nationality_to_region = create_nationality_to_region_mapping()
    nationality_merged['region'] = nationality_merged['nationality'].map(nationality_to_region)
    nationality_merged = nationality_merged.dropna(subset=['region'])
    
    # Get all unique regions for max entropy calculation
    all_regions = nationality_merged['region'].unique()
    max_entropy = np.log(len(all_regions))
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    genders = ['female', 'male']
    colors = ['lightcoral', 'lightblue']
    
    for gender, color in zip(genders, colors):
        gender_data = nationality_merged[nationality_merged['gender_consensus'] == gender]
        
        if len(gender_data) == 0:
            continue
        
        yearly_entropy = []
        yearly_counts = []
        valid_years = []
        
        for year in sorted(gender_data['year'].unique()):
            year_data = gender_data[gender_data['year'] == year]
            
            if len(year_data) >= 20:  # Minimum for reliable entropy
                region_counts = year_data['region'].value_counts()
                probabilities = region_counts.values / region_counts.sum()
                year_entropy = entropy(probabilities)
                
                # Normalize entropy to 0-1 scale
                normalized_entropy = year_entropy / max_entropy if max_entropy > 0 else 0
                
                yearly_entropy.append(normalized_entropy)
                yearly_counts.append(len(year_data))
                valid_years.append(year)
        
        if len(yearly_entropy) > 0:
            ax.plot(valid_years, yearly_entropy, 
                    color='black', linewidth=2, marker='o', markersize=6,
                    markerfacecolor=color, markeredgecolor='black',
                    label=f'{gender.title()} Models')
            
            # Add trend line
            if len(valid_years) > 2:
                z = np.polyfit(valid_years, yearly_entropy, 1)
                p = np.poly1d(z)
                trend_direction = "↗️" if z[0] > 0 else "↘️"
                ax.plot(valid_years, p(valid_years), 
                        color=color, linestyle='--', linewidth=2, alpha=0.8,
                        label=f'{gender.title()} Trend ({z[0]*10:.4f}/decade) {trend_direction}')
    
    ax.set_title('Nationality Diversity Evolution by Gender (Normalized Entropy)', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Normalized Entropy (0-1)', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    remove_spines(ax)
    
    # Add interpretation text
    interpretation_text = (
        "Higher entropy = More diverse representation\\n"
        "Lower entropy = More concentrated in fewer categories"
    )
    ax.text(0.02, 0.98, interpretation_text, transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nationality_entropy_evolution_by_gender.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_gender_separated_analysis(merged_df, start_year=2000):
    """
    Create separate analysis for male and female models.
    """
    genders = ['female', 'male']
    
    for gender in genders:
        gender_data = merged_df[merged_df['gender_consensus'] == gender]
        if len(gender_data) == 0:
            print(f"No data for {gender} models")
            continue
            
        print(f"\n{'='*50}")
        print(f"ANALYZING {gender.upper()} MODELS")
        print(f"{'='*50}")
        print(f"Total {gender} records: {len(gender_data):,}")
        print(f"Unique {gender} models: {gender_data['model_id'].nunique():,}")
        
        # Create evolution plots (means with std as error bars)
        print(f"\nCreating evolution plots for {gender} models...")
        for measurement_system in ['us', 'eu']:
            plot_measurements_evolution_by_gender(merged_df, gender, start_year=start_year, measurement_system=measurement_system)
        
        # Create standard deviation evolution plots
        print(f"\nCreating standard deviation evolution plots for {gender} models...")
        for measurement_system in ['us', 'eu']:
            plot_measurement_std_evolution_by_gender(merged_df, gender, start_year=start_year, measurement_system=measurement_system)
        
        # Create IQR evolution plots (NEW)
        print(f"\nCreating IQR evolution plots for {gender} models...")
        for measurement_system in ['us', 'eu']:
            plot_measurement_iqr_evolution_by_gender(merged_df, gender, start_year=start_year, measurement_system=measurement_system)

def main():
    """
    Main analysis function.
    """
    # Configuration
    start_year = 2000
    
    print("=" * 80)
    print("FASHION MODEL DIRECTORY - COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Analysis from year: {start_year}")
    
    # Ensure required directories exist
    ensure_directories_exist()
    
    # Load all datasets
    datasets = load_core_datasets()
    
    # Create master dataset
    master_df = create_master_dataset(datasets)
    
    if master_df.empty:
        print("\nError: Could not create master dataset. Check data files.")
        return None
    
    # Analyze data quality
    analyze_data_quality(master_df)
    
    # Prepare merged data for detailed analysis
    print("\nPreparing data for comprehensive analysis...")
    merged_df = prepare_shows_and_measurements_data(datasets)
    
    if not merged_df.empty:
        # Create yearly counts visualization
        print("\nCreating yearly counts visualization...")
        plot_yearly_counts_by_gender(merged_df, start_year=start_year)

        
        # Create hair and eye color analysis by gender
        print("\nCreating hair and eye color analysis by gender...")
        plot_hair_eye_color_by_gender(merged_df, start_year=start_year)
        
        # Create separate hair color evolution by gender
        print("\nCreating hair color evolution by gender...")
        plot_hair_color_evolution_by_gender(merged_df, start_year=start_year)
        
        # Create separate eye color evolution by gender
        print("\nCreating eye color evolution by gender...")
        plot_eye_color_evolution_by_gender(merged_df, start_year=start_year)
        
        # Create nationality evolution by region and gender
        print("\nCreating nationality evolution by region and gender...")
        plot_nationality_evolution_by_region_and_gender(merged_df, start_year=start_year)
        
        # Create entropy evolution analysis
        print("\nCreating measurement entropy evolution for female models...")
        plot_measurement_entropy_evolution_by_gender(merged_df, 'female', start_year=start_year)
        
        print("\nCreating measurement entropy evolution for male models...")
        plot_measurement_entropy_evolution_by_gender(merged_df, 'male', start_year=start_year)
        
        print("\nCreating hair color entropy evolution by gender...")
        plot_hair_color_entropy_evolution_by_gender(merged_df, start_year=start_year)
        
        print("\nCreating eye color entropy evolution by gender...")
        plot_eye_color_entropy_evolution_by_gender(merged_df, start_year=start_year)
        
        print("\nCreating nationality entropy evolution by gender...")
        plot_nationality_entropy_evolution_by_gender(merged_df, start_year=start_year)
        
        # Create gender-separated analysis
        print("\nCreating gender-separated analysis...")
        create_gender_separated_analysis(merged_df, start_year=start_year)
        
        # Save merged dataset for further analysis
        merged_output_path = DATA_DIR / "merged_shows_measurements.csv"
        merged_df.to_csv(merged_output_path, index=False)
        print(f"\nMerged dataset saved to: {merged_output_path}")
        print(f"   Size: {len(merged_df):,} rows × {len(merged_df.columns)} columns")
    else:
        print("\nWarning: Could not create merged dataset for detailed analysis")
    
    # Save master dataset
    master_output_path = DATA_DIR / "master_dataset.csv"
    master_df.to_csv(master_output_path, index=False)
    print(f"\nMaster dataset saved to: {master_output_path}")
    print(f"   Size: {len(master_df):,} rows × {len(master_df.columns)} columns")
    
    print(f"\nComprehensive analysis complete!")
    print(f"Analysis finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    return master_df

if __name__ == "__main__":
    master_dataset = main()