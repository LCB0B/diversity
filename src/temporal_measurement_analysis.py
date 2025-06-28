"""
Temporal measurement distribution analysis.

This script analyzes how model measurements have changed over time,
identifying shifts in body type preferences in the fashion industry.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from scipy import stats
from scipy.stats import entropy
import json

warnings.filterwarnings('ignore')

# Import centralized path management
from src.paths import (
    DATA_DIR,
    FIGURES_DIR,
    ensure_directories_exist
)

# Use master dataset and shows data
MASTER_DATASET = DATA_DIR / "master_dataset.csv"
MODELS_SHOWS_ALL = DATA_DIR / "models_shows_all.json"

def load_temporal_data():
    """
    Load and merge measurement data with temporal show information.
    
    Returns:
        pd.DataFrame: Combined dataset with measurements and show years
    """
    print("Loading temporal measurement data...")
    
    # Load master dataset
    try:
        master_df = pd.read_csv(MASTER_DATASET)
        print(f"Loaded {len(master_df):,} model records")
    except FileNotFoundError:
        print(f"Error: {MASTER_DATASET} not found")
        return pd.DataFrame()
    
    # Convert height to numeric
    master_df['height-metric'] = pd.to_numeric(master_df['height-metric'], errors='coerce')
    
    # Load shows data for temporal information
    try:
        with open(MODELS_SHOWS_ALL, 'r', encoding='utf-8') as f:
            shows_data = json.load(f)
        print(f"Loaded shows data for temporal analysis")
    except FileNotFoundError:
        print(f"Error: {MODELS_SHOWS_ALL} not found")
        return pd.DataFrame()
    
    # Extract show years for each model
    model_years = {}
    for model_data in shows_data:
        model_id = model_data.get('model_id', '')
        shows = model_data.get('shows', [])
        
        # Get all years this model appeared in shows
        years = []
        for show in shows:
            year = show.get('year', 0)
            try:
                year = int(year) if year else 0
                if year > 1990 and year <= 2024:  # Reasonable year range
                    years.append(year)
            except (ValueError, TypeError):
                continue
        
        if years:
            # Use the first year the model appeared (debut year)
            model_years[model_id] = min(years)
    
    print(f"Found temporal data for {len(model_years):,} models")
    
    # Merge with master dataset
    master_df['debut_year'] = master_df['model_id'].map(model_years)
    
    # Filter for models with temporal data and complete measurements
    measurement_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu']
    temporal_df = master_df.dropna(subset=measurement_cols + ['debut_year'])
    
    # Filter for reasonable year range
    temporal_df = temporal_df[
        (temporal_df['debut_year'] >= 2000) & 
        (temporal_df['debut_year'] <= 2024)
    ]
    
    print(f"Final dataset: {len(temporal_df):,} models with temporal measurement data (2000-2024)")
    
    return temporal_df

def analyze_measurement_trends_by_year(temporal_df):
    """
    Analyze how average measurements change by year.
    
    Args:
        temporal_df (pd.DataFrame): Dataset with temporal information
        
    Returns:
        pd.DataFrame: Year-by-year measurement statistics
    """
    print("Analyzing measurement trends by year...")
    
    measurement_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu']
    
    # Calculate yearly statistics
    yearly_stats = []
    
    for year in sorted(temporal_df['debut_year'].unique()):
        year_data = temporal_df[temporal_df['debut_year'] == year]
        
        if len(year_data) < 10:  # Skip years with too few models
            continue
            
        stats_row = {'year': year, 'count': len(year_data)}
        
        for col in measurement_cols:
            values = year_data[col].dropna()
            if len(values) > 0:
                stats_row[f'{col}_mean'] = values.mean()
                stats_row[f'{col}_std'] = values.std()
                stats_row[f'{col}_median'] = values.median()
                stats_row[f'{col}_q25'] = values.quantile(0.25)
                stats_row[f'{col}_q75'] = values.quantile(0.75)
        
        yearly_stats.append(stats_row)
    
    yearly_df = pd.DataFrame(yearly_stats)
    print(f"Calculated trends for {len(yearly_df)} years")
    
    return yearly_df

def analyze_measurement_distributions_by_period(temporal_df):
    """
    Compare measurement distributions across different time periods.
    
    Args:
        temporal_df (pd.DataFrame): Dataset with temporal information
        
    Returns:
        dict: Distribution comparisons by period
    """
    print("Analyzing measurement distributions by time period...")
    
    # Define time periods
    periods = {
        'Early 2000s': (2000, 2005),
        'Late 2000s': (2006, 2010), 
        'Early 2010s': (2011, 2015),
        'Late 2010s': (2016, 2020),
        'Early 2020s': (2021, 2024)
    }
    
    measurement_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu']
    results = {}
    
    for period_name, (start_year, end_year) in periods.items():
        period_data = temporal_df[
            (temporal_df['debut_year'] >= start_year) & 
            (temporal_df['debut_year'] <= end_year)
        ]
        
        if len(period_data) < 50:  # Skip periods with too few models
            continue
            
        period_stats = {
            'period': period_name,
            'years': f'{start_year}-{end_year}',
            'count': len(period_data)
        }
        
        for col in measurement_cols:
            values = period_data[col].dropna()
            if len(values) > 0:
                period_stats[f'{col}_mean'] = values.mean()
                period_stats[f'{col}_std'] = values.std()
                period_stats[f'{col}_median'] = values.median()
        
        results[period_name] = period_stats
        print(f"  {period_name}: {len(period_data):,} models")
    
    return results

def calculate_distribution_shifts(temporal_df):
    """
    Calculate statistical significance of distribution shifts over time.
    
    Args:
        temporal_df (pd.DataFrame): Dataset with temporal information
        
    Returns:
        dict: Statistical test results
    """
    print("Calculating distribution shift significance...")
    
    measurement_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu']
    
    # Compare early period (2000-2010) vs late period (2015-2024)
    early_period = temporal_df[
        (temporal_df['debut_year'] >= 2000) & 
        (temporal_df['debut_year'] <= 2010)
    ]
    
    late_period = temporal_df[
        (temporal_df['debut_year'] >= 2015) & 
        (temporal_df['debut_year'] <= 2024)
    ]
    
    results = {}
    
    for col in measurement_cols:
        early_values = early_period[col].dropna()
        late_values = late_period[col].dropna()
        
        if len(early_values) > 30 and len(late_values) > 30:
            # T-test for mean difference
            t_stat, t_pvalue = stats.ttest_ind(early_values, late_values)
            
            # Kolmogorov-Smirnov test for distribution difference
            ks_stat, ks_pvalue = stats.ks_2samp(early_values, late_values)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(early_values)-1)*early_values.var() + 
                                 (len(late_values)-1)*late_values.var()) / 
                                (len(early_values) + len(late_values) - 2))
            cohens_d = (late_values.mean() - early_values.mean()) / pooled_std
            
            results[col] = {
                'early_mean': early_values.mean(),
                'late_mean': late_values.mean(),
                'mean_change': late_values.mean() - early_values.mean(),
                'early_std': early_values.std(),
                'late_std': late_values.std(),
                't_statistic': t_stat,
                't_pvalue': t_pvalue,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'cohens_d': cohens_d,
                'early_n': len(early_values),
                'late_n': len(late_values)
            }
    
    return results

def plot_temporal_trends(yearly_df, temporal_df):
    """
    Create visualizations of temporal measurement trends.
    
    Args:
        yearly_df (pd.DataFrame): Year-by-year statistics
        temporal_df (pd.DataFrame): Full temporal dataset
    """
    ensure_directories_exist()
    
    measurement_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu']
    measurement_names = ['Height (cm)', 'Bust (cm)', 'Waist (cm)', 'Hips (cm)']
    
    # 1. Yearly trends plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (col, name) in enumerate(zip(measurement_cols, measurement_names)):
        ax = axes[i]
        
        # Plot mean with confidence interval
        years = yearly_df['year']
        means = yearly_df[f'{col}_mean']
        stds = yearly_df[f'{col}_std']
        
        ax.plot(years, means, 'o-', linewidth=2, markersize=4, label='Mean')
        ax.fill_between(years, means - stds, means + stds, alpha=0.3, label='±1 STD')
        
        # Add trend line
        z = np.polyfit(years, means, 1)
        p = np.poly1d(z)
        ax.plot(years, p(years), '--', color='red', alpha=0.7, label='Trend')
        
        ax.set_xlabel('Debut Year')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Trends Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add trend direction indicator
        slope = z[0]
        direction = "↗️" if slope > 0 else "↘️" if slope < 0 else "→"
        ax.text(0.02, 0.98, f'Trend: {direction} {slope:.2f}/year', 
                transform=ax.transAxes, va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'temporal_measurement_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Distribution comparison by periods
    periods = {
        'Early 2000s': (2000, 2005),
        'Late 2010s': (2016, 2020)
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    colors = ['lightblue', 'lightcoral']
    
    for i, (col, name) in enumerate(zip(measurement_cols, measurement_names)):
        ax = axes[i]
        
        for j, (period_name, (start_year, end_year)) in enumerate(periods.items()):
            period_data = temporal_df[
                (temporal_df['debut_year'] >= start_year) & 
                (temporal_df['debut_year'] <= end_year)
            ]
            
            values = period_data[col].dropna()
            if len(values) > 0:
                ax.hist(values, bins=30, alpha=0.7, label=period_name, 
                       color=colors[j], density=True)
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name} Distribution Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'temporal_distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Box plots by decade
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Create decade groups
    temporal_df['decade_group'] = pd.cut(temporal_df['debut_year'], 
                                        bins=[2000, 2005, 2010, 2015, 2020, 2024],
                                        labels=['2000-2005', '2006-2010', '2011-2015', 
                                               '2016-2020', '2021-2024'])
    
    for i, (col, name) in enumerate(zip(measurement_cols, measurement_names)):
        ax = axes[i]
        
        # Create box plot data
        box_data = []
        labels = []
        for decade in ['2000-2005', '2006-2010', '2011-2015', '2016-2020', '2021-2024']:
            decade_values = temporal_df[temporal_df['decade_group'] == decade][col].dropna()
            if len(decade_values) > 10:
                box_data.append(decade_values)
                labels.append(decade)
        
        if box_data:
            ax.boxplot(box_data, labels=labels)
            ax.set_xlabel('Period')
            ax.set_ylabel(name)
            ax.set_title(f'{name} by Time Period')
            ax.grid(True, alpha=0.3)
            plt.setp(ax.get_xticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'temporal_measurement_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Temporal trend plots saved to {FIGURES_DIR}")

def save_temporal_analysis_results(yearly_df, period_stats, shift_results):
    """
    Save temporal analysis results to files.
    
    Args:
        yearly_df (pd.DataFrame): Year-by-year statistics
        period_stats (dict): Period comparison results
        shift_results (dict): Distribution shift test results
    """
    ensure_directories_exist()
    
    # Save yearly trends
    yearly_output = DATA_DIR / "temporal_measurement_trends.csv"
    yearly_df.to_csv(yearly_output, index=False)
    print(f"Yearly trends saved to: {yearly_output}")
    
    # Save period comparisons
    period_df = pd.DataFrame(period_stats.values())
    period_output = DATA_DIR / "temporal_period_comparison.csv"
    period_df.to_csv(period_output, index=False)
    print(f"Period comparison saved to: {period_output}")
    
    # Save shift analysis
    shift_df = pd.DataFrame(shift_results).T
    shift_output = DATA_DIR / "temporal_distribution_shifts.csv"
    shift_df.to_csv(shift_output, index=True)
    print(f"Distribution shift analysis saved to: {shift_output}")

def print_shift_analysis_summary(shift_results):
    """
    Print a summary of the distribution shift analysis.
    
    Args:
        shift_results (dict): Distribution shift test results
    """
    print("\n" + "="*80)
    print("TEMPORAL DISTRIBUTION SHIFT ANALYSIS SUMMARY")
    print("="*80)
    print("Comparing Early Period (2000-2010) vs Late Period (2015-2024)")
    print()
    
    measurement_names = {
        'height-metric': 'Height',
        'bust-eu': 'Bust',
        'waist-eu': 'Waist', 
        'hips-eu': 'Hips'
    }
    
    for col, results in shift_results.items():
        name = measurement_names.get(col, col)
        print(f"{name.upper()}:")
        print(f"  Early period (2000-2010): {results['early_mean']:.1f}±{results['early_std']:.1f} cm (n={results['early_n']:,})")
        print(f"  Late period (2015-2024):  {results['late_mean']:.1f}±{results['late_std']:.1f} cm (n={results['late_n']:,})")
        print(f"  Change: {results['mean_change']:+.1f} cm")
        print(f"  Effect size (Cohen's d): {results['cohens_d']:.3f}")
        
        # Interpret effect size
        abs_d = abs(results['cohens_d'])
        if abs_d < 0.2:
            effect = "negligible"
        elif abs_d < 0.5:
            effect = "small"
        elif abs_d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        
        print(f"  Effect size: {effect}")
        
        # Statistical significance
        if results['t_pvalue'] < 0.001:
            t_sig = "p < 0.001 (highly significant)"
        elif results['t_pvalue'] < 0.01:
            t_sig = f"p = {results['t_pvalue']:.3f} (significant)"
        elif results['t_pvalue'] < 0.05:
            t_sig = f"p = {results['t_pvalue']:.3f} (significant)"
        else:
            t_sig = f"p = {results['t_pvalue']:.3f} (not significant)"
        
        print(f"  Mean difference significance: {t_sig}")
        
        if results['ks_pvalue'] < 0.001:
            ks_sig = "p < 0.001 (highly significant)"
        elif results['ks_pvalue'] < 0.01:
            ks_sig = f"p = {results['ks_pvalue']:.3f} (significant)"
        elif results['ks_pvalue'] < 0.05:
            ks_sig = f"p = {results['ks_pvalue']:.3f} (significant)"
        else:
            ks_sig = f"p = {results['ks_pvalue']:.3f} (not significant)"
        
        print(f"  Distribution difference significance: {ks_sig}")
        print()

def main():
    """
    Main function to run the complete temporal measurement analysis.
    """
    print("=" * 60)
    print("TEMPORAL MEASUREMENT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Load temporal data
    temporal_df = load_temporal_data()
    
    if temporal_df.empty:
        print("No temporal data available for analysis")
        return
    
    # Analyze yearly trends
    yearly_df = analyze_measurement_trends_by_year(temporal_df)
    
    # Analyze period comparisons
    period_stats = analyze_measurement_distributions_by_period(temporal_df)
    
    # Calculate distribution shifts
    shift_results = calculate_distribution_shifts(temporal_df)
    
    # Print summary
    print_shift_analysis_summary(shift_results)
    
    # Create visualizations
    plot_temporal_trends(yearly_df, temporal_df)
    
    # Save results
    save_temporal_analysis_results(yearly_df, period_stats, shift_results)
    
    print("\n" + "=" * 60)
    print("TEMPORAL MEASUREMENT ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return temporal_df, yearly_df, period_stats, shift_results

if __name__ == "__main__":
    temporal_df, yearly_df, period_stats, shift_results = main()