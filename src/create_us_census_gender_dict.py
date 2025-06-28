"""
Create a comprehensive gender dictionary from US Census yearly name data.

This script processes all yearly files from 1880-2024 and creates a consolidated
dataframe with name, total male count, total female count, and determined gender.
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import json

# Import our centralized paths
from src.paths import DATA_DIR

# US Census data directory
US_CENSUS_DIR = DATA_DIR / "us_census_names"

def load_yearly_census_data():
    """
    Load and compile all yearly US Census name data files.
    
    Returns:
        pd.DataFrame: DataFrame with columns [name, male_count, female_count, total_count]
    """
    print("🗃️  Compiling US Census name data from 1880-2024...")
    
    # Initialize aggregation dictionary
    name_counts = {}
    
    # Get all year files
    year_files = sorted([f for f in os.listdir(US_CENSUS_DIR) if f.startswith('yob') and f.endswith('.txt')])
    
    print(f"Found {len(year_files)} yearly files from {year_files[0]} to {year_files[-1]}")
    
    # Process each year file
    for year_file in tqdm(year_files, desc="Processing yearly files"):
        year_path = US_CENSUS_DIR / year_file
        year = year_file.replace('yob', '').replace('.txt', '')
        
        try:
            # Read the year file
            year_df = pd.read_csv(year_path, names=['name', 'gender', 'count'])
            
            # Process each row
            for _, row in year_df.iterrows():
                name = row['name'].lower().strip()  # Normalize to lowercase
                gender = row['gender'].upper().strip()
                count = int(row['count'])
                
                # Initialize name entry if not exists
                if name not in name_counts:
                    name_counts[name] = {'male_count': 0, 'female_count': 0}
                
                # Add to appropriate gender count
                if gender == 'M':
                    name_counts[name]['male_count'] += count
                elif gender == 'F':
                    name_counts[name]['female_count'] += count
                    
        except Exception as e:
            print(f"Warning: Error processing {year_file}: {e}")
            continue
    
    print(f"✅ Processed {len(name_counts)} unique names across all years")
    
    # Convert to DataFrame
    records = []
    for name, counts in name_counts.items():
        male_count = counts['male_count']
        female_count = counts['female_count']
        total_count = male_count + female_count
        
        records.append({
            'name': name,
            'male_count': male_count,
            'female_count': female_count,
            'total_count': total_count
        })
    
    df = pd.DataFrame(records)
    df = df.sort_values('total_count', ascending=False).reset_index(drop=True)
    
    return df

def determine_gender(row, min_count=5):
    """
    Determine gender based on 90% threshold rule.
    
    Args:
        row: DataFrame row with male_count, female_count, total_count
        min_count: Minimum total count to make a determination
        
    Returns:
        str: 'male', 'female', or 'unknown'
    """
    if row['total_count'] < min_count:
        return 'unknown'
    
    male_ratio = row['male_count'] / row['total_count']
    
    if male_ratio >= 0.9:  # 90% or more male
        return 'male'
    elif male_ratio <= 0.1:  # 10% or less male (90% or more female)
        return 'female'
    else:
        return 'unknown'  # Less than 90% consensus

def create_gender_analysis(df):
    """Create detailed analysis of the gender determination."""
    
    # Add gender determination with 90% threshold
    df['gender'] = df.apply(lambda row: determine_gender(row, min_count=5), axis=1)
    
    # Calculate male ratio
    df['male_ratio'] = df['male_count'] / df['total_count']
    
    print("\n📊 GENDER DISTRIBUTION ANALYSIS (90% threshold):")
    print("="*50)
    
    counts = df['gender'].value_counts()
    total = len(df)
    
    print(f"\n90% Threshold (min 10 occurrences):")
    for gender in ['male', 'female', 'unknown']:
        count = counts.get(gender, 0)
        percentage = count / total * 100
        print(f"  {gender}: {count:,} ({percentage:.1f}%)")
    
    return df

def save_census_gender_dict(df):
    """
    Save the US Census gender dictionary in multiple formats.
    
    Args:
        df: DataFrame with gender determinations (90% threshold)
    """
    
    # Create simple dictionary for 90% threshold
    simple_dict = {}
    
    for _, row in df.iterrows():
        if row['gender'] != 'unknown':
            simple_dict[row['name']] = row['gender']
    
    print(f"\n💾 SAVING US CENSUS GENDER DICTIONARY:")
    print("="*50)
    
    # Save comprehensive CSV
    csv_path = DATA_DIR / "us_census_gender_comprehensive.csv"
    df.to_csv(csv_path, index=False)
    print(f"📋 Comprehensive data: {csv_path}")
    print(f"   - {len(df):,} total names")
    print(f"   - Columns: name, male_count, female_count, total_count, gender, male_ratio")
    
    # Save simple JSON dictionary (90% threshold)
    json_path = DATA_DIR / "us_census_gender_dict_90.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(simple_dict, f, ensure_ascii=False, indent=2)
    print(f"📖 Simple dictionary (90% threshold): {json_path}")
    print(f"   - {len(simple_dict):,} names with gender assignments")
    
    # Save top names for reference
    top_names_path = DATA_DIR / "us_census_top_names.csv"
    top_1000 = df.head(1000)[['name', 'total_count', 'male_count', 'female_count', 'male_ratio', 'gender']]
    top_1000.to_csv(top_names_path, index=False)
    print(f"🏆 Top 1000 names: {top_names_path}")
    
    return simple_dict

def print_interesting_statistics(df):
    """Print interesting statistics about the US Census name data."""
    
    print(f"\n🎯 INTERESTING STATISTICS:")
    print("="*50)
    
    # Most popular names overall
    print(f"\n📈 Most Popular Names (All Time):")
    top_10 = df.head(10)
    for i, row in top_10.iterrows():
        print(f"  {i+1:2d}. {row['name'].title():<12} - {row['total_count']:,} total ({row['male_ratio']*100:.1f}% male)")
    
    # Most ambiguous names (close to 50/50)
    df_substantial = df[df['total_count'] >= 1000]  # Only names with substantial counts
    df_substantial['ambiguity'] = abs(df_substantial['male_ratio'] - 0.5)
    most_ambiguous = df_substantial.nsmallest(10, 'ambiguity')
    
    print(f"\n⚖️  Most Gender-Ambiguous Names (min 1000 total):")
    for i, row in most_ambiguous.iterrows():
        print(f"  {row['name'].title():<12} - {row['male_ratio']*100:.1f}% male ({row['total_count']:,} total)")
    
    # Largest counts
    print(f"\n🔝 Highest Single Counts:")
    max_male = df.loc[df['male_count'].idxmax()]
    max_female = df.loc[df['female_count'].idxmax()]
    print(f"  Male:   {max_male['name'].title()} - {max_male['male_count']:,}")
    print(f"  Female: {max_female['name'].title()} - {max_female['female_count']:,}")

def main():
    """Main function to create US Census gender dictionary."""
    
    print("="*60)
    print("US CENSUS NAME DATA COMPILATION")
    print("="*60)
    
    # Load and compile yearly data
    census_df = load_yearly_census_data()
    
    # Create gender analysis
    census_df = create_gender_analysis(census_df)
    
    # Print interesting statistics
    print_interesting_statistics(census_df)
    
    # Save results
    gender_dict = save_census_gender_dict(census_df)
    
    print(f"\n✅ US Census gender dictionary creation complete!")
    print(f"📊 Summary: {len(census_df):,} unique names processed")
    print(f"🎯 Gender assignments: {len(gender_dict):,} names with definitive gender")
    print("="*60)
    
    return census_df, gender_dict

if __name__ == "__main__":
    df, gender_dict = main()