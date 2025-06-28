"""
Gender detection script for models.

This script uses multiple methods to assign genders to models:
1. gender-guesser library
2. Small gender dictionary 
3. Large gender dictionary 
4. US Census gender dictionary (90% threshold)

Each method has progress tracking with tqdm and uses majority vote for consensus.
"""

import pandas as pd
import json
import re
from tqdm import tqdm
from pathlib import Path

# Import our centralized paths
from src.paths import MODELS_SHOWS_ALL,MODELS_MEASURE, GENDER_DICT, NAME_GENDER_DICT, NAME_GENDER_DICT_LARGE, DATA_DIR, US_CENSUS_GENDER_COMPREHENSIVE, US_CENSUS_GENDER_DICT

# Try to import gender_guesser
try:
    import gender_guesser.detector as gender
    GENDER_GUESSER_AVAILABLE = True
except ImportError:
    print("Warning: gender_guesser not available. Installing...")
    import subprocess23
    subprocess.check_call(["pip", "install", "gender-guesser"])
    import gender_guesser.detector as gender
    GENDER_GUESSER_AVAILABLE = True

def load_models_data():
    """Load models data and extract model_id and model_name."""
    print("Loading models data...")
    
    df = pd.read_csv(MODELS_MEASURE)
    df = df[['model_id', 'name']].drop_duplicates()
    #rename name column to model_name for consistency
    return df

def extract_first_name(full_name, capitalize=False):
    """Extract first name from full name for gender detection."""
    if pd.isna(full_name) or not full_name.strip():
        return ""
    
    # Clean the name: remove special characters, split by spaces
    cleaned_name = re.sub(r'[^\w\s]', ' ', str(full_name))
    names = cleaned_name.strip().split()
    
    if names:
        first_name = names[0]
        if capitalize:
            return first_name.capitalize()  # First letter upper, rest lower
        else:
            return first_name.lower()  # All lowercase
    return ""

def gender_detection_guesser(models_df):
    """
    Use gender-guesser library to detect gender based on first names.
    
    Args:
        models_df (pd.DataFrame): DataFrame with model_id and model_name columns
    
    Returns:
        pd.Series: Series with gender predictions
    """
    print("\n🔍 Running gender detection with gender-guesser library...")
    
    detector = gender.Detector()
    genders = []
    
    for _, row in tqdm(models_df.iterrows(), total=len(models_df), desc="Gender-guesser"):
        first_name = extract_first_name(row['name'], capitalize=True)  # Capitalize for gender-guesser
        
        if not first_name:
            genders.append('unknown')
            continue
        
        # gender-guesser returns: male, female, mostly_male, mostly_female, andy (androgynous), unknown
        result = detector.get_gender(first_name)
        
        # Map results to simpler categories
        if result in ['male', 'mostly_male']:
            genders.append('male')
        elif result in ['female', 'mostly_female']:
            genders.append('female')
        else:
            genders.append('unknown')
    
    return pd.Series(genders, name='gender_guesser')

def gender_detection_small_dict(models_df):
    """
    Use small curated gender dictionary for gender detection.
    
    Args:
        models_df (pd.DataFrame): DataFrame with model_id and model_name columns
    
    Returns:
        pd.Series: Series with gender predictions
    """
    print("\n📖 Running gender detection with small curated dictionary...")
    
    # Load the small gender dictionary
    try:
        with open(GENDER_DICT, 'r', encoding='utf-8') as f:
            gender_dict = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {GENDER_DICT} not found")
        return pd.Series(['unknown'] * len(models_df), name='gender_small_dict')
    
    genders = []
    
    for _, row in tqdm(models_df.iterrows(), total=len(models_df), desc="Small dict"):
        model_name = row['name']
        
        # Direct lookup by full name (case-sensitive as in the dictionary)
        if model_name in gender_dict:
            genders.append(gender_dict[model_name])
        else:
            genders.append('unknown')
    
    print(f"Small dict matches: {sum(1 for g in genders if g != 'unknown')}/{len(genders)}")
    return pd.Series(genders, name='gender_small_dict')

def gender_detection_large_dict(models_df):
    """
    Use large comprehensive gender dictionary for gender detection.
    
    Args:
        models_df (pd.DataFrame): DataFrame with model_id and model_name columns
    
    Returns:
        pd.Series: Series with gender predictions
    """
    print("\n📚 Running gender detection with large name dictionary...")
    
    # Load the large gender dictionary from CSV
    try:
        large_gender_df = pd.read_csv(NAME_GENDER_DICT_LARGE)
        
        # Create dictionary from CSV data
        large_gender_dict = {}
        for _, row in large_gender_df.iterrows():
            name = str(row['name']).lower().strip()
            gender = str(row['gender']).upper().strip()
            
            # Clean name (remove quotes and special characters if needed)
            name = name.strip('"\'')
            
            # Normalize gender values
            if gender == 'M':
                gender = 'male'
            elif gender == 'F':
                gender = 'female'
            else:
                continue  # Skip unknown genders
            
            # For large dataset, just take first occurrence
            if name not in large_gender_dict:
                large_gender_dict[name] = gender
        
        print(f"Loaded {len(large_gender_dict)} unique names from large CSV dataset")
        
    except FileNotFoundError:
        print(f"Warning: {NAME_GENDER_DICT_LARGE} not found")
        return pd.Series(['unknown'] * len(models_df), name='gender_large_dict')
    except Exception as e:
        print(f"Warning: Could not load {NAME_GENDER_DICT_LARGE}: {e}")
        return pd.Series(['unknown'] * len(models_df), name='gender_large_dict')
    
    genders = []
    
    for _, row in tqdm(models_df.iterrows(), total=len(models_df), desc="Large dict"):
        first_name = extract_first_name(row['name'], capitalize=False)  # Lowercase for large dict
        
        if not first_name:
            genders.append('unknown')
            continue
        
        # Lookup by first name (lowercase)
        if first_name in large_gender_dict:
            genders.append(large_gender_dict[first_name])
        else:
            genders.append('unknown')
    
    print(f"Large dict matches: {sum(1 for g in genders if g != 'unknown')}/{len(genders)}")
    return pd.Series(genders, name='gender_large_dict')

def gender_detection_us_census(models_df):
    """
    Use US Census gender dictionary (90% threshold) for gender detection.
    
    Args:
        models_df (pd.DataFrame): DataFrame with model_id and model_name columns
    
    Returns:
        pd.Series: Series with gender predictions
    """
    print("\n🏛️  Running gender detection with US Census dictionary (90% threshold)...")
    
    # Load the US Census gender dictionary
    try:
        with open(US_CENSUS_GENDER_DICT, 'r', encoding='utf-8') as f:
            us_census_dict = json.load(f)
        
        print(f"Loaded {len(us_census_dict)} names from US Census dictionary")
        
    except FileNotFoundError:
        print(f"Warning: {US_CENSUS_GENDER_DICT} not found")
        return pd.Series(['unknown'] * len(models_df), name='gender_us_census')
    except Exception as e:
        print(f"Warning: Could not load {US_CENSUS_GENDER_DICT}: {e}")
        return pd.Series(['unknown'] * len(models_df), name='gender_us_census')
    
    genders = []
    
    for _, row in tqdm(models_df.iterrows(), total=len(models_df), desc="US Census"):
        first_name = extract_first_name(row['name'], capitalize=False)  # Lowercase for US Census dict
        
        if not first_name:
            genders.append('unknown')
            continue
        
        # Lookup by first name (lowercase)
        if first_name in us_census_dict:
            genders.append(us_census_dict[first_name])
        else:
            genders.append('unknown')
    
    print(f"US Census dict matches: {sum(1 for g in genders if g != 'unknown')}/{len(genders)}")
    return pd.Series(genders, name='gender_us_census')

def create_consensus_gender(guesser_series, small_dict_series, large_dict_series, us_census_series):
    """
    Create a consensus gender prediction from all four methods using majority voting.
    
    Uses majority vote among the four methods. If no majority exists, returns 'unknown'.
    
    Args:
        guesser_series (pd.Series): Gender-guesser predictions
        small_dict_series (pd.Series): Small dictionary predictions  
        large_dict_series (pd.Series): Large dictionary predictions
        us_census_series (pd.Series): US Census dictionary predictions
        
    Returns:
        pd.Series: Consensus gender predictions
    """
    print("\n🤝 Creating consensus gender predictions using majority voting...")
    
    consensus = []
    stats = {
        'unanimous_female': 0, 'unanimous_male': 0, 'unanimous_unknown': 0,
        'majority_female': 0, 'majority_male': 0, 
        'no_majority': 0, 'total_unknown': 0
    }
    
    for i in range(len(guesser_series)):
        # Get the four predictions
        predictions = [
            guesser_series.iloc[i],
            small_dict_series.iloc[i], 
            large_dict_series.iloc[i],
            us_census_series.iloc[i]
        ]
        
        # Count votes for each category (excluding 'unknown')
        female_votes = predictions.count('female')
        male_votes = predictions.count('male')
        unknown_votes = predictions.count('unknown')
        
        # Determine consensus based on majority voting (need at least 3 out of 4 for strong consensus)
        if female_votes >= 3:  # At least 3 out of 4 say female (strong consensus)
            consensus.append('female')
            if female_votes == 4:
                stats['unanimous_female'] += 1
            else:
                stats['majority_female'] += 1
        elif male_votes >= 3:  # At least 3 out of 4 say male (strong consensus)
            consensus.append('male')
            if male_votes == 4:
                stats['unanimous_male'] += 1
            else:
                stats['majority_male'] += 1
        elif female_votes == 2 and male_votes == 0:  # 2 female, 2 unknown (weak female consensus)
            consensus.append('female')
            stats['majority_female'] += 1
        elif male_votes == 2 and female_votes == 0:  # 2 male, 2 unknown (weak male consensus)
            consensus.append('male')
            stats['majority_male'] += 1
        # if one male or one female but it gender_guesser or us_census, we consider it a majority
        elif male_votes == 1 and unknown_votes == 3:
            if guesser_series.iloc[i] == 'male' or us_census_series.iloc[i] == 'male':
                consensus.append('male')
                stats['majority_male'] += 1
            else:
                consensus.append('unknown')
                stats['no_majority'] += 1
        elif female_votes == 1 and unknown_votes == 3:
            if guesser_series.iloc[i] == 'female' or us_census_series.iloc[i] == 'female':
                consensus.append('female')
                stats['majority_female'] += 1
            else:
                consensus.append('unknown')
                stats['no_majority'] += 1
        else:
            # No majority (tie, mixed, or all unknown)
            consensus.append('unknown')
            if unknown_votes == 4:
                stats['unanimous_unknown'] += 1
            else:
                stats['no_majority'] += 1
    
    # Calculate final stats
    total_known = stats['unanimous_female'] + stats['unanimous_male'] + stats['majority_female'] + stats['majority_male']
    stats['total_unknown'] = len(consensus) - total_known
    
    print(f"Majority voting results:")
    print(f"  - Unanimous agreement: {stats['unanimous_female'] + stats['unanimous_male'] + stats['unanimous_unknown']:,} cases")
    print(f"    • Female: {stats['unanimous_female']:,}, Male: {stats['unanimous_male']:,}, Unknown: {stats['unanimous_unknown']:,}")
    print(f"  - Majority vote (2/3): {stats['majority_female'] + stats['majority_male']:,} cases")
    print(f"    • Female: {stats['majority_female']:,}, Male: {stats['majority_male']:,}")
    print(f"  - No majority: {stats['no_majority']:,} cases")
    print(f"  - Total predictions: Female: {stats['unanimous_female'] + stats['majority_female']:,}, Male: {stats['unanimous_male'] + stats['majority_male']:,}, Unknown: {stats['total_unknown']:,}")
    
    return pd.Series(consensus, name='gender_consensus')

def create_gender_dataframe():
    """
    Create a comprehensive gender dataframe using all available methods.
    
    Returns:
        pd.DataFrame: DataFrame with model_id, model_name, and all gender predictions
    """
    print("="*60)
    print("FASHION MODEL DIRECTORY - GENDER DETECTION")
    print("="*60)
    
    # Load models data
    models_df = load_models_data()
    
    # Run all gender detection methods
    gender_guesser_series = gender_detection_guesser(models_df)
    gender_small_dict_series = gender_detection_small_dict(models_df) 
    gender_large_dict_series = gender_detection_large_dict(models_df)
    gender_us_census_series = gender_detection_us_census(models_df)
    
    # Create consensus
    gender_consensus_series = create_consensus_gender(
        gender_guesser_series, 
        gender_small_dict_series, 
        gender_large_dict_series,
        gender_us_census_series
    )
    
    # Combine all results
    result_df = models_df.copy()
    result_df['gender_guesser'] = gender_guesser_series
    result_df['gender_small_dict'] = gender_small_dict_series
    result_df['gender_large_dict'] = gender_large_dict_series
    result_df['gender_us_census'] = gender_us_census_series
    result_df['gender_consensus'] = gender_consensus_series
    
    # Add first name for reference
    result_df['first_name'] = result_df['name'].apply(extract_first_name)
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"Total models: {len(result_df)}")
    print(f"Gender distribution (consensus):")
    gender_counts = result_df['gender_consensus'].value_counts()
    for gender, count in gender_counts.items():
        print(f"  - {gender}: {count} ({count/len(result_df)*100:.1f}%)")
    
    return result_df

def save_gender_results(df, filename="models_gender.csv"):
    """Save gender detection results to CSV file."""
    output_path = DATA_DIR / filename
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to: {output_path}")
    return output_path

def print_sample_results(df, n=10):
    """Print sample results for verification."""
    print(f"\n🔍 Sample results (first {n} rows):")
    print("-" * 100)
    sample_cols = ['model_id', 'name', 'gender_guesser', 'gender_small_dict', 'gender_large_dict', 'gender_us_census', 'gender_consensus']
    print(df[sample_cols].head(n).to_string(index=False))

def create_confusion_matrices(df):
    """Create confusion matrices comparing the four gender detection methods."""
    print("\n" + "="*80)
    print("GENDER DETECTION METHOD COMPARISON - CONFUSION MATRICES")
    print("="*80)
    
    methods = ['gender_guesser', 'gender_small_dict', 'gender_large_dict', 'gender_us_census']
    method_names = ['Gender-Guesser', 'Small Dict', 'Large Dict', 'US Census']
    
    # Create comparison matrices for each pair of methods
    comparisons = [
        ('gender_guesser', 'gender_small_dict', 'Gender-Guesser vs Small Dict'),
        ('gender_guesser', 'gender_large_dict', 'Gender-Guesser vs Large Dict'),
        ('gender_guesser', 'gender_us_census', 'Gender-Guesser vs US Census'),
        ('gender_small_dict', 'gender_large_dict', 'Small Dict vs Large Dict'),
        ('gender_small_dict', 'gender_us_census', 'Small Dict vs US Census'),
        ('gender_large_dict', 'gender_us_census', 'Large Dict vs US Census')
    ]
    
    for method1, method2, title in comparisons:
        print(f"\n📊 {title}:")
        print("-" * 60)
        
        # Create crosstab
        confusion = pd.crosstab(df[method1], df[method2], margins=True, margins_name="Total")
        print(confusion)
        
        # Calculate agreement statistics (excluding 'unknown' vs 'unknown')
        both_known = df[(df[method1] != 'unknown') & (df[method2] != 'unknown')]
        if len(both_known) > 0:
            agreement = (both_known[method1] == both_known[method2]).sum()
            total_both_known = len(both_known)
            agreement_rate = agreement / total_both_known * 100
            print(f"\nAgreement when both methods have predictions: {agreement:,}/{total_both_known:,} ({agreement_rate:.1f}%)")
        
        # Calculate disagreement details
        disagreements = both_known[both_known[method1] != both_known[method2]]
        if len(disagreements) > 0:
            print(f"Disagreements: {len(disagreements):,} cases")
            disagree_breakdown = disagreements.groupby([method1, method2]).size()
            for (pred1, pred2), count in disagree_breakdown.items():
                print(f"  - {method1.replace('gender_', '').replace('_', ' ').title()}: {pred1}, {method2.replace('gender_', '').replace('_', ' ').title()}: {pred2} → {count:,} cases")
    
    # Overall method coverage comparison
    print(f"\n📈 METHOD COVERAGE COMPARISON:")
    print("-" * 60)
    coverage_stats = []
    for method, name in zip(methods, method_names):
        total = len(df)
        known = (df[method] != 'unknown').sum()
        female = (df[method] == 'female').sum()
        male = (df[method] == 'male').sum()
        coverage_rate = known / total * 100
        
        coverage_stats.append({
            'Method': name,
            'Total_Predictions': f"{known:,}",
            'Coverage_%': f"{coverage_rate:.1f}%",
            'Female': f"{female:,}",
            'Male': f"{male:,}",
            'Unknown': f"{total-known:,}"
        })
    
    coverage_df = pd.DataFrame(coverage_stats)
    print(coverage_df.to_string(index=False))
    
    # Method reliability comparison (against consensus)
    print(f"\n🎯 METHOD ACCURACY vs CONSENSUS:")
    print("-" * 60)
    accuracy_stats = []
    for method, name in zip(methods, method_names):
        # Only compare where both method and consensus are not unknown
        comparable = df[(df[method] != 'unknown') & (df['gender_consensus'] != 'unknown')]
        if len(comparable) > 0:
            correct = (comparable[method] == comparable['gender_consensus']).sum()
            accuracy = correct / len(comparable) * 100
            accuracy_stats.append({
                'Method': name,
                'Comparable_Cases': f"{len(comparable):,}",
                'Correct_Predictions': f"{correct:,}",
                'Accuracy_%': f"{accuracy:.1f}%"
            })
    
    if accuracy_stats:
        accuracy_df = pd.DataFrame(accuracy_stats)
        print(accuracy_df.to_string(index=False))

if __name__ == "__main__":
    # Create comprehensive gender dataframe
    gender_df = create_gender_dataframe()
    
    # Print sample results
    print_sample_results(gender_df)
    
    # Print confusion matrices and method comparison
    create_confusion_matrices(gender_df)
    
    # Save results
    output_file = save_gender_results(gender_df)
    
    print("\n" + "="*60)
    print("GENDER DETECTION COMPLETE!")
    print("="*60)