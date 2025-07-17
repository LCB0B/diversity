import sys
from pathlib import Path

# Add the project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# --- Imports ---
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import re
from matplotlib import gridspec


from src.paths import (
    DATA_DIR, FIGURES_DIR, ensure_directories_exist
)
from src.nationality_mappings import get_mappings
from src.analysis import (
    load_core_datasets,
    create_master_dataset,
    prepare_shows_and_measurements_data,
)

# --- Ensure necessary folders exist ---
ensure_directories_exist()

# --- Load country mappings ---

# --- Load core data and filter for females only ---
def load_female_model_data():
    core = load_core_datasets()
    df = create_master_dataset(core)
    nationality2country, country_to_global, country_to_region, country_to_super_region = get_mappings()

    # Map nationalities to geographic levels
    df["country"] = df["nationality"].map(nationality2country)
    df["region"] = df["country"].map(country_to_region)
    df["super_region"] = df["country"].map(country_to_super_region)
    df["global_region"] = df["country"].map(country_to_global)

    skincolor_data = pd.read_csv(DATA_DIR / "model_info_from_profilepic.csv")
    df["model_name"] = df["filename"].apply(lambda x: x.split(".")[0])
    skincolor_data["model_name"] = skincolor_data["image_file"].apply(lambda x: x.split(".")[0])

    df = df.merge(skincolor_data, on = "model_name")
    df = df.loc[df["face_detected"]]
    df = df.loc[df["predicted_race"] != "Unknown"]
    df["is_white"] = (df["predicted_race"]=="White") * 1

    return df[df["gender_consensus"] == "female"]

# --- Load geographic data (world shapefile, exclude Antarctica) ---
def load_world_geometry():
    shp_path = DATA_DIR / "110m_cultural/ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shp_path)
    return world[world['NAME'] != 'Antarctica']



# Define this once globally
EU_TO_US_DRESS = {
    30: 0, 32: 2, 34: 4, 36: 6, 38: 8, 40: 10, 42: 12,
    44: 14, 46: 16, 48: 18, 50: 20, 52: 22, 54: 24
}

def parse_eu_dress_to_us(val):
    """Convert EU dress size (or size range) to US equivalent."""
    if pd.isnull(val):
        return None
    try:
        cleaned = re.sub(r"[^\d\-]", "", str(val))
        if '-' in cleaned:
            parts = cleaned.split('-')
            nums = []
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    continue
            if len(nums) == 2:
                avg = round(np.mean(nums))
                return EU_TO_US_DRESS.get(avg)
        else:
            return EU_TO_US_DRESS.get(int(cleaned))
    except:
        return None

def preprocess_model_data(df):
    """
    Clean and prepare model data by:
    - Cleaning waist-us values (e.g., converting cm to inches)
    - Parsing dress-eu to US sizes
    - Dropping rows without valid sizes
    - Creating 'plus_sized' label
    """
    df = df.copy()
    df['dress-us_clean'] = df['dress-eu'].apply(parse_eu_dress_to_us)
    df = df.dropna(subset=['dress-us_clean'])
    df['plus_sized'] = df['dress-us_clean'] >= 12
    return df

def assign_year_bin(year):
    if 2011 <= year <= 2013:
        return "2011–2013"
    elif 2014 <= year <= 2016:
        return "2014–2016"
    elif 2017 <= year <= 2019:
        return "2017–2019"
    elif 2020 <= year <= 2024:
        return "2020–2024"
    else:
        None

def enrich_shows_with_model_data(core_data, model_data):
    """
    Load and enrich show-level data by mapping model-level attributes.
    Only includes years between 2011 and 2024 and models with known data.
    """
    shows = prepare_shows_and_measurements_data(core_data)

    # Filter shows to valid years and known models
    shows = shows[(shows["year"] >= 2011) & (shows["year"] < 2025)]
    shows_data['year_bin'] = shows_data['year'].apply(assign_year_bin)

    shows = shows[shows["model_id"].isin(model_data["model_id"].unique())]

    # Create mapping dicts from model_id to attributes
    for col in ["plus_sized", "is_white", "predicted_race",
                "country", "region", "super_region", "global_region"]:
        shows[col] = shows["model_id"].map(dict(zip(model_data["model_id"], model_data[col])))

    shows = shows.dropna(subset=["global_region", "super_region"])
    shows["plus_sized"] = shows["plus_sized"].astype(int)

    return shows

def compute_odds_ratio_by_year(data):
    """
    Estimate year-varying odds ratio of being plus-sized in Global South vs Global North.
    Returns: DataFrame with odds ratio per year.
    """
    model = smf.logit(
        formula="plus_sized ~ C(global_region, Treatment(reference='Global North')) * year",
        data=data
    ).fit(maxiter=100, disp=0)  # suppress fit output

    region_base_coef = model.params.get("C(global_region, Treatment(reference='Global North'))[T.Global South]", 0)
    interaction_coef = model.params.get("C(global_region, Treatment(reference='Global North'))[T.Global South]:year", 0)

    years = np.arange(data['year'].min(), data['year'].max() + 1)
    return pd.DataFrame({
        "year": years,
        "odds_ratio": [np.exp(region_base_coef + interaction_coef * y) for y in years]
    })


def compute_yearly_summary(data):
    """
    Compute yearly summaries of total models and plus-sized shares,
    including % of plus-sized coming from the Global South.
    """
    summary = data.groupby('year').agg(
        total_models=('model_id', 'count'),
        plus_sized=('plus_sized', 'sum'),
        global_south_plus_sized=('plus_sized', lambda x: (
            (data.loc[x.index, 'global_region'] == 'Global South') & (x == 1)
        ).sum())
    ).reset_index()

    summary['pct_plus_sized'] = 100 * summary['plus_sized'] / summary['total_models']
    summary['pct_plus_sized_from_south'] = 100 * summary['global_south_plus_sized'] / summary['plus_sized'].replace(0, pd.NA)
    return summary


def plot_odds_ratio_and_share(plot_df, save_path=None):
    """
    Dual-axis line plot:
      - Odds ratio (Global South vs North)
      - % of plus-sized models (overall and from Global South)
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Primary y-axis: Odds ratio
    ax1.plot(plot_df['year'], plot_df['odds_ratio'], color='#d62728', marker='o', label='Odds Ratio (GS vs GN)')
    ax1.axhline(1.0, color='gray', linestyle='--')
    ax1.set_ylabel("Odds Ratio", color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')

    # Secondary y-axis: % shares
    ax2 = ax1.twinx()
    ax2.plot(plot_df['year'], plot_df['pct_plus_sized'], color='#1f77b4', marker='s', label='% Plus-Sized Overall')
    ax2.plot(plot_df['year'], plot_df['pct_plus_sized_from_south'], color='#9467bd', linestyle='--', marker='x', label='% from Global South')
    ax2.set_ylabel("Percentage (%)")
    ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title("Odds Ratio and Plus-Sized Representation Over Time")
    plt.xlabel("Year")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

from matplotlib import gridspec

def plot_odds_ratio_maps_by_year(world, or_by_year, save_path=None):
    """
    Generate maps showing odds ratio of plus-sized representation in the Global South for each selected year.
    """
    nationality2country, country_to_global, country_to_region, country_to_super_region = get_mappings()
    global_north_countries = [k for k, v in country_to_global.items() if v == "Global North"]
    global_north_countries += ["Greenland"]
    selected_years = [2011, 2015, 2020, 2024]

    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=or_by_year['odds_ratio'].min(), vmax=or_by_year['odds_ratio'].max())

    fig = plt.figure(figsize=(26, 6))
    gs = gridspec.GridSpec(1, len(selected_years) + 1, width_ratios=[1]*len(selected_years) + [0.05], wspace=0)

    for i, year in enumerate(selected_years):
        ax = fig.add_subplot(gs[0, i])
        or_value = or_by_year.loc[or_by_year['year'] == year, 'odds_ratio'].values[0]
        or_str = f"{or_value:.2f}"

        world_copy = world.copy()
        world_copy['global_region'] = world_copy['NAME'].apply(
            lambda x: "Global North" if x in global_north_countries else "Global South"
        )
        world_copy['odds_ratio'] = world_copy['global_region'].apply(
            lambda region: or_value if region == "Global South" else np.nan
        )

        world_copy.plot(
            column='odds_ratio',
            cmap=cmap,
            norm=norm,
            linewidth=0.8,
            edgecolor='black',
            legend=False,
            ax=ax,
            missing_kwds={"color": "#1f77b4", "label": "Global North"}
        )

        ax.text(
            -160, -55,
            f"Year: {year}\nGlobal South OR = {or_str}",
            fontsize=11,
            bbox=dict(facecolor='white', edgecolor='black'),
            verticalalignment='bottom'
        )
        ax.set_title(f"{year}", fontsize=14)
        ax.set_axis_off()

    # Colorbar
    cax = fig.add_subplot(gs[0, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Odds Ratio (Global South)", fontsize=12)

    fig.suptitle("Odds of Plus-Size Model Representation in the Global South (Selected Years)", fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_super_region_odds_ratios(data, ref_category="North America"):
    """
    Logistic regression predicting plus-sized status using super_region.
    Returns:
        - or_map: odds ratios by region
        - enhanced_or_map: odds ratios + SEs + significance stars
    """

    model = smf.logit(
        formula=f"plus_sized ~ C(super_region, Treatment(reference='{ref_category}'))",
        data=data
    ).fit(disp=0)

    odds_ratios = np.exp(model.params)
    conf_ints = np.exp(model.conf_int())
    pvals = model.pvalues

    or_map = {ref_category: 1.0}
    enhanced_or_map = {
        ref_category: {"or": 1.0, "se": 0.0, "stars": ""}
    }

    for idx in model.params.index:
        if idx.startswith("C(super_region"):
            region = idx.split("T.")[-1].rstrip("]")
            or_val = odds_ratios[idx]
            se_val = or_val * model.bse[idx]  # approximate SE of odds ratio

            # Significance stars
            pval = pvals[idx]
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"
            else:
                stars = ""

            or_map[region] = or_val
            enhanced_or_map[region] = {"or": or_val, "se": se_val, "stars": stars}

    return or_map, enhanced_or_map

def plot_super_region_odds_map(world, or_map, enhanced_or_map, save_path=None):
    """
    Plot world map colored by super_region odds ratios.
    Annotate each region with odds ratio, SE, and significance stars.
    """
    nationality2country, country_to_global, country_to_region, country_to_super_region = get_mappings()
    region_annotations = {
        "North America": (-100, 45),
        "South and Latin America": (-60, -15),
        "Europe": (10, 50),
        "Africa": (20, 0),
        "Asia": (90, 30),
        "Oceania": (150, -25)
    }
    world = world.copy()
    world['super_region'] = world['NAME'].map(country_to_super_region)
    world['odds_ratio'] = world['super_region'].map(or_map)

    fig, ax = plt.subplots(figsize=(14, 8))

    world.plot(
        column='odds_ratio',
        cmap='coolwarm',
        linewidth=0.8,
        edgecolor='black',
        legend=True,
        ax=ax,
        missing_kwds={"color": "lightgrey", "label": "Missing data"}
    )

    for region, (x, y) in region_annotations.items():
        if region in enhanced_or_map:
            val = enhanced_or_map[region]
            ax.text(
                x, y,
                f"{region}\nOR = {val['or']:.2f} (SE = {val['se']:.2f}) {val['stars']}",
                fontsize=11,
                ha='center',
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, boxstyle='round,pad=0.3')
            )

    ax.set_title("Odds Ratio of Being Plus-Sized by Super Region\n(Compared to North America)", fontsize=16)
    ax.set_axis_off()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

def compute_odds_ratio_by_race_year(data):
    """
    Estimate year-varying odds ratio of being plus-sized for non-white vs white.
    """
    model = smf.logit(
        formula="plus_sized ~ C(is_white, Treatment(reference=1)) * year",
        data=data
    ).fit(maxiter=100, disp=0)

    base_coef = model.params.get("C(is_white, Treatment(reference=1))[T.0]", 0)
    interaction_coef = model.params.get("C(is_white, Treatment(reference=1))[T.0]:year", 0)

    years = np.arange(data['year'].min(), data['year'].max() + 1)
    return pd.DataFrame({
        "year": years,
        "odds_ratio": [np.exp(base_coef + interaction_coef * y) for y in years]
    })



def compute_yearly_race_summary(data):
    """
    Compute yearly summaries including:
    - % plus-sized overall
    - % of plus-sized who are non-white
    - % of non-white who are not plus-sized
    """
    summary = data.groupby('year').agg(
        total_models=('model_id', 'count'),
        plus_sized=('plus_sized', 'sum'),
        nonwhite_plus_sized=('plus_sized', lambda x: (
            (data.loc[x.index, 'is_white'] == 0) & (x == 1)
        ).sum()),
        nonwhite_not_plus_sized=('plus_sized', lambda x: (
            (data.loc[x.index, 'is_white'] == 0) & (x == 0)
        ).sum()),
        total_nonwhite=('is_white', lambda x: (x == 0).sum())
    ).reset_index()

    summary['pct_plus_sized'] = 100 * summary['plus_sized'] / summary['total_models']
    summary['pct_plus_sized_nonwhite'] = 100 * summary['nonwhite_plus_sized'] / summary['plus_sized'].replace(0, pd.NA)
    summary['pct_nonwhite_not_plus'] = 100 * summary['nonwhite_not_plus_sized'] / summary['total_nonwhite'].replace(0, pd.NA)

    return summary


def plot_odds_ratio_and_share_by_race(plot_df, save_path=None):
    """
    Plot odds ratio and racial shares per year.
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # OR line
    ax1.plot(plot_df['year'], plot_df['odds_ratio'], color='#9467bd', marker='o', label='Odds Ratio (Non-white vs White)')
    ax1.axhline(1.0, color='gray', linestyle='--')
    ax1.set_ylabel("Odds Ratio", color='#9467bd')
    ax1.tick_params(axis='y', labelcolor='#9467bd')

    # Share lines
    ax2 = ax1.twinx()
    ax2.plot(plot_df['year'], plot_df['pct_plus_sized'], color='#1f77b4', marker='s', label='% Plus-Sized Overall')
    ax2.plot(plot_df['year'], plot_df['pct_plus_sized_nonwhite'], color='#ff7f0e', linestyle='--', marker='x', label='% Plus-Sized Who Are Non-White')
    #ax2.plot(plot_df['year'], plot_df['pct_nonwhite_not_plus'], color='#d62728', linestyle=':', marker='^', label='% Non-White Who Are Not Plus-Sized')

    ax2.set_ylabel("Percentage (%)")
    ax2.tick_params(axis='y', labelcolor='black')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.title("Odds Ratio and Racial Representation Over Time")
    plt.xlabel("Year")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()




def plot_racial_group_shares_over_time(data, save_path=None):
    """
    Plot % of plus-sized models by year and predicted_race.
    """
    yearly = data.groupby(['year', 'predicted_race']).agg(
        total=('model_id', 'count'),
        plus_sized=('plus_sized', 'sum')
    ).reset_index()

    yearly['pct_plus_sized'] = 100 * yearly['plus_sized'] / yearly['total']
    pivoted = yearly.pivot(index='year', columns='predicted_race', values='pct_plus_sized').fillna(0)

    plt.figure(figsize=(14, 7))
    for col in pivoted.columns:
        plt.plot(pivoted.index, pivoted[col], marker='o', label=col)

    plt.title("% Plus-Sized Models by Predicted Race Over Time")
    plt.xlabel("Year")
    plt.ylabel("Percentage (%)")
    plt.legend(title="Predicted Race")
    plt.grid(True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def compute_race_odds_ratios(data, ref_category="White"):
    """
    Logistic regression predicting plus-sized status using predicted_race.
    Returns:
        - or_map: odds ratios by race
        - enhanced_or_map: dict with ORs, SEs, and significance stars
    """
    model = smf.logit(
        formula=f"plus_sized ~ C(predicted_race, Treatment(reference='{ref_category}'))",
        data=data
    ).fit(disp=0)

    odds_ratios = np.exp(model.params)
    conf_ints = np.exp(model.conf_int())
    pvals = model.pvalues

    or_map = {ref_category: 1.0}
    enhanced_or_map = {
        ref_category: {"or": 1.0, "se": 0.0, "stars": ""}
    }

    for idx in model.params.index:
        if idx.startswith("C(predicted_race"):
            race = idx.split("T.")[-1].rstrip("]")
            or_val = odds_ratios[idx]
            se_val = or_val * model.bse[idx]  # approximate SE

            # Significance stars
            pval = pvals[idx]
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"
            else:
                stars = ""

            or_map[race] = or_val
            enhanced_or_map[race] = {"or": or_val, "se": se_val, "stars": stars}

    return or_map, enhanced_or_map


def plot_race_odds_barplot(enhanced_or_map, ref_category="White", save_path=None):
    """
    Centered bar plot of odds ratios for predicted_race (ref = White).
    Bars grow up (OR > 1) or down (OR < 1), centered at OR = 1.
    """
    import matplotlib.pyplot as plt

    races = list(enhanced_or_map.keys())
    or_vals = [enhanced_or_map[r]["or"] for r in races]
    se_vals = [enhanced_or_map[r]["se"] for r in races]
    stars = [enhanced_or_map[r]["stars"] for r in races]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(races))
    baseline = 1.0
    bar_colors = ['#1f77b4' if r != ref_category else 'gray' for r in races]

    # Plot bars centered on y=1
    for i, (race, or_val, se, star) in enumerate(zip(races, or_vals, se_vals, stars)):
        bar_height = or_val - baseline
        bar_bottom = baseline if bar_height >= 0 else or_val
        ax.bar(x[i], abs(bar_height), bottom=bar_bottom, color=bar_colors[i], alpha=0.8, capsize=4)

        # Add error bar
        ax.errorbar(x[i], or_val, yerr=se, fmt='none', ecolor='black', capsize=5)

        # Add significance star
        if star:
            y_star = or_val + (0.05 if or_val >= baseline else -0.05)
            va = 'bottom' if or_val >= baseline else 'top'
            ax.text(x[i], y_star, star, ha='center', va=va, fontsize=12)

    # Formatting
    ax.axhline(baseline, color='black', linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(races, rotation=45)
    ax.set_ylabel("Odds Ratio (ref = White)")
    ax.set_title("Odds of Being Plus-Sized by Predicted Race")
    ax.set_ylim(bottom=min(0.5, min(or_vals) - 0.2), top=max(1.8, max(or_vals) + 0.7))
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()

##################main


def main():
    # --- Setup ---
    ensure_directories_exist()
    print("Loading data...")
    core_data = load_core_datasets()
    nationality2country, country_to_global, country_to_region, country_to_super_region = get_mappings()
    world = load_world_geometry()

    # --- Preprocess models ---
    print("Preprocessing model data...")
    model_data = load_female_model_data()
    model_data = preprocess_model_data(model_data)

    # --- Enrich shows with model attributes ---
    shows_data = enrich_shows_with_model_data(core_data, model_data)

    # --- Global North vs South Analysis ---
    print("Running Global North vs South analysis...")
    or_by_year = compute_odds_ratio_by_year(shows_data)
    year_summary = compute_yearly_summary(shows_data)
    plot_df = pd.merge(or_by_year, year_summary, on="year")
    plot_odds_ratio_and_share(plot_df, save_path=FIGURES_DIR / "odds_ratio_over_time.png")

    # --- Global Maps Over Time ---
    plot_odds_ratio_maps_by_year(
        world=world,
        or_by_year=or_by_year,
        save_path=FIGURES_DIR / "odds_ratio_maps_by_year.png"
    )

    # --- Super Region Analysis ---
    print("Running Regional analysis...")
    or_map, enhanced_or_map = compute_super_region_odds_ratios(shows_data)
    plot_super_region_odds_map(
        world=world,
        or_map=or_map,
        enhanced_or_map=enhanced_or_map,
        save_path=FIGURES_DIR / "super_region_odds_map.png"
    )

    # --- Race-Based Analysis ---
    print("Running Race-Based analysis...")
    race_or_by_year = compute_odds_ratio_by_race_year(shows_data)
    race_summary = compute_yearly_race_summary(shows_data)
    race_plot_df = pd.merge(race_or_by_year, race_summary, on="year")
    plot_odds_ratio_and_share_by_race(race_plot_df, save_path=FIGURES_DIR / "odds_ratio_race_over_time.png")

    # --- Race Group Trends ---
    print("Plotting racial group shares over time...")
    plot_racial_group_shares_over_time(shows_data, save_path=FIGURES_DIR / "racial_group_trends.png")
    
    print("Running Race Odds Ratio Model (no time)...")
    race_or_map, race_enhanced_or_map = compute_race_odds_ratios(shows_data)
    plot_race_odds_barplot(race_enhanced_or_map, save_path=FIGURES_DIR / "race_odds_ratio.png")

if __name__ == "__main__":
    main()


