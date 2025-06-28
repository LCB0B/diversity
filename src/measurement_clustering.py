"""
Alternative clustering analysis based on measurement patterns.

This script performs cluster analysis on model measurements to identify
distinct body type patterns and their evolution over time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import seaborn as sns

warnings.filterwarnings('ignore')

# Import centralized path management
from src.paths import (
    DATA_DIR,
    FIGURES_DIR,
    ensure_directories_exist
)

# Use master dataset instead of models_measure.csv
MASTER_DATASET = DATA_DIR / "master_dataset.csv"

def load_and_prepare_measurement_data():
    """
    Load model measurement data and prepare for clustering analysis.
    
    Returns:
        pd.DataFrame: Cleaned measurement dataframe
    """
    print("Loading model measurement data...")
    
    try:
        df = pd.read_csv(MASTER_DATASET)
        print(f"Loaded {len(df):,} measurement records")
    except FileNotFoundError:
        print(f"Error: {MASTER_DATASET} not found")
        return pd.DataFrame()
    
    # Focus on core measurements for clustering
    measurement_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu']
    
    # Convert height-metric to numeric (it might be stored as string)
    df['height-metric'] = pd.to_numeric(df['height-metric'], errors='coerce')
    
    # Filter for records with complete measurements
    complete_measurements = df.dropna(subset=measurement_cols)
    print(f"Records with complete measurements: {len(complete_measurements):,}")
    
    # Remove obvious outliers (extreme values that are likely data errors)
    # Height: reasonable range 150-200cm
    # Bust: reasonable range 70-120cm  
    # Waist: reasonable range 50-100cm
    # Hips: reasonable range 70-130cm
    
    filters = (
        (complete_measurements['height-metric'] >= 150) & 
        (complete_measurements['height-metric'] <= 200) &
        (complete_measurements['bust-eu'] >= 70) & 
        (complete_measurements['bust-eu'] <= 120) &
        (complete_measurements['waist-eu'] >= 50) & 
        (complete_measurements['waist-eu'] <= 100) &
        (complete_measurements['hips-eu'] >= 70) & 
        (complete_measurements['hips-eu'] <= 130)
    )
    
    clean_data = complete_measurements[filters].copy()
    print(f"Records after outlier removal: {len(clean_data):,}")
    
    # Calculate derived measurements for additional clustering features
    clean_data['bust_waist_ratio'] = clean_data['bust-eu'] / clean_data['waist-eu']
    clean_data['waist_hip_ratio'] = clean_data['waist-eu'] / clean_data['hips-eu']
    clean_data['bust_hip_ratio'] = clean_data['bust-eu'] / clean_data['hips-eu']
    
    return clean_data

def determine_optimal_clusters(measurement_data, max_k=10):
    """
    Determine optimal number of clusters using elbow method and silhouette analysis.
    
    Args:
        measurement_data (pd.DataFrame): Measurement data for clustering
        max_k (int): Maximum number of clusters to test
        
    Returns:
        tuple: (optimal_k, elbow_scores, silhouette_scores)
    """
    print("Determining optimal number of clusters...")
    
    # Prepare features for clustering
    feature_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu', 
                   'bust_waist_ratio', 'waist_hip_ratio', 'bust_hip_ratio']
    
    X = measurement_data[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Test different numbers of clusters
    k_range = range(2, max_k + 1)
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"  Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    # Find elbow point (simple heuristic)
    # Calculate the rate of change in inertia
    deltas = np.diff(inertias)
    second_deltas = np.diff(deltas)
    
    # Find the point where the rate of change levels off
    elbow_k = k_range[np.argmax(second_deltas) + 2]  # +2 because of double diff
    
    # Find best silhouette score
    best_silhouette_k = k_range[np.argmax(silhouette_scores)]
    
    print(f"Elbow method suggests k={elbow_k}")
    print(f"Best silhouette score at k={best_silhouette_k} (score: {max(silhouette_scores):.3f})")
    
    # Use silhouette score as primary criterion
    optimal_k = best_silhouette_k
    
    return optimal_k, inertias, silhouette_scores, scaler

def perform_measurement_clustering(measurement_data, n_clusters):
    """
    Perform K-means clustering on measurement data.
    
    Args:
        measurement_data (pd.DataFrame): Measurement data
        n_clusters (int): Number of clusters
        
    Returns:
        tuple: (clustered_data, kmeans_model, scaler)
    """
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    # Prepare features for clustering
    feature_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu', 
                   'bust_waist_ratio', 'waist_hip_ratio', 'bust_hip_ratio']
    
    X = measurement_data[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to data
    clustered_data = measurement_data.copy()
    clustered_data['cluster'] = cluster_labels
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    print(f"Average silhouette score: {silhouette_avg:.3f}")
    
    return clustered_data, kmeans, scaler

def analyze_cluster_characteristics(clustered_data):
    """
    Analyze and describe characteristics of each cluster.
    
    Args:
        clustered_data (pd.DataFrame): Data with cluster assignments
        
    Returns:
        pd.DataFrame: Cluster characteristics summary
    """
    print("Analyzing cluster characteristics...")
    
    measurement_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu',
                       'bust_waist_ratio', 'waist_hip_ratio', 'bust_hip_ratio']
    
    cluster_summary = []
    
    for cluster_id in sorted(clustered_data['cluster'].unique()):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster_id]
        
        summary = {
            'cluster': cluster_id,
            'count': len(cluster_data),
            'percentage': len(cluster_data) / len(clustered_data) * 100
        }
        
        # Calculate mean and std for each measurement
        for col in measurement_cols:
            summary[f'{col}_mean'] = cluster_data[col].mean()
            summary[f'{col}_std'] = cluster_data[col].std()
        
        cluster_summary.append(summary)
    
    summary_df = pd.DataFrame(cluster_summary)
    
    # Print readable summary
    print("\nCluster Characteristics Summary:")
    print("=" * 80)
    
    for _, row in summary_df.iterrows():
        cluster_id = int(row['cluster'])
        count = int(row['count'])
        percentage = row['percentage']
        
        print(f"\nCluster {cluster_id}: {count:,} models ({percentage:.1f}%)")
        print(f"  Height: {row['height-metric_mean']:.1f}±{row['height-metric_std']:.1f} cm")
        print(f"  Bust:   {row['bust-eu_mean']:.1f}±{row['bust-eu_std']:.1f} cm")
        print(f"  Waist:  {row['waist-eu_mean']:.1f}±{row['waist-eu_std']:.1f} cm")
        print(f"  Hips:   {row['hips-eu_mean']:.1f}±{row['hips-eu_std']:.1f} cm")
        print(f"  Bust/Waist: {row['bust_waist_ratio_mean']:.2f}±{row['bust_waist_ratio_std']:.2f}")
        print(f"  Waist/Hip:  {row['waist_hip_ratio_mean']:.2f}±{row['waist_hip_ratio_std']:.2f}")
    
    return summary_df

def plot_cluster_analysis(measurement_data, optimal_k, inertias, silhouette_scores):
    """
    Create visualizations for cluster analysis.
    
    Args:
        measurement_data (pd.DataFrame): Clustered measurement data
        optimal_k (int): Optimal number of clusters
        inertias (list): Elbow method inertias
        silhouette_scores (list): Silhouette scores
    """
    ensure_directories_exist()
    
    # 1. Elbow method and silhouette analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    k_range = range(2, len(inertias) + 2)
    
    # Elbow method plot
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia (WCSS)')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette score plot
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='green', linestyle='--', alpha=0.7, 
                label=f'Selected k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.set_title('Silhouette Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_optimization_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. PCA visualization of clusters
    feature_cols = ['height-metric', 'bust-eu', 'waist-eu', 'hips-eu', 
                   'bust_waist_ratio', 'waist_hip_ratio', 'bust_hip_ratio']
    
    X = measurement_data[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(12, 8))
    
    # Create color map for clusters
    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))
    
    for i in range(optimal_k):
        cluster_mask = measurement_data['cluster'] == i
        plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                   c=[colors[i]], label=f'Cluster {i}', alpha=0.6, s=30)
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('Measurement Clusters in PCA Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'measurement_clusters_pca.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cluster characteristics heatmap
    summary_df = analyze_cluster_characteristics(measurement_data)
    
    # Prepare data for heatmap (means only)
    mean_cols = [col for col in summary_df.columns if '_mean' in col]
    heatmap_data = summary_df[['cluster'] + mean_cols].set_index('cluster')
    
    # Rename columns for better display
    column_names = {
        'height-metric_mean': 'Height (cm)',
        'bust-eu_mean': 'Bust (cm)',
        'waist-eu_mean': 'Waist (cm)', 
        'hips-eu_mean': 'Hips (cm)',
        'bust_waist_ratio_mean': 'Bust/Waist Ratio',
        'waist_hip_ratio_mean': 'Waist/Hip Ratio',
        'bust_hip_ratio_mean': 'Bust/Hip Ratio'
    }
    heatmap_data = heatmap_data.rename(columns=column_names)
    
    plt.figure(figsize=(12, 8))
    
    # Normalize data for better color scaling
    heatmap_normalized = heatmap_data.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    
    sns.heatmap(heatmap_normalized.T, 
                annot=heatmap_data.T, 
                fmt='.1f', 
                cmap='RdYlBu_r',
                center=0,
                cbar_kws={'label': 'Standardized Value'})
    
    plt.title('Cluster Characteristics Heatmap')
    plt.xlabel('Cluster ID')
    plt.ylabel('Measurement Features')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cluster_characteristics_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cluster analysis plots saved to {FIGURES_DIR}")

def save_clustering_results(clustered_data, cluster_summary, optimal_k):
    """
    Save clustering results to CSV files.
    
    Args:
        clustered_data (pd.DataFrame): Data with cluster assignments
        cluster_summary (pd.DataFrame): Cluster characteristics summary
        optimal_k (int): Number of clusters used
    """
    ensure_directories_exist()
    
    # Save clustered data
    clustered_output = DATA_DIR / f"measurement_clusters_k{optimal_k}.csv"
    clustered_data.to_csv(clustered_output, index=False)
    print(f"Clustered data saved to: {clustered_output}")
    
    # Save cluster summary
    summary_output = DATA_DIR / f"cluster_summary_k{optimal_k}.csv"
    cluster_summary.to_csv(summary_output, index=False)
    print(f"Cluster summary saved to: {summary_output}")

def main():
    """
    Main function to run the complete measurement clustering analysis.
    """
    print("=" * 60)
    print("MEASUREMENT CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Load and prepare data
    measurement_data = load_and_prepare_measurement_data()
    
    if measurement_data.empty:
        print("No measurement data available for clustering")
        return
    
    # Determine optimal number of clusters
    optimal_k, inertias, silhouette_scores, scaler = determine_optimal_clusters(measurement_data)
    
    # Perform clustering
    clustered_data, kmeans_model, scaler = perform_measurement_clustering(measurement_data, optimal_k)
    
    # Analyze cluster characteristics
    cluster_summary = analyze_cluster_characteristics(clustered_data)
    
    # Create visualizations
    plot_cluster_analysis(clustered_data, optimal_k, inertias, silhouette_scores)
    
    # Save results
    save_clustering_results(clustered_data, cluster_summary, optimal_k)
    
    print("\n" + "=" * 60)
    print("MEASUREMENT CLUSTERING ANALYSIS COMPLETE!")
    print("=" * 60)
    
    return clustered_data, cluster_summary

if __name__ == "__main__":
    clustered_data, cluster_summary = main()