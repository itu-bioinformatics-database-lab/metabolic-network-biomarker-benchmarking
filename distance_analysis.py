import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.spatial.distance import canberra
import os

# --- High-End Visual Setup ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.dpi': 300, 'font.family': 'sans-serif'})

# Algorithm names updated to CAPS LOCK and E-TIMBR for consistency
COLORS = {
    'E-TIMBR': '#003049', 
    'TIMBR': '#D62828', 
    'TAMBOOR': '#F77F00', 
    'RFE (TRUTH)': '#4CAF50'
}

ALGO_MAP = {
    'ModTIMBR_Rank': 'E-TIMBR', 
    'TIMBR_Rank': 'TIMBR', 
    'TAMBOOR_Rank': 'TAMBOOR'
}

OUTPUT_DIR = "Distance_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def analyze_advanced_benchmarks(files):
    all_data = []
    canberra_results = []
    
    for f in files:
        if not os.path.exists(f): 
            print(f"Warning: {f} not found, skipping...")
            continue
            
        disease = f.split('_')[0]
        df = pd.read_csv(f).dropna()
        
        # 1. Canberra Distance Calculation
        for col, name in ALGO_MAP.items():
            dist = canberra(df['RFE_Rank'], df[col])
            canberra_results.append({
                'Disease': disease, 
                'Algorithm': name, 
                'Canberra_Distance': dist
            })
        
        df['Disease'] = disease
        all_data.append(df)

    if not all_data:
        print("Error: No data to analyze.")
        return

    master_df = pd.concat(all_data)
    canberra_df = pd.DataFrame(canberra_results)

    # --- Plot 1: Canberra Distance ---
    # Used to measure ranking fidelity against the reference
    plt.figure(figsize=(10, 6))
    sns.barplot(data=canberra_df, x='Disease', y='Canberra_Distance', hue='Algorithm', palette=COLORS)
    plt.ylabel("Rank Fidelity (Canberra Distance Metric)")
    plt.xlabel("") 
    plt.savefig(os.path.join(OUTPUT_DIR, "canberra_distance_benchmark.png"), bbox_inches='tight')

    # --- Plot 2: Ranking Space PCA ---
    # Visualizing ranking philosophy proximity in 2D space
    pca_results = []
    for disease in master_df['Disease'].unique():
        sub = master_df[master_df['Disease'] == disease]
        rank_matrix = sub[['RFE_Rank'] + list(ALGO_MAP.keys())].T
        pca = PCA(n_components=2)
        components = pca.fit_transform(rank_matrix)
        
        temp_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
        temp_df['Algorithm'] = ['RFE (TRUTH)'] + list(ALGO_MAP.values())
        temp_df['Disease'] = disease
        pca_results.append(temp_df)

    pca_df = pd.concat(pca_results)
    
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Algorithm', style='Disease', s=200, palette=COLORS)
    plt.xlabel("Principal Component 1 (Ranking Philosophy)")
    plt.ylabel("Principal Component 2 (Ranking Philosophy)")
    plt.savefig(os.path.join(OUTPUT_DIR, "algorithm_pca_proximity.png"), bbox_inches='tight')

    # --- Plot 3: Unique & Shared Hits (Discovery Rate) ---
    top_n = 30
    consensus_list = []
    for disease in master_df['Disease'].unique():
        sub = master_df[master_df['Disease'] == disease]
        rfe_top = set(sub.nsmallest(top_n, 'RFE_Rank')['Metabolite'])
        
        for col, name in ALGO_MAP.items():
            algo_top = set(sub.nsmallest(top_n, col)['Metabolite'])
            hits = len(rfe_top.intersection(algo_top))
            consensus_list.append({
                'Disease': disease, 
                'Algorithm': name, 
                'Top_30_Hits': hits
            })

    consensus_df = pd.DataFrame(consensus_list)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=consensus_df, x='Disease', y='Top_30_Hits', hue='Algorithm', marker='o', palette=COLORS, linewidth=3)
    plt.ylabel(f"Metabolite Discovery Rate (Top {top_n} Hits)")
    plt.xlabel("")
    plt.savefig(os.path.join(OUTPUT_DIR, "top_n_discovery_rate.png"), bbox_inches='tight')

    # Export metrics to CSV for further review
    canberra_df.to_csv(os.path.join(OUTPUT_DIR, "canberra_metrics.csv"), index=False)
    pca_df.to_csv(os.path.join(OUTPUT_DIR, "pca_ranking_coordinates.csv"), index=False)

# Execution
report_files = ["Colon_Comparison_Report.csv", "Breast_Comparison_Report.csv", "Alzheimer_Comparison_Report.csv"]
analyze_advanced_benchmarks(report_files)
print(f"Scientific analysis suite completed. Results saved in: {OUTPUT_DIR}")