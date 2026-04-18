import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import os

# --- Professional Visual Configuration ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.frameon': True
})

# Color definitions for algorithms
COLORS = {
    'E-TIMBR': '#003049', 
    'TIMBR': '#D62828', 
    'TAMBOOR': '#F77F00', 
    'RFE (Data-driven Reference)': '#4CAF50'
}

ALGO_MAP = {
    'ModTIMBR_Rank': 'E-TIMBR', 
    'TIMBR_Rank': 'TIMBR', 
    'TAMBOOR_Rank': 'TAMBOOR'
}

OUTPUT_DIR = "PCA_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_publication_figure(files):
    all_data = []
    indiv_pca_results = []
    
    # 1. Data Preparation and Individual PCA Calculations
    for f in files:
        if not os.path.exists(f): 
            print(f"Warning: {f} not found, skipping...")
            continue
            
        disease = f.split('_')[0].upper()
        df = pd.read_csv(f).dropna()
        
        # Disease-specific PCA for each dataset
        rank_cols = ['RFE_Rank'] + list(ALGO_MAP.keys())
        rank_matrix = df[rank_cols].T
        pca_ind = PCA(n_components=2)
        comps_ind = pca_ind.fit_transform(rank_matrix)
        var_ind = pca_ind.explained_variance_ratio_ * 100
        
        temp_ind_df = pd.DataFrame(comps_ind, columns=['PC1', 'PC2'])
        temp_ind_df['Algorithm'] = ['RFE (Data-driven Reference)'] + list(ALGO_MAP.values())
        temp_ind_df['Disease'] = disease
        temp_ind_df['Var_Expl'] = [var_ind] * len(temp_ind_df)
        indiv_pca_results.append(temp_ind_df)
        
        df['Disease'] = disease
        all_data.append(df)

    if not all_data:
        print("Error: No data to process.")
        return

    master_df = pd.concat(all_data)
    
    # Calculate Mean Variance (for Panel A)
    global_variances = [res['Var_Expl'].iloc[0] for res in indiv_pca_results]
    avg_pc1 = np.mean([v[0] for v in global_variances])
    avg_pc2 = np.mean([v[1] for v in global_variances])
    
    # 2. Plotting Layout (1 Top Panel, 3 Bottom Panels)
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, len(files), hspace=0.3, wspace=0.25)
    
    # --- PANEL A: CONSOLIDATED PCA (TOP) ---
    ax_top = fig.add_subplot(gs[0, :])
    sns.scatterplot(
        data=pd.concat(indiv_pca_results), x='PC1', y='PC2', hue='Algorithm', 
        style='Disease', s=200, palette=COLORS, ax=ax_top, edgecolor='w', zorder=3
    )
    ax_top.set_xlabel(f"Principal Component 1 (Avg. Explained Var: {avg_pc1:.1f}%)", fontweight='bold')
    ax_top.set_ylabel(f"Principal Component 2 (Avg. Explained Var: {avg_pc2:.1f}%)", fontweight='bold')
    ax_top.text(-0.06, 1.05, "A", transform=ax_top.transAxes, fontsize=24, fontweight='bold')
    ax_top.set_title("Global Ranking Strategy Overview", pad=20, fontsize=14)
    ax_top.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # --- PANELS B, C, D: DISEASE SPECIFIC (BOTTOM) ---
    panel_labels = ["B", "C", "D"]
    for i, res in enumerate(indiv_pca_results):
        ax = fig.add_subplot(gs[1, i])
        var = res['Var_Expl'].iloc[0]
        disease_name = res['Disease'].iloc[0]
        
        sns.scatterplot(
            data=res, x='PC1', y='PC2', hue='Algorithm', 
            palette=COLORS, s=150, ax=ax, legend=False, edgecolor='k', zorder=3
        )
        
        # --- LABELLING (Spaced-out to avoid collisions) ---
        for _, row in res.iterrows():
            ax.annotate(
                row['Algorithm'], 
                xy=(row['PC1'], row['PC2']),
                xytext=(8, 5), # (8 units right, 5 units up offset)
                textcoords='offset points',
                fontsize=9,
                fontweight='semibold',
                alpha=0.85
            )
            
        ax.set_xlabel(f"PC1 ({var[0]:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]:.1f}%)")
        
        # Disease name inside the panel for a cleaner aesthetic
        ax.text(0.05, 0.95, f"{disease_name}", transform=ax.transAxes, 
                fontsize=11, fontweight='bold', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        
        # Panel Identifier Labels (B, C, D)
        ax.text(-0.15, 1.08, panel_labels[i], transform=ax.transAxes, fontsize=20, fontweight='bold')

    # 3. Export Procedure
    output_path = os.path.join(OUTPUT_DIR, "Master_PCA_Benchmark_Figure.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Figure successfully saved as '{output_path}'.")

# Execution
report_files = ["Colon_Comparison_Report.csv", "Breast_Comparison_Report.csv", "Alzheimer_Comparison_Report.csv"]
generate_publication_figure(report_files)