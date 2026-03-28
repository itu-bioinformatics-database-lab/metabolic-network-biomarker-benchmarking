import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

# Directory Setup
OUTPUT_DIR = "advanced_omics_research_v4"
SUBDIRS = ["plots/fusion_maps", "reports/metabolic_hubs"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def load_data_safely(path):
    """Robustly separates 'Factors' and returns numeric data."""
    df = pd.read_csv(path)
    label_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    y_raw = df[label_col]
    X = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in X.columns if "Unnamed" in c or "ID" in c or "Sample" in c]
    X = X.drop(columns=cols_to_drop)
    return X, y_raw

def run_fusion_pipeline():
    print("Initiating Multi-Omics Fusion & Hub Discovery...")
    
    for disease in DISEASES:
        print(f"\n>>> ANALYZING: {disease.upper()}")
        t_file, m_file = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        
        if not (os.path.exists(t_file) and os.path.exists(m_file)):
            continue

        # 1. Load and Align
        X_t, labels = load_data_safely(t_file)
        X_m, _ = load_data_safely(m_file)
        
        n = min(len(X_t), len(X_m))
        X_t, X_m, labels = X_t.iloc[:n], X_m.iloc[:n], labels.iloc[:n]

        # 2. Analysis A: Similarity Network Fusion (SNF)
        scaler = StandardScaler()
        dist_t = pairwise_distances(scaler.fit_transform(X_t), metric='euclidean')
        dist_m = pairwise_distances(scaler.fit_transform(X_m), metric='euclidean')
        
        # Fuse the layers
        fused_similarity = (dist_t + dist_m) / 2
        
        # Plotting the Fused Map (Fixed TypeError here)
        plot_fused_network(disease, fused_similarity, labels)

        # 3. Analysis B: Cross-Omics Hub Discovery
        print(f"      Calculating Metabolic Impact for top 500 genes...")
        top_v_genes = X_t.std().sort_values(ascending=False).head(500).index
        
        hub_results = []
        # Pre-convert metabolomics to numpy for speed
        X_m_np = X_m.to_numpy()
        
        for gene in top_v_genes:
            gene_vals = X_t[gene].to_numpy()
            # Vectorized Spearman-like approach or simple loop
            corrs = [spearmanr(gene_vals, X_m_np[:, j])[0] for j in range(X_m_np.shape[1])]
            hub_score = np.nanmean(np.abs(corrs))
            hub_results.append({"Gene": gene, "Metabolic_Impact_Score": hub_score})
            
        hub_df = pd.DataFrame(hub_results).sort_values(by="Metabolic_Impact_Score", ascending=False)
        hub_df.head(50).to_csv(os.path.join(OUTPUT_DIR, f"reports/metabolic_hubs/{disease}_top_regulators.csv"), index=False)

    print(f"\n[SUCCESS] Integrated Analysis Completed. Folder: {OUTPUT_DIR}")

def plot_fused_network(disease, matrix, labels):
    """Visualizes the fused similarity using Clustermap with fixed alignment."""
    plt.figure(figsize=(12, 10))
    
    # Create color palette for labels
    unique_labels = labels.unique()
    lut = dict(zip(unique_labels, sns.color_palette("Set1", len(unique_labels))))
    
    # CRITICAL FIX: Convert Series to numpy to avoid index mismatch with the matrix
    row_colors_np = labels.map(lut).to_numpy()
    
    # We plot the distance matrix (Dissimilarity)
    g = sns.clustermap(matrix, 
                       cmap="rocket_r", 
                       standard_scale=1,
                       row_colors=row_colors_np, 
                       col_colors=row_colors_np,
                       figsize=(12, 12), 
                       tree_kws=dict(linewidths=0.5))
    
    # Fix for title overlap
    g.fig.suptitle(f"{disease.upper()} - Similarity Network Fusion\n(Fused Patient Landscape)", fontsize=16, y=1.02)
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/fusion_maps/{disease}_fused_landscape.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_fusion_pipeline()