import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "network_rewiring_research"
SUBDIRS = ["plots/heatmaps", "plots/global_comparison", "reports"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_data_loader(path):
    """Safely extracts features and identifies the 'Factors' column."""
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    labels = df[target_col]
    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=[c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])], errors='ignore')
    return X, labels

def calculate_network_rewiring(disease, df_trans, df_meta, top_n=30):
    """Analyzes the shift in correlations between Healthy and Disease states."""
    print(f"\n>>> ANALYZING REWIRING: {disease.upper()}")
    
    # 1. Feature Selection (Focus on high-variance to reduce noise)
    top_genes = df_trans.var().sort_values(ascending=False).head(top_n).index
    top_mets = df_meta.var().sort_values(ascending=False).head(top_n).index
    
    X_t = df_trans[top_genes]
    X_m = df_meta[top_mets]
    
    # 2. Split by Group
    # We use the index from the labels we extracted earlier
    healthy_idx = labels[labels == 'healthy'].index
    disease_idx = labels[labels == 'c'].index
    
    # Calculate Cross-Correlations for both states
    # (Genetics vs Metabolomics)
    corr_h = X_t.loc[healthy_idx].corrwith(X_m.loc[healthy_idx], axis=0) # Simple example logic
    # More robust: Full Pairwise Correlation Matrix
    def get_cross_corr(df1, df2, idx):
        c_mat = pd.concat([df1.loc[idx], df2.loc[idx]], axis=1).corr()
        return c_mat.loc[top_genes, top_mets]

    c_h = get_cross_corr(X_t, X_m, healthy_idx)
    c_d = get_cross_corr(X_t, X_m, disease_idx)
    
    # Rewiring Score = Absolute Delta
    rewiring_matrix = (c_d - c_h).abs()
    
    # 3. Individual Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(rewiring_matrix, cmap="YlOrBr", annot=False, cbar_kws={'label': 'Rewiring Score ($|\Delta r|$)'})
    plt.title(f"{disease.upper()} - Regulatory Network Rewiring\n(Transcriptome ↔ Metabolome)", fontweight='bold')
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/heatmaps/{disease}_rewiring_map.png"))
    plt.close()
    
    return rewiring_matrix

def run_rewiring_pipeline():
    print("Initiating Global Network Rewiring Analysis...")
    global_rewiring_stats = []

    for disease in DISEASES:
        t_path, m_path = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        X_t, labels_t = robust_data_loader(t_path)
        X_m, _ = robust_data_loader(m_path)
        
        if X_t is not None and X_m is not None:
            # Inject labels globally for the function to use
            global labels
            labels = labels_t
            
            rewiring_mat = calculate_network_rewiring(disease, X_t, X_m)
            
            # Record global metrics
            global_rewiring_stats.append({
                "Disease": disease,
                "Avg_Rewiring": rewiring_mat.values.mean(),
                "Max_Rewiring": rewiring_mat.values.max()
            })
            rewiring_mat.to_csv(os.path.join(OUTPUT_DIR, f"reports/{disease}_rewiring_scores.csv"))

    # 4. Global Comparative Plot
    summary_df = pd.DataFrame(global_rewiring_stats)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "reports/global_summary.csv"), index=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary_df, x="Disease", y="Avg_Rewiring", palette="viridis")
    plt.title("Cross-Disease Network Rewiring Comparison", fontsize=16, fontweight='bold')
    plt.ylabel("Average Rewiring Score ($|\Delta r|$)")
    plt.savefig(os.path.join(OUTPUT_DIR, "plots/global_comparison/master_rewiring_benchmark.png"))
    plt.close()
    
    print(f"\n[SUCCESS] Rewiring analysis complete. Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_rewiring_pipeline()