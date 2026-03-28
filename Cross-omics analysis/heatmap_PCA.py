import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from scipy.stats import spearmanr

# --- Configuration ---
sns.set_theme(style="white")
DPI = 300
OUTPUT_DIR = "advanced_omics_analysis"
# Create directories for plots and reports
for d in ["plots", "reports"]: 
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate", "alzheimer"]

def manual_pca_svd(X, n_components=2):
    """Intel Mac Safe SVD-based PCA."""
    X_scaled = StandardScaler().fit_transform(X)
    X_centered = X_scaled - np.mean(X_scaled, axis=0)
    u, s, vt = np.linalg.svd(X_centered, full_matrices=False)
    return u[:, :n_components] * s[:n_components]

def smart_load_and_split(file_path):
    """
    Finds the target column containing 'healthy' and 'c' regardless of the column name 
    and splits the data into features (X) and target (y).
    """
    df = pd.read_csv(file_path)
    
    # 1. Search for the name 'Factors', otherwise check for specific keywords
    target_col = None
    if 'Factors' in df.columns:
        target_col = 'Factors'
    else:
        # Find the column containing 'healthy' or 'c' (case-insensitive)
        for col in df.columns:
            if df[col].astype(str).str.contains('healthy|c', case=False).any():
                target_col = col
                break
    
    if target_col is None:
        raise ValueError(f"Target column (Factors) not found in {file_path}!")

    # 2. Separation of X and y
    X = df.drop(columns=[target_col])
    
    # If the first column is an ID/Index (numeric increasing or like 'Sample_1'), drop it as well
    if X.iloc[:, 0].astype(str).str.contains('Sample|GSM|patient', case=False).any() or X.columns[0] == 'Unnamed: 0':
        X = X.iloc[:, 1:]

    y_raw = df[target_col]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw)
    
    return X, y_encoded, y_raw

def run_advanced_analysis():
    global_stats = []

    for disease in DISEASES:
        print(f"\n>>> Analyzing: {disease.upper()}")
        t_file, m_file = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        
        if not (os.path.exists(t_file) and os.path.exists(m_file)): 
            continue

        try:
            # Smart loading eliminates KeyError risks
            X_t, y_t_enc, y_t_raw = smart_load_and_split(t_file)
            X_m, y_m_enc, y_m_raw = smart_load_and_split(m_file)

            # --- PCA & Silhouette ---
            pc_t = manual_pca_svd(X_t)
            pc_m = manual_pca_svd(X_m)
            
            sil_t = silhouette_score(pc_t, y_t_enc)
            sil_m = silhouette_score(pc_m, y_m_enc)

            # Visualization
            plot_dual_pca(disease, pc_t, pc_m, y_t_raw, y_m_raw)
            
            global_stats.append({
                "Disease": disease,
                "Transcriptomics_Silhouette": sil_t,
                "Metabolomics_Silhouette": sil_m
            })

            # --- Cross-Omics Correlation (Common Samples) ---
            # Aligning via sample IDs (optional but recommended for robust analysis)
            common_len = min(len(X_t), len(X_m))
            X_t_sub = X_t.iloc[:common_len]
            X_m_sub = X_m.iloc[:common_len]

            # Select the top 15 features with the highest variance (Top Biomarkers)
            top_genes = X_t_sub.std().sort_values(ascending=False).head(15).index.tolist()
            top_mets = X_m_sub.std().sort_values(ascending=False).head(15).index.tolist()
            
            corr_matrix = pd.DataFrame(index=top_genes, columns=top_mets)
            for g in top_genes:
                for m in top_mets:
                    rho, _ = spearmanr(X_t_sub[g], X_m_sub[m])
                    corr_matrix.loc[g, m] = rho
            
            plot_heatmap(disease, corr_matrix.astype(float))

        except Exception as e:
            print(f"      ! Error in {disease}: {e}")

    # Reporting
    pd.DataFrame(global_stats).to_csv(os.path.join(OUTPUT_DIR, "reports/clustering_quality.csv"), index=False)

def plot_dual_pca(disease, pc_t, pc_m, labels_t, labels_m):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.scatterplot(x=pc_t[:,0], y=pc_t[:,1], hue=labels_t, ax=axes[0], palette="viridis", s=100)
    axes[0].set_title(f"{disease.upper()} - Transcriptomics PCA")
    sns.scatterplot(x=pc_m[:,0], y=pc_m[:,1], hue=labels_m, ax=axes[1], palette="magma", s=100)
    axes[1].set_title(f"{disease.upper()} - Metabolomics PCA")
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/{disease}_dual_pca.png"), dpi=DPI)
    plt.close()

def plot_heatmap(disease, df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, cmap="vlag", center=0, fmt=".2f")
    plt.title(f"Inter-Omics Interaction: {disease.upper()}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/{disease}_correlation.png"), dpi=DPI)
    plt.close()

if __name__ == "__main__":
    run_advanced_analysis()
    print(f"\n[DONE] Results are in {OUTPUT_DIR}")