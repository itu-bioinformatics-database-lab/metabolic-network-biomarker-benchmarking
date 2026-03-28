import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import StandardScaler

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "multi_block_pls_research_v2"
SUBDIRS = ["plots/latent_alignment", "reports/statistics"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_data_loader(path):
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    labels = df[target_col]
    X = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    return X, labels

def multi_block_analysis(disease, df_trans, df_meta, labels):
    print(f"\n>>> INITIATING MULTI-BLOCK PLS: {disease.upper()}")
    
    scaler = StandardScaler()
    X_t = scaler.fit_transform(df_trans)
    X_m = scaler.fit_transform(df_meta)
    
    pls = PLSRegression(n_components=2)
    pls.fit(X_t, X_m)
    t_scores, m_scores = pls.transform(X_t, X_m)
    
    # Extract first components for correlation
    x_latent = t_scores[:, 0]
    y_latent = m_scores[:, 0]
    correlation = np.corrcoef(x_latent, y_latent)[0, 1]
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=x_latent, y=y_latent, hue=labels, 
                    palette='coolwarm', s=120, edgecolor='black', alpha=0.8)
    
    # --- FIXED LINE: linestyle moved inside line_kws ---
    sns.regplot(x=x_latent, y=y_latent, scatter=False, color='gray', 
                line_kws={"linestyle": "--", "linewidth": 2})
    
    plt.title(f"{disease.upper()} - Multi-Omic Latent Alignment\nCorrelation (R) = {correlation:.4f}", fontsize=14, fontweight='bold')
    plt.xlabel("Transcriptome Latent Component 1", fontsize=12)
    plt.ylabel("Metabolomics Latent Component 1", fontsize=12)
    plt.legend(title='Condition', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/latent_alignment/{disease}_pls_alignment.png"), bbox_inches='tight')
    plt.close()
    
    return correlation

def run_pls_pipeline():
    print("Launching Global Multi-Block PLS Integration...")
    global_results = []

    for disease in DISEASES:
        t_path, m_path = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        X_t, labels_t = robust_data_loader(t_path)
        X_m, _ = robust_data_loader(m_path)
        
        if X_t is not None and X_m is not None:
            common_idx = X_t.index.intersection(X_m.index)
            r_val = multi_block_analysis(disease, X_t.loc[common_idx], X_m.loc[common_idx], labels_t.loc[common_idx])
            global_results.append({"Disease": disease, "Latent_Correlation": r_val})

    if global_results:
        summary_df = pd.DataFrame(global_results).sort_values(by="Latent_Correlation", ascending=False)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "reports/statistics/global_alignment.csv"), index=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(data=summary_df, x="Disease", y="Latent_Correlation", palette="viridis")
        plt.title("Cross-Disease Omics Latent Alignment Comparison", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "plots/global_pls_benchmark.png"))
        plt.close()
        print(f"\n[SUCCESS] Pipeline complete. Check '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_pls_pipeline()