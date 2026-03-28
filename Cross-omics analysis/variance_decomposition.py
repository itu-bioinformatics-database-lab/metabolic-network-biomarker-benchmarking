import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "variance_decomposition_research"
SUBDIRS = ["plots/individual_factors", "plots/global_benchmarks", "reports"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_data_loader(path):
    """Safely extracts features and identifies the 'Factors' column."""
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=[c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])], errors='ignore')
    return X

def analyze_variance_decomposition(disease, df_trans, df_meta, n_factors=3):
    """
    Decomposes variance into latent factors to determine 
    which omic layer dominates specific biological signals.
    """
    print(f"\n>>> DECOMPOSING VARIANCE: {disease.upper()}")
    
    # 1. Alignment and Combination
    common_idx = df_trans.index.intersection(df_meta.index)
    X_t = df_trans.loc[common_idx]
    X_m = df_meta.loc[common_idx]
    X_combined = pd.concat([X_t, X_m], axis=1)
    
    # Scaling is mandatory for Factor Analysis
    X_scaled = StandardScaler().fit_transform(X_combined)
    
    # 2. Factor Analysis
    fa = FactorAnalysis(n_components=n_factors, random_state=42)
    fa.fit(X_scaled)
    
    # 3. Loadings Analysis (Contribution of each feature to the factors)
    loadings = pd.DataFrame(
        np.abs(fa.components_.T), 
        index=X_combined.columns, 
        columns=[f'Factor_{i+1}' for i in range(n_factors)]
    )
    
    # 4. Calculate Contribution Ratios
    n_genes = X_t.shape[1]
    gene_cont = loadings.iloc[:n_genes].mean()
    met_cont = loadings.iloc[n_genes:].mean()
    
    summary = pd.DataFrame({
        'Transcriptomics': gene_cont, 
        'Metabolomics': met_cont
    })
    
    # Normalize to 100%
    summary_norm = summary.div(summary.sum(axis=1), axis=0) * 100

    # 5. Individual Plotting
    ax = summary_norm.plot(kind='bar', stacked=True, figsize=(10, 7), color=['#3498db', '#e74c3c'], alpha=0.8)
    plt.title(f"{disease.upper()} - Multi-Omic Variance Decomposition", fontsize=14, fontweight='bold')
    plt.ylabel("Contribution Percentage (%)")
    plt.xlabel("Latent Factors (Hidden Biological Drivers)")
    plt.xticks(rotation=0)
    plt.legend(title="Omic Layer", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/individual_factors/{disease}_variance_decomp.png"))
    plt.close()

    return summary_norm

def run_variance_pipeline():
    print("Initiating Global Multi-Omic Factor Analysis...")
    global_records = []

    for disease in DISEASES:
        X_t = robust_data_loader(f"{disease}_transcriptomics.csv")
        X_m = robust_data_loader(f"{disease}_metabolomics.csv")
        
        if X_t is not None and X_m is not None:
            summary = analyze_variance_decomposition(disease, X_t, X_m)
            
            # Store for global comparison (Mean across factors)
            avg_cont = summary.mean()
            global_records.append({
                "Disease": disease,
                "Transcriptomics_Mean": avg_cont['Transcriptomics'],
                "Metabolomics_Mean": avg_cont['Metabolomics']
            })
            summary.to_csv(os.path.join(OUTPUT_DIR, f"reports/{disease}_factor_loadings.csv"))

    # 6. Global Benchmark Plot
    master_df = pd.DataFrame(global_records)
    master_df.to_csv(os.path.join(OUTPUT_DIR, "reports/global_summary.csv"), index=False)
    
    master_df.set_index('Disease').plot(kind='bar', stacked=True, figsize=(12, 6), color=['#3498db', '#e74c3c'])
    plt.title("Cross-Disease Omic Dominance Benchmark", fontsize=16, fontweight='bold')
    plt.ylabel("Mean Variance Contribution (%)")
    plt.axhline(50, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots/global_benchmarks/master_variance_comparison.png"))
    plt.close()
    
    print(f"\n[SUCCESS] Variance decomposition complete. Results in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_variance_pipeline()