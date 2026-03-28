import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "sparse_module_discovery_results"
SUBDIRS = ["plots/modules", "reports/top_weights", "reports/global_summary"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_data_loader(path):
    """Safely extracts features and handles the 'Factors' column issue."""
    if not os.path.exists(path): return None
    df = pd.read_csv(path)
    # Detect 'Factors' or the first column as the target
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    # Isolate numeric features
    X = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    return X

def perform_sparse_module_discovery(disease, df_trans, df_meta, n_top=15, alpha=0.1):
    """
    Finds sparse modules using a Lasso-regularized CCA approach. 
    Identifies the 'needle in the haystack' master regulators.
    """
    print(f"\n>>> DISCOVERING SPARSE MODULES: {disease.upper()}")
    
    # 1. Sample Alignment
    common_idx = df_trans.index.intersection(df_meta.index)
    if len(common_idx) == 0:
        print(f"      [Error] No common samples for {disease}")
        return None

    # Standardize data (Z-score)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_trans.loc[common_idx])
    Y = scaler.fit_transform(df_meta.loc[common_idx])
    
    gene_names = df_trans.columns
    met_names = df_meta.columns

    # 2. Iterative Sparse CCA Logic
    # Component 1 represents the strongest global axis of correlation
    cca = CCA(n_components=1)
    X_scores, Y_scores = cca.fit_transform(X, Y)
    
    # Sparsification via Lasso: 
    # Regress Transcriptome on Metabolomics scores & vice-versa
    lasso_genes = Lasso(alpha=alpha).fit(X, Y_scores.ravel())
    lasso_mets = Lasso(alpha=alpha).fit(Y, X_scores.ravel())
    
    # 3. Extract Non-Zero Weights
    top_genes = pd.Series(np.abs(lasso_genes.coef_), index=gene_names).sort_values(ascending=False)
    top_mets = pd.Series(np.abs(lasso_mets.coef_), index=met_names).sort_values(ascending=False)
    
    # Filter out features shrunk to zero
    top_genes = top_genes[top_genes > 0].head(n_top)
    top_mets = top_mets[top_mets > 0].head(n_top)

    # 4. Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    if not top_genes.empty:
        sns.barplot(x=top_genes.values, y=top_genes.index, ax=ax1, palette="viridis")
        ax1.set_title("Transcriptomic Module (Lasso Weights)", fontweight='bold')
    
    if not top_mets.empty:
        sns.barplot(x=top_mets.values, y=top_mets.index, ax=ax2, palette="magma")
        ax2.set_title("Metabolomic Module (Lasso Weights)", fontweight='bold')

    plt.suptitle(f"{disease.upper()}: Sparse Multi-Omic Module discovery\n(Alpha={alpha})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/modules/{disease}_sparse_module.png"))
    plt.close()

    # Return results for global aggregation
    return {"disease": disease, "genes": top_genes.index.tolist(), "metabolites": top_mets.index.tolist()}

def run_pipeline():
    print("Initiating Global Sparse Module Discovery...")
    global_modules = []

    for disease in DISEASES:
        X_t = robust_data_loader(f"{disease}_transcriptomics.csv")
        X_m = robust_data_loader(f"{disease}_metabolomics.csv")
        
        if X_t is not None and X_m is not None:
            module_data = perform_sparse_module_discovery(disease, X_t, X_m)
            if module_data:
                global_modules.append(module_data)
                # Save disease-specific CSV
                pd.DataFrame({
                    "Genes": pd.Series(module_data['genes']),
                    "Metabolites": pd.Series(module_data['metabolites'])
                }).to_csv(os.path.join(OUTPUT_DIR, f"reports/top_weights/{disease}_sparse_features.csv"), index=False)

    # Global Summary Table
    if global_modules:
        summary_df = pd.DataFrame(global_modules)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, "reports/global_summary/master_module_list.csv"), index=False)
        print(f"\n[SUCCESS] Pipeline complete. Check '{OUTPUT_DIR}' for the master regulator list.")

if __name__ == "__main__":
    run_pipeline()