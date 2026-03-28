import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import StandardScaler

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

# Directory Setup
OUTPUT_DIR = "cca_research_results"
for d in ["plots", "reports"]: 
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def load_data_safely(path):
    """Robustly separates 'Factors' and returns numeric data."""
    df = pd.read_csv(path)
    label_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    y_raw = df[label_col]
    X = df.select_dtypes(include=[np.number])
    # Drop index-like or ID columns
    cols_to_drop = [c for c in X.columns if "Unnamed" in c or "ID" in c or "Sample" in c]
    X = X.drop(columns=cols_to_drop)
    return X, y_raw

def run_cca_pipeline():
    print("Initiating Canonical Correlation Analysis (CCA) Pipeline...")
    global_cca_results = []

    for disease in DISEASES:
        print(f"\n>>> ANALYZING: {disease.upper()}")
        t_file, m_file = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        
        if not (os.path.exists(t_file) and os.path.exists(m_file)): continue

        # 1. Load and Align
        X_t, labels = load_data_safely(t_file)
        X_m, _ = load_data_safely(m_file)
        
        # Alignment check
        n = min(len(X_t), len(X_m))
        X_t, X_m, labels = X_t.iloc[:n], X_m.iloc[:n], labels.iloc[:n]

        # 2. Pre-processing for Dimensionality Stability
        # CCA requires n_samples > n_features to be stable.
        # We use the top 500 highest variance genes to represent the transcriptome.
        top_v_genes = X_t.var().sort_values(ascending=False).head(500).index
        X_t_sub = X_t[top_v_genes]

        scaler = StandardScaler()
        X_t_std = scaler.fit_transform(X_t_sub)
        X_m_std = scaler.fit_transform(X_m)

        # 3. Canonical Correlation Analysis
        # n_components=2 allows us to find the two strongest axes of correlation
        cca = CCA(n_components=2)
        cca.fit(X_t_std, X_m_std)
        
        # Transform data to Canonical Variate scores
        X_c, Y_c = cca.transform(X_t_std, X_m_std)

        # Calculate the Canonical Correlations (R)
        r1 = np.corrcoef(X_c[:, 0], Y_c[:, 0])[0, 1]
        r2 = np.corrcoef(X_c[:, 1], Y_c[:, 1])[0, 1]

        # 4. Plotting & Reporting
        plot_cca_variates(disease, X_c, Y_c, labels, r1)
        
        global_cca_results.append({
            "Disease": disease,
            "Canonical_R_Comp1": r1,
            "Canonical_R_Comp2": r2
        })

    # Save Global Benchmark
    summary_df = pd.DataFrame(global_cca_results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "reports/cca_global_summary.csv"), index=False)
    print(f"\n[SUCCESS] CCA Analysis complete. Folder: {OUTPUT_DIR}")

def plot_cca_variates(disease, X_c, Y_c, labels, r_score):
    """Plots the 1st Transcriptomics Variate against the 1st Metabolomics Variate."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=X_c[:, 0], y=Y_c[:, 0], hue=labels, palette="Set1", s=100, alpha=0.8, edgecolor='w')
    
    # Draw identity line
    plt.plot([X_c[:,0].min(), X_c[:,0].max()], [X_c[:,0].min(), X_c[:,0].max()], 
             color='black', linestyle='--', alpha=0.5)
    
    plt.title(f"CCA Global Linkage: {disease.upper()}\nCorrelation (R) = {r_score:.4f}", fontsize=15, fontweight='bold')
    plt.xlabel("Transcriptomics Canonical Variate 1", fontsize=12)
    plt.ylabel("Metabolomics Canonical Variate 1", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/{disease}_cca_linkage.png"))
    plt.close()

if __name__ == "__main__":
    run_cca_pipeline()