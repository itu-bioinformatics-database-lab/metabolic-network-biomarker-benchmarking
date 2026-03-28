import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "nonlinear_dependency_research"
for d in ["plots/heatmaps", "reports/pairs"]: 
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def dist_corr(X, Y):
    """
    Vectorized implementation of Distance Correlation (dCor).
    Captures both linear and nonlinear dependencies.
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    
    n = X.shape[0]
    if n < 2: return 0.0

    # Distance matrices
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    
    # Double centering
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    # Distance covariance
    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)
    
    return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

def robust_load(path):
    df = pd.read_csv(path)
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=[c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])], errors='ignore')
    return X

def run_nonlinear_pipeline():
    print("Launching Nonlinear Pairwise Dependency Analysis...")
    
    for disease in DISEASES:
        print(f"\n>>> SCANNING PAIRS: {disease.upper()}")
        t_file, m_file = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        if not (os.path.exists(t_file) and os.path.exists(m_file)): continue

        # 1. Load and Select Top Features
        # Scanning 16k x 500 is too slow, we focus on the top 30 most variable of each
        X_t = robust_load(t_file)
        X_m = robust_load(m_file)
        
        top_genes = X_t.var().sort_values(ascending=False).head(30).index
        top_mets = X_m.var().sort_values(ascending=False).head(30).index
        
        X_t_sub = X_t[top_genes]
        X_m_sub = X_m[top_mets]

        # 2. Compute Distance Correlation Matrix
        dcor_matrix = pd.DataFrame(index=top_genes, columns=top_mets)
        
        for g in top_genes:
            for m in top_mets:
                dcor_matrix.loc[g, m] = dist_corr(X_t_sub[g].values, X_m_sub[m].values)

        # 3. Save and Plot
        dcor_matrix.to_csv(os.path.join(OUTPUT_DIR, f"reports/pairs/{disease}_dcor_matrix.csv"))
        plot_dcor_heatmap(disease, dcor_matrix.astype(float))

    print(f"\n[SUCCESS] Nonlinear analysis complete. Folder: {OUTPUT_DIR}")

def plot_dcor_heatmap(disease, df):
    plt.figure(figsize=(14, 12))
    sns.heatmap(df, cmap="viridis", annot=False, cbar_kws={'label': 'Distance Correlation (dCor)'})
    plt.title(f"{disease.upper()} - Nonlinear Gene-Metabolite Dependency\n(Top 30x30 Variance Features)", fontweight='bold')
    plt.xlabel("Metabolites"); plt.ylabel("Genes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/heatmaps/{disease}_dcor_map.png"))
    plt.close()

if __name__ == "__main__":
    run_nonlinear_pipeline()import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "nonlinear_dependency_research"
for d in ["plots/heatmaps", "reports/pairs"]: 
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def dist_corr(X, Y):
    """
    Vectorized implementation of Distance Correlation (dCor).
    Captures both linear and nonlinear dependencies.
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if X.ndim == 1: X = X[:, None]
    if Y.ndim == 1: Y = Y[:, None]
    
    n = X.shape[0]
    if n < 2: return 0.0

    # Distance matrices
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    
    # Double centering
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    # Distance covariance
    dcov2_xy = (A * B).sum() / (n * n)
    dcov2_xx = (A * A).sum() / (n * n)
    dcov2_yy = (B * B).sum() / (n * n)
    
    return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))

def robust_load(path):
    df = pd.read_csv(path)
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=[c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])], errors='ignore')
    return X

def run_nonlinear_pipeline():
    print("Launching Nonlinear Pairwise Dependency Analysis...")
    
    for disease in DISEASES:
        print(f"\n>>> SCANNING PAIRS: {disease.upper()}")
        t_file, m_file = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        if not (os.path.exists(t_file) and os.path.exists(m_file)): continue

        # 1. Load and Select Top Features
        # Scanning 16k x 500 is too slow, we focus on the top 30 most variable of each
        X_t = robust_load(t_file)
        X_m = robust_load(m_file)
        
        top_genes = X_t.var().sort_values(ascending=False).head(30).index
        top_mets = X_m.var().sort_values(ascending=False).head(30).index
        
        X_t_sub = X_t[top_genes]
        X_m_sub = X_m[top_mets]

        # 2. Compute Distance Correlation Matrix
        dcor_matrix = pd.DataFrame(index=top_genes, columns=top_mets)
        
        for g in top_genes:
            for m in top_mets:
                dcor_matrix.loc[g, m] = dist_corr(X_t_sub[g].values, X_m_sub[m].values)

        # 3. Save and Plot
        dcor_matrix.to_csv(os.path.join(OUTPUT_DIR, f"reports/pairs/{disease}_dcor_matrix.csv"))
        plot_dcor_heatmap(disease, dcor_matrix.astype(float))

    print(f"\n[SUCCESS] Nonlinear analysis complete. Folder: {OUTPUT_DIR}")

def plot_dcor_heatmap(disease, df):
    plt.figure(figsize=(14, 12))
    sns.heatmap(df, cmap="viridis", annot=False, cbar_kws={'label': 'Distance Correlation (dCor)'})
    plt.title(f"{disease.upper()} - Nonlinear Gene-Metabolite Dependency\n(Top 30x30 Variance Features)", fontweight='bold')
    plt.xlabel("Metabolites"); plt.ylabel("Genes")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/heatmaps/{disease}_dcor_map.png"))
    plt.close()

if __name__ == "__main__":
    run_nonlinear_pipeline()