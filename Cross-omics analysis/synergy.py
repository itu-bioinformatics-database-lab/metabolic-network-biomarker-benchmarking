import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = "multi_omics_fixed_research"
SUBDIRS = ["plots/correlations", "plots/synergy_auc", "reports"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_data_loader(path):
    """Safely extracts features and ensures numeric-only X and clean y."""
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    
    # 1. Identify and extract Target (Factors)
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    y = df[target_col].map({'healthy': 0, 'c': 1})
    
    # 2. Extract numeric features only
    X = df.select_dtypes(include=[np.number])
    
    # 3. Drop Target and any Non-Gene/Non-Metabolite columns
    cols_to_drop = [c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    # 4. Filter out Zero-Variance columns (Crucial for correlation)
    X = X.loc[:, (X.var() > 1e-9)]
    
    return X, y

def get_realistic_auc(X, y):
    """Calculates AUC using 5-Fold Cross-Validation to prevent overfitting (AUC=1.0 issue)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # cross_val_predict gets predictions for each sample when it was in the 'test' set
    try:
        y_probs = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:, 1]
        return roc_auc_score(y, y_probs)
    except:
        # In case a class has too few samples for 5-fold
        return np.nan

def run_fixed_pipeline():
    print("Launching Fixed Multi-Omic Research Pipeline...")
    global_results = []

    for disease in DISEASES:
        print(f"\n>>> ANALYZING: {disease.upper()}")
        t_path, m_path = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        
        X_t, y_t = robust_data_loader(t_path)
        X_m, y_m = robust_data_loader(m_path)
        
        if X_t is not None and X_m is not None:
            # Alignment
            common_idx = X_t.index.intersection(X_m.index)
            X_t_a, X_m_a, y_a = X_t.loc[common_idx], X_m.loc[common_idx], y_t.loc[common_idx]

            # --- CORRELATION ANALYSIS (Fixing the blank heatmap) ---
            # Use top 20 by variance but ensure they are actually in the data
            top_t = X_t_a.var().sort_values(ascending=False).head(20).index
            top_m = X_m_a.var().sort_values(ascending=False).head(20).index
            
            corr_matrix = pd.DataFrame(index=top_t, columns=top_m)
            for g in top_t:
                for m in top_m:
                    rho, _ = spearmanr(X_t_a[g], X_m_a[m])
                    corr_matrix.loc[g, m] = rho
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix.astype(float), cmap='RdBu_r', center=0, annot=False)
            plt.title(f"{disease.upper()} - Cross-Omic Correlation")
            plt.savefig(os.path.join(OUTPUT_DIR, f"plots/correlations/{disease}_corr.png"))
            plt.close()

            # --- SYNERGY ANALYSIS (Fixing the AUC=1.0 issue) ---
            auc_t = get_realistic_auc(X_t_a, y_a)
            auc_m = get_realistic_auc(X_m_a, y_a)
            
            X_combined = pd.concat([X_t_a, X_m_a], axis=1)
            auc_combined = get_realistic_auc(X_combined, y_a)
            
            gain = auc_combined - max(auc_t, auc_m)
            print(f"      Results: T_AUC={auc_t:.3f} | M_AUC={auc_m:.3f} | Combined={auc_combined:.3f} | Gain={gain:.3f}")

            global_results.append({
                "Disease": disease, "Transcriptomics_AUC": auc_t,
                "Metabolomics_AUC": auc_m, "Integrated_AUC": auc_combined, "Gain": gain
            })

    # Global Reporting
    summary_df = pd.DataFrame(global_results)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "reports/synergy_report.csv"), index=False)
    print(f"\n[SUCCESS] Pipeline completed. Folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_fixed_pipeline()