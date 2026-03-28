import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from itertools import combinations

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "biomarker_stability_results"
SUBDIRS = ["plots/individual_stability", "plots/global_comparison", "reports"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_data_loader(path):
    """Safely extracts features and labels without KeyError."""
    if not os.path.exists(path): return None, None
    df = pd.read_csv(path)
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    y = df[target_col]
    X = df.select_dtypes(include=[np.number])
    X = X.drop(columns=[c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])], errors='ignore')
    return X, y

def calculate_stability(disease, X, y, omics_type, n_iterations=30, top_k=20):
    """Measures feature selection stability using Jaccard Index over Bootstrap iterations."""
    print(f"      Processing {omics_type} stability for {disease.upper()}...")
    
    selected_sets = []
    feature_names = X.columns
    
    # 1. Bootstrap Iterations
    for i in range(n_iterations):
        # Resample with replacement
        X_boot, y_boot = resample(X, y, replace=True, random_state=i)
        
        # Train Random Forest to get importance
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        rf.fit(X_boot, y_boot)
        
        # Select Top K features
        importances = pd.Series(rf.feature_importances_, index=feature_names)
        top_features = set(importances.sort_values(ascending=False).head(top_k).index)
        selected_sets.append(top_features)
    
    # 2. Pairwise Jaccard Similarity
    jaccard_scores = []
    for set_a, set_b in combinations(selected_sets, 2):
        intersection = len(set_a.intersection(set_b))
        union = len(set_a.union(set_b))
        jaccard_scores.append(intersection / union)
        
    return jaccard_scores

def run_stability_pipeline():
    print("Initiating Global Biomarker Stability Analysis...")
    master_records = []

    for disease in DISEASES:
        print(f"\n>>> ANALYZING: {disease.upper()}")
        t_path = f"{disease}_transcriptomics.csv"
        m_path = f"{disease}_metabolomics.csv"
        
        for path, dtype in [(t_path, "Transcriptomics"), (m_path, "Metabolomics")]:
            X, y = robust_data_loader(path)
            if X is not None:
                scores = calculate_stability(disease, X, y, dtype)
                
                # Store data for global plotting
                for s in scores:
                    master_records.append({"Disease": disease, "Omics": dtype, "Jaccard_Index": s})
                
                # Individual Disease Plot
                plt.figure(figsize=(8, 5))
                sns.histplot(scores, kde=True, color="indigo")
                plt.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.3f}')
                plt.title(f"{disease.upper()} {dtype} Stability Distribution")
                plt.xlabel("Jaccard Similarity")
                plt.legend()
                plt.savefig(os.path.join(OUTPUT_DIR, f"plots/individual_stability/{disease}_{dtype}_stability.png"))
                plt.close()

    # 3. Global Comparative Analysis
    master_df = pd.DataFrame(master_records)
    master_df.to_csv(os.path.join(OUTPUT_DIR, "reports/global_stability_metrics.csv"), index=False)
    
    # Global Plot: Comparing all diseases and both omics layers
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=master_df, x="Disease", y="Jaccard_Index", hue="Omics", palette="Set2")
    plt.title("Cross-Disease Biomarker Selection Stability (Jaccard Index)", fontsize=16, fontweight='bold')
    plt.ylabel("Jaccard Similarity (Pairwise Bootstrap)")
    plt.axhline(0.5, color='black', linestyle=':', alpha=0.5, label="Stability Threshold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "plots/global_comparison/master_stability_benchmark.png"))
    plt.close()
    
    print(f"\n[SUCCESS] Stability analysis complete. Folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_stability_pipeline()