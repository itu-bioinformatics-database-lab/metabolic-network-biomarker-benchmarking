import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dcor
import warnings

# --- Research Aesthetics & Config ---
warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 300
DPI = 300

# Output Structure
OUTPUT_DIR = "multi_omics_nonlinear_discovery"
SUBDIRS = ["plots/individual_diseases", "plots/global_comparison", "reports/disease_rankings", "reports/global_summary"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_data_loader(path):
    """Safely extracts numeric features and handles 'Factors' column."""
    if not os.path.exists(path):
        return None, None
    
    df = pd.read_csv(path)
    # Detect 'Factors' or the first column as the target
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    labels = df[target_col]
    
    # Isolate numeric features and drop common non-gene/non-metabolite columns
    X = df.select_dtypes(include=[np.number])
    cols_to_drop = [c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    
    return X, labels

def analyze_nonlinear_dependency(disease, df_trans, df_meta, top_n=30):
    """Calculates pairwise Distance Correlation for the top variable features."""
    print(f"\n>>> PROCESSING: {disease.upper()}")
    
    # 1. Feature Selection (Top N High Variance Features)
    # Focuses on biological signal and reduces O(n^2) computational load
    top_genes = df_trans.var().sort_values(ascending=False).head(top_n).index
    top_mets = df_meta.var().sort_values(ascending=False).head(top_n).index
    
    results = []
    print(f"      Scanning {len(top_genes)}x{len(top_mets)} Gene-Metabolite pairs...")
    
    for gene in top_genes:
        for met in top_mets:
            # dCor captures any dependency (linear, parabolic, periodic, etc.)
            d_val = dcor.distance_correlation(df_trans[gene].values, df_meta[met].values)
            results.append({
                'Disease': disease,
                'Gene': gene,
                'Metabolite': met,
                'dCor': d_val,
                'Pair': f"{gene} ↔ {met}"
            })
            
    dcor_df = pd.DataFrame(results).sort_values(by='dCor', ascending=False)
    
    # Save Individual Report
    dcor_df.to_csv(os.path.join(OUTPUT_DIR, f"reports/disease_rankings/{disease}_dcor_results.csv"), index=False)
    
    # 2. Visualization: Top 15 Nonlinear Pairs
    plt.figure(figsize=(12, 8))
    sns.barplot(data=dcor_df.head(15), x='dCor', y='Pair', palette='magma')
    plt.title(f"{disease.upper()} - Top Nonlinear Dependencies\n(Distance Correlation Rank)", fontsize=15, fontweight='bold')
    plt.xlabel("Distance Correlation Coefficient ($dCor$)", fontsize=12)
    plt.axvline(0.7, color='red', linestyle='--', alpha=0.5, label='High Dependency')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/individual_diseases/{disease}_top_pairs.png"), bbox_inches='tight')
    plt.close()
    
    return dcor_df

def run_main_pipeline():
    print("Initiating Global Nonlinear Multi-Omics Discovery...")
    global_accumulator = []

    for disease in DISEASES:
        t_path = f"{disease}_transcriptomics.csv"
        m_path = f"{disease}_metabolomics.csv"
        
        X_t, _ = robust_data_loader(t_path)
        X_m, _ = robust_data_loader(m_path)
        
        if X_t is not None and X_m is not None:
            # Analyze and collect results
            disease_df = analyze_nonlinear_dependency(disease, X_t, X_m)
            global_accumulator.append(disease_df)
        else:
            print(f"      [SKIP] Missing data files for {disease}")

    if global_accumulator:
        # 3. Global Cross-Disease Comparison
        master_df = pd.concat(global_accumulator)
        master_df.to_csv(os.path.join(OUTPUT_DIR, "reports/global_summary/master_nonlinear_rankings.csv"), index=False)
        
        # Plot Global Overview
        plt.figure(figsize=(14, 7))
        sns.boxplot(data=master_df, x='Disease', y='dCor', palette='viridis')
        plt.title("Cross-Disease Comparison of Transcriptome-Metabolome Dependency", fontsize=16, fontweight='bold')
        plt.ylabel("Distance Correlation Distribution")
        plt.savefig(os.path.join(OUTPUT_DIR, "plots/global_comparison/cross_disease_dcor_distribution.png"))
        plt.close()
        
        print(f"\n[COMPLETE] Global analysis saved in '{OUTPUT_DIR}'")
    else:
        print("\n[ERROR] No data processed. Please check file names and directory.")

if __name__ == "__main__":
    run_main_pipeline()