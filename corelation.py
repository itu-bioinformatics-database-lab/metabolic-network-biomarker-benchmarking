import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, combine_pvalues, friedmanchisquare
import warnings

# Suppress mathematical warnings (log2 division by zero, etc.)
warnings.filterwarnings('ignore')

# --- Global Configuration ---
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
BASE_DIR = "processed_for_analysis"

# Mapping diseases to their respective processed metabolomics files
DISEASE_MAP = {
    'Alzheimer': 'processed_alzheimer_metabolomics.csv',
    'Breast': 'processed_breast_metabolomics.csv',
    'Colon': 'processed_colon_metabolomics.csv'
}

# Expected algorithm output files in each disease directory
ALGORITHMS = ['timbr_results.csv', 'modified_timbr_results.csv', 'tamboor_results.csv']

# Output File Specifications
OUTPUT_CSV = "discriminatory_benchmark_results.csv"
OUTPUT_PLOT = "discriminatory_unified_report_300dpi.png"

def clean_metabolite_name(name):
    """Standardizes metabolite names by stripping suffixes and whitespace."""
    if pd.isna(name): return ""
    name = str(name).strip().lower().replace("'", "").replace('"', "")
    # Remove compartment suffixes common in Recon3D (e.g., _e, _c)
    if name.endswith('_e') or name.endswith('_c'):
        name = name[:-2]
    return name

def get_real_discriminatory_power(df):
    """
    Quantifies biological change intensity using absolute Log2 Fold Change.
    Formula: |Log2(Patient_Mean / Healthy_Mean)|
    """
    if 'factors' in df.columns:
        df = df.set_index('factors')
    elif 'Factors' in df.columns:
        df = df.set_index('Factors')
    
    df.index = df.index.astype(str).str.lower().str.strip()
    
    # Calculate means for the two cohorts
    healthy_mean = df[df.index == 'healthy'].mean(numeric_only=True)
    patient_mean = df[df.index == 'c'].mean(numeric_only=True)
    
    if healthy_mean.empty or patient_mean.empty:
        return pd.Series(dtype=float)

    healthy_mean = pd.to_numeric(healthy_mean, errors='coerce').astype(float)
    patient_mean = pd.to_numeric(patient_mean, errors='coerce').astype(float)

    # Handle negative values for fold change or use log2 for intensity
    if (healthy_mean < 0).any() or (patient_mean < 0).any():
        power = (patient_mean - healthy_mean).abs()
    else:
        eps = 1e-9 # Prevent division by zero
        power = np.log2((patient_mean + eps) / (healthy_mean + eps)).abs()
    
    return power.dropna()

def run_master_benchmark():
    """Matches biological data with algorithm scores and performs statistical evaluation."""
    all_results = []

    print("--- STEP 1: Correlation Analysis (Biological vs. Computational) ---")
    
    for disease, metab_file in DISEASE_MAP.items():
        disease_path = os.path.join(BASE_DIR, disease)
        metab_full_path = os.path.join(disease_path, metab_file)
        
        if not os.path.exists(metab_full_path):
            print(f"!!! Warning: {metab_full_path} not found. Skipping {disease}.")
            continue

        df_metab = pd.read_csv(metab_full_path)
        df_metab.columns = [c.lower().strip() for c in df_metab.columns]
        actual_power = get_real_discriminatory_power(df_metab)

        for algo_file in ALGORITHMS:
            algo_path = os.path.join(disease_path, algo_file)
            if not os.path.exists(algo_path): continue
            
            # Read algorithm outputs (Score, Metabolite ID)
            df_algo = pd.read_csv(algo_path, header=None)
            df_algo.columns = ['score', 'metabolite']
            df_algo['clean_id'] = df_algo['metabolite'].apply(clean_metabolite_name)
            df_algo['abs_score'] = df_algo['score'].abs()
            
            # Aggregate scores for metabolites mapped multiple times
            algo_power = df_algo.groupby('clean_id')['abs_score'].max()
            
            # Align biological data with algorithm scores
            common_ids = list(set(actual_power.index) & set(algo_power.index))
            if len(common_ids) > 1:
                rho, p_val = spearmanr(actual_power.loc[common_ids], algo_power.loc[common_ids])
                
                # Standardize Algorithm Naming
                raw_name = algo_file.replace('_results.csv', '').upper()
                display_name = "E-TIMBR" if raw_name == "MODIFIED_TIMBR" else raw_name
                
                all_results.append({
                    'Condition': disease,
                    'Algorithm': display_name,
                    'Spearman_Rho': rho,
                    'P_Value': p_val,
                    'Metabolite_Count': len(common_ids)
                })
                print(f"  Processed: {disease} - {display_name} (Overlapping N={len(common_ids)})")

    if not all_results:
        print("CRITICAL ERROR: No metabolite overlaps found across datasets.")
        return

    # Save quantitative results
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Benchmark results exported to '{OUTPUT_CSV}'.")

    print("--- STEP 2: Statistical Visualization and Reporting ---")
    
    # Initialize a multi-panel figure
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel A: Performance Distribution (Boxplot with Friedman Test)
    try:
        pivot_data = df_results.pivot(index='Condition', columns='Algorithm', values='Spearman_Rho')
        if pivot_data.shape[1] >= 3:
            _, f_p = friedmanchisquare(*[pivot_data[col] for col in pivot_data.columns])
            p_label = f"(Friedman p={f_p:.3f})"
        else:
            p_label = ""
    except:
        p_label = ""

    sns.boxplot(data=df_results, x='Algorithm', y='Spearman_Rho', palette='Set2', ax=axes[0])
    sns.swarmplot(data=df_results, x='Algorithm', y='Spearman_Rho', color='.25', size=8, ax=axes[0])
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_title(f"A: Performance Distribution Across Datasets\n{p_label}", fontweight='bold')
    axes[0].set_ylabel("Correlation Coefficient (Spearman Rho)")

    # Panel B: Global Statistical Reliability (Fisher's Combined Probability)
    fisher_stats = []
    for algo in df_results['Algorithm'].unique():
        p_values = df_results[df_results['Algorithm'] == algo]['P_Value'].values
        _, combined_p = combine_pvalues(p_values, method='fisher')
        fisher_stats.append({'Algorithm': algo, 'neg_log_p': -np.log10(combined_p)})
    
    df_fisher = pd.DataFrame(fisher_stats)
    sns.barplot(data=df_fisher, x='Algorithm', y='neg_log_p', palette='viridis', ax=axes[1])
    axes[1].axhline(-np.log10(0.05), color='red', linestyle=':', label='Significance Threshold (p=0.05)')
    axes[1].set_ylabel("Global Confidence (-log10 Combined P)")
    axes[1].set_title("B: Global Statistical Meta-Analysis", fontweight='bold')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"🚀 Visual summary report generated: '{OUTPUT_PLOT}'.")

if __name__ == "__main__":
    run_master_benchmark()