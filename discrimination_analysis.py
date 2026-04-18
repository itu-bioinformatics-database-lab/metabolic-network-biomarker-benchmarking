import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, combine_pvalues, friedmanchisquare
import warnings

# Suppress mathematical warnings (log2 errors, etc.)
warnings.filterwarnings('ignore')

# --- Configuration ---
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300
BASE_DIR = "processed_for_analysis"

# Mapping diseases to their respective processed files
DISEASE_MAP = {
    'Alzheimer': 'processed_alzheimer_metabolomics.csv',
    'Breast': 'processed_breast_metabolomics.csv',
    'Colon': 'processed_colon_metabolomics.csv'
}

# List of result files from different algorithms
ALGORITHMS = ['timbr_results.csv', 'modified_timbr_results.csv', 'tamboor_results.csv']

# Output File Specifications
OUTPUT_CSV = "discriminatory_benchmark_results.csv"
OUTPUT_PLOT = "discriminatory_unified_report_300dpi.png"

def clean_metabolite_name(name):
    """Standardizes metabolite names by cleaning suffixes and whitespace."""
    if pd.isna(name): return ""
    name = str(name).strip().lower().replace("'", "").replace('"', "")
    # Remove compartment suffixes (e.g., extracellular '_e' or cytosolic '_c')
    if name.endswith('_e') or name.endswith('_c'):
        name = name[:-2]
    return name

def get_real_discriminatory_power(df):
    """
    Calculates biological change intensity: |Log2(Patient/Healthy)|
    Quantifies the 'Ground Truth' discriminatory power from experimental data.
    """
    if 'factors' in df.columns:
        df = df.set_index('factors')
    elif 'Factors' in df.columns:
        df = df.set_index('Factors')
    
    df.index = df.index.astype(str).str.lower().str.strip()
    
    # Calculate group means
    healthy_mean = df[df.index == 'healthy'].mean(numeric_only=True)
    patient_mean = df[df.index == 'c'].mean(numeric_only=True)
    
    if healthy_mean.empty or patient_mean.empty:
        return pd.Series(dtype=float)

    healthy_mean = pd.to_numeric(healthy_mean, errors='coerce').astype(float)
    patient_mean = pd.to_numeric(patient_mean, errors='coerce').astype(float)

    # Calculate absolute difference or absolute Log2 Fold Change
    if (healthy_mean < 0).any() or (patient_mean < 0).any():
        power = (patient_mean - healthy_mean).abs()
    else:
        eps = 1e-9 # Prevent division by zero
        power = np.log2((patient_mean + eps) / (healthy_mean + eps)).abs()
    
    return power.dropna()

def run_master_benchmark_with_csv():
    """Main execution loop to match biological changes with algorithm scores."""
    all_results = []

    print("--- STEP 1: Matching Biological Changes with Algorithm Scores ---")
    
    for disease, metab_file in DISEASE_MAP.items():
        disease_path = os.path.join(BASE_DIR, disease)
        metab_full_path = os.path.join(disease_path, metab_file)
        
        if not os.path.exists(metab_full_path):
            print(f"!!! Error: {metab_full_path} not found. Skipping {disease}.")
            continue

        df_metab = pd.read_csv(metab_full_path)
        df_metab.columns = [c.lower().strip() for c in df_metab.columns]
        actual_power = get_real_discriminatory_power(df_metab)

        for algo_file in ALGORITHMS:
            algo_path = os.path.join(disease_path, algo_file)
            if not os.path.exists(algo_path): continue
            
            # Load algorithm results
            df_algo = pd.read_csv(algo_path, header=None)
            df_algo.columns = ['score', 'metabolite']
            df_algo['clean_id'] = df_algo['metabolite'].apply(clean_metabolite_name)
            df_algo['abs_score'] = df_algo['score'].abs()
            
            # Aggregate scores for metabolites
            algo_power = df_algo.groupby('clean_id')['abs_score'].max()
            
            # Find common metabolites between experimental data and model results
            common_ids = list(set(actual_power.index) & set(algo_power.index))
            if len(common_ids) > 1:
                # Perform Spearman correlation
                rho, p_val = spearmanr(actual_power.loc[common_ids], algo_power.loc[common_ids])
                
                # Standardize Algorithm Naming (CAPS and E-TIMBR alias)
                raw_name = algo_file.replace('_results.csv', '').upper()
                display_name = "E-TIMBR" if raw_name == "MODIFIED_TIMBR" else raw_name
                
                all_results.append({
                    'Condition': disease,
                    'Algorithm': display_name,
                    'Spearman_Rho': rho,
                    'P_Value': p_val,
                    'Metabolite_Count': len(common_ids)
                })
                print(f"  Success: {disease} - {display_name} (N={len(common_ids)})")

    if not all_results:
        print("Critical Error: No metabolite matches found across datasets!")
        return

    # Export results to CSV
    df_final = pd.DataFrame(all_results)
    df_final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Results successfully saved to '{OUTPUT_CSV}'.")

    print("--- STEP 2: Generating Visual Report ---")
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Plot A: Discriminatory Rank Heatmap
    pivot_df = df_final.pivot(index="Algorithm", columns="Condition", values="Spearman_Rho")
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", center=0, fmt=".3f", ax=axes[0], cbar_kws={'label': 'Spearman Rho'})
    axes[0].set_title("A: Discriminatory Rank Heatmap\n(|Score| vs. |Log2FC|)", fontweight='bold')

    # Plot B: Performance Distribution
    try:
        pivot_f = df_final.pivot(index='Condition', columns='Algorithm', values='Spearman_Rho')
        if pivot_f.shape[1] >= 3:
            # Friedman test to check if algorithms perform significantly differently
            _, f_p = friedmanchisquare(*[pivot_f[col] for col in pivot_f.columns])
            f_title = f"B: Performance Distribution\n(Friedman p={f_p:.3f})"
        else:
            f_title = "B: Performance Distribution"
    except:
        f_title = "B: Performance Distribution"

    sns.boxplot(data=df_final, x='Algorithm', y='Spearman_Rho', palette='Set2', ax=axes[1])
    sns.swarmplot(data=df_final, x='Algorithm', y='Spearman_Rho', color='.25', size=8, ax=axes[1])
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[1].set_title(f_title, fontweight='bold')

    # Plot C: Global Statistical Reliability (Fisher's Method)
    fisher_list = []
    for algo in df_final['Algorithm'].unique():
        p_vals = df_final[df_final['Algorithm'] == algo]['P_Value'].values
        # Combine p-values across different datasets
        _, combined_p = combine_pvalues(p_vals, method='fisher')
        fisher_list.append({'Algorithm': algo, 'neg_log_p': -np.log10(combined_p)})
    
    df_fisher = pd.DataFrame(fisher_list)
    sns.barplot(data=df_fisher, x='Algorithm', y='neg_log_p', palette='viridis', ax=axes[2])
    axes[2].axhline(-np.log10(0.05), color='red', linestyle=':', label='p=0.05 Threshold')
    axes[2].set_ylabel("Global Confidence (-log10 Combined P)")
    axes[2].set_title("C: Global Statistical Reliability", fontweight='bold')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"🚀 Visual report saved as '{OUTPUT_PLOT}'.")

if __name__ == "__main__":
    run_master_benchmark_with_csv()