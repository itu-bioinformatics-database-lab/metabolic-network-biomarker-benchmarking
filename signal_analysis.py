import os
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, ttest_ind
import warnings

# Suppress mathematical warnings (e.g., log10 of zero)
warnings.filterwarnings('ignore')
sns.set_theme(style="white", font_scale=1.2)

# --- Configuration ---
BASE_DIR = "processed_for_analysis"
DISEASE_MAP = {
    'Alzheimer': 'processed_alzheimer_metabolomics.csv',
    'Breast': 'processed_breast_metabolomics.csv',
    'Colon': 'processed_colon_metabolomics.csv'
}
ALGO_MAPPING = {
    'modified_timbr_results.csv': 'E-TIMBR',
    'tamboor_results.csv': 'TAMBOOR',
    'timbr_results.csv': 'TIMBR'
}
OUTPUT_PLOT = "interpretability_vs_crushing_report_300dpi.png"
OUTPUT_CSV = "interpretability_final_metrics.csv"

def gini(x):
    """
    Calculates the Gini Coefficient as a measure of score inequality.
    1.0 = Extreme variance crushing (signal concentrated in few metabolites).
    0.0 = Equal distribution of scores.
    """
    if len(x) == 0: return 0
    x = np.array(x)
    x = x[x > 0] # Consider only positive scores
    if len(x) == 0: return 1.0
    sorted_x = np.sort(x)
    n = len(x)
    cum_x = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cum_x) / cum_x[-1]) / n

def clean_name(name):
    """Standardizes metabolite names and strips compartment suffixes."""
    if pd.isna(name): return ""
    name = str(name).strip().lower().replace("'", "").replace('"', "")
    if name.endswith('_e') or name.endswith('_c'): name = name[:-2]
    return name

def run_updated_benchmark():
    final_data = []
    distribution_data = []

    print("--- STEP 1: Gini Coefficient and Signal Analysis Started ---")
    
    for disease, metab_file in DISEASE_MAP.items():
        d_path = os.path.join(BASE_DIR, disease)
        m_path = os.path.join(d_path, metab_file)
        if not os.path.exists(m_path):
            print(f"Warning: {m_path} not found. Skipping...")
            continue

        # Establish Biological Ground Truth (Importance via T-test p-values)
        df_metab = pd.read_csv(m_path)
        is_p = df_metab['Factors'].str.lower() == 'c' # Case / Patient
        is_h = df_metab['Factors'].str.lower() == 'healthy' # Healthy Control
        
        sig_truth = {}
        for col in df_metab.columns:
            if col == 'Factors': continue
            # Calculate negative log p-values as the ground truth signal
            _, p = ttest_ind(df_metab[is_p][col].dropna(), df_metab[is_h][col].dropna())
            sig_truth[clean_name(col)] = -np.log10(p) if p > 0 else 0

        for f_name, d_name in ALGO_MAPPING.items():
            a_path = os.path.join(d_path, f_name)
            if not os.path.exists(a_path): continue
            
            df_a = pd.read_csv(a_path, header=None, names=['score', 'metab'])
            df_a['abs_score'] = df_a['score'].abs()
            df_a['clean_id'] = df_a['metab'].apply(clean_name)
            
            # Prepare log-scores for distribution visualization
            temp_dist = df_a[['abs_score']].copy()
            temp_dist['Algorithm'] = d_name
            temp_dist['Condition'] = disease
            distribution_data.append(temp_dist)

            # Performance Metrics
            gini_val = gini(df_a['abs_score'].values)
            
            # Signal Capture Assessment (Correlation with Biological Truth)
            algo_ranks = df_a.groupby('clean_id')['abs_score'].max()
            common = list(set(sig_truth.keys()) & set(algo_ranks.index))
            rho, _ = spearmanr([sig_truth[m] for m in common], algo_ranks.loc[common])

            final_data.append({
                'Condition': disease,
                'Algorithm': d_name,
                'Signal_Capture_Rho': rho,
                'Variance_Crushing_Gini': gini_val,
                'Dynamic_Range_Log10': np.log10(df_a['abs_score'].max() / (df_a['abs_score'].min() + 1e-12))
            })

    # Consolidate and Save Results
    df_res = pd.DataFrame(final_data)
    df_dist = pd.concat(distribution_data)
    df_res.to_csv(OUTPUT_CSV, index=False)

    # --- STEP 2: Visualization and Reporting ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Plot A: Signal Capture Efficiency
    sns.barplot(data=df_res, x='Algorithm', y='Signal_Capture_Rho', hue='Condition', ax=axes[0], palette='rocket')
    axes[0].set_title("A: Signal Capture Efficiency\n(Spearman Rho Correlation)", fontweight='bold')
    axes[0].set_ylim(df_res['Signal_Capture_Rho'].min()-0.05, df_res['Signal_Capture_Rho'].max()+0.1)

    # Plot B: Variance Crushing (Gini Index)
    sns.barplot(data=df_res, x='Algorithm', y='Variance_Crushing_Gini', hue='Condition', ax=axes[1], palette='magma')
    axes[1].set_title("B: Variance Crushing (Gini Index)\n1.0 = High Inequality / Low Interpretability", fontweight='bold')
    axes[1].set_ylim(0, 1.1)

    # Plot C: Score Compression Analysis
    df_dist['log_score'] = np.log10(df_dist['abs_score'] + 1e-12)
    sns.boxplot(data=df_dist, x='Algorithm', y='log_score', palette='Set2', ax=axes[2])
    axes[2].set_title("C: Score Compression (Log-Scale)\nGap indicates Model Constraint Impact", fontweight='bold')
    axes[2].set_ylabel("Log10(Absolute Score)")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"--- Analysis Complete! Report generated: {OUTPUT_PLOT} ---")

if __name__ == "__main__":
    run_updated_benchmark()