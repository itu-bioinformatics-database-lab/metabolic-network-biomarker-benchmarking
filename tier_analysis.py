import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Visual Setup ---
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 13
})

# Color palette and name mapping
ALGO_COLORS = {
    'E-TIMBR': '#1D3557',        # Deep Navy
    'Original TIMBR': '#E63946',  # Crimson Red
    'TAMBOOR': '#2A9D8F'          # Teal
}

ALGO_MAP = {
    'ModTIMBR_Rank': 'E-TIMBR',
    'TIMBR_Rank': 'Original TIMBR',
    'TAMBOOR_Rank': 'TAMBOOR'
}

OUTPUT_DIR = "Tiered_Benchmark_Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def define_tier(rank):
    if rank <= 20: return 'Tier 1 (Top 20)'
    elif rank <= 50: return 'Tier 2 (21-50)'
    elif rank <= 100: return 'Tier 3 (51-100)'
    else: return 'Tier 4 (100+)'

def calculate_f1(precision, recall):
    if (precision + recall) == 0: return 0
    return 2 * (precision * recall) / (precision + recall)

def run_tiered_analysis(files):
    all_data = []
    for f in files:
        if os.path.exists(f):
            disease = f.split('_')[0]
            df = pd.read_csv(f)
            df['Disease'] = disease
            all_data.append(df)
        else:
            print(f"Warning: {f} not found, skipping...")
    
    if not all_data:
        print("Error: No data files found! Please check the CSV file names.")
        return

    master_df = pd.concat(all_data, ignore_index=True)
    master_df['Tier'] = master_df['RFE_Rank'].apply(define_tier)
    
    # 1. Displacement Calculation
    for col, name in ALGO_MAP.items():
        master_df[f'{name}_Displacement'] = np.abs(master_df[col] - master_df['RFE_Rank'])

    # 2. Plot: Mean Rank Displacement by Tier
    metrics_melted = master_df.melt(
        id_vars=['Disease', 'Tier'], 
        value_vars=[f'{v}_Displacement' for v in ALGO_MAP.values()],
        var_name='Algorithm', value_name='Displacement'
    )
    metrics_melted['Algorithm'] = metrics_melted['Algorithm'].str.replace('_Displacement', '')

    g = sns.catplot(
        data=metrics_melted[metrics_melted['Tier'] != 'Tier 4 (100+)'], 
        x='Tier', y='Displacement', hue='Algorithm', col='Disease',
        kind='bar', palette=ALGO_COLORS, errorbar='se', height=5, aspect=0.8
    )
    g.set_axis_labels("Metabolite Importance Tier", "Mean Rank Displacement (Lower is Better)")
    g.set_titles("{col_name} Dataset")
    plt.savefig(os.path.join(OUTPUT_DIR, "tier_displacement_analysis.png"), bbox_inches='tight')

    # 3. Plot: Precision, Recall & F1-Score
    diseases = master_df['Disease'].unique()
    K_limit = 100
    x_range = np.arange(1, K_limit + 1)
    
    fig, axes = plt.subplots(3, len(diseases), figsize=(18, 12), squeeze=False)
    
    for i, disease in enumerate(diseases):
        subset = master_df[master_df['Disease'] == disease].sort_values('RFE_Rank')
        
        for algo_col, algo_name in ALGO_MAP.items():
            precisions, recalls, f1s = [], [], []
            for k in x_range:
                top_k_rfe = set(subset.nsmallest(k, 'RFE_Rank')['Metabolite'])
                top_k_algo = set(subset.nsmallest(k, algo_col)['Metabolite'])
                hits = len(top_k_rfe.intersection(top_k_algo))
                
                p = hits / k
                r = hits / K_limit 
                precisions.append(p)
                recalls.append(r)
                f1s.append(calculate_f1(p, r))
            
            axes[0, i].plot(x_range, precisions, color=ALGO_COLORS[algo_name], lw=2, label=algo_name)
            axes[1, i].plot(x_range, recalls, color=ALGO_COLORS[algo_name], lw=2)
            axes[2, i].plot(x_range, f1s, color=ALGO_COLORS[algo_name], lw=2)

        axes[0, i].set_title(f"{disease}\nPrecision @ K")
        axes[1, i].set_title(f"Recall (vs RFE Top {K_limit})")
        axes[2, i].set_title(f"F1-Score")
        
    axes[0, 0].legend(loc='upper right')
    for ax in axes[2, :]: ax.set_xlabel("Top-K Features")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "classification_metrics_curves.png"), bbox_inches='tight')

    # ---------------------------------------------------------
    # 4. NEW SECTION: UPPER LIMIT POSITION ANALYSIS (Written Report)
    # ---------------------------------------------------------
    # Calculates the maximum (worst) ranking value assigned by algorithms within each Tier.
    
    upper_limits = master_df.groupby(['Disease', 'Tier'])[list(ALGO_MAP.keys())].max()
    upper_limits.rename(columns=ALGO_MAP, inplace=True)
    
    print("\n" + "="*70)
    print("      UPPER LIMIT POSITION ANALYSIS (Maximum Rank Values)")
    print("Description: The WORST (highest) rank assigned to a metabolite in each Tier.")
    print("="*70)
    print(upper_limits.to_string())
    print("="*70 + "\n")
    
    upper_limit_path = os.path.join(OUTPUT_DIR, "upper_limit_positions.csv")
    upper_limits.to_csv(upper_limit_path)

    # General performance summary
    summary = master_df.groupby(['Disease', 'Tier'])[[f'{n}_Displacement' for n in ALGO_MAP.values()]].mean()
    summary.to_csv(os.path.join(OUTPUT_DIR, "E-TIMBR_performance_summary.csv"))

    # ---------------------------------------------------------
    # 5. GLOBAL UPPER LIMIT (Highest Value Regardless of Tier)
    # ---------------------------------------------------------
    # The absolute highest rank assigned by algorithms across all metabolites in the dataset.
    
    global_upper_limits = master_df.groupby('Disease')[list(ALGO_MAP.keys())].max()
    global_upper_limits.rename(columns=ALGO_MAP, inplace=True)
    
    print("\n" + "="*70)
    print("      GLOBAL UPPER LIMIT POSITION (Across All Tiers)")
    print("Description: The HIGHEST (worst) rank assigned by each algorithm in the dataset.")
    print("="*70)
    print(global_upper_limits.to_string())
    
    # Global Max values across all diseases
    print("-" * 30)
    print("MAX VALUES ACROSS ENTIRE DATASET:")
    for algo in ALGO_MAP.values():
        max_val = global_upper_limits[algo].max()
        print(f"{algo:15}: {max_val}")
    print("="*70 + "\n")

    # Save as CSV
    global_upper_limits.to_csv(os.path.join(OUTPUT_DIR, "global_upper_limits.csv"))

# Execution
report_files = ["Colon_Comparison_Report.csv", "Breast_Comparison_Report.csv", "Alzheimer_Comparison_Report.csv"]
run_tiered_analysis(report_files)

print(f"Analysis complete. Outputs are saved in the '{OUTPUT_DIR}' directory.")