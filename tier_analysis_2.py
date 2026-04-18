import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
from scipy.stats import ttest_ind

# Set professional plotting standards
sns.set_theme(style="white", font_scale=1.1)
plt.rcParams['figure.dpi'] = 300

def clean_metabolite_name(name):
    """Standardizes names to ensure matching between files."""
    if pd.isna(name): return ""
    name = str(name).strip().lower().replace("'", "").replace('"', "")
    if name.endswith('_e') or name.endswith('_c'):
        name = name[:-2]
    return name

def run_local_tier_analysis():
    # Folder and File Configuration
    base_path = "processed_for_analysis"
    disease_map = {
        'Alzheimer': 'processed_alzheimer_metabolomics.csv',
        'Breast': 'processed_breast_metabolomics.csv',
        'Colon': 'processed_colon_metabolomics.csv'
    }
    algos = ['timbr_results.csv', 'modified_timbr_results.csv', 'tamboor_results.csv']
    
    master_list = []

    print(f"--- Starting Analysis in: {os.path.abspath(base_path)} ---")

    for disease, metab_file in disease_map.items():
        disease_folder = os.path.join(base_path, disease)
        metab_path = os.path.join(disease_folder, metab_file)
        
        if not os.path.exists(metab_path):
            print(f"!!! Error: Could not find {metab_path}")
            continue
        
        # 1. Process Ground Truth (Independent T-Test)
        df_metab = pd.read_csv(metab_path)
        if 'Factors' not in df_metab.columns:
            print(f"!!! Error: 'Factors' column missing in {metab_file}")
            continue
            
        patients = df_metab[df_metab['Factors'].str.lower() == 'c'].drop(columns=['Factors'])
        healthy = df_metab[df_metab['Factors'].str.lower() == 'healthy'].drop(columns=['Factors'])
        
        stats_lookup = {}
        for col in patients.columns:
            _, p_val = ttest_ind(patients[col].dropna(), healthy[col].dropna())
            clean_m = clean_metabolite_name(col)
            stats_lookup[clean_m] = -np.log10(p_val) if p_val > 0 else 0

        # 2. Merge with Algorithm Results
        for algo_file in algos:
            algo_path = os.path.join(disease_folder, algo_file)
            if not os.path.exists(algo_path):
                print(f"  - Missing: {algo_path}")
                continue
                
            df_algo = pd.read_csv(algo_path, header=None)
            df_algo.columns = ['score', 'metab']
            
            # Format Algorithm Names (Upper Case and E-TIMBR alias check)
            raw_algo_name = algo_file.replace('_results.csv', '').upper()
            display_algo = "E-TIMBR" if raw_algo_name == "MODIFIED_TIMBR" else raw_algo_name
            
            for _, row in df_algo.iterrows():
                m_id = clean_metabolite_name(row['metab'])
                m_id_base = m_id[:-2] if (m_id.endswith('_e') or m_id.endswith('_c')) else m_id
                
                if m_id in stats_lookup or m_id_base in stats_lookup:
                    sig_val = stats_lookup.get(m_id, stats_lookup.get(m_id_base))
                    master_list.append({
                        'Disease': disease,
                        'Algorithm': display_algo,
                        'Abs_Score': abs(row['score']),
                        'Significance': sig_val
                    })

    if not master_list:
        print("Analysis Failed: No metabolites matched between files.")
        return

    df_master = pd.DataFrame(master_list)
    df_master.to_csv("tier_analysis_results.csv", index=False)
    print("CSV data saved as: tier_analysis_results.csv")

    # 3. Matrix Plotting
    g = sns.FacetGrid(df_master, row="Disease", col="Algorithm", 
                      height=4, aspect=1.3, margin_titles=True, sharex=False)

    def draw_tier_layers(data, **kwargs):
        ax = plt.gca()
        if data.empty: return

        s_min, s_max = data['Abs_Score'].min(), data['Abs_Score'].max()
        s_range = s_max - s_min
        
        # Define Tiers based on score distribution
        t_plat = s_max - (0.10 * s_range)
        t_gold = s_max - (0.25 * s_range)
        t_silv = s_max - (0.50 * s_range)
        sig_limit = 1.3 # Standard p=0.05 threshold

        ax.set_xlim(s_min * 0.9, s_max * 1.1)
        ax.set_ylim(0, data['Significance'].max() * 1.1)

        sns.scatterplot(data=data, x="Abs_Score", y="Significance", 
                        alpha=0.45, s=40, color="#2c3e50", edgecolor=None)

        # Apply Layer Shading
        ax.axvspan(t_plat, ax.get_xlim()[1], ymin=sig_limit/ax.get_ylim()[1], ymax=1, color='#1b5e20', alpha=0.3)
        ax.axvspan(t_gold, t_plat, ymin=sig_limit/ax.get_ylim()[1], ymax=1, color='#4caf50', alpha=0.2)
        ax.axvspan(t_silv, t_gold, ymin=sig_limit/ax.get_ylim()[1], ymax=1, color='#81c784', alpha=0.1)
        ax.axhline(sig_limit, color='#e74c3c', linestyle='--', linewidth=1.5)

    g.map_dataframe(draw_tier_layers)
    
    # Custom Legend Configuration
    legend_elements = [
        Line2D([0], [0], color='#e74c3c', lw=2, linestyle='--', label='Significance Limit ($p < 0.05$)'),
        mpatches.Patch(color='#1b5e20', alpha=0.4, label='Platinum Tier (Top 10% Range)'),
        mpatches.Patch(color='#4caf50', alpha=0.3, label='Gold Tier (10-25% Range)'),
        mpatches.Patch(color='#81c784', alpha=0.2, label='Silver Tier (25-50% Range)'),
        Line2D([0], [0], marker='o', color='w', label='Metabolites', 
               markerfacecolor='#2c3e50', markersize=8, alpha=0.6)
    ]

    g.fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                  bbox_to_anchor=(0.5, -0.08), frameon=True, fontsize=12)

    g.set_axis_labels("Predictive Modeling (Abs Score)", "Clinical Significance ($-\log_{10} p$)")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    
    # Adjust layout to accommodate the legend
    plt.subplots_adjust(top=1.0, bottom=0.15)

    # Save visualization to the working directory
    plt.savefig("tier_analysis_plot.png", bbox_inches='tight', dpi=300)
    print("Matrix plot saved successfully as: tier_analysis_plot.png")

if __name__ == "__main__":
    run_local_tier_analysis()