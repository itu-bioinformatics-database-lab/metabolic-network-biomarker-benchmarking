import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def create_clinical_plot(ax, file_path, disease_name):
    # 1. Load and Clean Data
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Log2FoldChange', 'HazardRatio'])
    
    # 2. Refined Logical Categorization
    def get_category(row):
        l2fc, hr = row['Log2FoldChange'], row['HazardRatio']
        if l2fc > 0 and hr > 1: return 'Poor Prognosis Oncogene (UP & HR > 1)'
        if l2fc > 0 and hr < 1: return 'Favorable Biomarker (UP & HR < 1)'
        if l2fc < 0 and hr < 1: return 'Protective Profile (DOWN & HR < 1)'
        if l2fc < 0 and hr > 1: return 'Tumor Suppressor Loss (DOWN & HR > 1)'
        return 'Neutral'

    df['Clinical_Status'] = df.apply(get_category, axis=1)
    
    # 3. Stats
    corr, pval = spearmanr(df['Log2FoldChange'], df['HazardRatio'])
    
    # 4. Aesthetic Palette
    palette = {
        'Poor Prognosis Oncogene (UP & HR > 1)': '#d62728', # Red
        'Favorable Biomarker (UP & HR < 1)': '#2ca02c',    # Green
        'Protective Profile (DOWN & HR < 1)': '#1f77b4',   # Blue
        'Tumor Suppressor Loss (DOWN & HR > 1)': '#ff7f0e', # Orange
        'Neutral': '#bcbd22'
    }

    # 5. Plotting
    sns.scatterplot(
        data=df, x='Log2FoldChange', y='HazardRatio', 
        hue='Clinical_Status', palette=palette, s=150, 
        edgecolor='black', alpha=0.8, ax=ax, zorder=3
    )

    # Threshold Lines
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax.axhline(1, color='red', linestyle='--', linewidth=1.2, alpha=0.6)

    # 6. Improved Gene Labeling (Readable)
    for i in range(df.shape[0]):
        ax.text(
            df.Log2FoldChange.iloc[i], 
            df.HazardRatio.iloc[i] + 0.03, 
            df.Gene.iloc[i], 
            fontsize=9, weight='bold', ha='center',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
        )

    # Formatting
    ax.set_title(f"{disease_name}\nSpearman Rho: {corr:.2f} (p={pval:.4f})", fontsize=15, pad=20)
    ax.set_xlabel('Expression Difference (Log2 Fold Change)', fontsize=12)
    ax.set_ylabel('Clinical Risk (Hazard Ratio)', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(title='Biological/Clinical Profile', loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=1)

# --- Execution ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))

# Generate the two plots
create_clinical_plot(ax1, 'breast_log2fc_HR.csv', 'Breast Cancer: Expression vs. Prognosis')
create_clinical_plot(ax2, 'colon_log2fc_HR.csv', 'Colon Cancer: Expression vs. Prognosis')

# Final adjustments for 300 DPI save
plt.tight_layout()
plt.subplots_adjust(bottom=0.25) # Space for the bottom legends
plt.savefig('combined_clinical_analysis_300dpi.png', dpi=300, bbox_inches='tight')

print("Success: Side-by-side high-resolution plots saved.")
plt.show()