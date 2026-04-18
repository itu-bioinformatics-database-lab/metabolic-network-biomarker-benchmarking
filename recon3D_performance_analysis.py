import pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
CSV_PATH = "recon3d_fast_analysis/fast_summary.csv"
OUTPUT_PLOT = "recon3d_performance_report.png"
DPI = 300

# Style Settings (Whitegrid background, professional font scaling)
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({'figure.dpi': DPI})

def create_comparison_plots():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found!")
        return

    df = pd.read_csv(CSV_PATH)

    # Figure initialization
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # --- PLOT 1: Forward Mapping (Trn -> Recon3D -> Met) ---
    df_fwd = df.melt(id_vars="Disease", 
                    value_vars=["AUC_Trn_Top20", "AUC_Met_Mapped"],
                    var_name="Type", value_name="AUC")
    
    df_fwd["Type"] = df_fwd["Type"].map({
        "AUC_Trn_Top20": "SOURCE: TOP-20 TRANSCRIPTOMICS",
        "AUC_Met_Mapped": "INFERRED: METABOLOMICS (VIA RECON3D)"
    })

    sns.barplot(data=df_fwd, x="Disease", y="AUC", hue="Type", ax=axes[0], palette="viridis")
    
    # Title formatting (keeping fontweight normal as requested)
    axes[0].set_title("Forward Inference Performance\n(Transcriptomics to Metabolomics)", fontweight='normal', pad=15)
    axes[0].set_ylabel("Predictive Performance (ROC-AUC)")
    axes[0].set_xlabel("Biological Condition")
    axes[0].set_ylim(0.4, 1.1)
    
    # Position legend outside/center to prevent overlapping data bars
    axes[0].legend(title="Data Stream", loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True)

    # --- PLOT 2: Backward Mapping (Met -> Recon3D -> Trn) ---
    df_bwd = df.melt(id_vars="Disease", 
                    value_vars=["AUC_Met_Top20", "AUC_Trn_Mapped"],
                    var_name="Type", value_name="AUC")
    
    df_bwd["Type"] = df_bwd["Type"].map({
        "AUC_Met_Top20": "SOURCE: TOP-20 METABOLOMICS",
        "AUC_Trn_Mapped": "INFERRED: TRANSCRIPTOMICS (VIA RECON3D)"
    })

    sns.barplot(data=df_bwd, x="Disease", y="AUC", hue="Type", ax=axes[1], palette="magma")
    
    axes[1].set_title("Backward Inference Performance\n(Metabolomics to Transcriptomics)", fontweight='normal', pad=15)
    axes[1].set_ylabel("Predictive Performance (ROC-AUC)")
    axes[1].set_xlabel("Biological Condition")
    axes[1].set_ylim(0.4, 1.1)
    
    # Move legend to the bottom center
    axes[1].legend(title="Data Stream", loc='upper center', bbox_to_anchor=(0.5, -0.15), frameon=True)

    # Adjust layout to accommodate external legends and prevent clipping
    plt.tight_layout()
    
    # Save output
    plt.savefig(OUTPUT_PLOT, bbox_inches='tight', dpi=DPI)
    print(f"✅ Visualization complete: {OUTPUT_PLOT}")

if __name__ == "__main__":
    create_comparison_plots()