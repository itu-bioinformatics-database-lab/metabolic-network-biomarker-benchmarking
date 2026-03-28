import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

# --- Configuration & Aesthetics ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

# Folder management
OUTPUT_DIR = "individual_omics_research"
SUBDIRS = ["plots/similarity", "reports/statistics"]
for sd in SUBDIRS: 
    os.makedirs(os.path.join(OUTPUT_DIR, sd), exist_ok=True)

# Dataset configuration (excluding Alzheimer's as discussed)
DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def load_and_preprocess(path):
    """Identifies the 'Factors' column and separates numerical features."""
    df = pd.read_csv(path)
    # Detect target column (usually 'Factors')
    label_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    labels = df[label_col]
    
    # Isolate numerical data only
    X = df.select_dtypes(include=[np.number])
    # Drop potential ID or index columns
    cols_to_drop = [c for c in X.columns if "Unnamed" in c or "Sample" in c or "ID" in c]
    X = X.drop(columns=cols_to_drop)
    return X, labels

def run_individual_relationship_analysis():
    print("Starting Individual Multi-Omics Relationship Analysis...")
    global_summary_data = []

    for disease in DISEASES:
        print(f"\n>>> PROCESSING: {disease.upper()}")
        t_file = f"{disease}_transcriptomics.csv"
        m_file = f"{disease}_metabolomics.csv"
        
        if not (os.path.exists(t_path := t_file) and os.path.exists(m_path := m_file)):
            print(f"      ! Missing files for {disease}. Skipping.")
            continue

        # 1. Load and Align Samples
        X_t, labels = load_and_preprocess(t_file)
        X_m, _ = load_and_preprocess(m_file)

        # Force sample alignment (assuming rows match as confirmed)
        min_n = min(len(X_t), len(X_m))
        X_t, X_m = X_t.iloc[:min_n], X_m.iloc[:min_n]
        labels = labels.iloc[:min_n]

        # 2. Dimensionality Neutralization (Distance Matrices)
        # We transform 16k genes and 500 metabolites into comparable N x N distance maps
        X_t_std = StandardScaler().fit_transform(X_t)
        X_m_std = StandardScaler().fit_transform(X_m)
        
        dist_t = pairwise_distances(X_t_std, metric='euclidean')
        dist_m = pairwise_distances(X_m_std, metric='euclidean')

        # 3. Individual Synchrony Calculation
        # Measures how a person's transcriptome 'neighborhood' matches their metabolic 'neighborhood'
        synchrony_scores = []
        for i in range(len(dist_t)):
            rho, _ = spearmanr(dist_t[i], dist_m[i])
            synchrony_scores.append(rho)

        # 4. Omics Discordance Calculation
        # Measures the absolute delta between the two omic layers
        discordance = np.abs(dist_t - dist_m).mean(axis=1)

        # 5. Plotting and Reporting
        plot_synchrony_distribution(disease, synchrony_scores, labels)
        plot_discordance_heatmap(disease, dist_t, dist_m)
        
        global_summary_data.append({
            "Disease": disease,
            "Avg_Synchrony": np.mean(synchrony_scores),
            "Discordance_Std": np.std(discordance)
        })

    # Save Global Summary
    summary_df = pd.DataFrame(global_summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "reports/statistics/global_summary.csv"), index=False)
    print(f"\n[SUCCESS] Analysis complete. Results in: {OUTPUT_DIR}")

def plot_synchrony_distribution(disease, scores, labels):
    plt.figure(figsize=(10, 6))
    df_plot = pd.DataFrame({"Synchrony": scores, "Status": labels})
    sns.violinplot(data=df_plot, x="Status", y="Synchrony", palette="coolwarm", inner="quart")
    sns.stripplot(data=df_plot, x="Status", y="Synchrony", color="black", alpha=0.3, size=4)
    plt.title(f"{disease.upper()} - Individual Omics Synchrony\n(Transcriptome vs. Metabolome Profile Consistency)")
    plt.ylabel("Synchrony Score (Spearman ρ)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/similarity/{disease}_synchrony_score.png"))
    plt.close()

def plot_discordance_heatmap(disease, dist_t, dist_m):
    plt.figure(figsize=(8, 6))
    # Calculate absolute difference between distance maps
    discordance_matrix = np.abs(dist_t - dist_m)
    sns.heatmap(discordance_matrix, cmap="YlOrRd", xticklabels=False, yticklabels=False)
    plt.title(f"{disease.upper()} - Omics Discordance Map\n(Red = High Contradiction Between Layers)")
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/similarity/{disease}_discordance_map.png"))
    plt.close()

if __name__ == "__main__":
    run_individual_relationship_analysis()