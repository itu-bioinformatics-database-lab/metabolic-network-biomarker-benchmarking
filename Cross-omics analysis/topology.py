import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.spatial import procrustes
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# --- Research Configuration ---
warnings.filterwarnings('ignore')
sns.set_theme(style="white")
plt.rcParams['figure.dpi'] = 300
DPI = 300

OUTPUT_DIR = "topology_congruence"
for d in ["plots/procrustes", "reports/topology"]: 
    os.makedirs(os.path.join(OUTPUT_DIR, d), exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate"]

def robust_load_numeric(path):
    """Safely extracts features and handles 'Factors' whether index or column."""
    df = pd.read_csv(path)
    # Identify target column
    target_col = 'Factors' if 'Factors' in df.columns else df.columns[0]
    labels = df[target_col]
    # Keep only numeric features
    X = df.select_dtypes(include=[np.number])
    # Drop IDs and target if it was numeric
    X = X.drop(columns=[c for c in X.columns if any(x in c.lower() for x in ["unnamed", "id", "sample", "factors"])], errors='ignore')
    return X, labels

def calculate_rv_coefficient(X, Y):
    """Computes the RV-coefficient: a robust multivariate correlation metric."""
    XXt = np.dot(X, X.T)
    YYt = np.dot(Y, Y.T)
    numerator = np.trace(np.dot(XXt, YYt))
    denominator = np.sqrt(np.trace(np.dot(XXt, XXt)) * np.trace(np.dot(YYt, YYt)))
    return numerator / denominator

def run_systems_biology_pipeline():
    print("Initiating Advanced Systems Biology Analysis...")
    topology_results = []

    for disease in DISEASES:
        print(f"\n>>> ANALYZING TOPOLOGY: {disease.upper()}")
        t_file, m_file = f"{disease}_transcriptomics.csv", f"{disease}_metabolomics.csv"
        if not (os.path.exists(t_file) and os.path.exists(m_file)): continue

        # 1. Load and Align
        X_t, labels = robust_load_numeric(t_file)
        X_m, _ = robust_load_numeric(m_file)
        
        n = min(len(X_t), len(X_m))
        X_t_std = StandardScaler().fit_transform(X_t.iloc[:n])
        X_m_std = StandardScaler().fit_transform(X_m.iloc[:n])
        current_labels = labels.iloc[:n]

        # 2. RV-Coefficient (Global Linkage)
        rv_score = calculate_rv_coefficient(X_t_std, X_m_std)

        # 3. Procrustes Analysis (Shape Congruence)
        # Use SVD to bring both to a 10-dimensional latent space for comparison
        svd = TruncatedSVD(n_components=min(n-1, 10))
        t_latent = svd.fit_transform(X_t_std)
        m_latent = svd.fit_transform(X_m_std)
        
        # Superimpose the two clouds
        mtx1, mtx2, disparity = procrustes(t_latent, m_latent)

        # 4. Plotting (Fixed to avoid label errors)
        plot_procrustes_safe(disease, mtx1, mtx2, current_labels, disparity)

        topology_results.append({
            "Disease": disease,
            "RV_Coefficient": rv_score,
            "Procrustes_Disparity": disparity,
            "Congruence": 1 - disparity
        })

    report_df = pd.DataFrame(topology_results)
    report_df.to_csv(os.path.join(OUTPUT_DIR, "reports/topology/congruence_report.csv"), index=False)
    print(f"\n[SUCCESS] Results saved in: {OUTPUT_DIR}")

def plot_procrustes_safe(disease, m1, m2, labels, d_score):
    """Fixed plotting function to avoid Seaborn's 'multiple values for label' error."""
    plt.figure(figsize=(10, 8))
    
    # We use explicit colors to avoid 'hue' label conflicts
    palette = sns.color_palette("Set1", len(labels.unique()))
    label_map = dict(zip(labels.unique(), palette))
    point_colors = labels.map(label_map)

    # Plot Transcriptome points
    plt.scatter(m1[:,0], m1[:,1], c=point_colors, marker="o", s=80, alpha=0.5, edgecolors='w')
    # Plot Metabolome points
    plt.scatter(m2[:,0], m2[:,1], c=point_colors, marker="X", s=100, alpha=0.5, edgecolors='w')
    
    # Draw linkage lines for each individual
    for i in range(len(m1)):
        plt.plot([m1[i,0], m2[i,0]], [m1[i,1], m2[i,1]], color='gray', alpha=0.2, lw=0.5)

    # Create manual legend to avoid Seaborn conflict
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Transcriptome', markerfacecolor='gray', markersize=10),
                       Line2D([0], [0], marker='X', color='w', label='Metabolome', markerfacecolor='gray', markersize=10)]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f"{disease.upper()} - Omics Topology Congruence\n(Congruence: {1-d_score:.4f})", fontweight='bold')
    plt.xlabel("Latent Coordinate 1"); plt.ylabel("Latent Coordinate 2")
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"plots/procrustes/{disease}_congruence.png"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    run_systems_biology_pipeline()