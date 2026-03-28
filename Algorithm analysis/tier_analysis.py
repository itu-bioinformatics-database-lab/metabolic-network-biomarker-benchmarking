import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cobra
import mygene
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy import stats

# --- Configuration & Styling ---
sns.set_theme(style="whitegrid")
COLORS = ["#1A5276", "#BA4A00", "#1E8449", "#7D3C98", "#2E4053"]
DPI = 300
OUTPUT_DIR = "tier_analysis_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate", "alzheimer"]
COMPARTMENTS = ['_c', '_m', '_e', '_n', '_r', '_g', '_l', '_x']

# --- 1. Mapping Helpers ---

def get_gene_symbol_map(model_genes):
    """
    Maps Recon3D gene IDs (e.g., 280_AT1) to Gene Symbols (e.g., ADORA2A).
    """
    print("Mapping gene IDs (Entrez ID -> Gene Symbol)...")
    mg = mygene.MyGeneInfo()
    
    # Extract Entrez ID by splitting suffixes (280_AT1 -> 280)
    gene_id_list = [g.id.split('_')[0] for g in model_genes]
    
    results = mg.querymany(gene_id_list, scopes='entrezgene', fields='symbol', species='human', verbose=False)
    
    mapping_dict = {}
    for res in results:
        if 'symbol' in res:
            mapping_dict[res['query']] = res['symbol']
            
    return mapping_dict

def find_metabolite_smart(model, query, compartments):
    """
    Searches for a metabolite by ID, compartment-specific ID, or common name.
    """
    query = str(query).strip()
    
    # 1. Direct ID match
    try:
        return model.metabolites.get_by_id(query)
    except KeyError:
        pass

    # 2. Compartment suffix match
    for comp in compartments:
        try:
            return model.metabolites.get_by_id(f"{query}{comp}")
        except KeyError:
            continue

    # 3. Case-insensitive Name or ID match
    query_lower = query.lower()
    for met in model.metabolites:
        if query_lower == met.id.lower() or query_lower == met.name.lower():
            return met
            
    return None

def get_metabolite_derived_genes(model, metabolite_list, gene_map):
    """
    Identifies genes associated with reactions producing the specified metabolites.
    """
    derived_genes = set()
    
    for met_id in metabolite_list:
        target_met = find_metabolite_smart(model, met_id, COMPARTMENTS)
        
        if target_met:
            # Filter reactions where the metabolite is a product (stoichiometry > 0)
            producing_rxns = [r for r in target_met.reactions if r.metabolites[target_met] > 0]
            for rxn in producing_rxns:
                for g in rxn.genes:
                    clean_id = g.id.split('_')[0]
                    symbol = gene_map.get(clean_id)
                    if symbol:
                        derived_genes.add(symbol)
                        
    return list(derived_genes)

# --- 2. Tier Analysis & Benchmarking ---

def run_tier_analysis():
    print("Loading Recon3D Model...")
    try:
        model = cobra.io.load_json_model("Recon3D.json")
        print("Model loaded successfully.")
        gene_map = get_gene_symbol_map(model.genes)
        print("Gene mapping initialized.\n")
    except Exception as e:
        print(f"Initialization Error: {e}")
        return

    global_comparison_data = []

    for disease in DISEASES:
        print(f"Analyzing {disease.upper()}...")
        
        # Path configuration
        rfe_met_path = f"analysis_results/tables/{disease}_metabolomics_rfe_features.txt"
        trans_data_path = f"{disease}_transcriptomics.csv"
        
        if not (os.path.exists(rfe_met_path) and os.path.exists(trans_data_path)):
            print(f"  Warning: Required files missing for {disease}. Skipping.")
            continue

        # Get Top Metabolites from RFE
        with open(rfe_met_path, "r") as f:
            top_metabolites = [line.strip() for line in f.readlines()]
        
        # Map Metabolites to Genes
        derived_gene_symbols = get_metabolite_derived_genes(model, top_metabolites, gene_map)
        
        # Load Transcriptomics Data
        df_trans = pd.read_csv(trans_data_path)
        X = df_trans.drop(columns=['Factors'])
        y = df_trans['Factors']
        
        # Feature Importance (Baseline)
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # Filter genes present in the transcriptomics dataset
        valid_derived_genes = [g for g in derived_gene_symbols if g in X.columns]
        
        # Validation: Skip if no matching genes are found to prevent statistical errors
        if len(valid_derived_genes) == 0:
            print(f"  Warning: No overlapping genes found for {disease}. Skipping statistical analysis.")
            continue

        derived_importances = importances[valid_derived_genes]
        other_importances = importances[~importances.index.isin(valid_derived_genes)]

        # --- Tier Analysis Statistics ---
        u_stat, p_val = stats.mannwhitneyu(derived_importances, other_importances, alternative='greater')
        
        # --- Benchmarking Performance ---
        cv = StratifiedKFold(n_splits=5)
        
        # Performance with Derived Gene Subset
        score_derived = cross_val_score(rf, X[valid_derived_genes], y, cv=cv, scoring='roc_auc').mean()
        
        # Performance with a Random Subset of the same size
        random_genes = np.random.choice(X.columns, size=len(valid_derived_genes), replace=False)
        score_random = cross_val_score(rf, X[random_genes], y, cv=cv, scoring='roc_auc').mean()
        
        disease_results = {
            "Disease": disease,
            "Derived_Count": len(valid_derived_genes),
            "Avg_Importance_Derived": derived_importances.mean(),
            "Avg_Importance_Other": other_importances.mean(),
            "P_Value_Significance": p_val,
            "AUC_Derived_Subset": score_derived,
            "AUC_Random_Subset": score_random
        }
        global_comparison_data.append(disease_results)

        # Visualizations & Reporting
        plot_individual_tier(disease, importances, valid_derived_genes)
        save_disease_report(disease, disease_results, valid_derived_genes)
        print(f"  Analysis complete for {disease}. AUC Ratio: {score_derived/score_random:.2f}")

    # Global Summarization
    if global_comparison_data:
        global_df = pd.DataFrame(global_comparison_data)
        global_df.to_csv(os.path.join(REPORTS_DIR, "global_tier_analysis_summary.csv"), index=False)
        plot_global_benchmarks(global_df)
        print("\nAll processes finished. Summary report generated.")
    else:
        print("\nNo valid data was processed.")

def plot_individual_tier(disease, all_importances, derived_genes):
    plt.figure(figsize=(10, 6))
    sns.kdeplot(all_importances, fill=True, label="All Genes", color="gray", alpha=0.3)
    sns.rugplot(all_importances[derived_genes], color="red", label="Metabolite-Derived Genes", alpha=0.8)
    plt.title(f"Predictive Importance Distribution: {disease.upper()}", fontsize=14)
    plt.xlabel("Random Forest Feature Importance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{disease}_tier_distribution.png"), dpi=DPI)
    plt.close()

def plot_global_benchmarks(df):
    df_melt = df.melt(id_vars="Disease", value_vars=["AUC_Derived_Subset", "AUC_Random_Subset"], 
                      var_name="Group", value_name="AUC")
    
    plt.figure(figsize=(14, 7))
    sns.barplot(data=df_melt, x="Disease", y="AUC", hue="Group", palette="mako")
    plt.title("Benchmarking Metabolite-Derived Gene Subsets", fontsize=16)
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, "global_subset_advantage_comparison.png"), dpi=DPI)
    plt.close()

def save_disease_report(disease, results, genes):
    report_path = os.path.join(REPORTS_DIR, f"{disease}_mapping_report.txt")
    with open(report_path, "w") as f:
        f.write(f"REVERSE MAPPING REPORT: {disease.upper()}\n" + "="*40 + "\n")
        f.write(f"Genes identified via Recon3D: {len(genes)}\n")
        f.write(f"Statistical Enrichment (p-value): {results['P_Value_Significance']:.6e}\n")
        f.write(f"AUC (Derived Genes): {results['AUC_Derived_Subset']:.4f}\n")
        f.write(f"AUC (Random Subset): {results['AUC_Random_Subset']:.4f}\n\n")
        f.write("Mapped Gene Symbols:\n" + ", ".join(genes))

if __name__ == "__main__":
    run_tier_analysis()