import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cobra
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
sns.set_theme(style="whitegrid")
DPI = 300
INPUT_BASE_DIR = "processed_for_analysis"
OUTPUT_DIR = "recon3d_fast_analysis"
DISEASES = ["Breast", "Colon", "Alzheimer"]
N_FEATURES = 20 
COMPARTMENTS = ['_c', '_m', '_e', '_n', '_r', '_g', '_l', '_x']

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 1. Building Fast Mapping Dictionaries ---
def build_fast_maps(model):
    print("-> Indexing Recon3D (Fast Access)...")
    sym_to_id = {}
    id_to_sym = {}
    
    # Gene Mapping
    for gene in model.genes:
        ann = gene.annotation
        symbols = set()
        if 'refseq_name' in ann:
            v = ann['refseq_name']
            symbols.update(v if isinstance(v, list) else [v])
        if 'refseq_synonym' in ann:
            v = ann['refseq_synonym']
            symbols.update(v if isinstance(v, list) else [v])
        symbols.add(gene.name)
        
        for s in symbols:
            if s: sym_to_id[s] = gene.id
        id_to_sym[gene.id] = gene.name # Use gene ID if symbol is not found

    # Metabolite - Reaction - Gene Mapping (Pre-computation)
    met_to_genes = {}
    for met in model.metabolites:
        # Strip compartment suffixes from metabolite ID (e.g., glc_c -> glc)
        base_id = met.id.split('_')[0]
        if base_id not in met_to_genes: met_to_genes[base_id] = set()
        
        for rxn in met.reactions:
            for g in rxn.genes:
                met_to_genes[base_id].add(g.id)

    return sym_to_id, id_to_sym, met_to_genes

# --- 2. Analysis Functions ---

def get_top_features(X, y, n):
    """Uses fast Feature Importance instead of RFE."""
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(n).index.tolist()

def run_pipeline():
    print("Loading Recon3D Model...")
    model = cobra.io.load_json_model("Recon3D.json")
    sym_to_id, id_to_sym, met_to_genes_map = build_fast_maps(model)
    
    results = []

    for disease in DISEASES:
        d_low = disease.lower()
        met_path = os.path.join(INPUT_BASE_DIR, disease, f"processed_{d_low}_metabolomics.csv")
        trn_path = f"{d_low}_transcriptomics.csv"

        if not (os.path.exists(met_path) and os.path.exists(trn_path)):
            print(f"Skipping: {disease}")
            continue

        print(f"\n--- Fast Analysis: {disease} ---")
        
        # Safe Reading and Header Cleaning
        df_met = pd.read_csv(met_path).rename(columns=lambda x: x.strip())
        df_trn = pd.read_csv(trn_path).rename(columns=lambda x: x.strip())

        target = 'Factors' if 'Factors' in df_met.columns else df_met.columns[0]
        y = LabelEncoder().fit_transform(df_met[target].astype(str))
        
        X_met = df_met.drop(columns=[target]).apply(pd.to_numeric, errors='coerce').fillna(0)
        X_trn = df_trn.drop(columns=[target]).apply(pd.to_numeric, errors='coerce').fillna(0)

        # 1. Fast Feature Selection
        top_genes = get_top_features(X_trn, y, N_FEATURES)
        top_mets = get_top_features(X_met, y, N_FEATURES)

        # 2. Recon3D Mapping (Forward/Backward)
        # Forward: Gene -> Met
        fwd_mets = set()
        for g_sym in top_genes:
            gid = sym_to_id.get(g_sym)
            if gid:
                for rxn in model.genes.get_by_id(gid).reactions:
                    for m in rxn.metabolites:
                        # Clean metabolite ID to match the dataset format
                        clean_m = m.id.split('_')[0]
                        fwd_mets.add(clean_m)
        valid_fwd_mets = [m for m in fwd_mets if m in X_met.columns]

        # Backward: Met -> Gene
        bwd_genes = set()
        for m_id in top_mets:
            clean_m = m_id.split('_')[0]
            g_ids = met_to_genes_map.get(clean_m, [])
            for gid in g_ids:
                sym = id_to_sym.get(gid)
                if sym: bwd_genes.add(sym)
        
        valid_bwd_genes = [g for g in bwd_genes if g in X_trn.columns]
        # (Redundant lines from original source kept as requested)
        valid_bwd_genes = [g for g in valid_bwd_genes if g in X_trn.columns]
        valid_bwd_genes = [g for g in valid_bwd_genes if g in X_trn.columns]

        # 3. Scoring
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf_eval = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)

        def quick_auc(X_sub):
            if X_sub.empty or X_sub.shape[1] == 0: return 0.5
            return cross_val_score(rf_eval, X_sub, y, cv=cv, scoring='roc_auc').mean()

        res = {
            "Disease": disease,
            "AUC_Trn_Top20": quick_auc(X_trn[top_genes]),
            "AUC_Met_Mapped": quick_auc(X_met[valid_fwd_mets]),
            "AUC_Met_Top20": quick_auc(X_met[top_mets]),
            "AUC_Trn_Mapped": quick_auc(X_trn[valid_bwd_genes])
        }
        results.append(res)
        print(f"  Finished: {disease} | Mapped AUC (Met): {res['AUC_Met_Mapped']:.2f}")

    # --- 4. Visualization ---
    if not results: return
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(OUTPUT_DIR, "fast_summary.csv"), index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    df_res.melt(id_vars="Disease", value_vars=["AUC_Trn_Top20", "AUC_Met_Mapped"]).pipe(
        (sns.barplot, "data"), x="Disease", y="value", hue="variable", ax=ax1, palette="viridis"
    )
    ax1.set_title("Forward: Trn -> Recon3D -> Met")
    
    df_res.melt(id_vars="Disease", value_vars=["AUC_Met_Top20", "AUC_Trn_Mapped"]).pipe(
        (sns.barplot, "data"), x="Disease", y="value", hue="variable", ax=ax2, palette="magma"
    )
    ax2.set_title("Backward: Met -> Recon3D -> Trn")

    plt.savefig(os.path.join(OUTPUT_DIR, "final_comparison.png"), dpi=DPI)
    print(f"\nProcess completed successfully. Check the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    run_pipeline()