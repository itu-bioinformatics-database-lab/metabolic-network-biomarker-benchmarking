import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score, f1_score, accuracy_score
from scipy import stats

# --- Configuration & Aesthetics ---
sns.set_theme(style="whitegrid")
COLORS = ["#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"]
DPI = 300
OUTPUT_DIR = "analysis_results"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
DATA_DIR = os.path.join(OUTPUT_DIR, "tables")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

DISEASES = ["breast", "ccrcc3", "ccrcc4", "colon", "pdac", "prostate", "alzheimer"]
DATA_TYPES = ["transcriptomics", "metabolomics"]

def tiered_rfe(X, y, estimator, disease, dtype):
    """
    Performs coarse-to-fine RFE: Step 100 -> 10 -> 1.
    """
    print(f"  > Running Tiered RFE for {disease} ({dtype})...")
    
    # Tier 1: Coarse (Step 100)
    selector = RFE(estimator, n_features_to_select=min(500, X.shape[1]), step=100)
    X_tier1 = selector.fit_transform(X, y)
    
    # Tier 2: Medium (Step 10)
    selector = RFE(estimator, n_features_to_select=min(100, X_tier1.shape[1]), step=10)
    X_tier2 = selector.fit_transform(X_tier1, y)
    
    # Tier 3: Fine (Step 1)
    selector = RFE(estimator, n_features_to_select=min(20, X_tier2.shape[1]), step=1)
    selector.fit(X_tier2, y)
    
    # Map back to original feature names
    # Keeping your original placeholder mapping logic exactly as requested
    ranking = RFE(estimator, n_features_to_select=20, step=100).fit(X, y)
    selected_features = X.columns[ranking.support_].tolist()
    
    return selected_features

def run_pipeline():
    all_results = []
    
    for disease in DISEASES:
        print(f"\nProcessing Disease: {disease.upper()}")
        data_dict = {}
        
        # 1. Load Data
        for dtype in DATA_TYPES:
            fname = f"{disease}_{dtype}.csv"
            if not os.path.exists(fname):
                print(f"    ! File {fname} not found. Skipping.")
                continue
            
            # FIXED: index_index corrected to index_col
            df = pd.read_csv(fname, index_col=0) if "index" in pd.read_csv(fname, nrows=1).columns else pd.read_csv(fname)
            
            X = df.drop(columns=['Factors']) if 'Factors' in df.columns else df.iloc[:, :-1]
            y_raw = df['Factors'] if 'Factors' in df.columns else df.iloc[:, -1]
            
            # FIXED: Map string labels to numeric for XGBoost compatibility
            y = y_raw.map({'healthy': 0, 'c': 1})
            
            data_dict[dtype] = (X, y)

        if not data_dict: continue

        # 2. RFE and Multi-Omics Join
        rfe_results = {}
        for dtype, (X, y) in data_dict.items():
            features = tiered_rfe(X, y, RandomForestClassifier(n_estimators=50), disease, dtype)
            rfe_results[dtype] = features
            
            # Save Feature List
            with open(os.path.join(DATA_DIR, f"{disease}_{dtype}_rfe_features.txt"), "w") as f:
                f.write("\n".join(features))

        # Create Joint Dataset
        if len(data_dict) == 2:
            X_t, y_t = data_dict["transcriptomics"]
            X_m, y_m = data_dict["metabolomics"]
            
            # Check if row counts (samples) match exactly
            if len(X_t) == len(X_m):
                t_features = rfe_results["transcriptomics"]
                m_features = rfe_results["metabolomics"]
                
                # Concatenate only if sizes are consistent
                X_joint = pd.concat([X_t[t_features], X_m[m_features]], axis=1)
                y_joint = y_t
                
                data_dict["joint"] = (X_joint, y_joint)
                rfe_results["joint"] = t_features + m_features
                print(f"  > {disease.upper()}: Samples match. Joint model created.")
            else:
                # Skip joint for Alzheimer's or any inconsistent data
                print(f"  ! Warning: {disease.upper()} sample counts differ (T:{len(X_t)}, M:{len(X_m)}). Skipping Joint analysis.")

        # 3. Classification with Cross-Validation
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100),
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        }
        
        disease_metrics = []

        for dtype, (X, y) in data_dict.items():
            X_sub = X[rfe_results[dtype]]
            
            for model_name, model in models.items():
                scoring = ['accuracy', 'f1_weighted', 'roc_auc']
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_validate(model, X_sub, y, cv=cv, scoring=scoring)
                
                res = {
                    "Disease": disease,
                    "DataType": dtype,
                    "Model": model_name,
                    "Accuracy_CV": scores['test_accuracy'].mean(),
                    "F1_CV": scores['test_f1_weighted'].mean(),
                    "AUC_CV": scores['test_roc_auc'].mean()
                }
                all_results.append(res)
                disease_metrics.append(res)

        # 4. Disease-Specific Visualization
        plot_disease_performance(disease, pd.DataFrame(disease_metrics))

    # 5. Global Analysis & Statistics
    final_df = pd.DataFrame(all_results)
    final_df.to_csv(os.path.join(DATA_DIR, "all_classification_results.csv"), index=False)
    
    plot_global_comparisons(final_df)
    perform_statistical_analysis(final_df)

def plot_disease_performance(disease, df):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x="Model", y="AUC_CV", hue="DataType", palette=COLORS)
    plt.title(f"Model Performance (AUC CV) - {disease.capitalize()}", fontsize=15)
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{disease}_performance_comparison.png"), dpi=DPI)
    plt.close()

def plot_global_comparisons(df):
    plt.figure(figsize=(16, 8))
    sns.boxplot(data=df, x="Disease", y="AUC_CV", hue="DataType", palette="viridis")
    plt.title("Global Multi-Omics Performance Comparison (AUC with CV)", fontsize=18)
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(PLOTS_DIR, "global_disease_comparison.png"), dpi=DPI)
    plt.close()

def perform_statistical_analysis(df):
    summary_path = os.path.join(DATA_DIR, "statistical_analysis_report.txt")
    with open(summary_path, "w") as f:
        f.write("STATISTICAL ANALYSIS REPORT\n" + "="*30 + "\n\n")
        
        # Compare Transcriptomics vs Metabolomics vs Joint
        for metric in ["AUC_CV", "F1_CV"]:
            f.write(f"--- Analysis for {metric} ---\n")
            unique_types = df["DataType"].unique()
            groups = [df[df["DataType"] == t][metric] for t in unique_types]
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                f.write(f"One-way ANOVA p-value: {p_val:.6f}\n\n")
            
    # Plotting ANOVA results (Restored pointplot)
    plt.figure(figsize=(10, 6))
    sns.pointplot(data=df, x="DataType", y="AUC_CV", capsize=.2, color=COLORS[4])
    plt.title("Statistical Confidence Intervals per Data Type (AUC CV)")
    plt.savefig(os.path.join(PLOTS_DIR, "statistical_significance.png"), dpi=DPI)
    plt.close()

if __name__ == "__main__":
    run_pipeline()