import pandas as pd
import numpy as np
import os
import glob

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold


# =====================
# SETTINGS
# =====================
base_dir = "processed_for_analysis"
diseases = ["Alzheimer", "Breast", "Colon"]

# Metadata columns to be excluded from the analysis
meta_cols = ["Sample ID", "Gender", "Race", "PMI", "Braak"]


# =====================
# HELPER FUNCTIONS
# =====================
def process_algo_file(file_path, algo_name):
    """
    Loads algorithm results (TIMBR, ModTIMBR, TAMBOOR)
    and ranks them based on the absolute value of the scores.
    """
    if not os.path.exists(file_path):
        print(f"  -> {algo_name} file not found, skipping.")
        return None

    df = pd.read_csv(file_path, header=None, names=["Score", "Metabolite"])

    df[f"{algo_name}_Rank"] = (
        df["Score"].abs().rank(ascending=False, method="min")
    )

    return df[["Metabolite", f"{algo_name}_Rank"]]


def get_labels(df, disease):
    """
    Selects the correct label column based on the disease type
    and converts it into a binary (0/1) format.
    """
    if disease == "Alzheimer":
        label_col = "Diagnosis"

        allowed = {"Control", "MCI", "MCI+", "AD", "AD+", "Other"}
        observed = set(df[label_col].astype(str).unique())

        if not observed.issubset(allowed):
            raise ValueError(
                f"Unexpected values found in Alzheimer Diagnosis column: {observed}"
            )

        # Exclude "Other" samples
        df = df[df[label_col] != "Other"].copy()

        # Binary encoding
        df[label_col] = df[label_col].map({
            "Control": 0,
            "MCI": 1,
            "MCI+": 1,
            "AD": 1,
            "AD+": 1
        })

    elif disease in ["Breast", "Colon"]:
        label_col = "Factors"

        allowed = {"healthy", "c"}
        observed = set(df[label_col].astype(str).unique())

        if not observed.issubset(allowed):
            raise ValueError(
                f"Unexpected values found in {disease} Factors column: {observed}"
            )

        df[label_col] = df[label_col].map({
            "healthy": 0,
            "c": 1
        })

    else:
        raise ValueError(f"Unknown disease type: {disease}")

    return df[label_col], label_col, df


# =====================
# MAIN WORKFLOW
# =====================
def main():

    for disease in diseases:
        print(f"\n>>> Starting analysis for: {disease}...")

        disease_path = os.path.join(base_dir, disease)
        if not os.path.exists(disease_path):
            print(f"Error: Directory {disease_path} not found.")
            continue

        # ---------------------
        # 1. Locate Metabolomics File
        # ---------------------
        metab_files = glob.glob(
            os.path.join(disease_path, "processed_*_metabolomics.csv")
        )

        if not metab_files:
            print(f"Warning: Metabolomics file for {disease} not found.")
            continue

        df_metab = pd.read_csv(metab_files[0])
        print(f"  -> Metabolomics file loaded: {os.path.basename(metab_files[0])}")

        # ---------------------
        # 2. Prepare Labels (y) and Features (X)
        # ---------------------
        y, label_col, df_metab = get_labels(df_metab, disease)

        drop_cols = [c for c in meta_cols if c in df_metab.columns]

        X = df_metab.drop(
            columns=drop_cols + [label_col],
            errors="ignore"
        )

        # Drop empty columns
        X = X.dropna(axis=1, how="all")

        # Impute missing values with median
        X = X.fillna(X.median(numeric_only=True))

        print(f"  -> Number of features for analysis: {X.shape[1]}")
        print(f"  -> Class distribution:\n{y.value_counts()}")

        # ---------------------
        # 3. RFECV (Class-aware Cross-Validation)
        # ---------------------
        min_class_size = y.value_counts().min()
        n_splits = min(5, min_class_size)

        if n_splits < 2:
            raise ValueError(
                f"Cannot perform CV for {disease} (minimum class size < 2)"
            )

        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )

        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        rfecv = RFECV(
            estimator=clf,
            step=1,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        rfecv.fit(X, y)

        rfe_ranking = pd.DataFrame({
            "Metabolite": X.columns,
            "RFE_Rank": rfecv.ranking_
        }).sort_values("RFE_Rank")

        print(f"  -> RFECV optimal number of features: {rfecv.n_features_}")

        # ---------------------
        # 4. Load Other Algorithms
        # ---------------------
        timbr = process_algo_file(
            os.path.join(disease_path, "timbr_results.csv"),
            "TIMBR"
        )

        mod_timbr = process_algo_file(
            os.path.join(disease_path, "modified_timbr_results.csv"),
            "ModTIMBR"
        )

        tamboor = process_algo_file(
            os.path.join(disease_path, "tamboor_results.csv"),
            "TAMBOOR"
        )

        # ---------------------
        # 5. Merge Results
        # ---------------------
        final_report = rfe_ranking.copy()

        for algo_df in [mod_timbr, timbr, tamboor]:
            if algo_df is not None:
                final_report = final_report.merge(
                    algo_df,
                    on="Metabolite",
                    how="left"
                )

        # ---------------------
        # 6. Save Report
        # ---------------------
        output_name = f"{disease}_Final_Comparison_Report_Python.csv"
        final_report.to_csv(output_name, index=False)

        print(f"  -> Report generated: {output_name}")

    print("\n--- ALL PROCESSES COMPLETED SUCCESSFULLY ---")


if __name__ == "__main__":
    main()