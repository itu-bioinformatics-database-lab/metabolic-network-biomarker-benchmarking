import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_regression

# --- Configuration ---
BASE_DIR = "processed_for_analysis"
CONDITIONS = ["Alzheimer", "Breast", "Colon"]
ALGORITHMS = ["timbr", "tamboor", "modified_timbr"]

def calculate_structural_metrics(algo_scores, raw_transcripts):
    # 1. Entropy
    prob_dist = np.abs(algo_scores) + 1e-12
    prob_dist /= np.sum(prob_dist)
    output_entropy = entropy(prob_dist)

    # 2. Dynamic Range
    # Adding a small epsilon to prevent division by zero
    min_val = np.min(algo_scores)
    max_val = np.max(algo_scores)
    dynamic_range = np.log10((max_val + 1e-15) / (min_val + 1e-15))

    # 3. Transcript Coupling (MI)
    input_signal = raw_transcripts.mean(axis=1).values
    min_len = min(len(input_signal), len(algo_scores))
    
    try:
        # Reshaping data into the correct format for regression
        mi_coupling = mutual_info_regression(
            algo_scores[:min_len].reshape(-1, 1), 
            input_signal[:min_len]
        )[0]
    except Exception as e:
        print(f"  [!] MI Calculation error: {e}")
        mi_coupling = 0

    return {
        "Entropy": output_entropy,
        "Dynamic_Range": dynamic_range,
        "GPR_Coupling_MI": mi_coupling
    }

def run_internal_benchmark():
    internal_bench_results = []

    # Directory check
    if not os.path.exists(BASE_DIR):
        print(f"Error: {BASE_DIR} directory not found!")
        return pd.DataFrame()

    for cond in CONDITIONS:
        print(f"Analyzing: {cond}")
        trans_path = f"{cond.lower()}_transcriptomics.csv"
        
        if not os.path.exists(trans_path):
            print(f"  [!] Missing file: {trans_path}")
            continue
        
        df_trans = pd.read_csv(trans_path, index_col=0)

        for algo in ALGORITHMS:
            algo_path = os.path.join(BASE_DIR, cond, f"{algo}_results.csv")
            
            if not os.path.exists(algo_path):
                print(f"  [!] Missing results: {algo_path}")
                continue
            
            df_algo = pd.read_csv(algo_path)
            scores = df_algo.iloc[:, 0].values # First column is scores
            
            metrics = calculate_structural_metrics(scores, df_trans)
            metrics.update({"Condition": cond, "Algorithm": algo})
            internal_bench_results.append(metrics)

    if not internal_bench_results:
        print("\n❌ No data processed. Check file names and paths.")
        return pd.DataFrame()

    # --- Visualization and Saving ---
    bench_df = pd.DataFrame(internal_bench_results)
    bench_df.to_csv("internal_structure_benchmark.csv", index=False)
    
    sns.set_theme(style="whitegrid")

    # Plot 1: MI vs Entropy
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=bench_df, x="Entropy", y="GPR_Coupling_MI", 
                    hue="Algorithm", style="Condition", s=150)
    plt.title("Information Preservation vs. Score Diversity")
    plt.savefig("internal_benchmark_scatter.png", dpi=300) # Added dpi for high quality
    plt.close() # Clears memory

    # Plot 2: Dynamic Range
    plt.figure(figsize=(10, 5))
    sns.barplot(data=bench_df, x="Algorithm", y="Dynamic_Range", hue="Condition")
    plt.title("Score Sensitivity (Log10 Dynamic Range)")
    plt.savefig("internal_dynamic_range.png", dpi=300)
    plt.close()

    print(f"\n✅ Success! Files saved:\n1. internal_structure_benchmark.csv\n2. internal_benchmark_scatter.png\n3. internal_dynamic_range.png")
    return bench_df

if __name__ == "__main__":
    results = run_internal_benchmark()
    if not results.empty:
        summary = results.groupby('Algorithm')[['Entropy', 'GPR_Coupling_MI', 'Dynamic_Range']].mean()
        print("\n--- Mean Performance per Algorithm ---")
        print(summary)