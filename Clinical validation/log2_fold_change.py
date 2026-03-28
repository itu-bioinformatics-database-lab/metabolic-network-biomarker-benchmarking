import pandas as pd

def calculate_gene_l2fc(data_path, features_path, output_path):
    """
    Calculates Log2 Fold Change for specific genes between 'c' (cancer) 
    and 'healthy' groups.
    """
    try:
        # 1. Load the gene list (features) from the text file
        # FIXED: Changed 'feature_path' to 'features_path' to match function argument
        with open(features_path, 'r') as f:
            selected_genes = [line.strip() for line in f if line.strip()]
        
        # 2. Load the transcriptomics dataset
        df = pd.read_csv(data_path)
        
        # Standardizing the grouping column name to 'Factors'
        group_col = 'Factors'
        
        if group_col not in df.columns:
            print(f"Error: '{group_col}' column not found in {data_path}")
            return

        # 3. Filter the dataframe to include only the Factors and the selected genes
        # We use intersection to ensure we only look for genes actually present in the CSV
        available_genes = list(set(selected_genes).intersection(df.columns))
        df_filtered = df[[group_col] + available_genes]
        
        # 4. Calculate the mean expression for each group
        means = df_filtered.groupby(group_col).mean()
        
        # 5. Calculate Log2 Fold Change
        # L2FC = Mean(Cancer) - Mean(Healthy)
        if 'c' in means.index and 'healthy' in means.index:
            l2fc = means.loc['c'] - means.loc['healthy']
            
            # 6. Format and save the results
            result_df = l2fc.reset_index()
            result_df.columns = ['Gene', 'Log2FoldChange']
            result_df.to_csv(output_path, index=False)
            print(f"Successfully saved L2FC results to: {output_path}")
        else:
            print(f"Error: Could not find both 'c' and 'healthy' factors in {data_path}")

    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Execution ---

# Process Colon Data
calculate_gene_l2fc(
    data_path='colon_transcriptomics.csv', 
    features_path='colon_transcriptomics_rfe_features.txt', 
    output_path='colon_l2fc_results.csv'
)

# Process Breast Data
calculate_gene_l2fc(
    data_path='breast_transcriptomics.csv', 
    features_path='breast_transcriptomics_rfe_features.txt', 
    output_path='breast_l2fc_results.csv'
)