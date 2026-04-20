import pandas as pd
from mygene import MyGeneInfo

# --- Load the input file ---
input_file = 'geneData_breast.csv'  # Replace with your actual filename
df = pd.read_csv(input_file)

# --- Extract gene symbols from header (skip the first column like "Factors") ---
gene_symbols = df.columns[1:].tolist()

# --- Use MyGeneInfo to map to Entrez IDs ---
mg = MyGeneInfo()
results = mg.querymany(gene_symbols, scopes='symbol', fields='entrezgene', species='human')

# --- Build mapping: symbol -> entrez_id ---
symbol_to_entrez = {res['query']: res['entrezgene'] for res in results if 'entrezgene' in res}

# --- Update column names using the mapping ---
mapped_columns = ['Factors'] + [symbol_to_entrez.get(gene) for gene in gene_symbols if gene in symbol_to_entrez]

# --- Keep only the columns that were successfully mapped ---
columns_to_keep = ['Factors'] + [gene for gene in gene_symbols if gene in symbol_to_entrez]
df_filtered = df[columns_to_keep]

# --- Rename columns (symbols to Entrez IDs) ---
df_filtered.columns = mapped_columns

# --- Save the new file ---
df_filtered.to_csv('entrez_mapped_expression.csv', index=False)

# --- Print mapping summary ---
total_genes = len(gene_symbols)
mapped_genes = len(symbol_to_entrez)
print(f"Mapping complete. {mapped_genes} out of {total_genes} gene symbols were mapped to Entrez IDs.")
print("Output saved as 'entrez_mapped_expression.csv'.")
