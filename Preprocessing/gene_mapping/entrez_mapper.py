import pandas as pd
import json

# # Example usage
# filtered_dataset_path = 'filtered_dataset.csv'  # Replace with your actual filtered dataset file path
# gene_dict_path = 'gene_dict.json'               # Replace with your actual gene dictionary file path

# entrez_mapper = EntrezMapper(filtered_dataset_path, gene_dict_path)
# entrez_mapper.save_mapped_dataset()

class EntrezMapper:
    def __init__(self, filtered_dataset_file, gene_dict_file, output_file, metadata_columns):
        self.filtered_dataset_file = filtered_dataset_file
        self.gene_dict_file = gene_dict_file
        self.output_file = output_file
        self.metadata_columns = metadata_columns

    def load_data(self):
        """Load the filtered dataset and gene dictionary from files."""
        df = pd.read_csv(self.filtered_dataset_file)
        with open(self.gene_dict_file, 'r') as file:
            entrez_to_ensembl = json.load(file)
        return df, entrez_to_ensembl

    def map_entrez_ids(self):
        """Map Ensembl gene IDs to Entrez IDs and calculate maximum expression values."""
        df, entrez_to_ensembl = self.load_data()
        
        # Initialize a new DataFrame with metadata columns
        new_df = df[self.metadata_columns].copy()

        # Map each Entrez ID to the maximum expression of corresponding Ensembl IDs
        for entrez_id, ensembl_ids in entrez_to_ensembl.items():
            # Check for Ensembl IDs present in the dataset columns
            valid_ensembl_ids = [ensembl_id for ensembl_id in ensembl_ids if ensembl_id in df.columns]
            
            if valid_ensembl_ids:
                # If only one Ensembl ID is present, use its expression values directly
                if len(valid_ensembl_ids) == 1:
                    new_df[entrez_id] = df[valid_ensembl_ids[0]]
                else:
                    # Otherwise, calculate the max expression across the multiple Ensembl IDs
                    new_df[entrez_id] = df[valid_ensembl_ids].max(axis=1)
        return new_df

    def save_mapped_dataset(self):
        """Save the new dataset with Entrez IDs and calculated expression values."""
        new_df = self.map_entrez_ids()
        new_df.to_csv(self.output_file, index=False)
        print(f"Mapped dataset saved as '{self.output_file}'")
