import pandas as pd

# # Example usage
# csv_file_path = 'transcriptomics.csv'  # Replace with your actual CSV file path
# gene_file_path = 'common_genes.txt'    # Replace with your actual text file path

# dataset_filter = DatasetFilter(csv_file_path, gene_file_path)
# dataset_filter.save_filtered_dataset()


class DatasetFilter:
    def __init__(self, csv_file, gene_file, output_file, metadata_columns):
        self.csv_file = csv_file
        self.gene_file = gene_file
        self.output_file = output_file
        self.metadata_columns = metadata_columns

    def load_dataset(self):
        """Load the dataset from a CSV file."""
        return pd.read_csv(self.csv_file)

    def load_required_genes(self):
        """Load the required genes from a text file."""
        with open(self.gene_file, 'r') as file:
            return [line.strip() for line in file]

    def filter_dataset(self):
        """Filter the dataset for required genes and metadata columns."""
        # Load the dataset and gene list
        df = self.load_dataset()
        required_genes = self.load_required_genes()

        # Remove gene version numbers from dataset column names
        df.columns = df.columns.str.split('.').str[0]

        # Combine metadata columns with the required gene columns
        columns_to_keep = self.metadata_columns + required_genes

        # Filter the dataframe to keep only the necessary columns
        filtered_df = df[columns_to_keep]
        return filtered_df

    def save_filtered_dataset(self):
        """Save the filtered dataset to a CSV file."""
        filtered_df = self.filter_dataset()
        filtered_df.to_csv(self.output_file, index=False)
        print(f"Filtered dataset saved as '{self.output_file}'")


