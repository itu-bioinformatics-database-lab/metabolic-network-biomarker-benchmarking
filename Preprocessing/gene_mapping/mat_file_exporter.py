import pandas as pd
from scipy.io import savemat

# # Example usage
# dataset_path = 'mapped_entrez_dataset.csv'  # Replace with your actual dataset file path

# mat_exporter = MATFileExporter(dataset_path)
# mat_exporter.save_to_mat()

class MATFileExporter:
    def __init__(self, dataset_file, output_file, metadata_columns):
        self.dataset_file = dataset_file
        self.output_file = output_file
        self.metadata_columns = metadata_columns

    def load_data(self):
        """Load the dataset from a CSV file."""
        return pd.read_csv(self.dataset_file)

    def process_data(self):
        """Process the dataset to calculate mean expression values by diagnosis."""
        df = self.load_data()

        # Identify gene columns by excluding metadata columns
        gene_columns = [col for col in df.columns if col not in self.metadata_columns]

        # Split data by diagnosis
        health_control_data = df[df['Factors'] == 'healthy'][gene_columns]
        
        # Alzheimer data is either 'AD' or 'AD+'
        alzheimer_data = df[df['Factors'].str.contains('c')][gene_columns]

        # Calculate mean expression values and reshape for MATLAB compatibility
        health_control_mean = health_control_data.mean(axis=0).values.reshape(-1, 1)
        alzheimer_mean = alzheimer_data.mean(axis=0).values.reshape(-1, 1)

        # Convert gene IDs from strings to floats and format as n x 1 lists for MATLAB
        gene_ids = [[float(col)] for col in gene_columns]

        # Return processed data
        return {
            'HealthControl': health_control_mean,
            'Alzheimer': alzheimer_mean,
            'GeneID': gene_ids
        }

    def save_to_mat(self):
        """Save the processed data to a MATLAB .mat file."""
        data_dict = self.process_data()
        savemat(self.output_file, data_dict)
        print(f"Data has been successfully saved to '{self.output_file}'")
