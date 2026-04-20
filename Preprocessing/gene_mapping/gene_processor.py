import csv
import re

# # Example usage
# csv_file_path = 'transcriptomics.csv'  # Replace with your actual CSV file path
# txt_file_path = 'ensembl_gene_ids.txt'  # Replace with your actual text file path
# gene_processor = GeneProcessor(csv_file_path, txt_file_path)

# # Find and save common genes
# gene_processor.save_common_genes()

class GeneProcessor:
    def __init__(self, csv_file, txt_file, output_file):
        self.csv_file = csv_file
        self.txt_file = txt_file
        self.output_file = output_file

    def get_values_in_first_line(self):
        """Extract gene values from the first line of the CSV file."""
        with open(self.csv_file, mode='r') as file:
            csv_reader = csv.reader(file)
            first_line = next(csv_reader)
            return first_line

    def get_values_from_lines(self):
        """Extract gene values from the text file, removing newline characters."""
        with open(self.txt_file, mode='r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]

    def process_gene_versions(self, genes):
        """Remove version numbers from gene IDs."""
        return [gene.split('.')[0] for gene in genes]

    def find_common_genes(self):
        """Find common genes between dataset genes and recon genes."""
        dataset_genes = self.get_values_in_first_line()
        recon_genes = self.get_values_from_lines()
        
        # Process dataset genes to ignore version numbers
        dataset_genes = self.process_gene_versions(dataset_genes)
        
        # Find common genes
        common_genes = set(dataset_genes).intersection(recon_genes)
        return common_genes

    def save_common_genes(self):
        """Save common genes to a specified output file."""
        common_genes = self.find_common_genes()
        with open(self.output_file, mode='w') as file:
            for gene in common_genes:
                file.write(gene + '\n')
