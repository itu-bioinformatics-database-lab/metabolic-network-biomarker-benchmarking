import re
import os
import mygene
import json
from cobra.io import load_json_model

class GeneMapper:
    def __init__(self, model_path, gene_file, common_genes_file ,gene_dict_file):
        self.model_path = model_path
        self.gene_file = gene_file
        self.gene_dict_file = gene_dict_file
        self.common_genes_file = common_genes_file
        self.gene_dict = {}
        self.gene_set = set()
        self.mg = mygene.MyGeneInfo()

    def load_genes_from_model(self):
        """Load gene IDs from a JSON model file."""
        model = load_json_model(self.model_path)
        gene_ids = set(gene.id for gene in model.genes)
        return gene_ids

    def write_gene_ids(self, gene_ids):
        """Write gene IDs to a text file."""
        with open(self.gene_file, 'w') as file:
            for gene in gene_ids:
                file.write(f'{gene}\n')

    def load_gene_ids(self):
        """Load gene IDs from a text file."""
        if os.path.exists(self.gene_file):
            with open(self.gene_file, 'r') as file:
                gene_ids = {line.strip() for line in file}
        else:
            gene_ids = self.load_genes_from_model()
            self.write_gene_ids(gene_ids)
        return gene_ids

    def convert_to_numeric_ids(self, gene_ids):
        """Extract numeric gene IDs from the loaded set."""
        return [re.match(r'(\d+)', gene_id).group(1) for gene_id in gene_ids if re.match(r'(\d+)', gene_id)]

    def map_names(self, genes):
        """Map Entrez gene IDs to gene symbols and update gene_dict and gene_set."""
        result = self.mg.querymany(genes, scopes="entrezgene", fields="symbol", species="human")
        for item in result:
            symbol = item.get('symbol', None)
            if symbol:
                self.gene_dict[item['query']] = [symbol]
                self.gene_set.add(symbol)
            else:
                print(f"Gene {item['query']} not found (no symbol available)")

    def process_multiple_ids(self, item, ensembl_id):
        """Handle multiple Ensembl IDs for a single gene."""
        for gene in ensembl_id:
            self.gene_set.add(gene['gene'])
            if item['query'] not in self.gene_dict:
                self.gene_dict[item['query']] = [gene['gene']]
            else:
                self.gene_dict[item['query']].append(gene['gene'])

    def save_gene_dict(self):
        """Save the gene dictionary to a JSON file."""
        with open(self.gene_dict_file, 'w') as file:
            json.dump(self.gene_dict, file)

    def save_gene_set(self):
        """Save the gene set to the gene file."""
        with open(self.common_genes_file, 'w') as file:
            for gene in self.gene_set:
                file.write(f'{gene}\n')

    def run_mapping(self):
        """Main function to run the mapping process."""
        gene_ids = self.load_gene_ids()
        numeric_gene_ids = self.convert_to_numeric_ids(gene_ids)
        self.map_names(numeric_gene_ids)
        self.save_gene_dict()
        self.save_gene_set()
