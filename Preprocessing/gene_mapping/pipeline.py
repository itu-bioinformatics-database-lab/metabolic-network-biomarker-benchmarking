from gene_mapper import GeneMapper
from gene_processor import GeneProcessor
from dataset_filter import DatasetFilter
from entrez_mapper import EntrezMapper
from mat_file_exporter import MATFileExporter

class Pipeline:
    def __init__(self, recon_json = 'Recon3D.json', entrez_ids_file='entrez_ids.txt', gene_dict_file='gene_dict.json',
                 csv_file='transcriptomics.csv', common_genes_txt_file='common_genes.txt', 
                 mapped_entrez_dataset_file='mapped_entrez_dataset.csv',
                 filtered_dataset_file='filtered_dataset.csv', metadata_columns = ['Factors'] 
                   ,output_mat_file='mean_output_dataset.mat'):
        
        # Initialize the pipeline with file paths for each class

        # metadata_columns = ['Sample ID', 'Gender', 'Race', 'PMI', 'Braak', 'Diagnosis']
        # Files should be in /data folder
        recon_json = 'data/' + recon_json
        # entrez_ids_file = 'data/' + entrez_ids_file
        # gene_dict_file = 'data/' + gene_dict_file
        csv_file = 'data/' + csv_file
        # common_genes_txt_file = 'data/' + common_genes_txt_file
        # mapped_entrez_dataset_file = 'data/' + mapped_entrez_dataset_file
        # filtered_dataset_file = 'data/' + filtered_dataset_file
        # output_mat_file = 'data/' + output_mat_file

        print("Initializing Pipeline...")
        self.gene_mapper = GeneMapper(recon_json, entrez_ids_file, common_genes_txt_file, gene_dict_file)
        print("GeneMapper initialized with files:", recon_json, entrez_ids_file, common_genes_txt_file, gene_dict_file)
        self.gene_processor = GeneProcessor(csv_file, common_genes_txt_file, common_genes_txt_file)
        print("GeneProcessor initialized with files:", csv_file, common_genes_txt_file)
        self.dataset_filter = DatasetFilter(csv_file, common_genes_txt_file, filtered_dataset_file, metadata_columns)
        print("DatasetFilter initialized with files:", csv_file, common_genes_txt_file, filtered_dataset_file)
        self.entrez_mapper = EntrezMapper(filtered_dataset_file = filtered_dataset_file, gene_dict_file = gene_dict_file, output_file = mapped_entrez_dataset_file, metadata_columns = metadata_columns)
        print("EntrezMapper initialized with files:", filtered_dataset_file, gene_dict_file, mapped_entrez_dataset_file)
        self.mat_exporter = MATFileExporter(dataset_file=mapped_entrez_dataset_file, output_file=output_mat_file, metadata_columns=metadata_columns)
        print("MATFileExporter initialized with files:", mapped_entrez_dataset_file, output_mat_file)

    def run(self):
        """Run the entire pipeline."""
        print("Starting Gene Mapping...")
        self.gene_mapper.run_mapping()  # Assuming run_mapping() is in GeneMapper

        print("Processing Genes...")
        self.gene_processor.save_common_genes()  # Assuming save_common_genes() is in GeneProcessor

        print("Filtering Dataset...")
        self.dataset_filter.save_filtered_dataset()  # Assuming save_filtered_dataset() is in DatasetFilter

        print("Mapping Entrez IDs...")
        self.entrez_mapper.save_mapped_dataset()  # Assuming save_mapped_dataset() is in EntrezMapper

        print("Exporting to MATLAB format...")
        self.mat_exporter.save_to_mat()  # Assuming save_to_mat() is in MATFileExporter

        print("Pipeline completed successfully!")


# Example usage of the Pipeline
if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
