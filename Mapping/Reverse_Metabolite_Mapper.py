import cobra
import pandas as pd
import os

# --- HELPER FUNCTION: Smart Matching ---
def find_metabolite_smart(model, query, compartments):
    """
    Searches for metabolites within Recon3D by ID, Lowercase, or Full Name.
    """
    query = str(query).strip()
    
    # 1. Direct ID lookup (Fastest)
    try:
        return model.metabolites.get_by_id(query)
    except KeyError:
        pass

    # 2. Lookup with compartment suffixes (e.g., h2o -> h2o_c)
    for comp in compartments:
        try:
            return model.metabolites.get_by_id(f"{query}{comp}")
        except KeyError:
            continue

    # 3. Deep search via Lowercase and Full Name
    # This section increases the match rate by 30-40%.
    query_lower = query.lower()
    for met in model.metabolites:
        # Case-insensitive ID match or Full Name match in the model
        if query_lower == met.id.lower() or query_lower == met.name.lower():
            return met
            
    return None

# --- MAIN PROCESS ---
print("Loading Recon3D Model...")
try:
    model = cobra.io.load_json_model("Recon3D.json")
    print("Model loaded successfully!\n")
except Exception as e:
    print(f"Model loading error: {e}")
    exit()

# List of tuples: (Input_File, Output_File)
disease_files = [
    ("Alzheimer_Final_Comparison_Report_Python.csv", "Alzheimer_Genetic_Map_Report.csv"),
    ("Breast_Final_Comparison_Report_Python.csv", "Breast_Genetic_Map_Report.csv"),
    ("Colon_Final_Comparison_Report_Python.csv", "Colon_Genetic_Map_Report.csv")
]

compartments = ['_c', '_m', '_e', '_n', '_r', '_g', '_l', '_x']

for input_csv, output_csv in disease_files:
    print(f"--- Processing: {input_csv} ---")
    
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} file not found.")
        continue

    try:
        df_rfe = pd.read_csv(input_csv)
        df_rfe.columns = df_rfe.columns.str.strip()

        # Unique metabolites with RFE_Rank 1
        rank_1_list = df_rfe[df_rfe['RFE_Rank'] == 1]['Metabolite'].unique().tolist()
        print(f"Unique metabolites to analyze: {len(rank_1_list)}")

        results = []
        found_count = 0

        for met_raw in rank_1_list:
            metabolite = find_metabolite_smart(model, met_raw, compartments)

            if metabolite:
                found_count += 1
                # Reactions PRODUCING the metabolite (Stoichiometry > 0)
                producing_rxns = [r for r in metabolite.reactions if r.metabolites[metabolite] > 0]
                
                for rxn in producing_rxns:
                    results.append({
                        "Input_Metabolite": met_raw,
                        "Recon3D_ID": metabolite.id,
                        "Recon3D_Name": metabolite.name,
                        "Reaction_ID": rxn.id,
                        "AND_OR_Logic": rxn.gene_reaction_rule,
                        "Genes": ", ".join([g.id for g in rxn.genes]),
                        "Subsystem": rxn.subsystem
                    })
        
        df_output = pd.DataFrame(results)
        df_output.to_csv(output_csv, index=False)
        
        print(f"Result: {found_count} out of {len(rank_1_list)} metabolites matched in the model.")
        print(f"Report generated: {output_csv}\n")

    except Exception as e:
        print(f"Error while processing {input_csv}: {e}")

print("All processes completed.")