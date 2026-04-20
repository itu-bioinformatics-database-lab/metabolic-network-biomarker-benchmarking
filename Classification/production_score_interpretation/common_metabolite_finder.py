import csv
import json

# Function to count the occurrences of '1' and '0' in the Comment column
def generate_dict(csv_file, map_dict):
    count_1 = 0
    count_0 = 0
    
    # Open the CSV file for reading
    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)  # Use DictReader to read the CSV into dictionaries
        
        # Loop through each row in the CSV
        for row in reader:
            comment = row['Comment']
            if comment == '1':
                count_1 += 1
                if map_dict.get(row['Match']) is None or map_dict[row['Match']] == '-':
                    map_dict[row['Match']] = row['Query']
                elif map_dict[row['Match']] != row['Query']:
                    print("Different match for the same metabolite: ", row['Match']," --> " , row['Query']," --> "  ,map_dict[row['Match']])

            elif comment == '0':
                count_0 += 1

tamboor_wb_dict = {}
metabolomics_wb_dict = {}

tamboor_refmet_dict = {}
metabolomics_refmet_dict = {}

isTogether = True

# Load tamboor_refmet.json and metabolomics_refmet.json as dictionaries

if isTogether:
    with open('tamboor_refmet.json') as f:
        tamboor_refmet_dict = json.load(f)

    with open('metabolomics_refmet.json') as f:
        metabolomics_refmet_dict = json.load(f)

# csv_file = 'name_map_metabolomics.csv'
# csv_file2 = 'name_map_tamboor.csv'

# # Use the loaded dictionaries to call the function
# generate_dict(csv_file, metabolomics_wb_dict)
# generate_dict(csv_file2, tamboor_wb_dict)

# Print the length of the dictionaries
print("Tamboor WB: ", len(tamboor_wb_dict))
print("Tamboor REFMET: ", len(tamboor_refmet_dict))
print("Metabolomics WB: ", len(metabolomics_wb_dict))
print("Metabolomics REFMET: ", len(metabolomics_refmet_dict))

metabolomics_wb_set =  set(metabolomics_wb_dict.keys())
tamboor_wb_set = set(tamboor_wb_dict.keys())

metabolomics_refmet_set = set(metabolomics_refmet_dict.keys())
tamboor_refmet_set = set(tamboor_refmet_dict.keys())

tamboor_metabolite_results = set.union(tamboor_wb_set, tamboor_refmet_set)
metabolomics_metabolite_results = set.union(metabolomics_refmet_set, metabolomics_wb_set)

print("Tamboor metabolites: ", len(tamboor_metabolite_results))
print("Metabolomics metabolites: ", len(metabolomics_metabolite_results))

common_metabolites = tamboor_metabolite_results.intersection(metabolomics_metabolite_results)

tamboor_file = "tamboor_results.xlsx"
print("Common metabolites total: ", len(common_metabolites))

# Find common metabolites for refmet mapping
common_metabolites_refmet = tamboor_refmet_set.intersection(metabolomics_refmet_set)

print("Common metabolites for refmet mapping: ", len(common_metabolites_refmet))

# Find common metabolites for wb mapping
common_metabolites_wb = tamboor_wb_set.intersection(metabolomics_wb_set)

print("Common metabolites for wb mapping: ", len(common_metabolites_wb))

# Save the common metabolites as a dictionary (Tamboor)
common_metabolites_dict_tamboor = {}
for met in common_metabolites:
    if met in tamboor_wb_dict:
        common_metabolites_dict_tamboor[met] = tamboor_wb_dict[met]
    else:
        common_metabolites_dict_tamboor[met] = tamboor_refmet_dict[met]

with open('common_metabolites_tamboor.json', 'w') as f:
    json.dump(common_metabolites_dict_tamboor, f)

# Save the common metabolites as a dictionary (Metabolomics)
common_metabolites_dict_metabolomics = {}
for met in common_metabolites:
    if met in metabolomics_refmet_dict:
        common_metabolites_dict_metabolomics[met] = metabolomics_refmet_dict[met]
    else:
        common_metabolites_dict_metabolomics[met] =  metabolomics_wb_dict[met]

with open('common_metabolites_metabolomics.json', 'w') as f:
    json.dump(common_metabolites_dict_metabolomics, f)