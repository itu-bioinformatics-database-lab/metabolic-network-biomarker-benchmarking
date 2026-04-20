import pandas as pd
import json

def get_n_matched_metabolites(n):
    # Go backwards from the last metabolite in the tamboor_metabolites list
    metabolites_list = []
    half_matched_number = 0
    # Loop from len(tamboor_metabolites) - 1 to 0
    for i in range(len(tamboor_metabolites) - 1, -1, -1):
        if tamboor_metabolites[i] in common_metabolites_dict.keys():
            metabolites_list.append(common_metabolites_dict[tamboor_metabolites[i]])
            if i >= half:
                half_matched_number += 1
        # if abs(tamboor_values[i]) < 0.01:
        #     break
        # if len(metabolites_list) == n:
        #     break
    return metabolites_list, half_matched_number


algorithm_name = "modified_timbr"  # Change this to the desired algorithm name

excel_path = algorithm_name + "_results.xlsx"
metabolomics_file_path = "metabolomics.csv"

tamboor_df = pd.read_excel(excel_path)
tamboor_values = tamboor_df.iloc[:, 0].tolist() # Get the metabolites from the first column
tamboor_metabolites = tamboor_df.iloc[:, 1].tolist() # Get the metabolites from the second column

# get the 50% of the metabolites 
half = len(tamboor_metabolites) // 2
print(half)

common_metabolites_dict = {}

# Read common_metaobolites_tamboor.json file to get the common metabolites dictionary
with open('common_metabolites_tamboor.json') as f:
    common_metabolites_dict = json.load(f)

# Reverse the dictionary to get the metabolite names as keys
common_metabolites_dict = {v: k for k, v in common_metabolites_dict.items()}

# # Remove the last part of the metabolite name 'tyrosine [Extracellular]' to tyrosine without quotes
# for i in range(len(tamboor_metabolites)):
#     met = clear_metabolite_name(tamboor_metabolites[i])
#     tamboor_metabolites[i] = met

metabolites_list, half_matched = get_n_matched_metabolites(100)

print(len(metabolites_list))
print(half_matched)

common_metabolites_dict_metabolomics = {}

# Read common_metaobolites_metabolomics.json file to get the common metabolites dictionary
with open('common_metabolites_metabolomics.json') as f:
    common_metabolites_dict_metabolomics = json.load(f)

result_list = []
for i in range(len(metabolites_list)):
    if metabolites_list[i] in common_metabolites_dict_metabolomics.keys():
        result_list.append(common_metabolites_dict_metabolomics[metabolites_list[i]])

print(len(result_list))

# Save the result list to a file
# with open(algorithm_name + "_result_list.txt", "w") as f:
with open("result_list.txt", "w") as f: 
    for i in range(len(result_list)):
        f.write(result_list[i] + '\n')