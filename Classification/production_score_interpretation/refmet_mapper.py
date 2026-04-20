import requests
import pandas as pd
import json

# import rpy2
# import rpy2.robjects as ro
# from rpy2.robjects.packages import importr
# from rpy2.robjects.vectors import StrVector

# def use_r_package():
#     # Import the R library
#     metaboliteIDmapping = importr("metaboliteIDmapping")

#     # Example metabolites to map
#     metabolites = ["glucose", "fructose", "citric acid"]

#     # Convert Python list to R vector
#     metabolite_vector = StrVector(metabolites)

#     # Call the mapping function (example function from the library)
#     # Replace 'mapIdentifiers' with the specific function you'd like to use
#     # For this demonstration, assume 'mapIdentifiers' maps metabolite names to KEGG IDs
#     mapped = metaboliteIDmapping.mapIdentifiers(metabolite_vector, database="KEGG")

#     # Print the result
#     print("Mapped Metabolites:", list(mapped))   


# def send_metaboanalyst_request():

#     url = "https://rest.xialab.ca/api/mapcompounds"

#     payload = "{\n\t\"queryList\": \"1,3-Diaminopropane;2-Ketobutyric acid;2-Hydroxybutyric acid;\",\n\t\"inputType\": \"name\"\n}"
#     headers = {
#         'Content-Type': "application/json",
#         'cache-control': "no-cache",
#         }

#     response = requests.request("POST", url, data=payload, headers=headers)

#     print(response.text)


# params is a string with metabolite names separated by newline
def get_refmet_metabolites(params):
    refmet_url = "https://www.metabolomicsworkbench.org/databases/refmet/name_to_refmet_new_min.php"
    res = requests.post(refmet_url, data=params).text.split('\n')
    res.pop(0)

    ref_dict = {}
    map_count = 0
    for line in res:
        if line == '':
            continue
        line = line.split('\t')
        met = line[0]
        ref = line[1]
        if(ref != "-"):
            map_count += 1
            if(ref_dict.get(ref) is None):
                ref_dict[ref] = met
            else:
                print("Different match for the same metabolite: ", met, " --> ", ref, " --> ", ref_dict[ref])

        print("Metabolite: ", met, "Refmet: ", ref)

    print("Mapped ", map_count, " out of ", len(res))

    return ref_dict


# Clear metabolite name from quotes and other parts
def clear_metabolite_name(metabolite):
    met = metabolite.split(' [')[0]
    # met = met.split('\'')[1]
    return met


if __name__ == "__main__":
    
    tamboor_file_path = "modified_timbr_results.xlsx"
    metabolomics_file_path = "metabolomics.csv"

    tamboor_df = pd.read_excel(tamboor_file_path)
    tamboor_metabolites = tamboor_df.iloc[:, 1].tolist() # Get the metabolites from the second column
    tamboor_metabolites_string = ""

    # Remove the last part of the metabolite name 'tyrosine [Extracellular]' to tyrosine without quotes
    for i in range(len(tamboor_metabolites)):
        met = clear_metabolite_name(tamboor_metabolites[i])
        tamboor_metabolites[i] = met
        tamboor_metabolites_string += met + '\n'

    # Read the first line of metabolomics.csv to get the column names excluding Sample ID,Gender,Race,PMI,Braak,Diagnosis columns
    metabolomics_df = pd.read_csv(metabolomics_file_path, nrows=1) 
    metabolomics_columns = metabolomics_df.columns.tolist()[6:] 
    print(len(metabolomics_columns))
    metabolomics_string = ""

    # save metabolomics columns and tamboor metabolites to a file
    with open("metabolomics_columns.txt", "w") as f:
        for i in range(len(metabolomics_columns)):
            f.write(metabolomics_columns[i] + '\n')

    with open("tamboor_metabolites.txt", "w") as f:
        for i in range(len(tamboor_metabolites)):
            f.write(tamboor_metabolites[i] + '\n')

    for i in range(len(metabolomics_columns)):
        metabolomics_string += metabolomics_columns[i] + '\n'

    data_params = {
            "metabolite_name": metabolomics_string
    }

    params = {
            "metabolite_name": tamboor_metabolites_string
    }
    

    ref_dict = get_refmet_metabolites(params)
    ref_dict_metabolomics = get_refmet_metabolites(data_params)

    # # Write the ids and refmet ids to a new csv file
    # tamboor_refmet_df = pd.DataFrame(list(ref_dict.items()), columns=['Metabolite', 'Refmet'])
    # tamboor_refmet_df.to_csv("tamboor_refmet.csv", index=False)

    # metabolomics_refmet_df = pd.DataFrame(list(ref_dict_metabolomics.items()), columns=['Metabolite', 'Refmet'])
    # metabolomics_refmet_df.to_csv("metabolomics_refmet.csv", index=False)

    # Write the ids and refmet ids to a new json file
    with open("tamboor_refmet.json", "w") as f:
        json.dump(ref_dict, f)

    with open("metabolomics_refmet.json", "w") as f:
        json.dump(ref_dict_metabolomics, f)
    