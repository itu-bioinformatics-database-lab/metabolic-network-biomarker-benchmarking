import scipy.io
import pandas as pd

def clear_metabolite_name(metabolite):
    met = metabolite.split(' [')[0]
    # met = met.split('\'')[1]
    return met

def extract_struct(mat_struct):
    """Recursively convert mat_struct to a Python dictionary."""
    data = {}
    for field in mat_struct._fieldnames:
        value = getattr(mat_struct, field)
        if isinstance(value, scipy.io.matlab.mio5_params.mat_struct):
            data[field] = extract_struct(value)
        else:
            data[field] = value
    return data

# Load and parse the .mat file
mat = scipy.io.loadmat('ScoreData.mat', struct_as_record=False, squeeze_me=True)
score_data_struct = mat['ScoreData']
score_data = extract_struct(score_data_struct)

# Extract relevant data
tamboor_scores = score_data['TAMBOORScore'][:, 3]            # 4th column
timbr_scores = score_data['TIMBRScore'][:, 2]                # 3rd column
mod_timbr_scores = score_data['ModifiedTIMBRScore'][:, 2]    # 3rd column

# Extract metabolite names
met_names_raw = score_data['metNames']
met_names = [str(m) for m in met_names_raw]

# Clean metabolite names
met_names = [clear_metabolite_name(met) for met in met_names]

# Function to write sorted Excel without headers
def write_sorted_excel(scores, names, filename):
    df = pd.DataFrame({'Score': scores, 'Metabolite': names})
    df_sorted = df.reindex(df['Score'].abs().sort_values().index)  # Sort by absolute score
    df_sorted.to_excel(filename, index=False, header=False)
    print(f"Saved to {filename}")

# Write each file
write_sorted_excel(tamboor_scores, met_names, 'tamboor_results.xlsx')
write_sorted_excel(timbr_scores, met_names, 'timbr_results.xlsx')
write_sorted_excel(mod_timbr_scores, met_names, 'modified_timbr_results.xlsx')
