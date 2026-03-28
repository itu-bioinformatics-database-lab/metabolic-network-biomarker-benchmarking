import pandas as pd
import requests
import json
import re
from pathlib import Path

# --- Configuration ---
base_dir = Path(".") 
diseases = ["Alzheimer", "Breast", "Colon"]
excel_files = ["tamboor_results.xlsx", "timbr_results.xlsx", "modified_timbr_results.xlsx"]
json_mapping_file = "new-synonym-mapping.json"

metabolomics_map = {
    "Alzheimer": "alzheimer_metabolomics.csv",
    "Breast": "breast_metabolomics.csv",
    "Colon": "colon_metabolomics.csv"
}

output_base_dir = Path("processed_for_analysis")
output_base_dir.mkdir(exist_ok=True)

# --- Enhanced Helper Functions ---

def clear_metabolite_name(name):
    """
    Standardizes metabolite names for maximum matching rate.
    Cleans compartment suffixes, quotes, and biochemical tags.
    """
    if pd.isna(name): return ""
    name = str(name).strip()
    
    # 1. Clean compartment suffixes (e.g., "Alanine [Extracellular]", "Alanine [e]", "Alanine (e)")
    name = re.sub(r'\s*[\[\(].*?[\]\)]', '', name)
    
    # 2. Clean special characters and quotes
    name = name.replace("'", "").replace('"', "").replace("_", " ")
    
    # 3. Clean common suffixes (case-insensitive)
    name = re.sub(r'(?i)(_pos|_neg|_id| - ion| ion)$', '', name)
    
    # 4. Normalize (lowercase and remove extra whitespace)
    return " ".join(name.lower().split())

def load_json_mapping(file_path):
    """Loads the JSON file and normalizes the keys."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            # Store both raw name and cleaned name as keys for guaranteed bidirectional matching
            optimized_map = {}
            for k, v in raw_data.items():
                optimized_map[k] = v
                optimized_map[clear_metabolite_name(k)] = v
            return optimized_map
    except Exception as e:
        print(f"⚠️ JSON loading error: {e}")
        return {}

def get_refmet_mapping(metabolite_list):
    """Performs batch mapping using the RefMet API."""
    refmet_url = "https://www.metabolomicsworkbench.org/databases/refmet/name_to_refmet_new_min.php"
    # Send only unique and cleaned names to increase API success rate
    clean_list = list(set(filter(None, metabolite_list)))
    if not clean_list: return {}
    
    print(f"🌐 Calling RefMet API... ({len(clean_list)} names)")
    try:
        res = requests.post(refmet_url, data={"metabolite_name": "\n".join(clean_list)}, timeout=60).text.split("\n")
        if res: res.pop(0) 
        
        mapping_dict = {}
        for line in res:
            if not line: continue
            parts = line.split("\t")
            if len(parts) >= 2:
                original, refmet = parts[0], parts[1]
                mapping_dict[original] = refmet if refmet != "-" else None
        return mapping_dict
    except Exception as e:
        print(f"❌ API error: {e}")
        return {}

# --- STEP 1: Data Collection ---
print("📂 Scanning files...")
all_raw_names = set()
data_registry = []
json_map = load_json_mapping(json_mapping_file)

for ds in diseases:
    ds_folder = base_dir / ds
    for f_name in excel_files:
        path = ds_folder / f_name
        if path.exists():
            df = pd.read_excel(path)
            raw_names = df.iloc[:, 1].dropna().astype(str).tolist()
            all_raw_names.update(raw_names)
            data_registry.append({'type': 'excel', 'disease': ds, 'file': f_name, 'df': df})
    
    csv_filename = metabolomics_map.get(ds)
    csv_path = base_dir / csv_filename
    if csv_path.exists():
        df_meto = pd.read_csv(csv_path)
        meto_cols = df_meto.columns.tolist()[6:]
        all_raw_names.update(meto_cols)
        data_registry.append({'type': 'csv', 'disease': ds, 'file': csv_filename, 'df': df_meto, 'cols': meto_cols})

# --- STEP 2: Advanced Hierarchical Mapping ---
# Query the API using cleaned names first
cleaned_to_raw = {clear_metabolite_name(name): name for name in all_raw_names}
refmet_results = get_refmet_mapping(list(cleaned_to_raw.keys()))

def get_best_id(raw_name):
    """
    Hierarchical validation for maximum success rate:
    1. Is raw name in JSON?
    2. Is cleaned name in JSON?
    3. Is cleaned name in RefMet?
    4. Fallback: Cleaned name.
    """
    clean = clear_metabolite_name(raw_name)
    
    # 1 & 2: JSON Check (Recon3D Guarantee)
    if raw_name in json_map: return json_map[raw_name]
    if clean in json_map: return json_map[clean]
    
    # 3: RefMet Check
    ref_id = refmet_results.get(clean)
    if ref_id: return ref_id
    
    # 4: Last resort
    return clean

master_map = {name: get_best_id(name) for name in all_raw_names}

# --- STEP 3: Saving Results ---
print("💾 Updating files...")
for item in data_registry:
    df = item['df'].copy()
    out_dir = output_base_dir / item['disease']
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if item['type'] == 'excel':
        df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: master_map.get(str(x), x))
        save_path = out_dir / item['file'].replace(".xlsx", ".csv")
    else:
        new_cols = df.columns.tolist()[:6]
        for col in item['cols']:
            new_cols.append(master_map.get(col, col))
        df.columns = new_cols
        save_path = out_dir / f"processed_{item['file']}"
    
    df.to_csv(save_path, index=False)

print(f"✅ Process completed. {len(all_raw_names)} metabolites analyzed.")