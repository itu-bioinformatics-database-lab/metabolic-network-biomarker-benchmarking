import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# 1. Load the data
data = pd.read_csv('metabolomics.csv')
# target_column = 'Diagnosis'
target_column = 'Factors'

# Biomarkers (selected features) read from result_list.txt separated by new line (read only first n lines)
n = 10
with open("result_list.txt", "r") as f:
    biomarkers = [next(f).strip() for _ in range(n)]

# All features
# non_metabolite_columns = ['Sample ID', 'Gender', 'Race', 'PMI', 'Braak', 'Diagnosis', "X - 24728", "X - 24807", "X - 25009", "X - 25020", "X - 25026", "X - 25047", "X - 25109", "X - 25180", "X - 25244", "X - 25422", "X - 25790", "X - 25828", "X - 25855", "X - 25884", "X - 25936", "X - 25948", "X - 25981", "X - 25982"]
non_metabolite_columns = ['Factors']

X_all = data.drop(columns=non_metabolite_columns)
X_biomarkers = data[biomarkers]

# Ensure 'Diagnosis' column is consistent
data[target_column] = data[target_column].replace('AD+', 'AD')

# Define binary classification groups
# binary_classifications = {
#     "Control vs. AD": ['Control', 'AD'],
#     "Control vs. MCI": ['Control', 'MCI'],
#     "MCI vs. AD": ['MCI', 'AD'],
#     "Control vs. MCI+AD": ['Control', 'MCI', 'AD']
# }

binary_classifications = {
    "Control vs. Cancer": ['healthy', 'c']
}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_model(model, X, y):
    accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy').mean()
    precision = cross_val_score(model, X, y, cv=kf, scoring='precision_weighted').mean()
    recall = cross_val_score(model, X, y, cv=kf, scoring='recall_weighted').mean()
    f1 = cross_val_score(model, X, y, cv=kf, scoring='f1_weighted').mean()
    return accuracy, precision, recall, f1

scaler_all = StandardScaler()
scaler_biomarkers = StandardScaler()
scaler_random = StandardScaler()

for label, classes in binary_classifications.items():
    binary_data = data[data[target_column].isin(classes)].copy()
    
    if label == "Control vs. MCI+AD":
        binary_data[target_column] = binary_data[target_column].replace({'MCI': 'AD'})
    
    binary_data[target_column] = LabelEncoder().fit_transform(binary_data[target_column])
    
    X_all_binary = scaler_all.fit_transform(binary_data.drop(columns=non_metabolite_columns))
    X_biomarkers_binary = scaler_biomarkers.fit_transform(binary_data[biomarkers])
    y_binary = binary_data[target_column]
    
    # Randomly select 179 features
    random_features = np.random.choice(X_all.columns, n, replace=False).tolist()
    X_random_binary = scaler_random.fit_transform(binary_data[random_features])
    
    clf_all = RandomForestClassifier(random_state=42)
    clf_biomarkers = RandomForestClassifier(random_state=42)
    clf_random = RandomForestClassifier(random_state=42)
    
    accuracy_all, precision_all, recall_all, f1_all = evaluate_model(clf_all, X_all_binary, y_binary)
    accuracy_biomarkers, precision_biomarkers, recall_biomarkers, f1_biomarkers = evaluate_model(clf_biomarkers, X_biomarkers_binary, y_binary)
    accuracy_random, precision_random, recall_random, f1_random = evaluate_model(clf_random, X_random_binary, y_binary)
    
    print(f"Results for {label}:")
    print("Model with All Features:")
    print(f"Accuracy: {accuracy_all:.4f}")
    print(f"Precision: {precision_all:.4f}")
    print(f"Recall: {recall_all:.4f}")
    print(f"F1 Score: {f1_all:.4f}")
    
    print("\nModel with Biomarkers:")
    print(f"Accuracy: {accuracy_biomarkers:.4f}")
    print(f"Precision: {precision_biomarkers:.4f}")
    print(f"Recall: {recall_biomarkers:.4f}")
    print(f"F1 Score: {f1_biomarkers:.4f}")
    
    print("\nModel with Randomly Selected Features:")
    print(f"Accuracy: {accuracy_random:.4f}")
    print(f"Precision: {precision_random:.4f}")
    print(f"Recall: {recall_random:.4f}")
    print(f"F1 Score: {f1_random:.4f}")
    print("\n" + "-"*50 + "\n")
