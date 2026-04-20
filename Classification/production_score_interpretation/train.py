import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Load the data
data = pd.read_csv('metabolomics.csv')
# target_column = 'Diagnosis'
target_column = 'Factors'

# Biomarkers (selected features) read from result_list.txt separated by new line (read only first n lines)
n = 50
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

# binary_classifications = {
#     "Control vs. AD": ['Control', 'AD']
# }

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

with open("training_results.txt", "w") as output_file:
    for label, classes in binary_classifications.items():
        binary_data = data[data[target_column].isin(classes)].copy()

        if label == "Control vs. MCI+AD":
            binary_data[target_column] = binary_data[target_column].replace({'MCI': 'AD'})

        binary_data[target_column] = LabelEncoder().fit_transform(binary_data[target_column])

        y_binary = binary_data[target_column]
        X_all_binary = binary_data.drop(columns=non_metabolite_columns)
        X_biomarkers_binary = binary_data[biomarkers]

        # Randomly select n features
        random_features = np.random.choice(X_all.columns, n, replace=False).tolist()
        X_random_binary = binary_data[random_features]

        # Pipelines to prevent data leakage
        pipe_all = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        pipe_biomarkers = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])
        pipe_random = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(random_state=42))
        ])

        accuracy_all = cross_val_score(pipe_all, X_all_binary, y_binary, cv=kf, scoring='accuracy').mean()
        precision_all = cross_val_score(pipe_all, X_all_binary, y_binary, cv=kf, scoring='precision_weighted').mean()
        recall_all = cross_val_score(pipe_all, X_all_binary, y_binary, cv=kf, scoring='recall_weighted').mean()
        f1_all = cross_val_score(pipe_all, X_all_binary, y_binary, cv=kf, scoring='f1_weighted').mean()

        accuracy_biomarkers = cross_val_score(pipe_biomarkers, X_biomarkers_binary, y_binary, cv=kf, scoring='accuracy').mean()
        precision_biomarkers = cross_val_score(pipe_biomarkers, X_biomarkers_binary, y_binary, cv=kf, scoring='precision_weighted').mean()
        recall_biomarkers = cross_val_score(pipe_biomarkers, X_biomarkers_binary, y_binary, cv=kf, scoring='recall_weighted').mean()
        f1_biomarkers = cross_val_score(pipe_biomarkers, X_biomarkers_binary, y_binary, cv=kf, scoring='f1_weighted').mean()

        accuracy_random = cross_val_score(pipe_random, X_random_binary, y_binary, cv=kf, scoring='accuracy').mean()
        precision_random = cross_val_score(pipe_random, X_random_binary, y_binary, cv=kf, scoring='precision_weighted').mean()
        recall_random = cross_val_score(pipe_random, X_random_binary, y_binary, cv=kf, scoring='recall_weighted').mean()
        f1_random = cross_val_score(pipe_random, X_random_binary, y_binary, cv=kf, scoring='f1_weighted').mean()

        print(f"Total number of samples in the dataset: {len(binary_data)}")
        print(f"Results for {label}:")
        print("Model with All Features:")
        print(f"Total Features: {X_all.shape[1]}")
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

        output_file.write(f"Results for {label}:\n")
        output_file.write("Model with All Features:\n")
        output_file.write(f"Accuracy: {accuracy_all:.4f}\n")
        output_file.write(f"Precision: {precision_all:.4f}\n")
        output_file.write(f"Recall: {recall_all:.4f}\n")
        output_file.write(f"F1 Score: {f1_all:.4f}\n")

        output_file.write("\nModel with Biomarkers:\n")
        output_file.write(f"Accuracy: {accuracy_biomarkers:.4f}\n")
        output_file.write(f"Precision: {precision_biomarkers:.4f}\n")
        output_file.write(f"Recall: {recall_biomarkers:.4f}\n")
        output_file.write(f"F1 Score: {f1_biomarkers:.4f}\n")

        output_file.write("\nModel with Randomly Selected Features:\n")
        output_file.write(f"Accuracy: {accuracy_random:.4f}\n")
        output_file.write(f"Precision: {precision_random:.4f}\n")
        output_file.write(f"Recall: {recall_random:.4f}\n")
        output_file.write(f"F1 Score: {f1_random:.4f}\n")
        output_file.write("\n" + "-"*50 + "\n\n")
