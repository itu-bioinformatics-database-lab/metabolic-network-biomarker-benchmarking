import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load data
data = pd.read_csv('metabolomics.csv')
target_column = 'Diagnosis'

# Middle step: Remove columns that start with 'X -'
cols_to_remove = [col for col in data.columns if col.startswith('X -')]
data.drop(columns=cols_to_remove, inplace=True)

# 2. Preprocess target
data[target_column] = data[target_column].replace('AD+', 'AD')
classes = ['Control', 'AD']  # adjust as needed

# Filter binary classification subset
binary_data = data[data[target_column].isin(classes)].copy()
binary_data[target_column] = LabelEncoder().fit_transform(binary_data[target_column])

# Drop non-metabolite columns
# non_metabolite_columns = ['Factors']  # extend if needed
non_metabolite_columns = ["Diagnosis", "Sample ID", "Gender", "Race", "Braak"]
X = binary_data.drop(columns=non_metabolite_columns)
y = binary_data[target_column]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== METHOD 1: Random Forest Feature Importance ==========
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# Feature importance
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("Top features by Random Forest Importance:")
n_features_to_select = 120
print(importance_df.head(n_features_to_select))


# ========== METHOD 2: Recursive Feature Elimination (RFE) ==========
# Use a smaller estimator for speed
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(max_iter=1000)

# RFE for top n features
rfe = RFE(log_reg, n_features_to_select=n_features_to_select)
rfe.fit(X_scaled, y)

# Selected features
selected_features = X.columns[rfe.support_].tolist()
print("\nTop features selected by RFE:")
print(selected_features)

# Calculate the intersection of features from both methods
intersection_features = set(importance_df['Feature'].head(n_features_to_select)) & set(selected_features)
print("\nIntersection of features from both methods:")
print(intersection_features)
print(f"Number of intersecting features: {len(intersection_features)}")


# Write intersection, RF importance, or RFE selected features to files according to user input(RF, RFE, I)
user_choice = input("Enter 'RF' for Random Forest features, 'RFE' for RFE features, or 'I' for intersection: ").strip().upper()
if user_choice == 'RF':
    features_to_save = importance_df['Feature'].head(n_features_to_select).tolist()
elif user_choice == 'RFE':
    features_to_save = selected_features
elif user_choice == 'I':
    features_to_save = list(intersection_features)
else:
    print("Invalid choice. Exiting.")
    exit()

with open('result_list.txt', 'w') as f:
    for feature in features_to_save:
        f.write(f"{feature}\n")
