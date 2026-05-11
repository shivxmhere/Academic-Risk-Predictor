import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Helper to ensure we resolve paths correctly whether running in root or notebooks dir
base_dir = '.'
if not os.path.exists('data/raw_students.csv'):
    base_dir = '..'

data_path = os.path.join(base_dir, 'data/raw_students.csv')
models_dir = os.path.join(base_dir, 'models')
data_dir = os.path.join(base_dir, 'data')

os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

print("STEP 1 - Load and inspect")
df = pd.read_csv(data_path)
initial_rows = len(df)
print(f"Shape before cleaning: {df.shape}")
print("Null counts before cleaning:")
print(df.isnull().sum())

print("\nSTEP 2 - Clean")
# Drop nulls
df.dropna(inplace=True)
# Drop duplicates
df.drop_duplicates(inplace=True)

final_rows = len(df)
print(f"Removed {initial_rows - final_rows} rows (nulls or duplicates).")
print(f"Shape after cleaning: {df.shape}")

# Encode risk_label: "High"=2, "Medium"=1, "Low"=0
label_mapping = {"High": 2, "Medium": 1, "Low": 0}
df['risk_label'] = df['risk_label'].map(label_mapping)

print("\nSTEP 3 - Feature engineering (IMPORTANT)")
# Create API (Academic Performance Index)
df['api'] = (df['attendance_pct'] * 0.3) + (df['internal_score'] * 0.4) + (df['prev_grade'] * 0.3)
print("Created 'api' feature.")

print("\nSTEP 4 - Define features")
features = ['attendance_pct', 'assignment_marks', 'internal_score', 'participation', 'prev_grade', 'study_hours', 'api']
X = df[features]
y = df['risk_label']

print("\nSTEP 5 - Stratified 80/20 split")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
print(f"Train size: {len(X_train)}")
print(f"Test size: {len(X_test)}")

print("\nSTEP 6 - Normalise (CRITICAL - fit on train only)")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Normalisation complete.")

print("\nSTEP 7 - Save everything")
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
np.save(os.path.join(models_dir, 'X_train.npy'), X_train_scaled)
np.save(os.path.join(models_dir, 'X_test.npy'), X_test_scaled)
np.save(os.path.join(models_dir, 'y_train.npy'), y_train.values)
np.save(os.path.join(models_dir, 'y_test.npy'), y_test.values)

df.to_csv(os.path.join(data_dir, 'cleaned_students.csv'), index=False)

print("\nPreprocessing complete. Files saved to models/ and data/.")
