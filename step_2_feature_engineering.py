import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')

print("=== Starting Step 2: Feature Engineering & Selection ===")

# --- 1. Load & Filter ---
# We must re-load the *original* file to get VISITYR
file_path = r"C:\Users\user\OneDrive\Desktop\aicomp'\Dementia Prediction Dataset.csv"

# Add 'VISITYR' to calculate Age. This is critical.
allowed_features = {
    # Metadata for Age
    'VISITYR',
    # Demographics
    'BIRTHMO', 'BIRTHYR', 'SEX', 'HISPANIC', 'RACE', 'RACESEC', 'RACETER', 
    'PRIMLANG', 'EDUC', 'MARISTAT', 'HANDED',
    # Living Situation
    'NACCLIVS', 'INDEPEND', 'RESIDENC',
    # Health Behaviors
    'TOBAC30', 'TOBAC100', 'SMOKYRS', 'PACKSPER', 'QUITSMOK', 'ALCOCCAS', 'ALCFREQ',
    # Self-Reported Medical History
    'CVHATT', 'CVAFIB', 'CVCHF', 'CVANGINA', 'CVBYPASS', 'CVANGIO', 'CVPACE', 
    'CBSTROKE', 'CBTIA', 'PD', 'SEIZURES', 'NACCTBI', 'TBI', 'DIABETES', 
    'HYPERTEN', 'HYPERCHO', 'B12DEF', 'THYROID', 'ARTHRIT',
    # Self-Reported Sleep
    'APNEA', 'RBD', 'INSOMN',
    # Self-Reported Psych
    'PTSD', 'BIPOLAR', 'SCHIZ', 'DEP2YRS', 'ANXIETY', 'OCD',
    # Vitals
    'HEIGHT', 'WEIGHT', 'NACCBMI', 'VISION', 'HEARING', 'BPSYS', 'BPDIAS',
    # Family History
    'NACCMOM', 'NACCDAD', 'NACCAM', 'NACCFM', 'NACCOM',
    # Target
    'DEMENTED', 'NORMCOG'
}

try:
    df = pd.read_csv(file_path, low_memory=False)
    # Filter to only the columns we are allowed to use
    existing_features = [col for col in allowed_features if col in df.columns]
    df_filtered = df[existing_features].copy()
    print(f"Data re-loaded and filtered. Shape: {df_filtered.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# --- 2. Clean Special Codes (Robust Version) ---

# These codes represent "Unknown", "Not Assessed", or "Not Applicable"
# We must replace them with NaN before imputation.
SPECIAL_CODES = [-4, 9, 88, 888, 999, 88.8, 888.8, 99.9, 999.9]

# Convert all special codes to np.nan
# This is more robust than the previous function.
print("Cleaning special codes...")
for col in df_filtered.columns:
    # Use .isin() for efficient replacement
    df_filtered[col] = df_filtered[col].replace(SPECIAL_CODES, np.nan)

print("Special codes replaced with NaN.")

# --- 3. Define Target Variable (y) ---

# Per the goal "at risk of dementia", we create a target 'IMPAIRED'
# 1 = Impaired (MCI or Dementia), 0 = Normal Cognition
# This uses the balanced NORMCOG variable.
df_filtered['IMPAIRED'] = 1 - df_filtered['NORMCOG']

# Drop the original target columns
df_filtered = df_filtered.drop(columns=['NORMCOG', 'DEMENTED'])

# Drop rows where the target is missing (if any)
df_filtered.dropna(subset=['IMPAIRED'], inplace=True)
df_filtered['IMPAIRED'] = df_filtered['IMPAIRED'].astype(int)

print("\n=== Target Variable Distribution (IMPAIRED) ===")
print(df_filtered['IMPAIRED'].value_counts(normalize=True))

# --- 4. Feature Engineering ---
print("\n=== Performing Feature Engineering ===")

# 1. Create 'Age'
# Drop rows where VISITYR or BIRTHYR is missing
df_filtered.dropna(subset=['VISITYR', 'BIRTHYR'], inplace=True)
df_filtered['Age'] = df_filtered['VISITYR'] - df_filtered['BIRTHYR']
# Remove people with improbable ages
df_filtered = df_filtered[(df_filtered['Age'] >= 40) & (df_filtered['Age'] <= 110)]

# 2. Family History of Cognitive Impairment
# 1 = Yes (any immediate family member), 0 = No, 9 = Unknown
fam_cols = ['NACCMOM', 'NACCDAD', 'NACCAM', 'NACCFM', 'NACCOM']
# Fill NaN with 9 (Unknown) *before* checking
df_filtered[fam_cols] = df_filtered[fam_cols].fillna(9)
# If any column is 1 (Yes), then 1. Else if all are 0 (No), then 0. Else 9.
df_filtered['FamHistory_CogImp'] = df_filtered[fam_cols].apply(
    lambda row: 1 if 1 in row.values else (0 if all(v == 0 for v in row.values) else 9),
    axis=1
).astype(int)

# 3. Simplify Race
# Create a simplified race/ethnicity variable
def simplify_race(row):
    if row['HISPANIC'] == 1:
        return 'Hispanic'
    if row['RACE'] == 1:
        return 'White'
    if row['RACE'] == 2:
        return 'Black'
    if row['RACE'] == 3:
        return 'American Indian'
    if row['RACE'] == 4:
        return 'Hawaiian/Pacific'
    if row['RACE'] == 5:
        return 'Asian'
    if row['RACE'] == 6:
        return 'Other'
    if row['RACE'] == 50:
        return 'Multiple'
    return 'Unknown' # Includes NaN, 99, etc.

df_filtered['Race_Simplified'] = df_filtered.apply(simplify_race, axis=1)

# 4. Handle other categorical 'Unknown'/'NA'
# For binary (0=No, 1=Yes) health history variables, 
# we'll impute NaN (which were -4 or 9) with a '9' (Unknown)
# This treats "Not Assessed" as its own category.
health_cols = [
    'CVHATT', 'CVAFIB', 'CVCHF', 'CVANGINA', 'CVBYPASS', 'CVANGIO', 'CVPACE', 
    'CBSTROKE', 'CBTIA', 'PD', 'SEIZURES', 'NACCTBI', 'TBI', 'DIABETES', 
    'HYPERTEN', 'HYPERCHO', 'B12DEF', 'THYROID', 'ARTHRIT', 'APNEA', 'RBD', 
    'INSOMN', 'PTSD', 'BIPOLAR', 'SCHIZ', 'DEP2YRS', 'ANXIETY', 'OCD',
    'TOBAC30', 'TOBAC100'
]

for col in health_cols:
    df_filtered[col] = df_filtered[col].fillna(9).astype(int)

# --- 5. Define Final Feature Set (X) and Target (y) ---

# Drop raw/source columns used for engineering
df_features = df_filtered.drop(columns=[
    'IMPAIRED', 'VISITYR', 'BIRTHYR', 'BIRTHMO',
    'NACCMOM', 'NACCDAD', 'NACCAM', 'NACCFM', 'NACCOM', # Raw fam history
    'RACE', 'RACESEC', 'RACETER', 'HISPANIC' # Raw race
])

y = df_filtered['IMPAIRED']
X = df_features

print(f"\nFinal feature set shape: {X.shape}")
print("Final features:", X.columns.tolist())

# --- 6. Preprocessing & Feature Selection ---

# Identify column types
numerical_cols = ['Age', 'EDUC', 'SMOKYRS', 'PACKSPER', 'QUITSMOK', 
                  'HEIGHT', 'WEIGHT', 'NACCBMI', 'BPSYS', 'BPDIAS']
categorical_cols = ['SEX', 'PRIMLANG', 'MARISTAT', 'HANDED', 'NACCLIVS', 
                    'INDEPEND', 'RESIDENC', 'VISION', 'HEARING', 'Race_Simplified']
# These are ordinal (0, 1, 9) but we'll treat them as categorical
health_categorical_cols = health_cols + ['FamHistory_CogImp']

# Remove cols that might be all NaN (if any)
X.dropna(axis=1, how='all', inplace=True)
# Update lists
numerical_cols = [col for col in numerical_cols if col in X.columns]
categorical_cols = [col for col in categorical_cols if col in X.columns]
health_categorical_cols = [col for col in health_categorical_cols if col in X.columns]

# Create Preprocessing Pipelines
# This ensures imputation and scaling happen correctly
numerical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# For health, 0=No, 1=Yes, 9=Unknown. SimpleImputer with 9 is fine.
health_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value=9)),
    ('onehot', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False))
])

# Create the full preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols),
    ('health', health_pipeline, health_categorical_cols)
], remainder='passthrough')

# Apply preprocessing to the data FOR FEATURE SELECTION
print("\nPreprocessing data for feature selection...")
X_processed = preprocessor.fit_transform(X)

# Get feature names after transformation
feature_names = preprocessor.get_feature_names_out()

# Convert to DataFrame for easier handling - handle both sparse and dense arrays
if hasattr(X_processed, 'toarray'):  # If it's a sparse matrix
    X_processed_df = pd.DataFrame.sparse.from_spmatrix(X_processed, columns=feature_names)
else:  # If it's already a dense array
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

# Check for any remaining NaN values
if X_processed_df.isna().any().any():
    nan_columns = X_processed_df.columns[X_processed_df.isna().any()].tolist()
    print(f"\nWarning: Found {len(nan_columns)} columns with NaN values after preprocessing.")
    print("Filling NaN values with 0 (for binary/one-hot encoded features, this represents the absence of that category).")
    X_processed_df.fillna(0, inplace=True)

# Convert back to numpy array for SelectKBest
X_processed = X_processed_df.values

print(f"Data processed. Final shape: {X_processed.shape}")

# --- 7. Run Feature Selection ---
# Use f_classif (ANOVA F-value) for numerical/categorical features
print("Running feature selection (SelectKBest)...")
selector = SelectKBest(f_classif, k='all')
selector.fit(X_processed, y)

# Get scores and names
scores = selector.scores_
p_values = selector.pvalues_

# Create a DataFrame for scores
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Score': scores,
    'P_Value': p_values
}).sort_values(by='Score', ascending=False).reset_index(drop=True)

print("\n=== Top 25 Most Predictive Features ===\n")
print(feature_importance.head(25))

# Plot the top 25 features
plt.figure(figsize=(12, 10))
sns.barplot(
    x='Score', 
    y='Feature', 
    data=feature_importance.head(25),
    palette='viridis'
)
plt.title('Top 25 Most Predictive Features (SelectKBest F-score)')
plt.xlabel('F-score (Higher is More Predictive)')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()


# Save the fully cleaned and engineered data for the next step
final_cleaned_df = pd.concat([X, y], axis=1)
output_path = r"c:\Users\user\OneDrive\Desktop\aicomp'\engineered_dementia_data.csv"
final_cleaned_df.to_csv(output_path, index=False)
print(f"\nFully engineered dataset saved to: {output_path}")

print("\n=== Step 2 Complete ===")