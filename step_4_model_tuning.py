import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import randint

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=== Starting Step 4: Hyperparameter Tuning & Optimization ===")

# --- 1. Load Engineered Data ---
file_path = r"c:\Users\user\OneDrive\Desktop\aicomp'\engineered_dementia_data.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Engineered data loaded. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading engineered data: {e}")
    raise

# --- 2. Define Target (y) and Feature Sets (X) ---
df.dropna(subset=['IMPAIRED'], inplace=True)
y = df['IMPAIRED']

# Model A: Full Feature Set
X_full = df.drop(columns=['IMPAIRED'])

# Model B: Risk Factor Set
functional_proxies = ['INDEPEND', 'NACCLIVS', 'RESIDENC']
X_risk = df.drop(columns=['IMPAIRED'] + functional_proxies)

print(f"Full feature set shape: {X_full.shape}")
print(f"Risk-only feature set shape: {X_risk.shape}")

# --- 3. Define Preprocessing Pipelines (Same as Step 3) ---
numerical_cols = ['Age', 'EDUC', 'SMOKYRS', 'PACKSPER', 'QUITSMOK', 
                  'HEIGHT', 'WEIGHT', 'NACCBMI', 'BPSYS', 'BPDIAS']
categorical_cols = ['SEX', 'PRIMLANG', 'MARISTAT', 'HANDED', 'NACCLIVS', 
                    'INDEPEND', 'RESIDENC', 'VISION', 'HEARING', 'Race_Simplified']
health_categorical_cols = [
    'CVHATT', 'CVAFIB', 'CVCHF', 'CVANGINA', 'CVBYPASS', 'CVANGIO', 'CVPACE', 
    'CBSTROKE', 'CBTIA', 'PD', 'SEIZURES', 'NACCTBI', 'TBI', 'DIABETES', 
    'HYPERTEN', 'HYPERCHO', 'B12DEF', 'THYROID', 'ARTHRIT', 'APNEA', 'RBD', 
    'INSOMN', 'PTSD', 'BIPOLAR', 'SCHIZ', 'DEP2YRS', 'ANXIETY', 'OCD',
    'TOBAC30', 'TOBAC100', 'FamHistory_CogImp'
]

def get_valid_cols(df, cols_list):
    return [col for col in cols_list if col in df.columns]

numerical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

health_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='constant', fill_value=9)),
    ('onehot', OneHotEncoder(categories='auto', handle_unknown='ignore', sparse_output=False))
])

# --- Preprocessor for Model A (Full) ---
preprocessor_full = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, get_valid_cols(X_full, numerical_cols)),
    ('cat', categorical_pipeline, get_valid_cols(X_full, categorical_cols)),
    ('health', health_pipeline, get_valid_cols(X_full, health_categorical_cols))
], remainder='passthrough')

# --- Preprocessor for Model B (Risk) ---
risk_cat_cols = [c for c in categorical_cols if c not in functional_proxies]
preprocessor_risk = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, get_valid_cols(X_risk, numerical_cols)),
    ('cat', categorical_pipeline, get_valid_cols(X_risk, risk_cat_cols)),
    ('health', health_pipeline, get_valid_cols(X_risk, health_categorical_cols))
], remainder='passthrough')

# --- 4. Split Data ---
X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)
X_risk_train, X_risk_test = train_test_split(
    X_risk, test_size=0.2, random_state=42, stratify=y
)

# --- 5. Hyperparameter Tuning Setup ---

# Create the full pipeline with the model
# 'model' is the prefix for the classifier's parameters
pipeline = Pipeline(steps=[
    ('preprocessor', None), # Will be replaced
    ('model', RandomForestClassifier(random_state=42, n_jobs=-1))
])

# Define the parameter grid to search
# These are parameters for the 'model' step of the pipeline
param_dist = {
    'model__n_estimators': randint(100, 500),       # Number of trees
    'model__max_depth': [None, 10, 20, 30],         # Max depth of trees
    'model__min_samples_split': randint(2, 11),     # Min samples to split
    'model__min_samples_leaf': randint(1, 11),      # Min samples at leaf
    'model__max_features': ['sqrt', 'log2', None]   # Features to use
}

# Set up RandomizedSearchCV
# n_iter=10 means it will try 10 different combinations.
# cv=3 is 3-fold cross-validation.
# We'll use n_iter=10 and cv=3 to get good results quickly.
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=10, 
    cv=3, 
    scoring='roc_auc', 
    verbose=2, 
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# --- 6. Tune Model A (Full Set) ---
print("\n=== Tuning Model A: Random Forest (Full Set) ===")
start_time = time.time()
# Set the correct preprocessor
random_search.estimator.set_params(preprocessor=preprocessor_full)

# Fit
random_search.fit(X_full_train, y_train)

print(f"Tuning finished in {time.time() - start_time:.2f} seconds")
print(f"Best AUC Score (Full Set): {random_search.best_score_:.4f}")
print("Best Parameters (Full Set):")
print(random_search.best_params_)

# Save the best model
best_model_full = random_search.best_estimator_

# --- 7. Tune Model B (Risk Set) ---
print("\n=== Tuning Model B: Random Forest (Risk Set) ===")
start_time = time.time()
# Set the correct preprocessor
random_search.estimator.set_params(preprocessor=preprocessor_risk)

# Fit
random_search.fit(X_risk_train, y_train)

print(f"Tuning finished in {time.time() - start_time:.2f} seconds")
print(f"Best AUC Score (Risk Set): {random_search.best_score_:.4f}")
print("Best Parameters (Risk Set):")
print(random_search.best_params_)

# Save the best model
best_model_risk = random_search.best_estimator_

# --- 8. Final Evaluation on Test Set ---
print("\n=== Final Evaluation on Held-Out Test Set ===")

# Model A (Full)
y_pred_full = best_model_full.predict(X_full_test)
y_proba_full = best_model_full.predict_proba(X_full_test)[:, 1]
print("\n--- Tuned Random Forest (Full Set) ---")
print(f"Test Set AUC: {roc_auc_score(y_test, y_proba_full):.4f}")
print(classification_report(y_test, y_pred_full))

# Model B (Risk)
y_pred_risk = best_model_risk.predict(X_risk_test)
y_proba_risk = best_model_risk.predict_proba(X_risk_test)[:, 1]
print("\n--- Tuned Random Forest (Risk Set) ---")
print(f"Test Set AUC: {roc_auc_score(y_test, y_proba_risk):.4f}")
print(classification_report(y_test, y_pred_risk))

print("\n=== Step 4 Complete ===")