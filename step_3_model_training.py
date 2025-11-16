import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, RocCurveDisplay

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')

print("=== Starting Step 3: Model Development & Training ===")

# --- 1. Load Engineered Data ---
file_path = r"c:\Users\user\OneDrive\Desktop\aicomp'\engineered_dementia_data.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Engineered data loaded. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading engineered data: {e}")
    raise

# --- 2. Define Target (y) and Feature Sets (X) ---
y = df['IMPAIRED']
# Drop rows where target is missing (should be none, but as a safeguard)
df.dropna(subset=['IMPAIRED'], inplace=True)
y = df['IMPAIRED']

# Model A: Full Feature Set
X_full = df.drop(columns=['IMPAIRED'])

# Model B: Risk Factor Set
# Exclude the functional status "proxy" variables
functional_proxies = ['INDEPEND', 'NACCLIVS', 'RESIDENC']
X_risk = df.drop(columns=['IMPAIRED'] + functional_proxies)

print(f"Full feature set shape: {X_full.shape}")
print(f"Risk-only feature set shape: {X_risk.shape}")

# --- 3. Define Preprocessing Pipelines ---
# We must use the *exact* same column lists and transforms as Step 2
# Note: We must re-define the column lists based on the *final* engineered file

# Define column types from the engineered file
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

# Ensure lists only contain columns present in their respective DataFrames
def get_valid_cols(df, cols_list):
    return [col for col in cols_list if col in df.columns]

# --- Pipeline for Model A (Full) ---
full_num_cols = get_valid_cols(X_full, numerical_cols)
full_cat_cols = get_valid_cols(X_full, categorical_cols)
full_health_cols = get_valid_cols(X_full, health_categorical_cols)

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

preprocessor_full = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, full_num_cols),
    ('cat', categorical_pipeline, full_cat_cols),
    ('health', health_pipeline, full_health_cols)
], remainder='passthrough')

# --- Pipeline for Model B (Risk) ---
# Note: categorical_cols list for this model *must not* include the proxies
risk_num_cols = get_valid_cols(X_risk, numerical_cols)
risk_cat_cols = get_valid_cols(X_risk, [c for c in categorical_cols if c not in functional_proxies])
risk_health_cols = get_valid_cols(X_risk, health_categorical_cols)

preprocessor_risk = ColumnTransformer(transformers=[
    ('num', numerical_pipeline, risk_num_cols),
    ('cat', categorical_pipeline, risk_cat_cols),
    ('health', health_pipeline, risk_health_cols)
], remainder='passthrough')

# --- 4. Split Data ---
# We split the *original* X and y, as preprocessing must be fit *only* on training data
X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)

X_risk_train, X_risk_test = train_test_split(
    X_risk, test_size=0.2, random_state=42, stratify=y
)

print(f"Training data shape: {X_full_train.shape[0]} samples")
print(f"Test data shape: {X_full_test.shape[0]} samples")

# --- 5. Define Models ---
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    "Random Forest": RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
}

# --- 6. Train and Evaluate ---
model_results = {}
fig, ax = plt.subplots(figsize=(10, 8))

# --- Train/Evaluate Model A (Full) ---
print("\n=== Training Models on FULL Feature Set (Model A) ===")
for name, model in models.items():
    print(f"\n--- {name} (Full Set) ---")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_full),
        ('final_imputer', SimpleImputer(strategy='most_frequent')),  # Handle any remaining NaNs
        ('model', model)
    ])
    
    # Train
    pipeline.fit(X_full_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_full_test)
    y_proba = pipeline.predict_proba(X_full_test)[:, 1]
    
    # Evaluate
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    
    print(f"Test Set AUC: {auc:.4f}")
    print("Classification Report:\n", report)
    
    model_results[f"{name} (Full)"] = {'auc': auc, 'pipeline': pipeline}
    RocCurveDisplay.from_estimator(pipeline, X_full_test, y_test, name=f"{name} (Full)", ax=ax)

# --- Train/Evaluate Model B (Risk) ---
print("\n=== Training Models on RISK-ONLY Feature Set (Model B) ===")
for name, model in models.items():
    print(f"\n--- {name} (Risk Set) ---")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor_risk),
        ('final_imputer', SimpleImputer(strategy='most_frequent')),  # Handle any remaining NaNs
        ('model', model)
    ])
    
    # Train
    pipeline.fit(X_risk_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_risk_test)
    y_proba = pipeline.predict_proba(X_risk_test)[:, 1]
    
    # Evaluate
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred)
    
    print(f"Test Set AUC: {auc:.4f}")
    print("Classification Report:\n", report)
    
    model_results[f"{name} (Risk)"] = {'auc': auc, 'pipeline': pipeline}
    RocCurveDisplay.from_estimator(pipeline, X_risk_test, y_test, name=f"{name} (Risk)", ax=ax, linestyle='--')

# Finalize and show plot
ax.set_title("ROC Curve Comparison: Full vs. Risk-Only Models")
ax.plot([0, 1], [0, 1], 'k--', label='Chance')
plt.legend()
plt.tight_layout()
plt.show()

print("\n=== Step 3 Complete ===")