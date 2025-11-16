import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from sklearn.model_selection import train_test_split

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=== Starting Step 6: Explainability & Insights (Faster Version) ===")
# This tells SHAP to play nice with matplotlib
shap.initjs()

# --- 1. Load Engineered Data (to get the test set) ---
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
X_full = df.drop(columns=['IMPAIRED'])
functional_proxies = ['INDEPEND', 'NACCLIVS', 'RESIDENC']
X_risk = df.drop(columns=['IMPAIRED'] + functional_proxies)

# --- 3. Split Data (to get the *exact same* test set) ---
X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)
X_risk_train, X_risk_test = train_test_split(
    X_risk, test_size=0.2, random_state=42, stratify=y
)

# --- 4. Load Final Models ---
try:
    pipeline_full = joblib.load('final_model_full.joblib')
    pipeline_risk = joblib.load('final_model_risk.joblib')
    print("Final models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# --- 5. Explain Model A (Full Set) ---
print("\n--- Generating SHAP plot for Model A (Full Set) ---")
# Extract the components
preprocessor_full = pipeline_full.named_steps['preprocessor']
model_full = pipeline_full.named_steps['model']
# Get feature names AFTER preprocessing
feature_names_full = preprocessor_full.get_feature_names_out()
# Transform the test set
X_test_transformed_full = preprocessor_full.transform(X_full_test)

# === THIS LINE IS CHANGED ===
print("Using a smaller sample (500) for faster results...")
X_test_sample_full = shap.sample(X_test_transformed_full, 500, random_state=42) # Changed 2000 to 500

# Create explainer
explainer_full = shap.TreeExplainer(model_full)
# Calculate SHAP values
shap_values_full = explainer_full.shap_values(X_test_sample_full)

# Plot
print("Showing SHAP Summary Plot for Model A...")
plt.title("Model A (Full Set) - Feature Importance")
shap.summary_plot(
    shap_values_full[1], # [1] is for the 'Impaired' class
    X_test_sample_full, 
    feature_names=feature_names_full,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.show()

# --- 6. Explain Model B (Risk Set) ---
print("\n--- Generating SHAP plot for Model B (Risk Set) ---")
# Extract components
preprocessor_risk = pipeline_risk.named_steps['preprocessor']
model_risk = pipeline_risk.named_steps['model']
# Get feature names
feature_names_risk = preprocessor_risk.get_feature_names_out()
# Transform the test set
X_test_transformed_risk = preprocessor_risk.transform(X_risk_test)

# === THIS LINE IS CHANGED ===
print("Using a smaller sample (500) for faster results...")
X_test_sample_risk = shap.sample(X_test_transformed_risk, 500, random_state=42) # Changed 2000 to 500

# Create explainer
explainer_risk = shap.TreeExplainer(model_risk)
# Calculate SHAP values
shap_values_risk = explainer_risk.shap_values(X_test_sample_risk)

# Plot
print("Showing SHAP Summary Plot for Model B...")
plt.title("Model B (Risk Set) - Feature Importance")
shap.summary_plot(
    shap_values_risk[1], # [1] is for the 'Imformed' class
    X_test_sample_risk, 
    feature_names=feature_names_risk,
    max_display=20,
    show=False
)
plt.tight_layout()
plt.show()

print("\n=== Step 6 Complete ===")