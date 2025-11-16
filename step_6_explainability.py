import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
import os
import time
from sklearn.model_selection import train_test_split

print("=== Starting Step 6: Explainability (Kernel Explainer - FINAL FIX) ===")
shap.initjs()

# Configuration:
SAMPLE_SIZE = 100  # We will keep this small to be fast
BACKGROUND_SIZE = 50 # Background data for the explainer
CACHE_DIR = "./shap_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- 1. Load Engineered Data ---
file_path = r"c:\Users\user\OneDrive\Desktop\aicomp'\engineered_dementia_data.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error loading engineered data: {e}")
    raise

df.dropna(subset=['IMPAIRED'], inplace=True)
y = df['IMPAIRED']
X_full = df.drop(columns=['IMPAIRED'])
functional_proxies = ['INDEPEND', 'NACCLIVS', 'RESIDENC']
X_risk = df.drop(columns=['IMPAIRED'] + functional_proxies)

# --- 2. Split Data ---
X_full_train, X_full_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)
X_risk_train, X_risk_test = train_test_split(
    X_risk, test_size=0.2, random_state=42, stratify=y
)
print("Data loaded and split.")

# --- 3. Load Final Models ---
try:
    pipeline_full = joblib.load('final_model_full.joblib')
    pipeline_risk = joblib.load('final_model_risk.joblib')
    print("Final models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    raise

# --- 4. Preprocess data (needed for explainers) ---
preprocessor_full = pipeline_full.named_steps['preprocessor']
preprocessor_risk = pipeline_risk.named_steps['preprocessor']

# Transform train and test sets
X_full_train_transformed = preprocessor_full.fit_transform(X_full_train)
X_full_test_transformed = preprocessor_full.transform(X_full_test)
feature_names_full = preprocessor_full.get_feature_names_out()

X_risk_train_transformed = preprocessor_risk.fit_transform(X_risk_train)
X_risk_test_transformed = preprocessor_risk.transform(X_risk_test)
feature_names_risk = preprocessor_risk.get_feature_names_out()

print("Data preprocessing is complete.")

# --- 5. Explain Model A (Full Set) using KernelExplainer ---
print("\n--- Generating SHAP plot for Model A (Full Set) using KernelExplainer ---")

# Create a small "background" dataset for the explainer
background_full = shap.kmeans(X_full_train_transformed, BACKGROUND_SIZE).data
# Get a small sample of the test set to explain
X_test_sample_full = shap.sample(X_full_test_transformed, SAMPLE_SIZE, random_state=42)

# Define the prediction function for the pipeline
def predict_proba_full(data):
    # Data is *already transformed* (as numpy), so just pass to model
    return pipeline_full.named_steps['model'].predict_proba(data)

# Cache path
cache_path_full = os.path.join(CACHE_DIR, f'shap_values_full_kernel_{SAMPLE_SIZE}.joblib')

if os.path.exists(cache_path_full):
    print(f"Loading cached SHAP values from {cache_path_full}")
    shap_values_full_raw = joblib.load(cache_path_full)
else:
    print(f"Computing SHAP values for Model A (Kernel)...")
    t0 = time.time()
    explainer_full = shap.KernelExplainer(predict_proba_full, background_full)
    
    # Pass the numpy array sample to the explainer
    shap_values_full_raw = explainer_full.shap_values(X_test_sample_full) 
    
    joblib.dump(shap_values_full_raw, cache_path_full)
    print(f"SHAP computation finished in {time.time()-t0:.1f}s and cached.")

# --- THIS IS THE FIX ---
# We need the SHAP values for Class 1 (Impaired)
# The output can be a list of 2 arrays, or a 3D array (samples, features, classes)
# This code handles both.
if isinstance(shap_values_full_raw, list):
    shap_values_full_class1 = shap_values_full_raw[1] # It's a list, take 2nd element
elif hasattr(shap_values_full_raw, 'ndim') and shap_values_full_raw.ndim == 3:
    shap_values_full_class1 = shap_values_full_raw[:, :, 1] # It's 3D, take 2nd class
else:
    shap_values_full_class1 = shap_values_full_raw # Not sure, just use it

# Create the DataFrame for plotting (must match SHAP values shape)
X_test_sample_full_df = pd.DataFrame(X_test_sample_full, columns=feature_names_full)

# Plot
print("Showing SHAP Summary Plot for Model A...")
plt.title("Model A (Full Set) - Feature Importance (Kernel SHAP)")
shap.summary_plot(
    shap_values_full_class1,
    X_test_sample_full_df, 
    max_display=20,
    plot_type="bar", 
    show=False,
)
plt.tight_layout()
plt.show()

# --- 6. Explain Model B (Risk Set) using KernelExplainer ---
print("\n--- Generating SHAP plot for Model B (Risk Set) using KernelExplainer ---")

# Create background and sample
background_risk = shap.kmeans(X_risk_train_transformed, BACKGROUND_SIZE).data
X_test_sample_risk = shap.sample(X_risk_test_transformed, SAMPLE_SIZE, random_state=42)

# Define prediction function
def predict_proba_risk(data):
    return pipeline_risk.named_steps['model'].predict_proba(data)

# Cache path
cache_path_risk = os.path.join(CACHE_DIR, f'shap_values_risk_kernel_{SAMPLE_SIZE}.joblib')

if os.path.exists(cache_path_risk):
    print(f"Loading cached SHAP values from {cache_path_risk}")
    shap_values_risk_raw = joblib.load(cache_path_risk)
else:
    print(f"Computing SHAP values for Model B (Kernel)...")
    t0 = time.time()
    explainer_risk = shap.KernelExplainer(predict_proba_risk, background_risk)
    shap_values_risk_raw = explainer_risk.shap_values(X_test_sample_risk) 
    joblib.dump(shap_values_risk_raw, cache_path_risk)
    print(f"SHAP computation finished in {time.time()-t0:.1f}s and cached.")

# --- THIS IS THE FIX ---
if isinstance(shap_values_risk_raw, list):
    shap_values_risk_class1 = shap_values_risk_raw[1]
elif hasattr(shap_values_risk_raw, 'ndim') and shap_values_risk_raw.ndim == 3:
    shap_values_risk_class1 = shap_values_risk_raw[:, :, 1]
else:
    shap_values_risk_class1 = shap_values_risk_raw

# Create DataFrame for plotting
X_test_sample_risk_df = pd.DataFrame(X_test_sample_risk, columns=feature_names_risk)

# Plot
print("Showing SHAP Summary Plot for Model B...")
plt.title("Model B (Risk Set) - Feature Importance (Kernel SHAP)")
shap.summary_plot(
    shap_values_risk_class1,
    X_test_sample_risk_df,
    max_display=20,
    plot_type="bar",
    show=False,
)
plt.tight_layout()
plt.show()

print("\n=== Step 6 Complete (Kernel Explainer - FINAL FIX) ===")