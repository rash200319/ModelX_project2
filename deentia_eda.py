import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')

# Define file path
file_path = r"C:\Users\user\OneDrive\Desktop\aicomp'\Dementia Prediction Dataset.csv"

# Load the data
try:
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Define allowed features based on the provided list
allowed_features = {
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

# Check which of our allowed features exist in the dataframe
existing_features = [col for col in allowed_features if col in df.columns]
missing_features = allowed_features - set(existing_features)

print(f"\nFound {len(existing_features)} out of {len(allowed_features)} features in the dataset")
print(f"Missing features: {missing_features}")

# Create filtered dataframe with only allowed features
filtered_df = df[existing_features].copy()

# Basic EDA
print("\n=== Basic Dataset Info ===")
filtered_df.info()

print("\n=== Summary Statistics ===")
print(filtered_df.describe(include='all').T)

# Check for missing values
print("\n=== Missing Values ===")
missing = filtered_df.isnull().sum()
missing_pct = (missing / len(filtered_df)) * 100
missing_df = pd.concat([missing, missing_pct], axis=1, 
                      keys=['Missing Count', 'Missing %'])
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing %', ascending=False))

# Check for special codes that represent missing/unknown values
special_codes = {
    '8': 'Unknown',
    '9': 'Unknown',
    '88': 'Unknown',
    '888': 'Unknown',
    '999': 'Unknown',
    '-4': 'Not assessed'
}

# Function to replace special codes with NaN
def clean_special_codes(series):
    if series.dtype == 'object':
        return series.replace({str(k): np.nan for k in special_codes.keys()})
    else:
        return series.replace({int(k): np.nan for k in special_codes.keys() 
                             if k.lstrip('-').isdigit()})

# Apply cleaning to all columns
for col in filtered_df.columns:
    filtered_df[col] = clean_special_codes(filtered_df[col])

# Check target variable distribution
if 'DEMENTED' in filtered_df.columns:
    print("\n=== Target Variable Distribution (DEMENTED) ===")
    print(filtered_df['DEMENTED'].value_counts(dropna=False))
    print("\n=== Target Variable Distribution (NORMCOG) ===")
    print(filtered_df['NORMCOG'].value_counts(dropna=False))

# Plot distributions of key variables
def plot_distributions(df, cols, n_cols=3):
    n_rows = (len(cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for i, col in enumerate(cols):
        if col in df.columns:
            if df[col].nunique() < 10:  # For categorical
                sns.countplot(data=df, x=col, ax=axes[i])
            else:  # For numerical
                sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
            axes[i].tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for j in range(len(cols), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()

# Select some key features to visualize
key_features = ['SEX', 'EDUC', 'MARISTAT', 'TOBAC30', 'ALCFREQ', 'HEIGHT', 'WEIGHT', 'NACCBMI']
plot_distributions(filtered_df, key_features)

# Correlation matrix for numerical variables
numerical_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns
if len(numerical_cols) > 1:
    plt.figure(figsize=(12, 10))
    corr = filtered_df[numerical_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                center=0, linewidths=0.5, cbar_kws={'shrink': 0.8})
    plt.title('Correlation Matrix of Numerical Variables')
    plt.tight_layout()
    plt.show()

# Save the cleaned dataframe for future use
output_path = r"c:\Users\user\OneDrive\Desktop\aicomp'\filtered_dementia_data.csv"
filtered_df.to_csv(output_path, index=False)
print(f"\nFiltered dataset saved to: {output_path}")

print("\n=== EDA Complete ===")