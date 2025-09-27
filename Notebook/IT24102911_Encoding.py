#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd

# Load dataset
df = pd.read_csv("output.csv")


# In[18]:


import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt  # Added missing import
import os

# Load the dataset
df = pd.read_csv('output.csv')

# Check the actual column names in your dataset
print("Available columns:", df.columns.tolist())

# Find the correct column name for accident severity
# After checking the available columns, use the correct column name
# For this example, let's assume the correct column name is 'Accident_Severity' (check the printed columns)
severity_column = 'Accident_Severity'  # Update this based on the actual column name in your dataset

# Check if the column exists before proceeding
if severity_column not in df.columns:
    print(f"Column '{severity_column}' not found. Please check the column names.")
    # You might want to list possible severity-related columns
    possible_columns = [col for col in df.columns if 'sever' in col.lower() or 'accid' in col.lower()]
    print(f"Possible severity-related columns: {possible_columns}")
else:
    # Encode target variable (Label Encoding)
    le = LabelEncoder()
    df[severity_column] = le.fit_transform(df[severity_column])

    # One-hot encode categorical columns (adjust column names if needed)
    weather_column = 'Weather_Conditions'  # Update based on actual column name
    light_column = 'Light_Conditions'      # Update based on actual column name

    # Get columns that actually exist in the dataset
    encode_columns = [col for col in [weather_column, light_column] if col in df.columns]
    df_encoded = pd.get_dummies(df, columns=encode_columns, drop_first=True)

    print("\nFirst 5 rows after encoding:")
    print(df_encoded.head())

    # Plot Accident Severity distribution
    plt.figure(figsize=(6,4))
    sns.countplot(x=severity_column, data=df)
    plt.title("Accident Severity Distribution")
    plt.xlabel("Severity (0=Slight, 1=Serious, 2=Fatal)")
    plt.ylabel("Count")
    plt.tight_layout()

    # Save plot
    os.makedirs("../results/eda_visualizations", exist_ok=True)
    plt.savefig("../results/eda_visualizations/encoding_accident_severity.png", dpi=150)
    plt.show()

    # Save encoded dataset for next member
    os.makedirs("../results/outputs", exist_ok=True)
    df_encoded.to_csv("../results/outputs/accidents_encoded.csv", index=False)
    print("\n✅ Encoded dataset saved to results/outputs/accidents_encoded.csv")

