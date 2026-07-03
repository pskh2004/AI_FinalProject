# src/data_loader.py
import os
import pandas as pd

def load_raw_data(data_path="data/v1/telco_customer_churn.csv"):
    """
    Loads raw dataset from the specified path.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Raw dataset not found at {data_path}. Please place the file first.")
    df = pd.read_csv(data_path)
    return df

def save_data_version(df, path):
    """
    Saves a specific version of the dataset to the specified path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[+] Data successfully saved to: {path}")