# src/data_loader.py

import os
import zipfile
import pandas as pd

def extract_zip_if_needed(zip_path, extract_to="data/"):
    """Extracts CSV from zip if it hasn't been extracted yet."""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith('.csv'):
                extracted_path = os.path.join(extract_to, file)
                if not os.path.exists(extracted_path):
                    zip_ref.extract(file, path=extract_to)
                    print(f"✅ Extracted: {file}")
                else:
                    print(f"ℹ️ Already extracted: {file}")
                return extracted_path
    raise FileNotFoundError("❌ No CSV file found in the ZIP archive.")

def load_data(filepath):
    """Loads CSV or extracts from zip and loads it."""
    if filepath.endswith('.zip'):
        filepath = extract_zip_if_needed(filepath)
    return pd.read_csv(filepath)
