import os
import zipfile
import pandas as pd

def extract_zip_if_needed(filepath):
    import zipfile

    # Step 1: Extract the outer ZIP
    if zipfile.is_zipfile(filepath):
        with zipfile.ZipFile(filepath, 'r') as outer_zip:
            outer_zip.extractall("data/")
            inner_zip_name = next((f for f in outer_zip.namelist() if f.endswith(".zip")), None)
            if inner_zip_name:
                inner_zip_path = os.path.join("data", inner_zip_name)

                # Step 2: Extract the inner ZIP
                with zipfile.ZipFile(inner_zip_path, 'r') as inner_zip:
                    inner_zip.extractall("data/")
                    csv_name = next((f for f in inner_zip.namelist() if f.endswith(".csv")), None)
                    if csv_name:
                        return os.path.join("data", csv_name)

    raise FileNotFoundError("❌ No CSV file found in the ZIP archive.")


def load_data(filepath):
    extracted_path = extract_zip_if_needed(filepath)
    # 👇 Add the separator for semicolon-delimited CSVs
    return pd.read_csv(extracted_path, sep=';')
