# main.py

from src.data_loader import load_data
from src.models import train_model
from src.evaluate import evaluate_model

import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Load and extract data
    df = load_data("data/student+performance.zip")  # updated zip path

    # Preprocessing
    df = pd.get_dummies(df, drop_first=True)
    target = "G3"  # You might want to adjust this depending on dataset

    # Train/test split
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(df, target_column=target)

    # Evaluate
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
