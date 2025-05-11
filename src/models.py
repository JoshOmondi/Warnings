# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

def train_model(df, target_column="charges"):
    # Prepare data
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    dump(model, "models/random_forest.pkl")
    dump(X.columns, "models/features.pkl")

    print("✅ Model trained and saved.")
    return model
