from src.data_loader import load_data
from src.models import train_model
from src.evaluate import evaluate_model
import pandas as pd  # ⬅️ Needed for get_dummies

def main():
    # Load data
    df = load_data("data/student+performance.zip")
    
    # Print columns for verification
    print("✅ Columns found in dataset:", df.columns.tolist())

    # Define your target column
    target = "G3"

    # Check if target column exists
    if target not in df.columns:
        print(f"❌ Target column '{target}' not found in dataset.")
        print("🧠 Tip: Use one of these columns as your target instead:", df.columns.tolist())
        return

    # Prepare features and labels
    X = df.drop(columns=[target])
    y = df[target]

    # Convert categorical features to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Train model
    model = train_model(X, y)

    # Evaluate model
    evaluate_model(model, X, y)

if __name__ == "__main__":
    main()
