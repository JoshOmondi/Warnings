# src/api.py

from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

app = Flask(__name__)

# Load model and feature order
model = load("models/random_forest.pkl")
features = load("models/features.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    try:
        df = pd.DataFrame([data])
        df = pd.get_dummies(df)
        df = df.reindex(columns=features, fill_value=0)

        prediction = model.predict(df)[0]
        return jsonify({"prediction": round(prediction, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
