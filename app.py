from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

# Get the absolute path of the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Load pre-trained Isolation Forest model
model = joblib.load("model.joblib")

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Anomaly Detection API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Ensure required columns exist
        required_columns = ["User", "API_Call", "Time", "IP_Numeric"]
        for col in required_columns:
            if col not in df.columns:
                return jsonify({"error": f"Missing required field: {col}"}), 400

        # Make predictions
        predictions = model.predict(df)

        # Convert predictions (-1 = anomaly, 1 = normal)
        df["Anomaly"] = ["Suspicious" if pred == -1 else "Normal" for pred in predictions]

        return jsonify(df.to_dict(orient="records"))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
