import pickle
import json
import numpy as np
import pandas as pd
import os

# -----------------------------
# Load model and columns
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "model.pkl")
columns_path = os.path.join(BASE_DIR, "model", "columns.json")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(columns_path, "r") as f:
    data_columns = json.load(f)


# -----------------------------
# Prediction function
# -----------------------------
def predict_price(location, total_sqft, bhk, bath):
    """
    Predict house price in Bangalore (in Lakhs)
    Model trained on log1p(price), so we reverse it here.
    """

    # Create input with all columns initialized to 0
    input_data = dict.fromkeys(data_columns, 0)

    # Numerical features
    input_data['total_sqft'] = total_sqft
    input_data['bath'] = bath
    input_data['bhk'] = bhk

    # Location (one-hot)
    if location in input_data:
        input_data[location] = 1
    else:
        # Handles unseen locations safely
        if 'other' in input_data:
            input_data['other'] = 1

    # Convert to DataFrame to preserve column order
    input_df = pd.DataFrame([input_data])

    # Model predicts log(price)
    log_price = model.predict(input_df)[0]

    # Reverse log transform
    price = np.expm1(log_price)

    return round(price, 2)


# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    price = predict_price(
        location="1st Phase JP Nagar",
        total_sqft=1000,
        bhk=2,
        bath=2
    )
    print("Predicted Price:", price, "Lakhs")
