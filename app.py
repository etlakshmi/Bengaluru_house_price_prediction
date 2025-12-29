import streamlit as st
import json
import os
from main import predict_price

# -----------------------------
# Load columns (for locations)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
columns_path = os.path.join(BASE_DIR, "model", "columns.json")

with open(columns_path, "r") as f:
    data_columns = json.load(f)

# First 3 columns are numerical features
locations = data_columns[3:]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Bangalore House Price Predictor", layout="centered")

st.title("🏠 Bengaluru House Price Prediction")
st.write("Enter the details below to predict the house price (in Lakhs)")

# Inputs
location = st.selectbox("📍 Location", sorted(locations))
total_sqft = st.number_input("📐 Total Square Feet", min_value=300.0, max_value=10000.0, step=50.0)
bhk = st.number_input("🛏️ BHK", min_value=1, max_value=10, step=1)
bath = st.number_input("🛁 Bathrooms", min_value=1, max_value=10, step=1)

# Predict button
if st.button("🔮 Predict Price"):
    price = predict_price(location, total_sqft, bhk, bath)

    if price < 0:
        st.error("Prediction failed. Please check inputs.")
    else:
        st.success(f"💰 Estimated Price: **₹ {price} Lakhs**")
