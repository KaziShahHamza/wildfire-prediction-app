import streamlit as st
from datetime import date
import joblib
from utils.weather import get_weather
from utils.ndvi import get_ndvi_data
from utils.predict import predict_wildfire
import numpy as np

st.set_page_config(page_title="California Wildfire Prediction", layout="wide")
st.title("California Wildfire Prediction (Live)")

# Load model and scaler
model = joblib.load("wildfire_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Get current date
today = date.today()
st.markdown(f"**Present Date:** {today}")

# Fetch live weather
st.info("Fetching live weather data...")
precip, max_temp, min_temp, wind = get_weather()

# Fetch NDVI
st.info("Fetching NDVI data from GEE...")
ndvi, ndvi_7, ndvi_30 = get_ndvi_data(today)

# Display features
st.subheader("Input Features")
st.write({
    "Precipitation (mm)": precip,
    "Max Temperature (°C)": max_temp,
    "Min Temperature (°C)": min_temp,
    "Wind Speed (m/s)": wind,
    "NDVI (today)": ndvi,
    "NDVI 7-day avg": ndvi_7,
    "NDVI 30-day avg": ndvi_30
})

# Prepare feature vector
features = np.array([[precip, max_temp, min_temp, wind, ndvi, ndvi_7, ndvi_30]])

# Predict wildfire
proba, risk = predict_wildfire(model, scaler, features)

# Display result
st.subheader("Wildfire Prediction")
st.write(f"Probability: {proba}%")
st.write(f"Risk Level: {risk}")
