import streamlit as st
import numpy as np
import joblib

# ---------------------
# Load model and scaler
# ---------------------
@st.cache_resource
def load_model():
    model = joblib.load("wildfire_rf_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model()

# ---------------------
# App UI
# ---------------------
st.title("Wildfire Prediction App (California)")
st.markdown("Enter today's weather and NDVI values:")

# Input fields (7 features)
precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, value=0.0, step=0.1)
max_temp = st.number_input("Max Temperature (°C)", min_value=-10.0, max_value=50.0, value=35.0, step=0.1)
min_temp = st.number_input("Min Temperature (°C)", min_value=-10.0, max_value=50.0, value=20.0, step=0.1)
avg_wind_speed = st.number_input("Average Wind Speed (m/s)", min_value=0.0, max_value=50.0, value=5.0, step=0.1)
ndvi = st.number_input("NDVI (today)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
ndvi_7day_avg = st.number_input("NDVI 7-day average", min_value=0.0, max_value=1.0, value=0.28, step=0.01)
ndvi_30day_avg = st.number_input("NDVI 30-day average", min_value=0.0, max_value=1.0, value=0.32, step=0.01)

# Predict button
if st.button("Predict Wildfire"):
    # Prepare feature array
    features = np.array([[precipitation, max_temp, min_temp, avg_wind_speed,
                          ndvi, ndvi_7day_avg, ndvi_30day_avg]])
    # Scale features
    scaled = scaler.transform(features)
    
    # Predict probability
    proba = model.predict_proba(scaled)[0][1]  # probability of wildfire
    risk_percent = proba * 100

    # Display with color
    if risk_percent < 30:
        st.success(f"✅ Wildfire probability: {risk_percent:.1f}% (Low risk)")
    elif risk_percent < 60:
        st.warning(f"⚠️ Wildfire probability: {risk_percent:.1f}% (Medium risk)")
    else:
        st.error(f"🔥 Wildfire probability: {risk_percent:.1f}% (High risk)")

# ---------------------
# Demo values for quick testing
# ---------------------
st.markdown("### Demo Values You Can Try:")
st.markdown("""
**High risk:**  
Precipitation=0, Max Temp=38, Min Temp=20, Wind=6.5, NDVI=0.22, NDVI_7=0.25, NDVI_30=0.30  

**Low risk:**  
Precipitation=5, Max Temp=25, Min Temp=15, Wind=2.0, NDVI=0.45, NDVI_7=0.42, NDVI_30=0.40  

**Moderate risk:**  
Precipitation=1, Max Temp=30, Min Temp=18, Wind=4.0, NDVI=0.30, NDVI_7=0.32, NDVI_30=0.35
""")
