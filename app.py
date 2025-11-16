import streamlit as st
import pandas as pd
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
st.title("Wildfire Prediction App (Batch CSV)")

st.markdown("""
Upload a CSV file with **up to 50 rows**.  
CSV should have exactly **7 columns** in this order:  
`PRECIPITATION, MAX_TEMP, MIN_TEMP, AVG_WIND_SPEED, NDVI, NDVI_7day_avg, NDVI_30day_avg`
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Limit to 50 rows
    if len(df) > 50:
        st.warning("Only the first 50 rows will be processed.")
        df = df.head(50)
    
    # Check for correct number of columns
    if df.shape[1] != 7:
        st.error("CSV must have exactly 7 columns.")
    else:
        # Scale features
        scaled = scaler.transform(df.values)
        
        # Predict probabilities
        probs = model.predict_proba(scaled)[:, 1]  # probability of wildfire
        
        # Create results DataFrame
        results = df.copy()
        results["Wildfire_Probability_%"] = np.round(probs * 100, 2)
        
        # Assign risk levels
        def risk_level(p):
            if p < 40:
                return "Low ✅"
            else:
                return "High 🔥"
        
        results["Risk_Level"] = [risk_level(p) for p in results["Wildfire_Probability_%"]]
        
        st.markdown("### Prediction Results")
        st.dataframe(results)
        
        # Optional: download results
        csv = results.to_csv(index=False).encode()
        st.download_button("Download Results as CSV", data=csv, file_name="wildfire_predictions.csv", mime="text/csv")
