import numpy as np

def predict_wildfire(model, scaler, features):
    """
    Scale features and predict wildfire probability and risk level.
    features: list or np.array of shape (1,7)
    """
    scaled = scaler.transform(features)
    proba = model.predict_proba(scaled)[0][1]
    risk = "Low ✅" if proba < 0.3 else ("Medium ⚠️" if proba < 0.6 else "High 🔥")
    return round(proba * 100, 2), risk
