# app.py
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Hyperlocal Delivery Time Predictor", page_icon="🚚", layout="centered")

st.title("🚚 Hyperlocal Delivery Time Predictor")
st.write("Enter details below and click **Predict**. This app uses a trained model to estimate delivery time in minutes.")

# Load model
@st.cache_resource
def load_pipeline():
    return joblib.load("models/delivery_time_pipeline.joblib")

pipeline = load_pipeline()

# Inputs
distance_km = st.number_input("Distance (km)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
num_stops = st.number_input("Number of stops", min_value=0, max_value=10, value=1, step=1)
traffic = st.selectbox("Traffic level", ["low", "moderate", "heavy", "very_heavy"])
weather = st.selectbox("Weather", ["sunny", "rainy", "windy", "snowy"])
pickup_hour = st.slider("Pickup hour (24h)", 0, 23, 18)
day_of_week = st.selectbox("Day of week (0=Mon,6=Sun)", [0,1,2,3,4,5,6])
package_size = st.selectbox("Package size", ["small", "medium", "large"])
delivery_experience_years = st.number_input("Delivery person's experience (years)", min_value=0, max_value=50, value=2, step=1)

if st.button("Predict delivery time"):
    input_df = pd.DataFrame([{
        "distance_km": distance_km,
        "num_stops": num_stops,
        "traffic": traffic,
        "weather": weather,
        "pickup_hour": pickup_hour,
        "day_of_week": day_of_week,
        "package_size": package_size,
        "delivery_experience_years": delivery_experience_years
    }])
    pred = pipeline.predict(input_df)[0]
    st.success(f"Estimated delivery time: **{pred:.1f} minutes**")
    st.write("This is a model estimate. Real-world times will vary.")