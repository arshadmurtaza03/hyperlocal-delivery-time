# app.py
import streamlit as st
import joblib
import pandas as pd
import os
import traceback
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Hyperlocal Delivery Time Predictor", page_icon="🚚", layout="centered")
st.title("🚚 Hyperlocal Delivery Time Predictor")
st.write("Enter details and click **Predict**. If model file can't be loaded, app will train a model automatically (one-time).")

MODEL_PATH = "models/delivery_time_pipeline.joblib"
DATA_PATH = "data/delivery_data.csv"

def build_and_train_pipeline(data_path=DATA_PATH):
    """Build preprocessing + RandomForest pipeline and train it on CSV dataset."""
    st.info("Training model now (this may take ~10-60 seconds). Please wait...")
    df = pd.read_csv(data_path)

    # Basic cleaning
    df = df.dropna()

    X = df.drop(columns=["delivery_time_min"])
    y = df["delivery_time_min"]

    numeric_features = ["distance_km", "num_stops", "pickup_hour", "day_of_week", "delivery_experience_years"]
    categorical_features = ["traffic", "weather", "package_size"]

    numeric_transformer = StandardScaler()
    # handle sklearn API change: older versions accept `sparse`, newer use `sparse_output`
    try:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    except TypeError:
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    pipeline.fit(X, y)

    # Try saving the trained pipeline (optional). Ignore errors while saving.
    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        st.success("Trained model saved to models/ directory.")
    except Exception as e:
        st.warning(f"Could not save model to disk (not critical). Error: {e}")

    return pipeline

@st.cache_resource
def load_pipeline():
    """Try to load pipeline, if fails then build & train one."""
    # 1) Try to load saved model
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            return pipeline
        except Exception as e:
            # Loading failed — fall through and train
            st.warning("Saved model exists but could not be loaded due to environment mismatch. We'll train a new model now.")
            # Print traceback to logs (not the user)
            st.text("Loading error (logged).")
            logging.error(traceback.format_exc())

    # 2) If no model or loading failed, train a new one
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. The app needs the CSV (data/delivery_data.csv).")
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    return build_and_train_pipeline(DATA_PATH)

# Load or train pipeline
pipeline = load_pipeline()

# --- UI Inputs ---
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
    try:
        pred = pipeline.predict(input_df)[0]
        st.success(f"Estimated delivery time: **{pred:.1f} minutes**")
        st.write("This is a model estimate. Real-world times will vary.")
    except Exception as e:
        st.error("Prediction failed. See logs.")
    logging.exception("Prediction error")
