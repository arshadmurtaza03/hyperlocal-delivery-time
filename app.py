# app.py
import streamlit as st
import joblib
import pandas as pd
import os
import logging
import traceback

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Logging for server logs (Streamlit Cloud)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Hyperlocal Delivery Time Predictor", page_icon="🚚", layout="centered")
st.title("🚚 Hyperlocal Delivery Time Predictor")
st.write("Enter details and click **Predict**. If necessary, the app will train a model automatically (one-time).")

MODEL_PATH = "models/delivery_time_pipeline.joblib"
DATA_PATH = "data/delivery_data.csv"

def make_onehot_encoder_compatible(**kwargs):
    """
    Return a OneHotEncoder that works with the sklearn version on this environment.
    Newer sklearn uses sparse_output=False, older uses sparse=False. We try both.
    """
    try:
        # try the newer arg name first (sklearn >= 1.2)
        return OneHotEncoder(handle_unknown=kwargs.get("handle_unknown", "ignore"), sparse_output=kwargs.get("sparse_output", False))
    except TypeError:
        # fall back to older arg name
        return OneHotEncoder(handle_unknown=kwargs.get("handle_unknown", "ignore"), sparse=kwargs.get("sparse", False))

def build_and_train_pipeline(data_path=DATA_PATH):
    """Build preprocessing + RandomForest pipeline and train it on CSV dataset."""
    st.info("Training model now (this may take ~10-60 seconds). Please wait...")
    try:
        df = pd.read_csv(data_path)
    except Exception:
        logger.exception("Could not read dataset")
        st.error(f"Dataset not found or unreadable at {data_path}. Deployment needs data/delivery_data.csv.")
        raise

    # Basic cleaning
    df = df.dropna()

    X = df.drop(columns=["delivery_time_min"])
    y = df["delivery_time_min"]

    numeric_features = ["distance_km", "num_stops", "pickup_hour", "day_of_week", "delivery_experience_years"]
    categorical_features = ["traffic", "weather", "package_size"]

    numeric_transformer = StandardScaler()
    # Create OneHotEncoder in a way that works across sklearn versions
    try:
        categorical_transformer = make_onehot_encoder_compatible(handle_unknown="ignore", sparse_output=False, sparse=False)
    except Exception:
        # If something unexpected happens, fall back to a safe default and log
        logger.exception("Failed to create OneHotEncoder with either parameter name; trying minimal fallback.")
        categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    pipeline.fit(X, y)

    # Try saving the trained pipeline (optional). If it fails, log but continue.
    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        logger.info("Trained model saved to %s", MODEL_PATH)
        st.success("Model trained and saved.")
    except Exception:
        logger.exception("Failed to save trained model to disk; continuing with in-memory model")
        st.info("Model trained (not saved).")

    return pipeline

@st.cache_resource
def load_pipeline():
    """Try to load pipeline; if fails, train a new one. Fail quietly for users, log details for devs."""
    if os.path.exists(MODEL_PATH):
        try:
            pipeline = joblib.load(MODEL_PATH)
            logger.info("Loaded pipeline from %s", MODEL_PATH)
            return pipeline
        except Exception:
            # Log full traceback for devs; show a friendly info for users.
            logger.exception("Failed to load existing model; rebuilding from data.")
            st.info("Existing model is incompatible with this environment; the app will build a new model now.")

    # No model or failed to load -> build & train
    if not os.path.exists(DATA_PATH):
        logger.error("Dataset missing: %s", DATA_PATH)
        st.error(f"Dataset not found at {DATA_PATH}. Please include data/delivery_data.csv in the repo.")
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    return build_and_train_pipeline(DATA_PATH)

# Load or build pipeline
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
    except Exception:
        logger.exception("Prediction failed")
        st.error("Prediction failed. See server logs for details.")
