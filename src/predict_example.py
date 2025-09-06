# src/predict_example.py
import joblib
import pandas as pd

pipeline = joblib.load("models/delivery_time_pipeline.joblib")

sample = pd.DataFrame([{
    "distance_km": 5.2,
    "num_stops": 1,
    "traffic": "moderate",
    "weather": "sunny",
    "pickup_hour": 18,
    "day_of_week": 2,
    "package_size": "medium",
    "delivery_experience_years": 2
}])

pred = pipeline.predict(sample)[0]
print(f"Predicted delivery time: {pred:.1f} minutes")
