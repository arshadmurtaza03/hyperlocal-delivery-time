# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np

# 1. Load data
df = pd.read_csv("data/delivery_data.csv")

# 2. Quick cleaning (if any NaNs, drop them for this beginner project)
df = df.dropna()

# 3. Features and target
X = df.drop(columns=["delivery_time_min"])
y = df["delivery_time_min"]

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Preprocessing: numeric and categorical
numeric_features = ["distance_km", "num_stops", "pickup_hour", "day_of_week", "delivery_experience_years"]
categorical_features = ["traffic", "weather", "package_size"]

numeric_transformer = StandardScaler()
try:
    # older sklearn versions accept `sparse` (bool)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
except TypeError:
    # newer sklearn versions use `sparse_output` instead
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 6. Pipeline: preprocessing + model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# 7. Train
pipeline.fit(X_train, y_train)

# 8. Evaluate
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
# compute RMSE without using the deprecated `squared` kwarg
try:
    # newer sklearn provides this utility
    from sklearn.metrics import root_mean_squared_error

    rmse = root_mean_squared_error(y_test, y_pred)
except Exception:
    # fallback: compute sqrt of MSE (avoids using `squared=False` which is deprecated)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R2: {r2:.3f}")

# 9. Save the pipeline
joblib.dump(pipeline, "models/delivery_time_pipeline.joblib")
print("Saved model to models/delivery_time_pipeline.joblib")
