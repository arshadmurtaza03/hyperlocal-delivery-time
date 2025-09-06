# src/data_generation.py
import numpy as np
import pandas as pd

N = 6000  # rows (>=5000)

rng = np.random.default_rng(42)

# Features
distance_km = np.round(rng.uniform(0.5, 30.0, N), 2)
num_stops = rng.integers(0, 6, N)  # 0 to 5 stops
traffic_levels = ["low", "moderate", "heavy", "very_heavy"]
traffic = rng.choice(traffic_levels, N, p=[0.3, 0.4, 0.25, 0.05])

weather_types = ["sunny", "rainy", "windy", "snowy"]
weather = rng.choice(weather_types, N, p=[0.6, 0.25, 0.12, 0.03])

pickup_hour = rng.integers(7, 22, N)  # work hours 7-21
day_of_week = rng.integers(0, 7, N)  # 0=Monday .. 6=Sunday
package_size = rng.choice(["small", "medium", "large"], N, p=[0.5, 0.35, 0.15])
delivery_experience_years = rng.integers(0, 11, N)

# Base delivery time (minutes)
base = 10 + distance_km * rng.uniform(1.5, 3.0, N)
base += num_stops * rng.uniform(2.0, 6.0, N)

# traffic multiplier
traffic_map = {"low": 0.95, "moderate": 1.0, "heavy": 1.25, "very_heavy": 1.6}
traffic_mult = np.array([traffic_map[t] for t in traffic])
base *= traffic_mult

# weather effect
weather_map = {"sunny": 1.0, "rainy": 1.12, "windy": 1.05, "snowy": 1.3}
weather_mult = np.array([weather_map[w] for w in weather])
base *= weather_mult

# package size effect
size_map = {"small": 0.98, "medium": 1.0, "large": 1.1}
size_mult = np.array([size_map[s] for s in package_size])
base *= size_mult

# time of day: rush hour penalty
rush = ((pickup_hour >= 8) & (pickup_hour <= 10)) | ((pickup_hour >= 17) & (pickup_hour <= 19))
base += rush * rng.uniform(3, 8, N)

# experience reduces time slightly
base -= delivery_experience_years * rng.uniform(0.2, 0.7, N)

# Add random noise
noise = rng.normal(0, 5, N)
delivery_time_min = np.clip(base + noise, a_min=5, a_max=None)  # minimum 5 minutes

df = pd.DataFrame({
    "distance_km": distance_km,
    "num_stops": num_stops,
    "traffic": traffic,
    "weather": weather,
    "pickup_hour": pickup_hour,
    "day_of_week": day_of_week,
    "package_size": package_size,
    "delivery_experience_years": delivery_experience_years,
    "delivery_time_min": np.round(delivery_time_min, 1)
})

# Save
df.to_csv("data/delivery_data.csv", index=False)
print("Saved data/delivery_data.csv with", len(df), "rows")
