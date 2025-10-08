import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("flight_data_delhi.csv")

# Keep only useful columns
df = df[["icao24", "lat", "lon", "altitude", "velocity", "heading"]]
df = df.dropna()

# ----------------------------
# Manual scaling setup
# ----------------------------
ranges = {
    "lat": (df["lat"].min(), df["lat"].max()),
    "lon": (df["lon"].min(), df["lon"].max()),
    "altitude": (df["altitude"].min(), df["altitude"].max()),
    "velocity": (df["velocity"].min(), df["velocity"].max()),
    "heading": (0, 360),  # fixed full range
}

def scale(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin)

def descale(val, vmin, vmax):
    return val * (vmax - vmin) + vmin

# Apply scaling
scaled = pd.DataFrame()
for col in ["lat", "lon", "altitude", "velocity", "heading"]:
    vmin, vmax = ranges[col]
    scaled[col] = df[col].apply(lambda x: scale(x, vmin, vmax))

# Save ranges for later inference
os.makedirs("models", exist_ok=True)
joblib.dump(ranges, "models/ranges.save")

# ----------------------------
# Sequence preparation
# ----------------------------
SEQ_LEN = 10
features = scaled[["lat", "lon", "altitude", "velocity", "heading"]].values

X, y = [], []
for i in range(len(features) - SEQ_LEN):
    X.append(features[i:i+SEQ_LEN])
    y.append(features[i+SEQ_LEN])

X, y = np.array(X), np.array(y)
print("✅ Training data shape:", X.shape, y.shape)

# ----------------------------
# Build LSTM
# ----------------------------
model = Sequential([
    LSTM(64, input_shape=(SEQ_LEN, X.shape[2])),
    Dense(X.shape[2])  # predict next [lat, lon, alt, vel, heading]
])

model.compile(optimizer="adam", loss="mse")

# ----------------------------
# Train
# ----------------------------
history = model.fit(
    X, y,
    epochs=50,           # 🔑 train longer
    batch_size=16,       # 🔑 smaller batches
    validation_split=0.2
)

# ----------------------------
# Save model
# ----------------------------
model.save("models/lstm_allfeatures.h5")
print("✅ Model + ranges saved in models/")
