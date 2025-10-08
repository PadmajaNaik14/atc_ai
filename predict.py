import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/lstm_fixed.h5")

# Load scalers
scalers = joblib.load("models/scalers_individual.pkl")

SEQ_LEN = 10

def predict_next(sequence):
    """
    sequence: last 10 timesteps [lat, lon, alt, vel, heading] (unscaled real values)
    returns: predicted (lat, lon, alt) in real-world units
    """

    # Scale input features with their respective scalers
    scaled_seq = np.zeros_like(sequence)
    for i, col in enumerate(["lat", "lon", "altitude", "velocity", "heading"]):
        scaled_seq[:, i] = scalers[col].transform(sequence[:, i].reshape(-1, 1)).flatten()

    # Reshape for LSTM: (1, timesteps, features)
    X_input = scaled_seq.reshape(1, SEQ_LEN, -1)

    # Predict scaled [lat, lon, alt]
    pred_scaled = model.predict(X_input, verbose=0)[0]

    # Inverse transform each output separately
    pred_real = [
        scalers["lat"].inverse_transform([[pred_scaled[0]]])[0, 0],
        scalers["lon"].inverse_transform([[pred_scaled[1]]])[0, 0],
        scalers["altitude"].inverse_transform([[pred_scaled[2]]])[0, 0],
    ]

    return tuple(pred_real)

# ------------------ DEMO ------------------
if __name__ == "__main__":
    # Example: fake last 10 timesteps for one flight
    example_seq = np.array([
        [28.6, 77.1, 5000, 220, 90],  # lat, lon, alt, vel, heading
        [28.61, 77.11, 5050, 221, 91],
        [28.62, 77.12, 5100, 219, 92],
        [28.63, 77.13, 5150, 222, 93],
        [28.64, 77.14, 5200, 223, 94],
        [28.65, 77.15, 5250, 224, 95],
        [28.66, 77.16, 5300, 225, 96],
        [28.67, 77.17, 5350, 223, 97],
        [28.68, 77.18, 5400, 222, 98],
        [28.69, 77.19, 5450, 221, 99],
    ])

    pred = predict_next(example_seq)
    print("Predicted next position:", pred)
