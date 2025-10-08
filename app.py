import threading
import time
import requests
from flask import Flask, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict, deque
import joblib
from geopy.distance import geodesic

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__, template_folder="templates")

# ----------------------------
# Config
# ----------------------------
lamin, lamax = 27.5, 29.5   # Delhi region
lomin, lomax = 76.0, 78.5

# Load model + scaling ranges
model = load_model("models/lstm_allfeatures.h5")
ranges = joblib.load("models/ranges.save")

def scale(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin)

def descale(val, vmin, vmax):
    return val * (vmax - vmin) + vmin

# Histories (10 timesteps per flight)
history = defaultdict(lambda: deque(maxlen=10))

# Store current results
flights_data = []
predictions_data = []


# ----------------------------
# Background worker
# ----------------------------
def fetch_loop():
    global flights_data, predictions_data
    while True:
        try:
            url = f"https://opensky-network.org/api/states/all?lamin={lamin}&lamax={lamax}&lomin={lomin}&lomax={lomax}"
            r = requests.get(url, timeout=10)
            data = r.json()

            flights = []
            if data and "states" in data and data["states"]:
                for s in data["states"]:
                    flight = {
                        "icao24": s[0],
                        "callsign": s[1].strip() if s[1] else "",
                        "origin_country": s[2],
                        "lat": s[6],
                        "lon": s[5],
                        "velocity": s[9],
                        "heading": s[10],
                        "altitude": s[7],
                    }
                    flights.append(flight)

                    # update history only if valid
                    if s[6] and s[5] and s[7] is not None:
                        history[s[0]].append([
                            scale(s[6], *ranges["lat"]),
                            scale(s[5], *ranges["lon"]),
                            scale(s[7], *ranges["altitude"]),
                            scale(s[9] or 0, *ranges["velocity"]),
                            scale(s[10] or 0, *ranges["heading"]),
                        ])

            flights_data = flights

            preds = []
            errors = []
            for fid, seq in history.items():
                if len(seq) == 10:
                    X = np.array(seq).reshape(1, 10, 5)
                    y_scaled = model.predict(X, verbose=0)[0]

                    # descale
                    y = [
                        descale(y_scaled[0], *ranges["lat"]),
                        descale(y_scaled[1], *ranges["lon"]),
                        descale(y_scaled[2], *ranges["altitude"]),
                        descale(y_scaled[3], *ranges["velocity"]),
                        descale(y_scaled[4], *ranges["heading"]),
                    ]

                    pred = {
                        "icao24": fid,
                        "lat": float(y[0]),
                        "lon": float(y[1]),
                        "altitude": float(y[2]),
                        "velocity": float(y[3]),
                        "heading": float(y[4]),
                    }
                    preds.append(pred)

                    # error in meters
                    for f in flights:
                        if f["icao24"] == fid and f["lat"] and f["lon"]:
                            err_m = geodesic((f["lat"], f["lon"]), (y[0], y[1])).meters
                            errors.append(err_m)
                            print(f"[DEBUG] {fid}: Real=({f['lat']:.4f},{f['lon']:.4f},alt={f['altitude']:.1f}) "
                                  f"| Pred=({y[0]:.4f},{y[1]:.4f},alt={y[2]:.1f}) | Err≈{err_m:.1f} m")

            predictions_data = preds

            if errors:
                avg_err = np.mean(errors)
                print(f"Fetched {len(flights)} flights, {len(preds)} predictions | Avg error ≈ {avg_err:.1f} m")

        except Exception as e:
            print("Error in fetch loop:", e)

        time.sleep(10)


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/flights")
def get_flights():
    return jsonify({
        "flights": flights_data,
        "predictions": predictions_data
    })


# ----------------------------
# Start background thread
# ----------------------------
if __name__ == "__main__":
    t = threading.Thread(target=fetch_loop, daemon=True)
    t.start()
    app.run(debug=True)
