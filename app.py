import threading
import time
import requests
from flask import Flask, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from collections import defaultdict, deque
import joblib
from geopy.distance import geodesic
from datetime import datetime, timedelta

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__, template_folder="templates")

# ----------------------------
# Config
# ----------------------------
lamin, lamax = 27.5, 29.5   # Delhi region
lomin, lomax = 76.0, 78.5

# Destination airports (lat, lon, name)
AIRPORTS = {
    "DEL": {"lat": 28.5562, "lon": 77.1000, "name": "Indira Gandhi International Airport"},
}

# Queuing config
APPROACH_RADIUS_KM = 50  # Planes within this radius are considered approaching
LANDING_INTERVAL_MINUTES = 3  # Minimum time between landings
landing_queue = {}  # airport_code -> list of {icao24, distance, eta, slot_time}

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
queue_data = {}


# ----------------------------
# Queue Management
# ----------------------------
def update_landing_queue(flights):
    """Manage landing queue for airports"""
    global landing_queue, queue_data
    
    # Reset queue for this cycle
    current_queue = {code: [] for code in AIRPORTS.keys()}
    
    for flight in flights:
        if not flight["lat"] or not flight["lon"]:
            continue
            
        flight_pos = (flight["lat"], flight["lon"])
        
        # Check distance to each airport
        for airport_code, airport_info in AIRPORTS.items():
            airport_pos = (airport_info["lat"], airport_info["lon"])
            distance_km = geodesic(flight_pos, airport_pos).kilometers
            
            # If within approach radius
            if distance_km <= APPROACH_RADIUS_KM:
                # Calculate ETA (minutes) based on velocity
                velocity_kmh = (flight["velocity"] or 0) * 3.6  # m/s to km/h
                eta_minutes = (distance_km / velocity_kmh * 60) if velocity_kmh > 0 else 999
                
                current_queue[airport_code].append({
                    "icao24": flight["icao24"],
                    "callsign": flight["callsign"],
                    "distance_km": round(distance_km, 2),
                    "eta_minutes": round(eta_minutes, 2),
                    "altitude": flight["altitude"],
                    "velocity": flight["velocity"]
                })
    
    # Sort each airport queue by distance (closest first)
    for airport_code in current_queue:
        current_queue[airport_code].sort(key=lambda x: x["distance_km"])
        
        # Assign landing slots
        base_time = datetime.now()
        for idx, plane in enumerate(current_queue[airport_code]):
            slot_time = base_time + timedelta(minutes=idx * LANDING_INTERVAL_MINUTES)
            plane["queue_position"] = idx + 1
            plane["assigned_slot"] = slot_time.strftime("%H:%M:%S")
            plane["wait_time_minutes"] = idx * LANDING_INTERVAL_MINUTES
    
    landing_queue = current_queue
    queue_data = {
        "airports": {
            code: {
                "name": AIRPORTS[code]["name"],
                "queue": landing_queue[code],
                "queue_length": len(landing_queue[code])
            }
            for code in AIRPORTS.keys()
        }
    }

# ----------------------------
# Background worker
# ----------------------------
def fetch_loop():
    global flights_data, predictions_data, queue_data
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
            
            # Update landing queue
            update_landing_queue(flights)

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
        "predictions": predictions_data,
        "landing_queue": queue_data
    })

@app.route("/queue")
def get_queue():
    """Dedicated endpoint for landing queue data"""
    return jsonify(queue_data)


# ----------------------------
# Start background thread
# ----------------------------
if __name__ == "__main__":
    t = threading.Thread(target=fetch_loop, daemon=True)
    t.start()
    app.run(debug=True)
