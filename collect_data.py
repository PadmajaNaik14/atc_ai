import requests
import pandas as pd
import time
from datetime import datetime
import os

# File to store flight data
CSV_FILE = "flight_data_delhi.csv"

# Delhi bounding box (approx 200 km radius around IGI Airport)
lamin, lamax = 25.0, 30.0     # latitude range
lomin, lomax = 75.0, 80.0     # longitude range

def fetch_data():
    url = f"https://opensky-network.org/api/states/all?lamin={lamin}&lamax={lamax}&lomin={lomin}&lomax={lomax}"
    response = requests.get(url, timeout=10)
    
    if response.status_code == 200:
        data = response.json()
        if data and "states" in data and data["states"] is not None:
            records = []
            for state in data["states"]:
                record = {
                    "time": datetime.utcfromtimestamp(data["time"]).strftime('%Y-%m-%d %H:%M:%S'),
                    "icao24": state[0],
                    "callsign": state[1],
                    "origin_country": state[2],
                    "lat": state[6],
                    "lon": state[5],
                    "altitude": state[7],
                    "velocity": state[9],
                    "heading": state[10]
                }
                records.append(record)
            
            # Save to CSV
            df = pd.DataFrame(records)
            file_exists = os.path.isfile(CSV_FILE)
            df.to_csv(CSV_FILE, mode='a', header=not file_exists, index=False)
            print(f"✅ {len(records)} records saved at {datetime.utcnow()}")
        else:
            print("⚠️ No flights found in response")
    else:
        print("❌ Error fetching data:", response.status_code)

if __name__ == "__main__":
    print("📡 Starting Delhi flight data collection...")
    while True:
        try:
            fetch_data()
            time.sleep(10)  # fetch every 10 seconds
        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")
            break