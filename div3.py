# collision_sim_with_gui_origin_check.py
# Full script: model loading, Tkinter flight planner GUI with origin-uniqueness check,
# then the simulator (multi-model) — updated so you cannot add/update a flight whose
# origin coordinate is already used by another flight in the current plan.
#
# Save as: collision_sim_with_gui_origin_check.py
# Run: python collision_sim_with_gui_origin_check.py
#
# Requirements: tensorflow, joblib, numpy, pandas, matplotlib, tkinter (standard library)

import os, time, json, csv, ast
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from math import radians, sin, cos, asin

# ---------------- CONFIG (same as your original) ----------------
SEQ_LEN = 10
FUTURE_STEPS = 20
NUM_FEATURES = 5
TIMESTEPS = 300
PLOT_INTERVAL = 0.12

HORIZ_KM = 5.0
ALT_THRESH_M = 300.0

PERSIST_ALT = 1500.0        # final separation (meters)
RAMP_UP_STEPS = 20         # steps to reach separation (gradual climb)
RAMP_DOWN_STEPS = 20       # steps to return to original
PASS_INCREASE_COUNT = 6    # consecutive increases to detect pass

FEATURES = ["latitude","longitude","altitude","speed","heading"]

# ---------------- Small airport map (extend as needed) ----------------
AIRPORTS = {
    "AMD": (23.0776, 72.6326),
    "BOM": (19.0896, 72.8656),
    "MUMBAI": (19.0896, 72.8656),
    "DELHI": (28.5562, 77.1000),
}

# ---------------- Helper math functions ----------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dl = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dl/2)**2
    return 2 * R * asin(np.sqrt(a))

def meters_to_deg_lat(m): return m / 111320.0
def meters_to_deg_lon(m, lat): return m / (111320.0 * cos(radians(lat)) + 1e-12)

# ---------------- Models / Scalers loading (multiple) ----------------
models_dir = "models"

model_registry = {}
candidates = {
    "AMD_BOM": {
        "scaler_X": os.path.join(models_dir, "scaler_X_AMD_BOM.save"),
        "scaler_Y": os.path.join(models_dir, "scaler_Y_AMD_BOM.save"),
        "model":    os.path.join(models_dir, "delta_multi_lstm_AMD_BOM.h5")
    },
    "GEN": {
        "scaler_X": os.path.join(models_dir, "scaler_X.save"),
        "scaler_Y": os.path.join(models_dir, "scaler_y.save"),
        "model":    os.path.join(models_dir, "delta_multi_lstm.h5")
    }
}

print("Loading model sets from", models_dir)
for key, paths in candidates.items():
    try:
        sx = joblib.load(paths["scaler_X"])
        sy = joblib.load(paths["scaler_Y"])
        m  = load_model(paths["model"])
        model_registry[key] = {"scaler_X": sx, "scaler_Y": sy, "model": m}
        print(f"Loaded model set '{key}'.")
    except Exception as e:
        print(f"Could not load model set '{key}': {e}")

if len(model_registry) == 0:
    print("Warning: No models loaded. GUI will still allow plan creation but simulation will fail unless models exist.")

# ---------------- Tkinter GUI to build flights_spec with origin-uniqueness check ----------------
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

def _parse_loc_entry(s, AIRPORTS):
    if s is None: raise ValueError("Empty")
    s = s.strip()
    if s == "": raise ValueError("Empty")
    code = s.upper()
    if code in AIRPORTS:
        return AIRPORTS[code]
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        arr = ast.literal_eval(s)
        return float(arr[0]), float(arr[1])
    if "," in s:
        parts = [p.strip() for p in s.split(",")]
        return float(parts[0]), float(parts[1])
    raise ValueError("Unrecognized location: " + s)

def _normalize_row(values, model_keys, AIRPORTS, auto_fid):
    fid = values.get("fid","").strip()
    if fid == "":
        fid = auto_fid()
    else:
        fid = int(float(fid))
    origin = _parse_loc_entry(values["origin"], AIRPORTS)
    dest   = _parse_loc_entry(values["dest"], AIRPORTS)
    heading = int(float(values.get("heading","0") or 0))
    model_key = values.get("model_key","").strip()
    if model_key not in model_keys:
        raise ValueError(f"Unknown model_key '{model_key}'. Available: {model_keys}")
    late_start = int(float(values.get("late_start","0") or 0))
    return {"fid": fid, "origin": origin, "dest": dest, "heading": heading, "model_key": model_key, "late_start": late_start}

def gui_build_flights(model_registry, AIRPORTS, initial_list=None):
    model_keys = list(model_registry.keys()) or ["AMD_BOM","GEN"]  # fallback if none loaded
    initial_list = initial_list or []

    root = tk.Tk()
    root.title("Flight Plan Builder (unique origin enforced)")
    root.geometry("950x520")
    root.resizable(True, True)

    flights = []
    used_fids = set()
    def set_used(f):
        used_fids.add(int(f))
    for r in initial_list:
        if "fid" in r:
            set_used(r["fid"])
    def next_auto_fid():
        i = 0
        while i in used_fids:
            i += 1
        used_fids.add(i)
        return i

    # ---------- helpers for origin uniqueness ----------
    def _origin_tuple_from_norm(norm):
        return (float(norm["origin"][0]), float(norm["origin"][1]))

    def origin_exists(origin_tuple, exclude_index=None):
        lat1, lon1 = origin_tuple
        for idx, r in enumerate(flights):
            if exclude_index is not None and idx == exclude_index:
                continue
            lat2, lon2 = float(r["origin"][0]), float(r["origin"][1])
            if abs(lat1 - lat2) < 1e-6 and abs(lon1 - lon2) < 1e-6:
                return True
        return False

    # ---------- UI layout ----------
    frm_left = ttk.Frame(root, padding=(8,8)); frm_left.pack(side="left", fill="y")
    frm_right = ttk.Frame(root, padding=(8,8)); frm_right.pack(side="right", fill="both", expand=True)

    ttk.Label(frm_left, text="FID (optional)").grid(row=0, column=0, sticky="w")
    ent_fid = ttk.Entry(frm_left, width=20); ent_fid.grid(row=0,column=1, pady=2)

    ttk.Label(frm_left, text="Origin (code or lat,lon)").grid(row=1, column=0, sticky="w")
    ent_origin = ttk.Combobox(frm_left, values=list(AIRPORTS.keys()), width=22); ent_origin.set(""); ent_origin.grid(row=1,column=1, pady=2)

    ttk.Label(frm_left, text="Dest (code or lat,lon)").grid(row=2, column=0, sticky="w")
    ent_dest = ttk.Combobox(frm_left, values=list(AIRPORTS.keys()), width=22); ent_dest.set(""); ent_dest.grid(row=2,column=1, pady=2)

    ttk.Label(frm_left, text="Heading (deg)").grid(row=3, column=0, sticky="w")
    ent_heading = ttk.Entry(frm_left, width=20); ent_heading.grid(row=3,column=1, pady=2); ent_heading.insert(0,"0")

    ttk.Label(frm_left, text="Model Key").grid(row=4, column=0, sticky="w")
    ent_model = ttk.Combobox(frm_left, values=model_keys, width=22)
    if model_keys: ent_model.set(model_keys[0])
    ent_model.grid(row=4,column=1, pady=2)

    ttk.Label(frm_left, text="Late start (timesteps)").grid(row=5, column=0, sticky="w")
    ent_late = ttk.Entry(frm_left, width=20); ent_late.grid(row=5,column=1, pady=2); ent_late.insert(0,"0")

    btn_add = ttk.Button(frm_left, text="Add flight")
    btn_update = ttk.Button(frm_left, text="Update selected")
    btn_remove = ttk.Button(frm_left, text="Remove selected")
    btn_clear = ttk.Button(frm_left, text="Clear form")

    btn_add.grid(row=6, column=0, columnspan=2, pady=(8,2), sticky="we")
    btn_update.grid(row=7, column=0, columnspan=2, pady=2, sticky="we")
    btn_remove.grid(row=8, column=0, columnspan=2, pady=2, sticky="we")
    btn_clear.grid(row=9, column=0, columnspan=2, pady=(2,8), sticky="we")

    ttk.Label(frm_right, text="Planned Flights (origin must be unique)").pack(anchor="w")
    listbox = tk.Listbox(frm_right, selectmode="browse")
    listbox.pack(fill="both", expand=True, padx=4, pady=4)

    frm_actions = ttk.Frame(frm_right); frm_actions.pack(fill="x")
    btn_import = ttk.Button(frm_actions, text="Import JSON", width=12)
    btn_export = ttk.Button(frm_actions, text="Export JSON", width=12)
    btn_export_csv = ttk.Button(frm_actions, text="Export CSV", width=12)
    btn_done = ttk.Button(frm_actions, text="Done", width=12)
    btn_cancel = ttk.Button(frm_actions, text="Cancel", width=12)
    btn_import.grid(row=0,column=0, padx=4); btn_export.grid(row=0,column=1, padx=4)
    btn_export_csv.grid(row=0,column=2, padx=4); btn_done.grid(row=0,column=3, padx=4); btn_cancel.grid(row=0,column=4, padx=4)

    def row_to_text(r):
        return f"fid={r.get('fid')}  {r.get('origin')} -> {r.get('dest')}  hdg={r.get('heading')}  model={r.get('model_key')}  late={r.get('late_start')}"

    def get_form_values():
        return {
            "fid": ent_fid.get().strip(),
            "origin": ent_origin.get().strip(),
            "dest": ent_dest.get().strip(),
            "heading": ent_heading.get().strip(),
            "model_key": ent_model.get().strip(),
            "late_start": ent_late.get().strip()
        }

    def refresh_listbox():
        listbox.delete(0, tk.END)
        for r in flights:
            listbox.insert(tk.END, row_to_text(r))

    # ---------- modified add/update that enforce origin uniqueness ----------
    def add_flight():
        vals = get_form_values()
        try:
            normalized = _normalize_row(vals, model_keys, AIRPORTS, next_auto_fid)
        except Exception as e:
            messagebox.showerror("Invalid entry", str(e)); return
        origin_t = _origin_tuple_from_norm(normalized)
        if origin_exists(origin_t):
            messagebox.showerror("Origin occupied",
                                 f"A flight with origin {normalized['origin']} already exists.\n"
                                 "Remove it first or choose a different origin.")
            return
        flights.append(normalized)
        refresh_listbox()

    def update_selected():
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("Select", "Select a flight to update.")
            return
        idx = sel[0]
        vals = get_form_values()
        try:
            normalized = _normalize_row(vals, model_keys, AIRPORTS, lambda: flights[idx]["fid"])
        except Exception as e:
            messagebox.showerror("Invalid entry", str(e)); return
        origin_t = _origin_tuple_from_norm(normalized)
        if origin_exists(origin_t, exclude_index=idx):
            messagebox.showerror("Origin occupied",
                                 f"Another flight already uses origin {normalized['origin']}.\n"
                                 "Choose a different origin or remove the other flight first.")
            return
        flights[idx] = normalized
        refresh_listbox()

    def remove_selected():
        sel = listbox.curselection()
        if not sel:
            messagebox.showinfo("Select", "Select a flight to remove.")
            return
        idx = sel[0]
        try:
            used_fids.remove(int(flights[idx]["fid"]))
        except Exception:
            pass
        flights.pop(idx)
        refresh_listbox()

    def clear_form():
        ent_fid.delete(0,"end"); ent_origin.set(""); ent_dest.set("")
        ent_heading.delete(0,"end"); ent_heading.insert(0,"0")
        if model_keys: ent_model.set(model_keys[0])
        ent_late.delete(0,"end"); ent_late.insert(0,"0")

    def on_select(evt):
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        r = flights[idx]
        ent_fid.delete(0,"end"); ent_fid.insert(0,str(r["fid"]))
        def find_code(latlon):
            for k,v in AIRPORTS.items():
                if abs(v[0]-latlon[0])<1e-6 and abs(v[1]-latlon[1])<1e-6:
                    return k
            return f"{latlon[0]:.6f},{latlon[1]:.6f}"
        ent_origin.set(find_code(r["origin"]))
        ent_dest.set(find_code(r["dest"]))
        ent_heading.delete(0,"end"); ent_heading.insert(0,str(r["heading"]))
        ent_model.set(r["model_key"])
        ent_late.delete(0,"end"); ent_late.insert(0,str(r["late_start"]))

    listbox.bind("<<ListboxSelect>>", on_select)

    # ---------- import/export with validation ----------
    def import_json():
        path = filedialog.askopenfilename(filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path: return
        try:
            with open(path,"r") as f:
                arr = json.load(f)
            new = []
            seen_origins = set()
            for r in arr:
                norm = _normalize_row({
                    "fid": r.get("fid",""),
                    "origin": r.get("origin"),
                    "dest": r.get("dest"),
                    "heading": r.get("heading",0),
                    "model_key": r.get("model_key",""),
                    "late_start": r.get("late_start",0)
                }, model_keys, AIRPORTS, next_auto_fid)
                ot = (float(norm["origin"][0]), float(norm["origin"][1]))
                if ot in seen_origins:
                    raise ValueError(f"Import contains duplicate origin {norm['origin']}. Import aborted.")
                seen_origins.add(ot)
                new.append(norm)
            # additionally check collisions with existing in-GUI flights
            for norm in new:
                if origin_exists((float(norm["origin"][0]), float(norm["origin"][1]))):
                    raise ValueError(f"Import would create origin conflict with existing plan for origin {norm['origin']}. Import aborted.")
            flights.clear()
            flights.extend(new)
            refresh_listbox()
        except Exception as e:
            messagebox.showerror("Import error", str(e))

    def export_json():
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files","*.json")])
        if not path: return
        try:
            with open(path,"w") as f:
                json.dump(flights, f, indent=2)
            messagebox.showinfo("Saved", f"Saved {len(flights)} flights to {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def export_csv():
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if not path: return
        try:
            with open(path,"w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["fid","origin","dest","heading","model_key","late_start"])
                for r in flights:
                    origin = f"{r['origin'][0]},{r['origin'][1]}"
                    dest   = f"{r['dest'][0]},{r['dest'][1]}"
                    writer.writerow([r["fid"], origin, dest, r["heading"], r["model_key"], r["late_start"]])
            messagebox.showinfo("Saved", f"Saved {len(flights)} flights to {path}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    result = {"done": False, "flights": []}
    def do_done():
        try:
            fids = [int(r["fid"]) for r in flights]
            if len(fids) != len(set(fids)):
                messagebox.showerror("Validation", "Duplicate fid detected.")
                return
        except Exception as e:
            messagebox.showerror("Validation", str(e)); return
        result["done"] = True
        result["flights"] = flights.copy()
        root.destroy()

    def do_cancel():
        if messagebox.askyesno("Cancel","Discard changes and close?"):
            result["done"] = False
            result["flights"] = []
            root.destroy()

    btn_add.configure(command=add_flight)
    btn_update.configure(command=update_selected)
    btn_remove.configure(command=remove_selected)
    btn_clear.configure(command=clear_form)
    btn_import.configure(command=import_json)
    btn_export.configure(command=export_json)
    btn_export_csv.configure(command=export_csv)
    btn_done.configure(command=do_done)
    btn_cancel.configure(command=do_cancel)

    # preload initial_list if provided
    for r in initial_list:
        try:
            norm = {
                "fid": int(r["fid"]),
                "origin": tuple(r["origin"]),
                "dest": tuple(r["dest"]),
                "heading": int(r.get("heading",0)),
                "model_key": r.get("model_key"),
                "late_start": int(r.get("late_start",0))
            }
            flights.append(norm); set_used(norm["fid"])
        except Exception as e:
            print("Skipping invalid initial flight:", r, e)
    refresh_listbox()

    root.focus_force()
    root.mainloop()

    if result["done"]:
        return result["flights"]
    else:
        return []

# ---------------- Get flights_spec from GUI ----------------
flights_spec = gui_build_flights(model_registry, AIRPORTS)
if not flights_spec:
    print("No flights selected. Exiting.")
    raise SystemExit(0)

# ---------------- The simulator (using your original code structure) ----------------
# Build df_all by generating flights
def generate_flight(fid, origin, dest, steps, heading_base):
    lat1, lon1 = origin
    lat2, lon2 = dest
    lats = np.linspace(lat1, lat2, steps)
    lons = np.linspace(lon1, lon2, steps)
    C = int(steps * 0.2)
    R = int(steps * 0.5)
    D = steps - C - R
    altitudes = np.concatenate([np.linspace(0, 35000, C), np.full(R, 35000), np.linspace(35000, 0, D)])
    speeds = np.concatenate([np.linspace(0,250,C), np.full(R,450), np.linspace(450,200,D)])
    headings = np.full(steps, heading_base)
    return pd.DataFrame({
        "flight_id": fid,
        "timestep": np.arange(steps),
        "latitude": lats,
        "longitude": lons,
        "altitude": altitudes,
        "speed": speeds,
        "heading": headings
    })

dfs = []
for spec in flights_spec:
    df = generate_flight(spec["fid"], spec["origin"], spec["dest"], TIMESTEPS, spec["heading"])
    if spec.get("late_start", 0):
        df["timestep"] += spec["late_start"]
    df["model_key"] = spec["model_key"]
    dfs.append(df)
df_all = pd.concat(dfs, ignore_index=True)

# ---------------- Prepare scaled arrays and sequences per-flight (use flight's model/scalers) ----------------
scaled_groups = {}
original_scaled = {}
seqs = {}
indices = {}
start_times = {}
flight_model_key = {}

for spec in flights_spec:
    fid = spec["fid"]
    g = df_all[df_all["flight_id"]==fid].sort_values("timestep").reset_index(drop=True)
    arr = g[FEATURES].values.astype(float)
    start_times[fid] = g["timestep"].iloc[0]
    model_key = spec["model_key"]
    flight_model_key[fid] = model_key
    sx = model_registry[model_key]["scaler_X"]
    scaled = sx.transform(arr)
    scaled_groups[fid] = scaled.copy()
    original_scaled[fid] = scaled.copy()
    original_scaled[fid].setflags(write=False)
    seqs[fid] = None
    indices[fid] = start_times[fid]

# persistent / ramp state per-flight
flights = [spec["fid"] for spec in flights_spec]
persistent_active = {fid: False for fid in flights}
persistent_offset = {fid: 0.0 for fid in flights}
ramp_remaining = {fid: None for fid in flights}
applied_offset_current = {fid: 0.0 for fid in flights}

dist_history = []

# ---------------- Prediction helper (per-flight uses its model/scalers) ----------------
def predict_horizon(seq_scaled, model_key):
    reg = model_registry[model_key]
    model = reg["model"]
    sx = reg["scaler_X"]
    sy = reg["scaler_Y"]

    y_s = model.predict(seq_scaled.reshape(1, SEQ_LEN, NUM_FEATURES), verbose=0)[0]
    y_s = y_s.reshape(FUTURE_STEPS, NUM_FEATURES)
    deltas = sy.inverse_transform(y_s)
    last = sx.inverse_transform(seq_scaled[-1].reshape(1,-1))[0].copy()
    st = last.copy()
    fut = []
    for d in deltas:
        st = st + d
        fut.append(st.copy())
    return np.array(fut)

# ---------------- Apply increment to all future samples (per-flight uses its scaler_X) ----------------
def add_increment_to_future(fid, increment_m):
    idx = indices[fid]
    if idx >= len(scaled_groups[fid]):
        return
    model_key = flight_model_key[fid]
    sx = model_registry[model_key]["scaler_X"]
    for i in range(idx, len(scaled_groups[fid])):
        base = sx.inverse_transform(scaled_groups[fid][i].reshape(1,-1))[0].copy()
        base[2] = float(base[2] + increment_m)
        scaled_groups[fid][i] = sx.transform(base.reshape(1,-1))[0]

# ---------------- Ramp restore to original smoothly (per-flight uses its scaler_X) ----------------
def ramp_restore_to_original(fid):
    start_idx = indices[fid]
    n = len(scaled_groups[fid])
    model_key = flight_model_key[fid]
    sx = model_registry[model_key]["scaler_X"]
    for j in range(RAMP_DOWN_STEPS):
        i = start_idx + j
        if i >= n:
            break
        cur = sx.inverse_transform(scaled_groups[fid][i].reshape(1,-1))[0].copy()
        orig = sx.inverse_transform(original_scaled[fid][i].reshape(1,-1))[0].copy()
        frac = (j+1) / float(RAMP_DOWN_STEPS)
        interp_alt = cur[2] + (orig[2] - cur[2]) * frac
        new_real = orig.copy()
        new_real[2] = interp_alt
        scaled_groups[fid][i] = sx.transform(new_real.reshape(1,-1))[0]
    for k in range(start_idx + RAMP_DOWN_STEPS, n):
        scaled_groups[fid][k] = original_scaled[fid][k].copy()
    applied_offset_current[fid] = 0.0
    ramp_remaining[fid] = None

# ---------------- Resolve conflict by setting up ramp increments ----------------
def resolve_and_start_ramp(fid_a, fid_b, collision_step):
    pa = predict_horizon(seqs[fid_a], flight_model_key[fid_a])
    pb = predict_horizon(seqs[fid_b], flight_model_key[fid_b])
    cs = min(collision_step, FUTURE_STEPS-1)

    if pa[cs][2] > pb[cs][2]:
        higher, lower = fid_a, fid_b
    else:
        higher, lower = fid_b, fid_a

    def minsep(p,q):
        md_km = float("inf"); md_alt = float("inf")
        for x,y in zip(p,q):
            d = haversine_km(x[0], x[1], y[0], y[1])
            a = abs(x[2] - y[2])
            if d < md_km:
                md_km = d; md_alt = a
        return md_km, md_alt

    before_km, before_alt = minsep(pa, pb)
    print(f"sep before: {before_km*1000:.1f} m / {before_alt:.1f} m")

    high_target = PERSIST_ALT
    low_target  = -PERSIST_ALT
    high_full = np.linspace(0.0, high_target, RAMP_UP_STEPS, endpoint=True)
    low_full  = np.linspace(0.0, low_target,  RAMP_UP_STEPS, endpoint=True)
    high_steps = (high_full[1:] - high_full[:-1]).tolist()
    low_steps  = (low_full[1:]  - low_full[:-1]).tolist()

    ramp_remaining[higher] = high_steps
    ramp_remaining[lower]  = low_steps
    applied_offset_current[higher] = 0.0
    applied_offset_current[lower]  = 0.0
    persistent_offset[higher] = high_target
    persistent_offset[lower]  = low_target
    persistent_active[higher] = True
    persistent_active[lower] = True

    for fid in (higher, lower):
        if ramp_remaining[fid] and len(ramp_remaining[fid]) > 0:
            inc = ramp_remaining[fid].pop(0)
            applied_offset_current[fid] += inc
            add_increment_to_future(fid, inc)
            ni = indices[fid]
            if ni < len(scaled_groups[fid]):
                model_key = flight_model_key[fid]
                sx = model_registry[model_key]["scaler_X"]
                base = sx.inverse_transform(scaled_groups[fid][ni].reshape(1,-1))[0].copy()
                base[2] = float(base[2] + inc)
                scaled_groups[fid][ni] = sx.transform(base.reshape(1,-1))[0]
                seqs[fid] = np.vstack([seqs[fid][1:], scaled_groups[fid][ni]])
    pa2 = predict_horizon(seqs[higher], flight_model_key[higher])
    pb2 = predict_horizon(seqs[lower], flight_model_key[lower])
    after_km, after_alt = minsep(pa2, pb2)
    print(f"sep after (post-apply): {after_km*1000:.1f} m / {after_alt:.1f} m")
    return True

# ---------------- Pass detection & restore ----------------
def check_pass_and_restore():
    if len(dist_history) < PASS_INCREASE_COUNT:
        return False
    recent = dist_history[-PASS_INCREASE_COUNT:]
    if all(recent[i+1] > recent[i] for i in range(len(recent)-1)):
        print("PASS DETECTED → ramping back to original profiles")
        for fid in flights:
            if persistent_active[fid]:
                ramp_restore_to_original(fid)
                persistent_active[fid] = False
                persistent_offset[fid] = 0.0
                ramp_remaining[fid] = None
        return True
    return False

# ---------------- Plot setup ----------------
plt.ion()
fig, ax = plt.subplots(figsize=(12,7))
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(True)

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
actual_lines = {}; pred_lines = {}; markers = {}; alt_texts = {}
for i, fid in enumerate(flights):
    actual_lines[fid], = ax.plot([], [], color=colors[i % len(colors)], linewidth=2)
    pred_lines[fid], = ax.plot([], [], color=colors[i % len(colors)], linestyle='--')
    markers[fid] = ax.scatter([], [], s=90, c=colors[i % len(colors)], edgecolor="black", zorder=5)
    alt_texts[fid] = ax.text(0,0,"", fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

all_lats = df_all["latitude"].to_numpy()
all_lons = df_all["longitude"].to_numpy()
ax.set_xlim(all_lons.min()-0.5, all_lons.max()+0.5)
ax.set_ylim(all_lats.min()-0.5, all_lats.max()+0.5)

plt.pause(0.2)

def draw_alt_label(fid):
    if seqs[fid] is None: return
    model_key = flight_model_key[fid]
    sx = model_registry[model_key]["scaler_X"]
    cur = sx.inverse_transform(seqs[fid][-1].reshape(1,-1))[0]
    lat, lon, alt = cur[0], cur[1], cur[2]
    tag = "(P)" if persistent_active[fid] else ""
    alt_texts[fid].set_position((lon, lat + meters_to_deg_lat(800)))
    alt_texts[fid].set_text(f"{alt:.0f} m {tag}")

# ---------------- Main loop ----------------
sim_time = 0
while True:

    # Activate late-start flights
    for fid in flights:
        if seqs[fid] is None and sim_time >= start_times[fid]:
            seqs[fid] = scaled_groups[fid][0:SEQ_LEN].copy()
            indices[fid] = SEQ_LEN
            print(f"✈ Flight {fid} ENTERED SIM at t={sim_time} (model={flight_model_key[fid]})")

    # For each active flight, if a ramp is ongoing, pop one increment and apply it now:
    for fid in flights:
        if seqs[fid] is None:
            continue
        if ramp_remaining[fid] is not None and len(ramp_remaining[fid]) > 0:
            inc = ramp_remaining[fid].pop(0)
            applied_offset_current[fid] += inc
            add_increment_to_future(fid, inc)
            ni = indices[fid]
            if ni < len(scaled_groups[fid]):
                model_key = flight_model_key[fid]
                sx = model_registry[model_key]["scaler_X"]
                base = sx.inverse_transform(scaled_groups[fid][ni].reshape(1,-1))[0].copy()
                base[2] = float(base[2] + inc)
                scaled_groups[fid][ni] = sx.transform(base.reshape(1,-1))[0]
                seqs[fid] = np.vstack([seqs[fid][1:], scaled_groups[fid][ni]])
            if len(ramp_remaining[fid]) == 0:
                ramp_remaining[fid] = None

    # record distances for pass detection between the first two flights of same route if available
    if len(flights) >= 2:
        f0, f1 = flights[0], flights[1]
        if seqs.get(f0) is not None and seqs.get(f1) is not None:
            mk0 = flight_model_key[f0]; mk1 = flight_model_key[f1]
            sx0 = model_registry[mk0]["scaler_X"]; sx1 = model_registry[mk1]["scaler_X"]
            cur0 = sx0.inverse_transform(seqs[f0][-1].reshape(1,-1))[0]
            cur1 = sx1.inverse_transform(seqs[f1][-1].reshape(1,-1))[0]
            dkm = haversine_km(cur0[0], cur0[1], cur1[0], cur1[1])
            dist_history.append(dkm)

    # predict & draw
    predictions = {}
    for fid in flights:
        if seqs[fid] is None:
            continue
        if indices[fid] >= len(scaled_groups[fid]) - FUTURE_STEPS:
            continue
        pred = predict_horizon(seqs[fid], flight_model_key[fid])
        predictions[fid] = pred
        model_key = flight_model_key[fid]
        sx = model_registry[model_key]["scaler_X"]
        actual_real = sx.inverse_transform(seqs[fid])
        actual_lines[fid].set_data(actual_real[:,1], actual_real[:,0])
        pred_lines[fid].set_data(pred[:,1], pred[:,0])
        markers[fid].set_offsets([[actual_real[-1,1], actual_real[-1,0]]])
        draw_alt_label(fid)

    # pairwise conflict detection among active flights
    active = [fid for fid in flights if seqs[fid] is not None]
    for i in range(len(active)):
        for j in range(i+1, len(active)):
            fa, fb = active[i], active[j]
            if fa not in predictions or fb not in predictions:
                continue
            pa, pb = predictions[fa], predictions[fb]
            conflict = False; step = None
            for s, (A, B) in enumerate(zip(pa, pb)):
                d = haversine_km(A[0], A[1], B[0], B[1])
                h = abs(A[2] - B[2])
                if d < HORIZ_KM and h < ALT_THRESH_M:
                    conflict = True; step = s; break
            if conflict:
                print(f"\n⚠️ Predicted conflict between {fa} & {fb} at step {step}")
                resolve_and_start_ramp(fa, fb, step)
                ax.set_title("Conflict -> Gradual persistent separation", color='red')
            else:
                ax.set_title("No imminent conflict", color='black')

    # check pass & restore if persistent active
    if any(persistent_active.values()):
        check_pass_and_restore()

    fig.canvas.draw(); fig.canvas.flush_events()
    time.sleep(PLOT_INTERVAL)

    # sliding window: advance sequences by one (uses modified scaled_groups)
    for fid in flights:
        if seqs[fid] is None:
            continue
        if indices[fid] < len(scaled_groups[fid]):
            seqs[fid] = np.vstack([seqs[fid][1:], scaled_groups[fid][indices[fid]]])
            indices[fid] += 1

    sim_time += 1

    # exit when all flights finished
    done = True
    for fid in flights:
        if seqs[fid] is None:
            done = False; break
        if indices[fid] < len(scaled_groups[fid]):
            done = False; break
    if done:
        break

plt.ioff()
plt.show()
print("Simulation finished.")
