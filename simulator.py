"""Streams simulated sensor readings to the API, like real machines would.
Usage: python simulator.py [--url http://localhost:8000] [--machines 12] [--interval 1.0] [--key SECRET]"""
import sys, time, json, random, urllib.request
import predictive_maintenance as pm
a = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
URL, N, DT, KEY = a("--url", "http://localhost:8000"), int(a("--machines", 12)), float(a("--interval", 1.0)), a("--key", None)

def post(recs):
    req = urllib.request.Request(URL + "/api/ingest", json.dumps(recs).encode(), {"Content-Type": "application/json", **({"X-API-Key": KEY} if KEY else {})})
    return json.load(urllib.request.urlopen(req))

fleet = pm.simulate_fleet(n=N, seed=2024); random.seed(1)
units = {mid: g.sort_values("cycle").to_dict("records") for mid, g in fleet.groupby("machine_id")}
pos = {}
for mid, rows in units.items():                      # start each machine at a different point in its life
    start = max(40, len(rows) - random.choice([8, 20, 35, 50, 80, 120, 200, 260]))
    post([{**{k: r[k] for k in ["cycle"] + pm.SENSORS}, "machine_id": f"M-{mid:03d}"} for r in rows[:start]]); pos[mid] = start
print(f"Preloaded {N} machines. Streaming live (1 cycle every {DT}s). Ctrl+C to stop.")
while True:
    batch = []
    for mid, rows in units.items():
        if pos[mid] < len(rows):
            r = rows[pos[mid]]; pos[mid] += 1
            batch.append({**{k: r[k] for k in ["cycle"] + pm.SENSORS}, "machine_id": f"M-{mid:03d}"})
    if not batch: print("All machines have run to failure."); break
    post(batch); time.sleep(DT)
