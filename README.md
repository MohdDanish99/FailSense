# Predictive Maintenance System (API + live dashboard)

Machines send sensor readings to an API. The system keeps each machine's history,
builds features, predicts failure risk and remaining useful life (RUL), and shows a
live ranked dashboard.

## Quick start
    pip install -r requirements.txt
    python train.py        # trains models -> models/pdm.joblib
    python server.py       # API + dashboard at http://localhost:8000
    python simulator.py    # (second terminal) streams fake machines into the API

Open http://localhost:8000.

## Send your own readings
    curl -X POST http://localhost:8000/api/ingest -H "Content-Type: application/json" \
      -d '{"machine_id":"PUMP-7","cycle":57,"vibration":2.3,"temperature":71.2,"pressure":29.8,"current":10.4,"rpm":1498,"oil_debris":5.3}'
Send one reading per cycle with `cycle` increasing. Accepts one object or a list.

| Endpoint | Purpose |
|---|---|
| POST /api/ingest | add readings, returns risk + RUL |
| GET /api/fleet | all machines ranked by risk |
| GET /api/machine/<id> | one machine |
| GET /api/health | liveness check |

Set `PDM_API_KEY=secret` before starting the server to require an `X-API-Key` header on ingest.

## Use real data
1. Edit `SENSORS` in `predictive_maintenance.py` to your sensor names.
2. Prepare a CSV of run-to-failure history: `machine_id, cycle, <sensors>` (+ `rul`, or it is derived).
3. `python train.py --csv your_data.csv`, then restart the server.
4. Have your machines or SCADA system POST readings to /api/ingest.

## Files
- `predictive_maintenance.py` simulation, features, offline evaluation
- `train.py` trains and saves models
- `server.py` API + dashboard server (standard-library HTTP)
- `simulator.py` demo data stream
- `dashboard.html` live UI

## Production notes
History is kept in memory and lost on restart; swap `HIST`/`SUMM` in `server.py` for a
database for real deployments. Put the server behind HTTPS and a reverse proxy, and
retrain as new failures are recorded. Thresholds are in `status()` in `server.py`.
