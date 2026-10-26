"""Live predictive-maintenance API + dashboard (needs only scikit-learn, pandas, numpy).
  python server.py [--port 8000]          optional: PDM_API_KEY=secret to protect POST /api/ingest
Endpoints:
  POST /api/ingest        {"machine_id":"M-1","cycle":12,"vibration":2.1,...}  (object or list)
  GET  /api/fleet         all machines, ranked by failure risk
  GET  /api/machine/<id>  one machine
  GET  /api/health        liveness check
  GET  /                  live dashboard
"""
import json, os, sys, threading, time, joblib, pandas as pd
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import predictive_maintenance as pm

B = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pdm.joblib"))
SENS, F = B["sensors"], B["features"]
API_KEY = os.environ.get("PDM_API_KEY")
HIST, SUMM, LOCK, KEEP = {}, {}, threading.Lock(), 50

def status(p, rul):
    return "critical" if p >= .5 or rul <= 30 else "warning" if p >= .1 or rul <= 60 else "healthy"

def ingest(rec):
    mid = str(rec["machine_id"]); miss = [s for s in SENS if s not in rec]
    if miss: raise ValueError(f"missing sensors: {miss}")
    row = {"cycle": int(rec["cycle"]), **{s: float(rec[s]) for s in SENS}}
    with LOCK:
        rows = HIST.setdefault(mid, [])
        if rows and row["cycle"] <= rows[-1]["cycle"]: raise ValueError("cycle must increase")
        rows.append(row); rows[:] = rows[-2000:]
        df = pd.DataFrame(rows); df["machine_id"] = mid
        feats, _ = pm.build_features(df); last = feats.tail(1)
        p = float(B["clf"].predict_proba(last[F])[0, 1]); r = float(B["reg"].predict(last[F])[0].clip(0, pm.RUL_CAP))
        s = SUMM.setdefault(mid, dict(id=mid, hp=[], hr=[], sv=[], st=[], sp=[]))
        for k, v in (("hp", round(p, 3)), ("hr", round(r)), ("sv", round(float(last.vibration_dev10.fillna(0).iloc[0]), 2)),
                     ("st", round(float(last.temperature_dev10.fillna(0).iloc[0]), 1)), ("sp", round(float(last.pressure_dev10.fillna(0).iloc[0]), 1))):
            s[k] = (s[k] + [v])[-KEEP:]
        s.update(cycle=row["cycle"], p=round(p, 3), rul=round(r), s=status(p, r), updated=time.time())
        return {k: s[k] for k in ("id", "cycle", "p", "rul", "s")}

def fleet():
    with LOCK:
        return sorted((dict(v) for v in SUMM.values()), key=lambda m: (-m["p"], m["rul"]))

class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*"); self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers(); self.wfile.write(data)
    def do_OPTIONS(self): self._send(204, b"")
    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._send(200, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html"), "rb").read(), "text/html; charset=utf-8")
        elif path == "/api/fleet": self._send(200, fleet())
        elif path.startswith("/api/machine/"):
            m = next((x for x in fleet() if x["id"] == path.rsplit("/", 1)[1]), None)
            self._send(200 if m else 404, m or {"error": "unknown machine"})
        elif path == "/api/health": self._send(200, {"ok": True, "machines": len(SUMM)})
        else: self._send(404, {"error": "not found"})
    def do_POST(self):
        if self.path != "/api/ingest": return self._send(404, {"error": "not found"})
        if API_KEY and self.headers.get("X-API-Key") != API_KEY: return self._send(401, {"error": "bad API key"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            out = [ingest(r) for r in (body if isinstance(body, list) else [body])]
            self._send(200, out if isinstance(body, list) else out[0])
        except (ValueError, KeyError, json.JSONDecodeError) as e: self._send(400, {"error": str(e)})
    def log_message(self, *a): pass

if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    print(f"Dashboard: http://localhost:{port}   (Ctrl+C to stop)")
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
