"""
Predictive Maintenance & Failure Forecasting - prototype
=========================================================
Pipeline:
  1. Simulate run-to-failure sensor data for a fleet of machines
  2. Engineer per-machine rolling / baseline-relative / trend features
  3. Three models:
       - Isolation Forest   -> unsupervised health index (anomaly score)
       - GradientBoosting   -> Remaining Useful Life (RUL) regression
       - GradientBoosting   -> "fails within HORIZON cycles" classifier
  4. Evaluate on held-out MACHINES (never split rows of one machine
     across train/test - that leaks the degradation trajectory)
  5. Translate predictions into an alerting policy and report lead time

Run:  python predictive_maintenance.py
Needs: numpy, pandas, scikit-learn, matplotlib
To use real data (e.g. NASA C-MAPSS), replace `simulate_fleet()` with a
loader that returns a DataFrame with columns: machine_id, cycle, <sensors>.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier, IsolationForest)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, roc_auc_score,
                             precision_recall_fscore_support, fbeta_score)

SEED = 42
N_MACHINES = 150
SENSORS = ["vibration", "temperature", "pressure", "current", "rpm", "oil_debris"]
RUL_CAP = 125        # RUL beyond this is "healthy"; capping is standard practice
HORIZON = 30         # classifier: will it fail within this many cycles?
BASELINE_CYCLES = 30
OUT = "outputs"


# ---------------------------------------------------------------- 1. DATA
def simulate_fleet(n=N_MACHINES, seed=SEED):
    """Run-to-failure data. Each machine runs healthy until a random onset,
    then degrades with accelerating (quadratic) wear until failure."""
    rng = np.random.default_rng(seed)
    # (baseline, load sensitivity, degradation gain, noise)
    spec = {
        "vibration":   (2.0, 0.30, 3.0, 0.25),
        "temperature": (70., 3.00, 18., 1.50),
        "pressure":    (30., 1.50, -6., 0.80),
        "current":     (10., 1.20, 4.0, 0.50),
        "rpm":         (1500, 40., -30, 25.0),   # weakly informative on purpose
        "oil_debris":  (5.0, 0.20, 12., 0.80),
    }
    frames = []
    for m in range(n):
        life = int(rng.integers(160, 360))
        onset = int(life * rng.uniform(0.45, 0.70))
        t = np.arange(1, life + 1)
        wear = np.clip((t - onset) / (life - onset), 0, None) ** 2   # 0 -> 1
        load = rng.normal(0, 1, life)                                # operating condition
        d = {"machine_id": m, "cycle": t, "life": life}
        for s, (base, ls, gain, noise) in spec.items():
            unit_offset = rng.normal(0, noise * 2)                   # machine-to-machine variation
            d[s] = base + unit_offset + ls * load + gain * wear * rng.uniform(0.7, 1.3) \
                   + rng.normal(0, noise, life)
        frames.append(pd.DataFrame(d))
    df = pd.concat(frames, ignore_index=True)
    df["rul"] = df["life"] - df["cycle"]
    return df


# ------------------------------------------------------- 2. FEATURES
def build_features(df):
    """Uses only past/current data per machine, so it works online."""
    out = []
    for _, g in df.groupby("machine_id"):
        g = g.sort_values("cycle").copy()
        feats = {}
        for s in SENSORS:
            x = g[s]
            base = x.expanding().mean().where(g["cycle"] <= BASELINE_CYCLES).ffill()
            dev = x - base                                   # removes unit-to-unit offsets
            m10 = dev.rolling(10, min_periods=3).mean()
            m30 = dev.rolling(30, min_periods=5).mean()
            feats[f"{s}_dev10"] = m10
            feats[f"{s}_std10"] = x.rolling(10, min_periods=3).std()
            feats[f"{s}_trend"] = m10 - m30                  # short vs long -> acceleration
            feats[f"{s}_ewm"] = dev.ewm(alpha=0.1).mean()
        g = pd.concat([g, pd.DataFrame(feats, index=g.index)], axis=1)
        out.append(g)
    full = pd.concat(out, ignore_index=True)
    feature_cols = [c for c in full.columns
                    if c.split("_")[0] in SENSORS and c not in SENSORS] + ["cycle"]
    return full, feature_cols


# -------------------------------------------------------------- 3. MODELS
def main():
    df = simulate_fleet()
    data, F = build_features(df)
    data["rul_cap"] = data["rul"].clip(upper=RUL_CAP)
    data["fail_soon"] = (data["rul"] <= HORIZON).astype(int)

    # machine-level split: train / validation (threshold tuning) / test
    ids = np.random.default_rng(SEED).permutation(data["machine_id"].unique())
    n = len(ids)
    tr_ids, va_ids, te_ids = ids[: int(.6 * n)], ids[int(.6 * n): int(.8 * n)], ids[int(.8 * n):]
    tr, va, te = (data[data.machine_id.isin(k)].copy() for k in (tr_ids, va_ids, te_ids))
    print(f"Machines  train/val/test: {len(tr_ids)}/{len(va_ids)}/{len(te_ids)}   "
          f"rows: {len(tr)}/{len(va)}/{len(te)}")

    # --- (a) unsupervised health index: fit on early-life rows only
    healthy = tr[(tr.cycle > BASELINE_CYCLES) & (tr.cycle <= 90)]
    iso = IsolationForest(n_estimators=200, random_state=SEED).fit(healthy[F].fillna(0))
    for d_ in (tr, va, te):
        d_["health_idx"] = -iso.score_samples(d_[F].fillna(0))   # higher = more abnormal

    # --- (b) RUL regression
    reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                        max_depth=6, random_state=SEED)
    reg.fit(tr[F], tr["rul_cap"])
    te["rul_pred"] = reg.predict(te[F]).clip(0, RUL_CAP)

    # --- (c) failure-within-horizon classifier
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                         max_depth=5, random_state=SEED)
    clf.fit(tr[F], tr["fail_soon"])
    va["p"] = clf.predict_proba(va[F])[:, 1]
    te["p"] = clf.predict_proba(te[F])[:, 1]

    # pick alert threshold on VALIDATION machines; recall-weighted (missed failure >> false alarm)
    grid = np.linspace(0.05, 0.95, 91)
    thr = grid[np.argmax([fbeta_score(va.fail_soon, va.p > t, beta=2) for t in grid])]

    # ------------------------------------------------------ 4. EVALUATION
    print("\n=== RUL regression (test machines) ===")
    mae_all = mean_absolute_error(te.rul_cap, te.rul_pred)
    late = te[te.rul <= 50]
    print(f"MAE, all rows (capped RUL):        {mae_all:6.2f} cycles")
    print(f"MAE, final 50 cycles before fail:  {mean_absolute_error(late.rul_cap, late.rul_pred):6.2f} cycles")
    # NASA C-MAPSS asymmetric score: late predictions punished harder than early
    dlt = te.rul_pred - te.rul_cap
    score = np.where(dlt < 0, np.exp(-dlt / 13) - 1, np.exp(dlt / 10) - 1).mean()
    print(f"Asymmetric score (lower=better):   {score:6.2f}")

    print(f"\n=== Failure within {HORIZON} cycles (threshold={thr:.2f}) ===")
    pred = (te.p > thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(te.fail_soon, pred, average="binary")
    print(f"ROC-AUC {roc_auc_score(te.fail_soon, te.p):.3f} | "
          f"precision {pr:.3f} | recall {rc:.3f} | F1 {f1:.3f}")

    # --- operational view: alert = 3 consecutive cycles above threshold
    rows = []
    for mid, g in te.groupby("machine_id"):
        g = g.sort_values("cycle")
        flag = (g.p > thr).rolling(3).sum().fillna(0) >= 3
        life = g.life.iloc[0]
        if flag.any():
            lead = life - g.loc[flag.idxmax(), "cycle"]
            rows.append((mid, life, lead))
        else:
            rows.append((mid, life, np.nan))
    ops = pd.DataFrame(rows, columns=["machine", "life", "lead_time"])
    caught = ops.lead_time.notna().mean()
    too_early = (ops.lead_time > 2 * HORIZON).mean()
    print(f"\n=== Alerting policy (3 consecutive cycles over threshold) ===")
    print(f"Failures warned before they happened: {caught:.0%}")
    print(f"Median warning lead time:             {ops.lead_time.median():.0f} cycles")
    print(f"Alerts firing very early (>{2*HORIZON} cycles, wasteful): {too_early:.0%}")

    # --- feature importance
    sub = te.sample(min(3000, len(te)), random_state=SEED)
    pi = permutation_importance(reg, sub[F], sub["rul_cap"], n_repeats=3,
                                random_state=SEED, scoring="neg_mean_absolute_error")
    imp = pd.Series(pi.importances_mean, index=F).sort_values(ascending=False)
    print("\nTop drivers of RUL prediction:")
    print(imp.head(8).round(3).to_string())

    # ------------------------------------------------------------ 5. PLOTS
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    ex = te[te.machine_id == te_ids[0]].sort_values("cycle")
    for s in ["vibration", "temperature", "oil_debris", "current"]:
        ax[0, 0].plot(ex.cycle, (ex[s] - ex[s].iloc[:BASELINE_CYCLES].mean()), label=s, lw=1)
    ax[0, 0].set(title=f"Raw sensors vs baseline (machine {te_ids[0]})", xlabel="cycle")
    ax[0, 0].legend(fontsize=8)

    for mid in te_ids[:3]:
        g = te[te.machine_id == mid].sort_values("cycle")
        l, = ax[0, 1].plot(g.cycle, g.rul_pred, lw=1.2, label=f"machine {mid} predicted")
        ax[0, 1].plot(g.cycle, g.rul_cap, "--", lw=0.9, color=l.get_color())
    ax[0, 1].set(title="RUL: predicted (solid) vs true (dashed)", xlabel="cycle", ylabel="RUL (cycles)")
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].plot(ex.cycle, ex.health_idx, color="tab:red", label="anomaly score")
    ax[1, 0].plot(ex.cycle, ex.p, color="tab:blue", label=f"P(fail ≤ {HORIZON})")
    ax[1, 0].axhline(thr, color="gray", ls=":", label="alert threshold")
    ax[1, 0].axvline(ex.life.iloc[0], color="k", ls="--", lw=0.8)
    ax[1, 0].set(title="Health index & failure probability", xlabel="cycle")
    ax[1, 0].legend(fontsize=8)

    imp.head(12)[::-1].plot.barh(ax=ax[1, 1])
    ax[1, 1].set(title="Permutation importance (RUL model)", xlabel="MAE increase")
    plt.tight_layout()
    plt.savefig(f"{OUT}/maintenance_report.png", dpi=130)
    data.to_csv(f"{OUT}/fleet_features.csv", index=False)
    print(f"\nSaved plot -> {OUT}/maintenance_report.png")


if __name__ == "__main__":
    main()
