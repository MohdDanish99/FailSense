"""Train models and save them to models/pdm.joblib.
Usage: python train.py [--csv your_data.csv]
CSV needs: machine_id, cycle, <sensor columns>, and either 'rul' or 'life'."""
import sys, joblib, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
import predictive_maintenance as pm

if "--csv" in sys.argv:
    df = pd.read_csv(sys.argv[sys.argv.index("--csv") + 1])
    if "rul" not in df:
        df["life"] = df.groupby("machine_id").cycle.transform("max"); df["rul"] = df.life - df.cycle
else:
    df = pm.simulate_fleet()
data, F = pm.build_features(df)
data["rul_cap"] = data.rul.clip(upper=pm.RUL_CAP)
data["fail_soon"] = (data.rul <= pm.HORIZON).astype(int)
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=.06, max_depth=6, random_state=42).fit(data[F], data.rul_cap)
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=.06, max_depth=5, random_state=42).fit(data[F], data.fail_soon)
joblib.dump(dict(reg=reg, clf=clf, features=F, sensors=pm.SENSORS), "models/pdm.joblib")
print(f"Trained on {data.machine_id.nunique()} machines, {len(F)} features -> models/pdm.joblib")
