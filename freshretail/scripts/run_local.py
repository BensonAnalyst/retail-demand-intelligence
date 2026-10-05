"""Run the full FreshRetailNet experiment with pandas (mirrors notebooks F1-F5).

    python freshretail/scripts/run_local.py --data <dir with data/train.parquet, data/eval.parquet> [--series 5000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from frn import data, pipeline, recovery  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--series", type=int, default=0, help="0 = all series")
ap.add_argument("--out", default=str(ROOT / "outputs"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)
t0 = time.time()
lap = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}")  # noqa: E731

tr_path, ev_path = f"{a.data}/data/train.parquet", f"{a.data}/data/eval.parquet"
keys = data.sample_series(tr_path, a.series or None)
train, eval_ = data.load_panel(tr_path, keys), data.load_panel(ev_path, keys)
full, last_day, chk = pipeline.prepare(train, eval_)
lap(f"loaded {len(full.df):,} rows; orientation check {chk}")

val = recovery.validate_recovery(full, last_day)
lap("recovery validation (masking test on held-out series):\n" + val.round(4).to_string(index=False))

results = []
for cutoff in (last_day - 7, last_day):
    res, rec, _ = pipeline.run_fold(full, cutoff, log=lap)
    results.append(res)
res = pd.concat(results)
print(res.round(4).to_string(index=False))

total, by_cat, by_store, by_hour = pipeline.lost_demand_tables(full, rec, last_day)
print(json.dumps(total, indent=2))
val.to_csv(out / "recovery_validation.csv", index=False)
res.to_csv(out / "forecast_results.csv", index=False)
by_cat.to_csv(out / "lost_demand_by_category.csv", index=False)
by_hour.to_csv(out / "oos_by_hour.csv", index=False)
json.dump(total, open(out / "summary.json", "w"), indent=2)
lap("done")
