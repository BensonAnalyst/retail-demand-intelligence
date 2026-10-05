"""Write a SMALL mock with exactly the FreshRetailNet-50K schema, for CI / offline testing.
It is NOT the real data. True demand is known here, so tests can check that recovery works.

    python freshretail/scripts/make_mock_data.py --out /tmp/frn_mock --series 400
"""
import argparse
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--series", type=int, default=400)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

n_stores, n_products = max(10, a.series // 20), 60
days = pd.date_range("2024-03-28", periods=97)
hour_shape = np.array([0, 0, 0, 0, 0, 0, .3, .9, 1.2, 1.0, .8, .9, 1.1, 1.0, .8, .8, 1.0, 1.5, 1.9, 1.6, 1.0, .6, .3, 0])
hour_shape = hour_shape / hour_shape.sum()

keys = set()
while len(keys) < a.series:
    keys.add((int(rng.integers(n_stores)), int(rng.integers(n_products))))
rows, true_rows = [], []
for store, prod in sorted(keys):
    cat3 = prod % 12
    base = rng.lognormal(1.5, 0.8)
    tightness = rng.uniform(0.75, 1.5)            # how generously the store is stocked
    for di, dt in enumerate(days):
        dow = dt.dayofweek
        disc = rng.choice([1.0, 1.0, 1.0, 0.9, 0.8])
        lam_day = base * (1 + 0.25 * (dow >= 5)) * (1 / disc) ** 1.5 * rng.lognormal(0, 0.2)
        demand_h = rng.poisson(lam_day * hour_shape * 3) / 3.0
        stock = lam_day * tightness * rng.uniform(0.7, 1.3)
        sold, oos = np.zeros(24), np.zeros(24, dtype=np.int32)
        for h in range(24):
            take = min(stock, demand_h[h])
            sold[h], stock = take, stock - take
            if stock <= 1e-9 and 6 <= h <= 21:
                oos[h] = 1
        rows.append(dict(city_id=store % 5, store_id=store, management_group_id=cat3 % 3,
                         first_category_id=cat3 % 4, second_category_id=cat3 % 8, third_category_id=cat3,
                         product_id=prod, dt=dt.strftime("%Y-%m-%d"), sale_amount=float(sold.sum()),
                         hours_sale=sold.tolist(), stock_hour6_22_cnt=int(oos[6:22].sum()),
                         hours_stock_status=oos.tolist(), discount=float(disc),
                         holiday_flag=int(dow >= 5), activity_flag=int(disc < 1),
                         precpt=float(rng.gamma(1, 2)), avg_temperature=float(rng.normal(22, 4)),
                         avg_humidity=float(rng.uniform(40, 90)), avg_wind_level=float(rng.uniform(1, 5))))
        true_rows.append(dict(store_id=store, product_id=prod, dt=rows[-1]["dt"], true_demand=float(demand_h.sum())))

df = pd.DataFrame(rows)
mx = df["sale_amount"].max()                          # "global normalization", like the real data
df["sale_amount"] /= mx
df["hours_sale"] = df["hours_sale"].apply(lambda v: [x / mx for x in v])
os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
schema = pa.schema([("city_id", pa.int64()), ("store_id", pa.int64()), ("management_group_id", pa.int64()),
                    ("first_category_id", pa.int64()), ("second_category_id", pa.int64()),
                    ("third_category_id", pa.int64()), ("product_id", pa.int64()), ("dt", pa.string()),
                    ("sale_amount", pa.float64()), ("hours_sale", pa.list_(pa.float64())),
                    ("stock_hour6_22_cnt", pa.int32()), ("hours_stock_status", pa.list_(pa.int32())),
                    ("discount", pa.float64()), ("holiday_flag", pa.int32()), ("activity_flag", pa.int32()),
                    ("precpt", pa.float64()), ("avg_temperature", pa.float64()), ("avg_humidity", pa.float64()),
                    ("avg_wind_level", pa.float64())])
cut = days[89].strftime("%Y-%m-%d")
for name, part in [("train", df[df.dt <= cut]), ("eval", df[df.dt > cut])]:
    pq.write_table(pa.Table.from_pandas(part, schema=schema, preserve_index=False), os.path.join(a.out, "data", f"{name}.parquet"))
truth = pd.DataFrame(true_rows)
truth["true_demand"] /= mx
truth.to_parquet(os.path.join(a.out, "true_demand.parquet"))
print(f"mock written: {len(df):,} rows, {len(keys)} series, stock-out day share "
      f"{(df.stock_hour6_22_cnt > 0).mean():.1%}")
