"""Load FreshRetailNet-50K into a compact panel: scalar columns in pandas,
the 24-hour sales / out-of-stock lists as dense numpy matrices (n_rows x 24).

Dataset: Dingdong-Inc/FreshRetailNet-50K (Hugging Face, CC BY 4.0)
  data/train.parquet  ~4.5M rows = 50,000 store-product series x 90 days
  data/eval.parquet   ~350k rows = the next 7 days
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO_ID = "Dingdong-Inc/FreshRetailNet-50K"
FILES = {"train": "data/train.parquet", "eval": "data/eval.parquet"}
KEY = ["store_id", "product_id"]
SCALARS = ["city_id", "store_id", "management_group_id", "first_category_id", "second_category_id",
           "third_category_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt", "discount",
           "holiday_flag", "activity_flag", "precpt", "avg_temperature", "avg_humidity", "avg_wind_level"]
OP_HOURS = slice(6, 23)  # store operating window 06:00-22:59, as in stock_hour6_22_cnt


@dataclass
class Panel:
    df: pd.DataFrame          # one row per store-product-day
    sales: np.ndarray         # (n, 24) hourly sales
    oos: np.ndarray           # (n, 24) 1 = out of stock in that hour

    def take(self, idx) -> "Panel":
        return Panel(self.df.iloc[idx].reset_index(drop=True), self.sales[idx], self.oos[idx])


def download(dest_dir: str) -> dict[str, str]:
    """Download the two parquet files from Hugging Face into dest_dir (e.g. a UC Volume)."""
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    out = {}
    for split, fname in FILES.items():
        target = os.path.join(dest_dir, f"{split}.parquet")
        if not os.path.exists(target):
            src = hf_hub_download(REPO_ID, fname, repo_type="dataset")
            shutil.copy(src, target)
        out[split] = target
    return out


def _list_to_matrix(col, width: int = 24) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.compute as pc
    arr = col.combine_chunks() if hasattr(col, "combine_chunks") else col
    lens = pc.list_value_length(arr).to_numpy(zero_copy_only=False)
    flat = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=float)
    if (lens == width).all():
        return flat.reshape(-1, width)
    out = np.zeros((len(lens), width))          # pad/truncate irregular rows (defensive)
    pos = 0
    for i, n in enumerate(lens):
        out[i, :min(n, width)] = flat[pos:pos + min(n, width)]
        pos += n
    return out


def sample_series(path: str, n_series: int | None, seed: int = 42) -> pd.DataFrame | None:
    """Pick whole stores at random until ~n_series store-product series are covered.
    Sampling by store (not by series) keeps every store's assortment complete and lets the
    parquet reader filter rows while reading, so the full 4.5M-row file never sits in memory."""
    if not n_series:
        return None
    keys = pd.read_parquet(path, columns=KEY).drop_duplicates()
    per_store = keys.groupby("store_id").size().sample(frac=1, random_state=seed)
    stores = per_store.index[: int((per_store.cumsum() < n_series).sum()) + 1]
    return keys[keys["store_id"].isin(stores)]


def load_panel(path: str, keys: pd.DataFrame | None = None) -> Panel:
    import pyarrow as pa
    import pyarrow.parquet as pq
    filters = [("store_id", "in", sorted(keys["store_id"].unique().tolist()))] if keys is not None else None
    t = pq.read_table(path, filters=filters)
    if keys is not None:
        k = t.select(KEY).to_pandas()
        mask = k.merge(keys.assign(_keep=1), on=KEY, how="left")["_keep"].fillna(0).astype(bool).to_numpy()
        t = t.filter(pa.array(mask))
    df = t.select(SCALARS).to_pandas()
    sales = _list_to_matrix(t.column("hours_sale"))
    oos = _list_to_matrix(t.column("hours_stock_status")).astype(np.int8)
    df["dt"] = pd.to_datetime(df["dt"])
    order = np.lexsort((df["dt"].to_numpy(), df["product_id"].to_numpy(), df["store_id"].to_numpy()))
    return Panel(df.iloc[order].reset_index(drop=True), sales[order], oos[order])


def check_oos_orientation(p: Panel) -> dict:
    """Sanity-check that hours_stock_status==1 means OUT of stock (sales should be ~0 there).
    Flips the matrix in place if the data uses the opposite convention."""
    s1 = p.sales[p.oos == 1].mean() if (p.oos == 1).any() else np.nan
    s0 = p.sales[p.oos == 0].mean() if (p.oos == 0).any() else np.nan
    flipped = bool(np.isfinite(s1) and np.isfinite(s0) and s1 > s0)
    if flipped:
        p.oos = (1 - p.oos).astype(np.int8)
    op_cnt = p.oos[:, OP_HOURS].sum(1)
    agree = float((op_cnt == p.df["stock_hour6_22_cnt"].to_numpy()).mean())
    return {"mean_sales_when_flag1": float(s1), "mean_sales_when_flag0": float(s0),
            "flipped": flipped, "share_rows_matching_stock_hour6_22_cnt": agree}


def add_day_index(train: Panel, eval_: Panel) -> int:
    """Integer day index shared by train and eval; returns the last train day (the cutoff)."""
    d0 = train.df["dt"].min()
    for p in (train, eval_):
        p.df["day"] = (p.df["dt"] - d0).dt.days.astype(int)
    return int(train.df["day"].max())


def concat(a: Panel, b: Panel) -> Panel:
    return Panel(pd.concat([a.df, b.df], ignore_index=True),
                 np.vstack([a.sales, b.sales]), np.vstack([a.oos, b.oos]))
