"""Execute the FreshRetailNet Databricks notebooks locally (usage: notebook_smoke_test.py <data_dir>) with lightweight Spark/dbutils shims.

Catches Python errors in notebook code before you burn serverless time.
Spark-only paths (applyInPandas) deliberately raise so the driver fallback runs;
SQL display cells are skipped. MLflow runs against a local sqlite registry.

    python scripts/notebook_smoke_test.py
"""
import io
import os
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
WORK = Path(tempfile.mkdtemp())
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import mlflow  # noqa: E402

if "pyspark" not in sys.modules:  # minimal pyspark stub (column expressions are never evaluated)
    for m in ["pyspark", "pyspark.sql"]:
        sys.modules[m] = types.ModuleType(m)
    sys.modules["pyspark.sql.functions"] = MagicMock()
    sys.modules["pyspark.sql"].functions = sys.modules["pyspark.sql.functions"]
    sys.modules["pyspark.sql"].DataFrame = type("DataFrame", (), {})
    sys.modules["pyspark"].sql = sys.modules["pyspark.sql"]

mlflow.set_tracking_uri(f"sqlite:///{WORK}/mlflow.db")
mlflow.set_registry_uri = lambda *_a, **_k: None  # UC registry -> local

TABLES: dict[str, bytes] = {}


class _Widgets:
    def __init__(self): self.v = {}
    def text(self, k, d, *_): self.v.setdefault(k, d)
    def dropdown(self, k, d, *_): self.v.setdefault(k, d)
    def get(self, k): return self.v[k]


class _Lib:
    def restartPython(self): pass


class dbutils:  # noqa: N801
    widgets = _Widgets()
    library = _Lib()


class _Writer:
    def __init__(self, pdf): self.pdf = pdf
    def mode(self, *_): return self
    def option(self, *_): return self
    def saveAsTable(self, name):  # parquet round-trip mimics Delta typing
        buf = io.BytesIO(); self.pdf.to_parquet(buf); TABLES[name] = buf.getvalue()


class _DF:
    def __init__(self, pdf): self.pdf = pdf
    @property
    def write(self): return _Writer(self.pdf)
    def toPandas(self): return self.pdf.copy()
    def select(self, *c): return _DF(self.pdf[list(c)])
    def where(self, *_): return self
    def groupBy(self, *_): return self
    def applyInPandas(self, *_): raise NotImplementedError("no spark locally")
    def first(self): return ["local@user"]


class spark:  # noqa: N801
    @staticmethod
    def sql(q): return _DF(pd.DataFrame())
    @staticmethod
    def table(name): return _DF(pd.read_parquet(io.BytesIO(TABLES[name])))
    @staticmethod
    def createDataFrame(pdf): return _DF(pdf)


def display(x):
    if isinstance(x, pd.DataFrame):
        print(x.head(12).to_string(max_cols=12))


def cells(path: Path):
    src = path.read_text()
    for cell in src.split("# COMMAND ----------"):
        lines = [l for l in cell.splitlines() if not l.startswith("# Databricks notebook source")]
        magic = [l for l in lines if l.startswith("# MAGIC")]
        if magic:
            first = magic[0].replace("# MAGIC", "").strip()
            if first.startswith("%run"):
                yield "RUN", first.split()[1]
            continue  # %md, %pip, %sql, %restart_python
        code = "\n".join(lines).strip()
        if code:
            yield "PY", code


def run_notebook(name: str, ns: dict):
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    for kind, payload in cells(NB / name):
        if kind == "RUN":
            run_notebook(Path(payload).name + ".py", ns)
            continue
        # notebook cwd is notebooks/ on Databricks -> os.getcwd()/../src works
        exec(compile(payload, name, "exec"), ns)


if __name__ == "__main__":
    os.chdir(NB)
    ns = {"spark": spark, "dbutils": dbutils, "display": display, "__name__": "__main__"}
    import matplotlib
    matplotlib.use("Agg")
    data_dir = sys.argv[1]                                  # folder holding train.parquet / eval.parquet
    dbutils.widgets.v.update({"data_dir": data_dir, "n_series": "0"})
    for nb in sorted(p.name for p in NB.glob("F[1-4]_*.py")):
        run_notebook(nb, ns)
    print("\nALL NOTEBOOKS RAN. tables:", sorted(TABLES))
