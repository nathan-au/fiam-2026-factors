"""Fetch the external files the research run uses into cache/ (git-ignored). Idempotent.
  1. FINRA mid-month short-interest files 2018-01..2020-05 (cdn.finra.org; the repo's cache already holds 2020-06 onward; files before 2018-01 return 403)
  2. JKP cluster labels and factor directions (github.com/bkelly-lab/ReplicationCrisis, GlobalFactors/)
Run: .venv/bin/python -m fiam_research.download_data
"""
import datetime as dt
import subprocess

import pandas as pd

from fiam_desks import config as C


def finra(start="2018-01-31", end="2020-05-31"):
    d0 = C.CACHE / "finra_short_interest"
    d0.mkdir(parents=True, exist_ok=True)
    for m in pd.date_range(start, end, freq="ME"):
        d = dt.date(m.year, m.month, 15)
        while d.weekday() >= 5:
            d -= dt.timedelta(days=1)
        for back in range(8):
            dd = d - dt.timedelta(days=back)
            f = d0 / f"shrt{dd:%Y%m%d}.csv"
            if f.exists():
                break
            r = subprocess.run(["curl", "-s", "-f", "-o", str(f), f"https://cdn.finra.org/equity/otcmarket/biweekly/{f.name}"])
            if r.returncode == 0:
                break
            f.unlink(missing_ok=True)


def jkp():
    d0 = C.CACHE / "jkp"
    d0.mkdir(parents=True, exist_ok=True)
    base = "https://raw.githubusercontent.com/bkelly-lab/ReplicationCrisis/master/GlobalFactors/"
    subprocess.run(["curl", "-s", "-L", "-o", str(d0 / "cluster_labels.csv"), base + "Cluster%20Labels.csv"], check=True)
    subprocess.run(["curl", "-s", "-L", "-o", str(d0 / "factor_details.xlsx"), base + "Factor%20Details.xlsx"], check=True)
    pd.read_excel(d0 / "factor_details.xlsx").to_csv(d0 / "factor_details.csv", index=False)


if __name__ == "__main__":
    finra()
    jkp()
    print("done")
