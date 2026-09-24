"""
desk_18_noise_floor - how much does the LP book's IR move for reasons that are NOT signal quality?

desk_15 showed that adding ~3 rows/month moved DEV net IR by 0.10. Before judging any variant by its book IR, measure the construction noise floor on DEV:
  N1 jitter: B1 score + iid N(0, k * sd_month) noise (k = 0.02, 0.05, 0.10), 20 seeds each -> spread of net IR and of IC
  N2 IC-matched random signals: the IR a book gets from a signal of known IC (B1 score mixed with shuffled score to hit IC ~0.00 / 0.01 / 0.02)
Output: sd of net IR under N1 (the minimum IR difference worth discussing) and the IR-vs-IC map (how much IC a book IR "means").
DEV only. B1 = baseline + desk_15 corrections.
"""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from fiam_desks import evaluate as E  # noqa: E402
from fiam_research import core, ledger  # noqa: E402

OUT = HERE / "output"
EXP = "desk_18_noise_floor"


def main():
    t0 = time.time()
    ctx = core.research_ctx(text=False)
    P = ctx.P
    s = core.b1_score(ctx)
    sd = pd.Series(s).groupby(P.df["eom"].to_numpy()).transform("std").to_numpy()
    rows = []
    for k in (0.02, 0.05, 0.10):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            x = s + rng.normal(0, 1, len(s)) * k * sd
            r = core.b1_book(ctx, x, "dev", keep_frames=False)
            ic = E.ic_series(P, x, "dev").mean()
            rows.append({"test": "jitter", "k": k, "seed": seed, "ir_net": r["ir_net"], "ir_gross": r["ir_gross"], "ic": ic})
        d = pd.DataFrame([r_ for r_ in rows if r_["k"] == k])
        print(f"jitter k={k}: net IR mean {d.ir_net.mean():+.3f} sd {d.ir_net.std():.3f} [{d.ir_net.min():+.3f}, {d.ir_net.max():+.3f}] | IC sd {d.ic.std():.4f}", flush=True)
    for alpha in (0.0, 0.25, 0.5):
        for seed in range(10):
            sh = E.shuffle_within(P, s, seed)
            x = alpha * s + (1 - alpha) * sh
            r = core.b1_book(ctx, x, "dev", keep_frames=False)
            rows.append({"test": "mix", "k": alpha, "seed": seed, "ir_net": r["ir_net"], "ir_gross": r["ir_gross"], "ic": E.ic_series(P, x, "dev").mean()})
        d = pd.DataFrame([r_ for r_ in rows if r_["test"] == "mix" and r_["k"] == alpha])
        print(f"mix alpha={alpha}: IC {d.ic.mean():+.4f} | net IR mean {d.ir_net.mean():+.3f} sd {d.ir_net.std():.3f}", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "noise_runs.csv", index=False)
    summ = D.groupby(["test", "k"]).agg(ir_mean=("ir_net", "mean"), ir_sd=("ir_net", "std"), ic_mean=("ic", "mean"), ic_sd=("ic", "std")).round(4)
    summ.to_csv(OUT / "summary.csv")
    ledger.record(EXP, "noise_floor", "jitter+mix", "construction noise floor", {"n_runs": len(D)}, "n/a", "n/a", "dev", 1, {}, decision="diagnostic")
    print(summ.to_string(), f"\n{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
