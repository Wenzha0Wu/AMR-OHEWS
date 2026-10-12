"""How much do the deep-tier results depend on the evaluation split and seed?

A fair question to ask of any result resting on 24 held-out events and a single
random seed. The answer matters differently for the two claims this project
makes about the spatial tier, and the distinction is the point of this script.

**The absolute numbers are expected to move.** Lead time on a panel this small is
a noisy statistic; changing where the train/test boundary falls changes which
episodes are scored at all.

**The permutation contrast is not.** Comparing a real adjacency against a
shuffled one holds the split, the seed, the architecture and the training budget
fixed and varies only the graph. Whatever the split does to the absolute
numbers, it does to both arms equally. That is why section 8.13 rests on the
contrast rather than on any single number.

Run with ``python scripts/robustness.py``. Everything is reported -- there is no
best configuration selected, because selecting one would be the error this
script exists to guard against.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from amrews.config import RESULTS                       # noqa: E402
from amrews.eval.benchmark import run_benchmark          # noqa: E402
from amrews.eval.experiments import run_us_deep          # noqa: E402

SPLITS = [(0.5, 0.7), (0.6, 0.8), (0.7, 0.85)]
SEEDS = [0, 1, 2, 3]


def us_deep_grid() -> pd.DataFrame:
    """Every split x seed combination on the real US state-month panel."""
    rows = []
    for split in SPLITS:
        for seed in SEEDS:
            out = run_us_deep(seed=seed, split=split, epochs=25, quiet=True)
            if out.get("status") != "ok":
                print(f"  split={split} seed={seed}: {out.get('status')}")
                continue
            c = out["comparisons"]
            summary = out["summary"].set_index("method")
            rows.append({
                "split": f"{split[0]}/{split[1]}", "seed": seed,
                "events_test": out["events_test"],
                "graph_vs_temporal": c.get("graph_stgnn_vs_temporal_tcn", {}).get("mean_delta_months", np.nan),
                "graph_sig": c.get("graph_stgnn_vs_temporal_tcn", {}).get("significant", False),
                "temporal_vs_fusion": c.get("temporal_tcn_vs_fusion", {}).get("mean_delta_months", np.nan),
                "graph_lead": summary.loc["graph_stgnn", "mean_lead_time"],
                "graph_pen": summary.loc["graph_stgnn", "mean_lead_time_penalised"],
                "fusion_pen": summary.loc["fusion", "mean_lead_time_penalised"],
            })
            print(f"  split={split} seed={seed}: graph-vs-temporal "
                  f"{rows[-1]['graph_vs_temporal']:+.2f} mo", flush=True)
    return pd.DataFrame(rows)


def benchmark_grid(dataset: str = "us_states") -> pd.DataFrame:
    """Seed variation of the permutation control on the largest benchmark."""
    rows = []
    for seed in SEEDS:
        out = run_benchmark(dataset, epochs=25, seed=seed)
        if out.get("status") != "ok":
            continue
        c = out["comparisons"]
        rows.append({
            "dataset": dataset, "seed": seed,
            "graph_vs_temporal": c["graph_stgnn_vs_temporal_tcn"]["mean_delta_weeks"],
            "graph_vs_shuffled": c["graph_stgnn_vs_graph_shuffled"]["mean_delta_weeks"],
            "shuffled_sig": c["graph_stgnn_vs_graph_shuffled"]["significant"],
        })
        print(f"  {dataset} seed={seed}: vs_temporal "
              f"{rows[-1]['graph_vs_temporal']:+.2f}  vs_shuffled "
              f"{rows[-1]['graph_vs_shuffled']:+.2f}", flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    print("=== US state-month panel: split x seed grid ===")
    us = us_deep_grid()
    print("\n=== US-States ILINet benchmark: seed variation ===")
    bench = benchmark_grid("us_states")

    us.to_csv(RESULTS / "robustness_us_deep.csv", index=False)
    bench.to_csv(RESULTS / "robustness_benchmark.csv", index=False)

    print("\n" + "=" * 78)
    print("US state-month panel, graph minus temporal (months):")
    if not us.empty:
        g = us["graph_vs_temporal"].dropna()
        print(us.pivot_table(index="split", columns="seed",
                             values="graph_vs_temporal").round(2).to_string())
        print(f"\n  across {len(g)} configurations: mean {g.mean():+.2f}, "
              f"range [{g.min():+.2f}, {g.max():+.2f}]")
        print(f"  configurations favouring the graph tier: {(g > 0).sum()}/{len(g)}")
        print(f"  configurations where that is significant: {int(us['graph_sig'].sum())}/{len(us)}")

    print("\nUS-States ILINet, permutation control (weeks):")
    if not bench.empty:
        print(bench[["seed", "graph_vs_temporal", "graph_vs_shuffled"]]
              .round(2).to_string(index=False))
        d = bench["graph_vs_shuffled"]
        print(f"\n  real minus shuffled: mean {d.mean():+.2f}, "
              f"range [{d.min():+.2f}, {d.max():+.2f}]")
        print(f"  seeds where the real graph beats the shuffled one significantly: "
              f"{int(bench['shuffled_sig'].sum())}/{len(bench)}")

    (RESULTS / "robustness.json").write_text(json.dumps({
        "us_deep": us.to_dict("records"),
        "benchmark": bench.to_dict("records"),
    }, indent=2, default=float))
    print("\nwrote results/robustness_*.csv")


if __name__ == "__main__":
    main()
