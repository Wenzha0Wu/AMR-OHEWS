"""Verify the spatio-temporal tiers on data where the graph is known to matter.

Every conclusion this project draws about Tier 3 rests on an assumption we never
tested: that the implementation works. We have only ever run it where it failed
-- on a simulated world with sixty events, and on a real AMR panel whose spatial
signal we independently showed to be undetectable. A model that is simply broken
would produce exactly the same results.

This module supplies the missing control. Influenza-like-illness surveillance
has strong, well-documented spatial autocorrelation: neighbouring states peak in
sequence, and the standard ST-GNN literature uses precisely these datasets as
benchmarks. If our Tier 3 cannot beat Tier 2 *here*, the implementation is at
fault and the AMR conclusions need revisiting. If it can, then the same code
succeeding where signal exists and failing where we showed there is none is
evidence for the AMR findings rather than against them.

Datasets (Cola-GNN, Deng et al., CIKM 2020; and PyTorch Geometric Temporal):

``us_states``
    ILINet, 360 weeks x 49 states, with a 49x49 state adjacency. Structurally
    the closest public analogue of our own US AMR panel.
``japan``
    348 weeks x 47 prefectures, with a 47x47 adjacency.
``chickenpox``
    521 weeks x 20 Hungarian counties, 102 edges.

The benchmark is deliberately reframed as *our* task rather than the regression
task the source papers use. We are not trying to reproduce Cola-GNN's RMSE; we
are testing whether the early-warning pipeline that produced our AMR results can
detect outbreak onsets earlier when it is allowed to see neighbouring regions.
Both tiers get a **single input channel** -- the region's own history -- so the
only thing Tier 3 has that Tier 2 lacks is the graph.

A shuffled-adjacency arm runs alongside the real one. If the real graph beats a
randomly permuted graph, the propagation is using genuine spatial structure
rather than merely adding parameters.
"""
from __future__ import annotations

import argparse
import json
import subprocess

import numpy as np
import pandas as pd

from ..config import DATA_RAW, RESULTS
from ..data.real_panel import RealEvent
from ..models.graph import GraphModel
from ..models.temporal import TemporalModel
from .metrics import (
    bootstrap_ci,
    calibrate_threshold,
    evaluate_scores,
    paired_lead_time_delta,
    probabilistic_metrics,
    wilcoxon_delta,
)

COLA_BASE = "https://raw.githubusercontent.com/amy-deng/colagnn/master/data"
PYGT_BASE = ("https://raw.githubusercontent.com/benedekrozemberczki/"
             "pytorch_geometric_temporal/master/dataset")

DATASETS = {
    "us_states": {"data": "state360.txt", "adj": "state-adj.txt"},
    "japan": {"data": "japan.txt", "adj": "japan-adj.txt"},
    "us_regions": {"data": "region785.txt", "adj": None},
}

# Weekly task parameters. Influenza onsets are worth knowing about a month out.
HORIZON = 4
MATCH_BEFORE = 8
MATCH_AFTER = 2
REFRACTORY = 6
PERIODS_PER_YEAR = 52
WINDOW = 52


def _fetch(url: str, name: str):
    """Download once into ``data/raw/benchmark`` and return the path."""
    dest = DATA_RAW / "benchmark" / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists() or dest.stat().st_size < 100:
        subprocess.run(["curl", "-sL", "--max-time", "180", url, "-o", str(dest)], check=True)
    return dest


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (T, R) observations and an (R, R) adjacency."""
    if name == "chickenpox":
        payload = json.loads(_fetch(f"{PYGT_BASE}/chickenpox.json", "chickenpox.json").read_text())
        series = np.asarray(payload["FX"], dtype=float)
        n = series.shape[1]
        adj = np.zeros((n, n))
        for a, b in payload["edges"]:
            adj[a, b] = adj[b, a] = 1.0
        return series, adj

    spec = DATASETS[name]
    series = np.loadtxt(_fetch(f"{COLA_BASE}/{spec['data']}", spec["data"]), delimiter=",")
    if spec["adj"]:
        adj = np.loadtxt(_fetch(f"{COLA_BASE}/{spec['adj']}", spec["adj"]), delimiter=",")
    else:
        # No published adjacency: fall back to a chain, and say so in the output.
        n = series.shape[1]
        adj = np.zeros((n, n))
        for i in range(n - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
    return series, adj


def to_panel(series: np.ndarray) -> pd.DataFrame:
    """Reshape (T, R) counts into the long panel the pipeline expects.

    Counts are scaled by each region's own 95th percentile so that the absolute
    thresholds in the event detector mean the same thing in a large state and a
    small one. Region identity is an integer because the deep tiers carry a
    learned per-region embedding.
    """
    T, R = series.shape
    scale = np.percentile(series, 95, axis=0)
    scale[scale <= 0] = 1.0
    scaled = np.clip(series / scale, 0, 1.5)
    rows = []
    for r in range(R):
        rows.append(pd.DataFrame({
            "country": r, "determinant": "ili", "period": np.arange(T),
            "human_p": scaled[:, r], "human_n": 100.0,
        }))
    return pd.concat(rows, ignore_index=True)


def build_task(panel: pd.DataFrame, horizon: int = HORIZON) -> tuple[pd.DataFrame, list]:
    """Detect outbreak onsets and build the causal feature table.

    The same event definition as the real AMR panel -- a sustained rise over a
    trailing baseline -- so that the benchmark exercises the same code, not a
    bespoke easier task.
    """
    from ..data.real_panel import find_real_events

    events = find_real_events(panel.rename(columns={"country": "country"}),
                              baseline_periods=26, k_sd=2.0,
                              min_absolute_rise=0.05, sustain=2, refractory=26)

    df = panel.rename(columns={"period": "week", "country": "region"}).copy()
    df = df.sort_values(["region", "determinant", "week"]).reset_index(drop=True)

    by_series: dict[tuple, list[int]] = {}
    for e in events:
        by_series.setdefault((e.region, e.determinant), []).append(e.human_emergence_week)

    week = df["week"].to_numpy()
    key = list(zip(df["region"], df["determinant"]))
    label = np.zeros(len(df), dtype=int)
    post = np.zeros(len(df), dtype=bool)
    for i, k in enumerate(key):
        weeks_of = by_series.get(k)
        if not weeks_of:
            continue
        t = week[i]
        upcoming = [w for w in weeks_of if w > t]
        if upcoming:
            label[i] = int(min(upcoming) <= t + horizon)
        post[i] = any(w <= t < w + horizon for w in weeks_of)
    df[f"label_h{horizon}"] = label
    df["post_event"] = post
    return df, events


def run_benchmark(name: str = "us_states", epochs: int = 25, seed: int = 0,
                  budget: float = 1.0) -> dict:
    """Tier 2 against Tier 3 (real graph and shuffled graph) on one benchmark."""
    series, adj = load_dataset(name)
    panel = to_panel(series)
    features, events = build_task(panel)

    T = int(features["week"].max()) + 1
    train_end, valid_end = int(T * 0.6), int(T * 0.8)
    train = features[features["week"] < train_end]
    valid = features[(features["week"] >= train_end) & (features["week"] < valid_end)]

    n_regions = int(features["region"].nunique())
    test_events = sum(1 for e in events if e.human_emergence_week >= valid_end)
    info = {"dataset": name, "regions": n_regions, "weeks": T,
            "adjacency_density": float((adj > 0).mean()),
            "events": len(events), "events_test": test_events,
            "positives_train": int(train[f"label_h{HORIZON}"].sum())}
    print(f"  {info}")
    if test_events < 10:
        return {"status": "insufficient_test_events", **info}

    eval_kw = dict(match_before=MATCH_BEFORE, match_after=MATCH_AFTER,
                   refractory=REFRACTORY, periods_per_year=PERIODS_PER_YEAR)
    channels = ["human_p"]          # own history only: the graph is Tier 3's only edge
    rows, per_event, probs = [], [], {}

    def score(frame, col, method):
        thr = calibrate_threshold(frame, col, events, train_end, valid_end, budget, **eval_kw)
        res = evaluate_scores(frame, col, events, method, thr, valid_end, T, **eval_kw)
        rows.append({**res.summary(), "dataset": name})
        per_event.append(res.per_event)
        print(f"     {method:22s} sens={res.sensitivity:.2f} lead={res.mean_lead_time:6.2f} "
              f"pen={res.mean_lead_time_penalised:6.2f} far={res.false_alarms_per_series_year:.2f}")
        return res

    temporal = TemporalModel(horizon=HORIZON, n_regions=n_regions, n_dets=1,
                             epochs=epochs, seed=seed, channels=channels,
                             window=WINDOW).fit(train, valid=valid)
    tsc = temporal.predict_frame(features, with_uncertainty=False)
    score(tsc, "score_temporal", "temporal_tcn")
    test_mask = (features["week"] >= valid_end) & (~features["post_event"])
    probs["temporal_tcn"] = probabilistic_metrics(
        features.loc[test_mask, f"label_h{HORIZON}"].to_numpy(),
        pd.DataFrame(features[["week", "region", "determinant"]]).merge(
            tsc, on=["week", "region", "determinant"], how="left"
        ).loc[test_mask.to_numpy(), "score_temporal"].fillna(0).to_numpy())

    steps = temporal.history[-1]["steps"]
    rng = np.random.default_rng(seed)
    for arm, matrix in (("graph_stgnn", adj),
                        ("graph_shuffled", adj[np.ix_(*(rng.permutation(n_regions),) * 2)])):
        graph = GraphModel(matrix.astype(np.float32), horizon=HORIZON, n_dets=1,
                           epochs=epochs, seed=seed, target_steps=steps,
                           channels=channels, window=WINDOW).fit(train, valid=valid)
        gsc = graph.predict_frame(features)
        score(gsc, "score_graph", arm)
        probs[arm] = probabilistic_metrics(
            features.loc[test_mask, f"label_h{HORIZON}"].to_numpy(),
            pd.DataFrame(features[["week", "region", "determinant"]]).merge(
                gsc, on=["week", "region", "determinant"], how="left"
            ).loc[test_mask.to_numpy(), "score_graph"].fillna(0).to_numpy())
        if arm == "graph_stgnn":
            info["graph_diagnostics"] = graph.diagnostics()

    ev = pd.concat(per_event, ignore_index=True)
    comparisons = {}
    for a, b in (("graph_stgnn", "temporal_tcn"), ("graph_stgnn", "graph_shuffled")):
        joined = paired_lead_time_delta(ev[ev["method"] == a], ev[ev["method"] == b])
        if joined.empty:
            continue
        mean, lo, hi = bootstrap_ci(joined["delta"].to_numpy())
        comparisons[f"{a}_vs_{b}"] = {
            "mean_delta_weeks": mean, "ci_low": lo, "ci_high": hi,
            "n_paired_events": int(len(joined)),
            "significant": bool(np.isfinite(lo) and lo > 0),
            "wilcoxon_p": wilcoxon_delta(joined["delta"].to_numpy())["p_value"],
        }
    return {"status": "ok", **info, "summary": pd.DataFrame(rows),
            "per_event": ev, "comparisons": comparisons, "prob_metrics": probs}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify the spatial tier on ST-GNN benchmarks.")
    ap.add_argument("--datasets", nargs="+",
                    default=["us_states", "japan", "chickenpox"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--tag", default="benchmark")
    args = ap.parse_args()

    outputs, summaries = {}, []
    for name in args.datasets:
        print(f"\n=== {name} ===")
        out = run_benchmark(name, epochs=args.epochs)
        if out.get("status") != "ok":
            print("   skipped:", out.get("status"))
            continue
        summaries.append(out["summary"])
        outputs[name] = {k: v for k, v in out.items()
                         if k not in ("summary", "per_event")}
        for k, c in out["comparisons"].items():
            star = "*" if c["significant"] else " "
            print(f"   {star} {k:34s} {c['mean_delta_weeks']:+.2f} wk "
                  f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] p={c['wilcoxon_p']:.3g}")
        print("     AUC-PR:", {k: round(v["auc_pr"], 4) for k, v in out["prob_metrics"].items()})

    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(
            RESULTS / f"benchmark_summary_{args.tag}.csv", index=False)
        (RESULTS / f"benchmark_{args.tag}.json").write_text(
            json.dumps(outputs, indent=2, default=float))
        print(f"\nwrote results/benchmark_*_{args.tag}")

        print("\n=== verdict ===")
        for name, o in outputs.items():
            c = o["comparisons"].get("graph_stgnn_vs_temporal_tcn", {})
            s = o["comparisons"].get("graph_stgnn_vs_graph_shuffled", {})
            print(f"  {name:12s} graph vs temporal {c.get('mean_delta_weeks', float('nan')):+.2f} wk"
                  f" (sig={c.get('significant')});  real vs shuffled graph "
                  f"{s.get('mean_delta_weeks', float('nan')):+.2f} wk (sig={s.get('significant')})")


if __name__ == "__main__":
    main()
