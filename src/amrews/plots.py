"""Figures for the report: simulator sanity checks and result summaries.

Every figure reads from the artefacts written by
:mod:`amrews.eval.experiments`, so the numbers in the report, the numbers in the
figures and the numbers in the dashboard cannot drift apart.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import (DATA_PROCESSED, FALSE_ALARM_BUDGET_PER_SERIES_YEAR, FIGURES,
                     MATCH_WINDOW_BEFORE, PALETTE, RESULTS, VALID_END_WEEK)

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.grid": True,
    "grid.color": PALETTE.grid,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "legend.frameon": False,
})

PRETTY = {
    "fusion_real": "One Health fusion",
    "baseline_ewma": "EWMA chart (clinical only)",
    "baseline_cusum": "CUSUM chart (clinical only)",
    "baseline_farrington": "Farrington-style (clinical only)",
    "baseline_manual": "Quarterly manual review",
    "fusion": "One Health fusion (Tier 1)",
    "temporal_tcn": "Temporal TCN (Tier 2)",
    "graph_stgnn": "Spatio-temporal GNN (Tier 3)",
    "ensemble": "Ensemble (Tiers 1-3)",
    "ablation_clinical_only": "Ablation: clinical only",
    "ablation_no_environment": "Ablation: no wastewater",
    "ablation_no_animal": "Ablation: no animal/AMU",
    "ablation_no_food": "Ablation: no food chain",
    "ablation_no_network": "Ablation: no trade network",
}


def _pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------- #
# Simulator sanity check
# --------------------------------------------------------------------------- #
def _takeoff_week(block: pd.DataFrame, onset: int, lookback: int = 104) -> int:
    """First week the latent animal reservoir passes a quarter of its eventual rise."""
    b = block.set_index("week")["latent_animal"]
    base = b.loc[max(0, onset - lookback): max(1, onset - 52)].mean()
    plateau = b.loc[onset: onset + lookback].max()
    if not np.isfinite(base) or not np.isfinite(plateau) or plateau <= base:
        return onset
    threshold = base + 0.25 * (plateau - base)
    window = b.loc[max(0, onset - lookback):]
    hits = window[window >= threshold]
    return int(hits.index[0]) if len(hits) else onset


def sanity_plots(seed: int = 0, out: Path | None = None, smooth: int = 13) -> Path:
    """One true event and one decoy, showing every sector on a shared timeline.

    Series are drawn as 13-week rolling means. At the weekly denominators real
    AMR surveillance works with, the raw proportion is almost pure binomial
    noise and the lead-lag structure the figure exists to show is invisible in
    it; the smoothing is a presentation choice and is stated on the axis. The
    models themselves see the unsmoothed series.
    """
    from .data.features import build_features
    from .data.simulate import simulate

    panel, events, trade = simulate(seed)
    feats = build_features(panel, events, trade)

    # Pick the clearest true event: the longest reservoir-to-human lag among
    # events whose emergence week sits well inside the archive, so that both the
    # pre-event baseline and the post-event plateau are visible.
    candidates = [e for e in events if e.is_true_event and e.human_emergence_week
                  and 150 <= e.human_emergence_week <= 440]
    true_ev = max(candidates, key=lambda e: e.lag_to_human) if candidates else next(
        e for e in events if e.is_true_event and e.human_emergence_week)
    decoys = [e for e in events if not e.is_true_event
              and 150 <= e.onset_week <= 440]
    decoy = max(decoys, key=lambda e: e.amplitude) if decoys else next(
        e for e in events if not e.is_true_event)

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 6.6))
    for ax, ev, title, centre in (
        (axes[0], true_ev, "True emergence event: the reservoir moves first",
         true_ev.human_emergence_week),
        (axes[1], decoy, "Decoy: the reservoir moves and nothing reaches the clinic",
         decoy.onset_week),
    ):
        block = feats[(feats["region"] == ev.region)
                      & (feats["determinant"] == ev.determinant)].sort_values("week")
        roll = lambda c: block[c].rolling(smooth, min_periods=3).mean()  # noqa: E731

        ax.plot(block["week"], roll("human_p"), color=PALETTE.human, lw=2.2,
                label="human clinical", zorder=4)
        ax.plot(block["week"], roll("animal_p"), color=PALETTE.animal, lw=1.7, label="animal")
        ax.plot(block["week"], roll("food_p"), color=PALETTE.food, lw=1.5, label="food")
        env = roll("env_x")
        top = max(roll("animal_p").max(), roll("human_p").max())
        env_scaled = (env - env.min()) / max(env.max() - env.min(), 1e-9) * top
        ax.plot(block["week"], env_scaled, color=PALETTE.environment, lw=1.4,
                ls="--", label="wastewater (rescaled)")

        lo, hi = max(0, centre - 160), min(int(block["week"].max()), centre + 110)
        ax.set_xlim(lo, hi)
        vis = block[(block["week"] >= lo) & (block["week"] <= hi)]
        ymax = max(roll("animal_p")[vis.index].max(), roll("human_p")[vis.index].max()) * 1.35
        ax.set_ylim(0, ymax)

        # Mark the reservoir take-off the same way the human ground truth is
        # defined -- a quarter of the way to its eventual plateau -- rather than
        # with the event's logistic midpoint parameter. The midpoint is not an
        # observable, and on a slow event it sits long after the reservoir has
        # visibly moved, which made the annotated "warning available" far
        # shorter than the warning a detector could actually earn.
        takeoff = _takeoff_week(block, ev.onset_week)
        ax.axvline(takeoff, color=PALETTE.animal, ls=":", lw=1.6)
        ax.annotate("reservoir take-off", (takeoff, ymax * 0.97),
                    xytext=(-5, 0), textcoords="offset points", fontsize=8,
                    color=PALETTE.animal, va="top", ha="right")
        if ev.human_emergence_week:
            ax.axvline(ev.human_emergence_week, color=PALETTE.truth, lw=1.8)
            ax.annotate("human emergence\n(ground truth)",
                        (ev.human_emergence_week, ymax * 0.97),
                        xytext=(6, 0), textcoords="offset points", fontsize=8,
                        va="top")
            ax.annotate(
                "", xy=(ev.human_emergence_week, ymax * 0.60),
                xytext=(takeoff, ymax * 0.60),
                arrowprops=dict(arrowstyle="<->", color=PALETTE.alert, lw=1.4))
            ax.annotate(f"{ev.human_emergence_week - takeoff} weeks of reservoir lead",
                        ((takeoff + ev.human_emergence_week) / 2, ymax * 0.635),
                        ha="center", fontsize=8.5, color=PALETTE.alert)
        ax.set_title(f"{title}   [region {ev.region}, {ev.determinant}]", loc="left")
        ax.set_ylabel("resistant proportion")
    # Lower right: at this zoom it is the only corner the curves and the event
    # annotations both leave free.
    axes[0].legend(ncol=2, fontsize=8, loc="lower right", framealpha=0.9,
                   frameon=True, edgecolor="none")
    axes[1].set_xlabel(f"week ({smooth}-week rolling mean; models see the unsmoothed series)")
    fig.tight_layout()

    out = out or FIGURES / "fig1_simulator.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def lead_time_figure(tag: str = "full", out: Path | None = None,
                     budget: float = FALSE_ALARM_BUDGET_PER_SERIES_YEAR) -> Path:
    """Per-event lead-time distributions, conventional baselines against the tiers."""
    ev = pd.read_csv(RESULTS / f"per_event_{tag}.csv")
    # Filter to one policy AND one budget: the results file holds the whole
    # sweep, and pooling it would plot every event three times over.
    ev = ev[(ev["policy"] == "prospective") & (np.isclose(ev["budget"], budget))]
    order = (ev.groupby("method")["lead_time"].mean().sort_values().index.tolist())
    order = [m for m in order if not m.startswith("ablation_")]

    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8.6, 0.52 * len(order) + 2.0))
    for i, method in enumerate(order):
        vals = ev.loc[ev["method"] == method, "lead_time"].dropna().to_numpy()
        if not len(vals):
            continue
        colour = PALETTE.baseline if method.startswith("baseline") else PALETTE.human
        ax.scatter(vals, np.full(len(vals), i) + rng.uniform(-0.3, 0.3, len(vals)),
                   s=9, alpha=0.3, color=colour, edgecolors="none", zorder=2)
        ax.boxplot([vals], positions=[i], vert=False, widths=0.5, showfliers=False,
                   medianprops=dict(color=PALETTE.truth, lw=1.4),
                   boxprops=dict(color=colour, lw=1.1),
                   whiskerprops=dict(color=colour, lw=1.0),
                   capprops=dict(color=colour, lw=1.0), zorder=3)
        ax.plot([vals.mean()], [i], marker="D", ms=7, color=PALETTE.alert, zorder=5)
    ax.axvline(0, color=PALETTE.truth, lw=1.4)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_pretty(m) for m in order])
    ax.set_xlabel("detection lead time (weeks). 0 = resistance already established in humans; "
                  "positive = early warning")
    ax.set_title(f"Lead time per detected event (diamond = mean, line = median)\n"
                 f"30 seeds, {budget} false alarms per series-year, prospective thresholds",
                 loc="left")
    fig.tight_layout()
    out = out or FIGURES / "fig2_lead_time.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def ablation_figure(tag: str = "full", out: Path | None = None,
                    budget: float = FALSE_ALARM_BUDGET_PER_SERIES_YEAR) -> Path:
    """What each surveillance stream is worth, as weeks of lead time lost by removing it.

    Plotted as the *cost of removal* rather than the absolute score of each
    ablated model: that is the quantity an ablation exists to estimate, and it
    keeps a stream whose contribution is near zero visible as a near-zero bar
    instead of an invisible one.
    """
    summary = pd.read_csv(RESULTS / f"summary_{tag}.csv")
    summary = summary[(summary["policy"] == "prospective")
                      & (np.isclose(summary["budget"], budget))
                      & (summary["tier"].isin(["ablation", "fusion"]))]
    full = summary.loc[summary["method"] == "fusion", "mean_lead_time_penalised"]
    if not len(full):
        raise RuntimeError("no fusion row in summary")
    full_value = float(full.iloc[0])

    abl = summary[summary["method"] != "fusion"].copy()
    abl["cost"] = full_value - abl["mean_lead_time_penalised"]
    abl = abl.sort_values("cost")

    labels = {
        "ablation_clinical_only": "All One Health streams\n(clinical only)",
        "ablation_no_animal": "Animal / veterinary AMU",
        "ablation_no_food": "Food chain",
        "ablation_no_environment": "Wastewater",
        "ablation_no_network": "Trade network",
    }
    colours = {"ablation_no_animal": PALETTE.animal, "ablation_no_food": PALETTE.food,
               "ablation_no_environment": PALETTE.environment,
               # Grey rather than another purple: the trade network is not a
               # surveillance sector, and it was hard to tell from wastewater.
               "ablation_no_network": PALETTE.baseline,
               "ablation_clinical_only": PALETTE.alert}

    fig, ax = plt.subplots(figsize=(7.8, 0.58 * len(abl) + 2.0))
    bars = ax.barh([labels.get(m, m) for m in abl["method"]], abl["cost"],
                   color=[colours.get(m, PALETTE.baseline) for m in abl["method"]],
                   height=0.6)
    for bar, value in zip(bars, abl["cost"]):
        ax.annotate(f"{value:+.1f} wk",
                    (value, bar.get_y() + bar.get_height() / 2),
                    xytext=(6 if value >= 0 else -6, 0), textcoords="offset points",
                    va="center", ha="left" if value >= 0 else "right", fontsize=8.5)
    ax.axvline(0, color=PALETTE.truth, lw=1.2)
    ax.margins(x=0.16)
    ax.set_xlabel("weeks of lead time lost when this stream is removed\n"
                  "(positive = the stream contributes)")
    ax.set_title(f"Sector ablations at {budget} false alarms per series-year:\n"
                 "what each One Health stream is worth", loc="left")
    fig.tight_layout()
    out = out or FIGURES / "fig3_ablation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def calibration_figure(tag: str = "full", out: Path | None = None) -> Path:
    """Reliability curve for the alerting score."""
    path = RESULTS / f"artifacts_{tag}" / "reliability.csv"
    if not path.exists():
        print(f"skipping calibration figure: {path} not found")
        return path
    rel = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    lim = max(rel["predicted"].max(), rel["observed"].max()) * 1.1 if len(rel) else 1
    ax.plot([0, lim], [0, lim], color=PALETTE.baseline, ls="--", lw=1)
    ax.plot(rel["predicted"], rel["observed"], "o-", color=PALETTE.human, ms=5)
    ax.set_xlabel("predicted probability of emergence")
    ax.set_ylabel("observed frequency")
    ax.set_title("Alert-score calibration (test window)")
    fig.tight_layout()
    out = out or FIGURES / "fig4_calibration.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def case_study_figure(out: Path | None = None) -> Path | None:
    """Real-data premise test: does the reservoir signal lead the clinical one?

    Two panels, because the honest answer needs both. Left: where the strongest
    correlation sits, over a lag window that is symmetric by construction, so
    that "the reservoir leads" is not the foregone conclusion it would be under
    a one-sided scan. Right: the head-to-head that the binomial test actually
    uses -- best correlation in the reservoir-leads direction against best
    correlation in the human-leads direction, for each series.
    """
    path = RESULTS / "real_case_study.json"
    if not path.exists():
        print("skipping case-study figure: run `python -m amrews.data.ncbi` first")
        return None
    study = json.loads(path.read_text())
    findings = pd.DataFrame(study["findings"])
    if findings.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.9))
    colours = {"animal": PALETTE.animal, "food": PALETTE.food,
               "environment": PALETTE.environment}
    lags = sorted(findings["best_lag_quarters"].unique())
    span = max(abs(min(lags)), abs(max(lags))) + 0.5
    bins = np.arange(-span, span + 1, 1)

    bottom = np.zeros(len(bins) - 1)
    for sector in ("animal", "food", "environment"):
        sub = findings[findings["reservoir"] == sector]
        if sub.empty:
            continue
        counts, _ = np.histogram(sub["best_lag_quarters"], bins=bins)
        ax1.bar(bins[:-1] + 0.5, counts, bottom=bottom, width=0.9,
                color=colours[sector], label=f"{sector} ({len(sub)})")
        bottom += counts
    ax1.axvline(0, color=PALETTE.truth, lw=1.4)
    ax1.set_xlabel("lag of strongest correlation (quarters)\n"
                   "positive = reservoir leads the clinic")
    ax1.set_ylabel("country x determinant series")
    ax1.legend(fontsize=8)
    ax1.set_title("Where the best lag sits", loc="left")

    paired = findings.dropna(subset=["max_corr_reservoir_leads", "max_corr_human_leads"])
    if not paired.empty:
        lim = [min(paired["max_corr_human_leads"].min(),
                   paired["max_corr_reservoir_leads"].min()) - 0.05,
               max(paired["max_corr_human_leads"].max(),
                   paired["max_corr_reservoir_leads"].max()) + 0.05]
        ax2.plot(lim, lim, color=PALETTE.baseline, ls="--", lw=1)
        for sector, sub in paired.groupby("reservoir"):
            ax2.scatter(sub["max_corr_human_leads"], sub["max_corr_reservoir_leads"],
                        s=38, color=colours.get(sector, PALETTE.baseline),
                        alpha=0.85, edgecolors="white", linewidths=0.8, label=sector)
        ax2.set_xlim(lim)
        ax2.set_ylim(lim)
        ax2.set_xlabel("best correlation when the clinic leads")
        ax2.set_ylabel("best correlation when the reservoir leads")
        ax2.annotate("above the line =\nreservoir leads", (lim[0] + 0.02, lim[1] - 0.02),
                     fontsize=8, va="top", color=PALETTE.baseline)
        ax2.set_title("Head-to-head by direction", loc="left")

    dt = study.get("direction_test", {})
    p_value = dt.get("binomial_p_one_sided")
    verdict = (f"reservoir-leads wins in {dt.get('reservoir_leads_wins')}/"
               f"{dt.get('n_series')} series, one-sided binomial p = {p_value:.2f}"
               if p_value is not None else "")
    fig.suptitle(f"NCBI Pathogen Detection ({study.get('organism', '')}, "
                 f"{study.get('n_isolates', 0):,} isolates): does the reservoir lead?\n"
                 f"{verdict}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out or FIGURES / "fig5_real_case_study.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def alert_timeline_figure(tag: str = "full", out: Path | None = None,
                          smooth: int = 9) -> Path | None:
    """A worked example: one test-window event, its alerts, and the ground truth.

    The event is chosen as the one the system warned about earliest, rather than
    the first in the file. An arbitrary pick lands on a miss about as often as
    on a detection, which illustrates nothing; the misses are counted properly
    in the sensitivity column of the results table, which is where they belong.
    """
    d = RESULTS / f"artifacts_{tag}"
    if not (d / "panel.csv").exists():
        print("skipping timeline figure: artifacts not found")
        return None
    panel = pd.read_csv(d / "panel.csv")
    scores = pd.read_csv(d / "scores.csv")
    alerts = pd.read_csv(d / "alerts.csv")
    balarms = pd.read_csv(d / "baseline_alarms.csv")
    meta = json.loads((d / "meta.json").read_text())

    events = [e for e in meta["events"]
              if e["is_true_event"] and e["human_emergence_week"]
              and e["human_emergence_week"] >= VALID_END_WEEK]
    if not events:
        return None

    # Prefer the event with the longest *credited* lead -- the first alert inside
    # the match window the evaluation actually scores. Taking the longest lead
    # over all earlier alerts instead would select for series with a spurious
    # early alarm, which is the opposite of a good illustration.
    best, best_score = None, -np.inf
    for e in events:
        mine = alerts[(alerts["region"] == e["region"])
                      & (alerts["determinant"] == e["determinant"])]
        window = mine[(mine["week"] >= e["human_emergence_week"] - MATCH_WINDOW_BEFORE)
                      & (mine["week"] <= e["human_emergence_week"])]
        if not len(window):
            continue
        lead = e["human_emergence_week"] - int(window["week"].min())
        # Tie-break towards a quiet series: an example cluttered with alarms
        # misrepresents how the system behaves at its calibrated budget.
        score = lead - 1.5 * len(mine)
        if score > best_score:
            best, best_score = e, score
    ev = best or events[0]

    sel = ((panel["region"] == ev["region"]) & (panel["determinant"] == ev["determinant"]))
    block = panel[sel].sort_values("week")
    score_col = "score_ensemble" if "score_ensemble" in scores.columns else "score_fusion"
    sblock = scores[(scores["region"] == ev["region"])
                    & (scores["determinant"] == ev["determinant"])].sort_values("week")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 5.8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.15]})
    roll = lambda c: block[c].rolling(smooth, min_periods=2).mean()  # noqa: E731
    ax1.plot(block["week"], roll("human_p"), color=PALETTE.human, lw=2.2,
             label="human clinical", zorder=4)
    ax1.plot(block["week"], roll("animal_p"), color=PALETTE.animal, lw=1.6, label="animal")
    ax1.plot(block["week"], roll("food_p"), color=PALETTE.food, lw=1.4, label="food")
    ax1.axvline(ev["human_emergence_week"], color=PALETTE.truth, lw=1.8)
    ax1.annotate("human emergence", (ev["human_emergence_week"], ax1.get_ylim()[1]),
                 xytext=(5, -10), textcoords="offset points", fontsize=8.5, va="top")
    ax1.set_ylabel("resistant proportion")
    ax1.legend(ncol=3, fontsize=8, loc="upper left")
    ax1.set_title(f"Worked example: region {ev['region']}, {ev['determinant']} "
                  f"({smooth}-week rolling mean)", loc="left")

    ax2.plot(sblock["week"], sblock[score_col], color=PALETTE.alert, lw=1.6,
             label="One Health risk score")
    ax2.axhline(meta["threshold"], color=PALETTE.baseline, ls="--", lw=1.1,
                label="alert threshold")
    ax2.axvline(ev["human_emergence_week"], color=PALETTE.truth, lw=1.8)

    my = alerts[(alerts["region"] == ev["region"])
                & (alerts["determinant"] == ev["determinant"])]
    for n, (_, a) in enumerate(my.iterrows()):
        ax2.axvline(a["week"], color=PALETTE.alert, lw=1.3,
                    label="system alert" if n == 0 else None)
    # Only the strongest conventional baseline is drawn; four overlaid alarm
    # rasters make the panel unreadable and say nothing extra.
    ref = balarms[(balarms["method"] == "ewma")
                  & (balarms["region"] == ev["region"])
                  & (balarms["determinant"] == ev["determinant"])]
    for n, (_, a) in enumerate(ref.iterrows()):
        ax2.axvline(a["week"], color=PALETTE.baseline, lw=1.1, ls=":",
                    label="EWMA chart alarm" if n == 0 else None)

    credited = my[(my["week"] >= ev["human_emergence_week"] - MATCH_WINDOW_BEFORE)
                  & (my["week"] <= ev["human_emergence_week"])]
    if len(credited):
        first = int(credited["week"].min())
        y = float(sblock[score_col].max()) * 1.10
        ax2.annotate("", xy=(ev["human_emergence_week"], y), xytext=(first, y),
                     arrowprops=dict(arrowstyle="<->", color=PALETTE.alert, lw=1.3))
        ax2.annotate(f"{ev['human_emergence_week'] - first} weeks of warning",
                     ((first + ev["human_emergence_week"]) / 2, y * 1.06),
                     ha="center", fontsize=8.5, color=PALETTE.alert)

    ax2.set_xlabel("week")
    ax2.set_ylabel("risk")
    # Headroom so the legend does not sit on top of the score trace.
    ax2.set_ylim(0, float(sblock[score_col].max()) * 1.75)
    ax2.legend(fontsize=8, loc="upper left", ncol=4, framealpha=0.9, frameon=True,
               edgecolor="none")
    ax2.set_xlim(max(0, ev["human_emergence_week"] - 130),
                 min(int(block["week"].max()), ev["human_emergence_week"] + 60))
    fig.tight_layout()
    out = out or FIGURES / "fig6_timeline.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out



# --------------------------------------------------------------------------- #
# Phase 2: spatial sweep, real-data evaluation, GLASS driver analysis
# --------------------------------------------------------------------------- #
def spatial_sweep_figure(tag: str = "sweep", out: Path | None = None) -> Path | None:
    """What the graph tier is worth as spatial transmission increases.

    Left: the paired graph-minus-temporal lead-time difference at each setting,
    split by whether the event was trade-seeded. Right: the learned gate, which
    is the model's own statement about how much it used the graph.

    The control at ``spatial_fraction = 0`` is the important column. A gain that
    appears only once the world contains spatial transmission is attributable to
    spatial structure; a gain present at zero would mean the architecture, not
    the mechanism, was doing the work.
    """
    comp_path = RESULTS / f"sweep_comparisons_{tag}.csv"
    diag_path = RESULTS / f"sweep_diagnostics_{tag}.csv"
    if not comp_path.exists():
        print("skipping sweep figure: run `--sweep` first")
        return None
    comp = pd.read_csv(comp_path)
    diag = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    styles = {"all": (PALETTE.truth, "o", "-"),
              "trade_seeded": (PALETTE.alert, "s", "-"),
              "independent": (PALETTE.baseline, "^", "--")}
    for label, (colour, marker, ls) in styles.items():
        sub = comp[comp["event_type"] == label].sort_values("spatial_fraction")
        if sub.empty:
            continue
        ax1.errorbar(sub["spatial_fraction"], sub["mean_delta_weeks"],
                     yerr=[sub["mean_delta_weeks"] - sub["ci_low"],
                           sub["ci_high"] - sub["mean_delta_weeks"]],
                     color=colour, marker=marker, ls=ls, capsize=3, lw=1.6,
                     label=label.replace("_", "-"))
    ax1.axhline(0, color=PALETTE.truth, lw=1.2)
    ax1.set_xlabel("share of events seeded along the trade network")
    ax1.set_ylabel("graph tier minus temporal tier\n(weeks of lead time)")
    ax1.set_title("Does the spatial model pay off?", loc="left")
    ax1.legend(fontsize=8)

    if not diag.empty and "gate" in diag:
        g = diag.groupby("spatial_fraction")["gate"]
        mean, sd = g.mean(), g.std()
        ax2.errorbar(mean.index, mean.to_numpy(), yerr=sd.to_numpy(),
                     color=PALETTE.environment, marker="o", capsize=3, lw=1.6)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("share of events seeded along the trade network")
        ax2.set_ylabel("learned gate on the graph branch")
        ax2.set_title("How much the model chose to use the graph", loc="left")
    fig.tight_layout()
    out = out or FIGURES / "fig7_spatial_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def real_evaluation_figure(tag: str = "real", out: Path | None = None) -> Path | None:
    """Lead time and sensitivity on real NCBI data, against the same baselines."""
    path = RESULTS / f"real_summary_{tag}.csv"
    if not path.exists():
        print("skipping real-evaluation figure: run `--real` first")
        return None
    summary = pd.read_csv(path)
    budget = sorted(summary["budget"].unique())[len(summary["budget"].unique()) // 2]
    sub = summary[np.isclose(summary["budget"], budget)].sort_values("mean_lead_time")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8), sharey=True)
    colours = [PALETTE.human if m == "fusion" else PALETTE.baseline for m in sub["method"]]
    labels = [_pretty(m) for m in sub["method"]]

    ax1.barh(labels, sub["mean_lead_time"], color=colours, height=0.6)
    ax1.axvline(0, color=PALETTE.truth, lw=1.2)
    ax1.set_xlabel("mean lead time when detected (quarters)")
    ax1.set_title("Warning time on real data", loc="left")

    ax2.barh(labels, sub["sensitivity"], color=colours, height=0.6)
    ax2.set_xlabel("sensitivity (share of real episodes detected)")
    ax2.set_title("How often it fires at all", loc="left")

    fig.suptitle(f"NCBI Pathogen Detection, {budget} false alarms per series-year. "
                 "The One Health model warns earlier when it fires, and fires less often.",
                 fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out or FIGURES / "fig8_real_evaluation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def glass_driver_figure(out: Path | None = None) -> Path | None:
    """WHO GLASS: does national antibiotic consumption lead national resistance?"""
    path = RESULTS / "glass_driver_study.json"
    if not path.exists():
        print("skipping GLASS figure: run `python -m amrews.data.glass` first")
        return None
    study = json.loads(path.read_text())
    findings = pd.DataFrame(study["findings"])
    if findings.empty:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    lags = findings["best_lag_years"]
    span = max(abs(lags.min()), abs(lags.max())) + 0.5
    ax1.hist(lags, bins=np.arange(-span, span + 1, 1), color=PALETTE.animal)
    ax1.axvline(0, color=PALETTE.truth, lw=1.4)
    ax1.set_xlabel("lag of strongest correlation (years)\n"
                   "positive = consumption leads resistance")
    ax1.set_ylabel("country x indicator series")
    ax1.set_title("Where the best lag sits", loc="left")

    paired = findings.dropna(subset=["max_corr_consumption_leads",
                                     "max_corr_resistance_leads"])
    if not paired.empty:
        lim = [min(paired["max_corr_resistance_leads"].min(),
                   paired["max_corr_consumption_leads"].min()) - 0.05,
               max(paired["max_corr_resistance_leads"].max(),
                   paired["max_corr_consumption_leads"].max()) + 0.05]
        ax2.plot(lim, lim, color=PALETTE.baseline, ls="--", lw=1)
        ax2.scatter(paired["max_corr_resistance_leads"],
                    paired["max_corr_consumption_leads"],
                    s=34, color=PALETTE.human, alpha=0.8,
                    edgecolors="white", linewidths=0.7)
        ax2.set_xlim(lim); ax2.set_ylim(lim)
        ax2.set_xlabel("best correlation when resistance leads")
        ax2.set_ylabel("best correlation when consumption leads")
        # Lower right: the half of the plane the points avoid, by construction
        # of the finding being reported.
        ax2.annotate("above the line =\nconsumption leads",
                     (lim[1] - 0.02, lim[0] + 0.02), fontsize=8, va="bottom",
                     ha="right", color=PALETTE.baseline)
        ax2.set_title("Head-to-head by direction", loc="left")

    dt = study["direction_test"]
    fig.suptitle(
        f"WHO GLASS, {study['n_countries']} countries, 2016-2023: does antimicrobial "
        f"use lead resistance?\nconsumption-leads wins in "
        f"{dt['consumption_leads_wins']}/{dt['n_series']} series, one-sided binomial "
        f"p = {dt['binomial_p_one_sided']:.3f}", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = out or FIGURES / "fig9_glass_driver.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out



def efsa_coupling_figure(out: Path | None = None) -> Path | None:
    """Designed-sampling evidence that the sectors are coupled across Member States."""
    path = RESULTS / "efsa_coupling_study.json"
    panel_path = DATA_PROCESSED / "efsa_panel.csv"
    if not path.exists() or not panel_path.exists():
        print("skipping EFSA figure: run `python -m amrews.data.efsa` first")
        return None
    study = json.loads(path.read_text())
    findings = pd.DataFrame(study["findings"])
    panel = pd.read_csv(panel_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0))

    # Left: the headline pair, one point per Member State.
    top = findings.sort_values("n_countries", ascending=False)
    top = top[top["spearman_rho"] > 0.5].head(1)
    if len(top):
        row = top.iloc[0]
        block = panel[(panel["serovar"] == row["serovar"])
                      & (panel["antimicrobial"] == row["antimicrobial"])]
        human = block[block["sector"] == "human"].groupby("country")["percent_resistant"].mean()
        animal = (block[(block["sector"] == "animal")
                        & (block["population"] == row["population"])]
                  .groupby("country")["percent_resistant"].mean())
        shared = human.index.intersection(animal.index)
        ax1.scatter(animal.loc[shared], human.loc[shared], s=46, color=PALETTE.human,
                    alpha=0.85, edgecolors="white", linewidths=0.8)
        for c in shared:
            ax1.annotate(c[:3], (animal[c], human[c]), fontsize=6.5,
                         xytext=(3, 3), textcoords="offset points", color=PALETTE.baseline)
        ax1.set_xlabel(f"resistance in {row['population'].lower()} (%)")
        ax1.set_ylabel("resistance in humans (%)")
        ax1.set_title(f"{row['serovar']}, {row['antimicrobial']}\n"
                      f"rho = {row['spearman_rho']:+.2f}, {row['n_countries']} countries",
                      loc="left")

    # Right: the distribution of correlations over every matched comparison.
    ax2.hist(findings["spearman_rho"], bins=np.arange(-1, 1.05, 0.1),
             color=PALETTE.animal, alpha=0.9)
    ax2.axvline(0, color=PALETTE.truth, lw=1.6)
    ax2.set_xlabel("Spearman rho, animal vs human resistance across Member States")
    ax2.set_ylabel("serovar x antimicrobial x population")
    ax2.set_title(f"{study['n_positive']}/{study['n_pairs']} positive "
                  f"(sign test p = {study['sign_test_p']:.1e})", loc="left")

    fig.suptitle("EFSA/ECDC EU Summary Report: are the One Health sectors coupled?",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = out or FIGURES / "fig10_efsa_coupling.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def us_deep_figure(tag: str = "usdeep", out: Path | None = None) -> Path | None:
    """The lead-time / sensitivity trade-off on real US state x month data."""
    path = RESULTS / f"us_deep_summary_{tag}.csv"
    head = RESULTS / f"us_deep_headline_{tag}.json"
    if not path.exists():
        print("skipping US deep figure: run `--us-deep` first")
        return None
    summary = pd.read_csv(path)
    info = json.loads(head.read_text()) if head.exists() else {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.0))
    colour = {"baseline": PALETTE.baseline, "fusion": PALETTE.human,
              "temporal": PALETTE.animal, "graph": PALETTE.alert}
    for _, r in summary.iterrows():
        c = colour.get(r["tier"], PALETTE.baseline)
        ax1.scatter(r["sensitivity"], r["mean_lead_time"], s=90, color=c,
                    edgecolors="white", linewidths=1.0, zorder=3)
        ax1.annotate(_pretty(r["method"]).replace(" (clinical only)", ""),
                     (r["sensitivity"], r["mean_lead_time"]), fontsize=7.5,
                     xytext=(6, 4), textcoords="offset points")
    ax1.axhline(0, color=PALETTE.truth, lw=1.2)
    ax1.set_xlabel("sensitivity (share of real episodes detected)")
    ax1.set_ylabel("mean lead time when detected (months)")
    ax1.set_title("The deeper the tier, the longer the warning\nand the rarer the alert",
                  loc="left")

    diag = info.get("graph_diagnostics", {})
    lags = diag.get("lag_weights", {})
    rels = diag.get("relation_weights", {})
    if lags and rels:
        names = [f"lag {k} mo" for k in lags] + list(rels)
        vals = list(lags.values()) + list(rels.values())
        cols = [PALETTE.environment] * len(lags) + [PALETTE.food] * len(rels)
        ax2.barh(names, vals, color=cols, height=0.6)
        ax2.set_xlabel("learned weight")
        ax2.set_title("What the graph tier learned:\nreal geography and multi-month lags",
                      loc="left")
    fig.suptitle(f"Real US panel: {info.get('states','?')} states, "
                 f"{info.get('months','?')} months, {info.get('events_test','?')} held-out episodes",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = out or FIGURES / "fig11_us_deep.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out



def benchmark_figure(tag: str = "benchmark_fixed", out: Path | None = None) -> Path | None:
    """The permutation control: does the real graph beat a shuffled one?

    The single most informative panel in the study. A graph tier that beats the
    temporal tier looks like a success until the same comparison is run against
    a randomly permuted adjacency; if the permuted graph does just as well, the
    spatial component is inert and the gain is capacity, not structure.
    """
    path = RESULTS / f"benchmark_summary_{tag}.csv"
    head = RESULTS / f"benchmark_{tag}.json"
    if not path.exists():
        print("skipping benchmark figure: run `python -m amrews.eval.benchmark` first")
        return None
    summary = pd.read_csv(path)
    info = json.loads(head.read_text()) if head.exists() else {}

    datasets = list(dict.fromkeys(summary["dataset"]))
    arms = ["temporal_tcn", "graph_stgnn", "graph_shuffled"]
    labels = {"temporal_tcn": "Temporal\nTCN", "graph_stgnn": "Graph\nreal adj.",
              "graph_shuffled": "Graph\nshuffled adj."}
    colours = {"temporal_tcn": PALETTE.baseline, "graph_stgnn": PALETTE.alert,
               "graph_shuffled": PALETTE.environment}

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.1 * len(datasets), 4.2),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ds in zip(axes, datasets):
        block = summary[summary["dataset"] == ds].set_index("method")
        vals = [block.loc[a, "mean_lead_time_penalised"] if a in block.index else np.nan
                for a in arms]
        ax.bar([labels[a] for a in arms], vals,
               color=[colours[a] for a in arms], width=0.62)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.annotate(f"{v:+.2f}", (i, v), ha="center",
                            va="bottom" if v >= 0 else "top",
                            xytext=(0, 4 if v >= 0 else -4),
                            textcoords="offset points", fontsize=8.5)
        n = info.get(ds, {})
        ax.set_title(f"{ds}\n{n.get('regions','?')} regions x {n.get('weeks','?')} weeks",
                     loc="left", fontsize=9.5)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.margins(y=0.15)
        ax.axhline(0, color=PALETTE.truth, lw=1.1)
    axes[0].set_ylabel("mean lead time, misses penalised (weeks)")
    fig.suptitle("Permutation control: a shuffled adjacency performs as well as the real one",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out or FIGURES / "fig12_benchmark_control.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out



def representation_figure(tag: str = "representation", out: Path | None = None) -> Path | None:
    """Separate the effect of the feature representation from the model class."""
    path = RESULTS / f"representation_{tag}.csv"
    cpath = RESULTS / f"representation_{tag}.json"
    if not path.exists():
        print("skipping representation figure: run `--representation` first")
        return None
    df = pd.read_csv(path)
    contrasts = json.loads(cpath.read_text()) if cpath.exists() else {}

    agg = (df.groupby(["model", "representation"])["mean_lead_time_penalised"]
             .mean().unstack())
    order_m = ["boosted_trees", "neural_net"]
    order_r = ["engineered", "raw"]
    agg = agg.reindex(index=order_m, columns=order_r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
    width, xs = 0.36, np.arange(len(order_m))
    for i, rep in enumerate(order_r):
        vals = agg[rep].to_numpy()
        ax1.bar(xs + (i - 0.5) * width, vals, width,
                color=PALETTE.human if rep == "engineered" else PALETTE.animal,
                label=f"{rep} features")
        for x, v in zip(xs + (i - 0.5) * width, vals):
            ax1.annotate(f"{v:+.2f}", (x, v), ha="center",
                         va="bottom" if v >= 0 else "top",
                         xytext=(0, 4 if v >= 0 else -4), textcoords="offset points",
                         fontsize=8.5)
    ax1.axhline(0, color=PALETTE.truth, lw=1.2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(["Gradient-boosted\ntrees", "Neural\nnetwork"])
    ax1.set_ylabel("mean lead time, misses penalised (weeks)")
    ax1.set_title("The gap is between model classes,\nnot between representations", loc="left")
    ax1.legend(fontsize=8.5)
    ax1.margins(y=0.2)

    labels = {"model_effect_on_engineered": "trees vs net\n(engineered)",
              "model_effect_on_raw": "trees vs net\n(raw)",
              "representation_effect_within_trees": "engineered vs raw\n(within trees)",
              "representation_effect_within_nn": "engineered vs raw\n(within net)"}
    keys = [k for k in labels if k in contrasts]
    means = [contrasts[k]["mean_delta_weeks"] for k in keys]
    los = [contrasts[k]["mean_delta_weeks"] - contrasts[k]["ci_low"] for k in keys]
    his = [contrasts[k]["ci_high"] - contrasts[k]["mean_delta_weeks"] for k in keys]
    cols = [PALETTE.alert if k.startswith("model") else PALETTE.baseline for k in keys]
    ax2.barh([labels[k] for k in keys], means, xerr=[los, his],
             color=cols, height=0.6, error_kw=dict(lw=1.2, capsize=3))
    ax2.axvline(0, color=PALETTE.truth, lw=1.2)
    ax2.set_xlabel("paired difference in lead time (weeks), 95% CI")
    ax2.set_title("Model class matters; the engineering does not", loc="left")

    fig.suptitle(f"Feature representation crossed with model class, {df['seed'].nunique()} seeds",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = out or FIGURES / "fig13_representation.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def all_figures(tag: str = "full") -> None:
    sanity_plots()
    for fn in (lead_time_figure, ablation_figure, calibration_figure, alert_timeline_figure):
        try:
            fn(tag)
        except Exception as exc:                      # a missing artefact must not
            print(f"skipped {fn.__name__}: {exc}")    # abort the whole figure run
    for fn in (case_study_figure, spatial_sweep_figure, real_evaluation_figure,
               glass_driver_figure, efsa_coupling_figure, us_deep_figure,
               benchmark_figure, representation_figure):
        try:
            fn()
        except Exception as exc:
            print(f"skipped {fn.__name__}: {exc}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Render report figures.")
    ap.add_argument("--tag", default="full")
    all_figures(ap.parse_args().tag)
