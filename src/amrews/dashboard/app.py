"""AMR-OHEWS dashboard: the operational face of the early-warning system.

Run with::

    streamlit run src/amrews/dashboard/app.py

Everything shown is read from ``results/`` -- no model is fitted and no network
call is made at render time, so the demo cannot fail in front of an audience
because a download stalled. Generate the inputs first with::

    python -m amrews.eval.experiments --full
    python -m amrews.data.ncbi

A note on the maps. The simulated regions are not real places, so the region
view is drawn as an explicit schematic rather than a map of anywhere; only the
real-data tab, which uses genuine country labels from NCBI Pathogen Detection,
draws a geographic map. Dressing synthetic regions in real geography would be
the most persuasive and least honest thing this dashboard could do.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from amrews.config import PALETTE, RESULTS, TRAIN_END_WEEK, VALID_END_WEEK  # noqa: E402

st.set_page_config(page_title="AMR One Health Early Warning", page_icon="🧫",
                   layout="wide", initial_sidebar_state="expanded")

SECTOR_COLOURS = {
    "human": PALETTE.human, "animal": PALETTE.animal,
    "food": PALETTE.food, "environment": PALETTE.environment,
    "network": "#805ad5", "context": PALETTE.baseline,
}
SEVERITY_COLOURS = {"watch": "#a0aec0", "elevated": "#d69e2e",
                    "high": "#dd6b20", "critical": PALETTE.alert}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_artifacts(tag: str) -> dict | None:
    d = RESULTS / f"artifacts_{tag}"
    if not (d / "panel.csv").exists():
        return None
    out = {name: pd.read_csv(d / f"{name}.csv")
           for name in ("panel", "scores", "baseline_scores", "baseline_alarms",
                        "alerts", "sector_shares", "reliability")
           if (d / f"{name}.csv").exists()}
    out["meta"] = json.loads((d / "meta.json").read_text())
    out["trade"] = np.load(d / "trade.npy")
    return out


@st.cache_data(show_spinner=False)
def load_summary(tag: str) -> tuple[pd.DataFrame | None, dict | None, pd.DataFrame | None]:
    s = RESULTS / f"summary_{tag}.csv"
    h = RESULTS / f"headline_{tag}.json"
    e = RESULTS / f"per_event_{tag}.csv"
    return (pd.read_csv(s) if s.exists() else None,
            json.loads(h.read_text()) if h.exists() else None,
            pd.read_csv(e) if e.exists() else None)


@st.cache_data(show_spinner=False)
def load_case_study() -> dict | None:
    p = RESULTS / "real_case_study.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_glass() -> dict | None:
    p = RESULTS / "glass_driver_study.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_real_eval(tag: str = "real") -> tuple[pd.DataFrame | None, dict | None]:
    s_path = RESULTS / f"real_summary_{tag}.csv"
    h_path = RESULTS / f"real_headline_{tag}.json"
    return (pd.read_csv(s_path) if s_path.exists() else None,
            json.loads(h_path.read_text()) if h_path.exists() else None)


@st.cache_data(show_spinner=False)
def load_benchmark(tag: str = "benchmark_fixed") -> tuple[pd.DataFrame | None, dict | None]:
    c = RESULTS / f"benchmark_summary_{tag}.csv"
    j = RESULTS / f"benchmark_{tag}.json"
    return (pd.read_csv(c) if c.exists() else None,
            json.loads(j.read_text()) if j.exists() else None)


@st.cache_data(show_spinner=False)
def load_robustness() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    u = RESULTS / "robustness_us_deep.csv"
    b = RESULTS / "robustness_benchmark.csv"
    return (pd.read_csv(u) if u.exists() else None,
            pd.read_csv(b) if b.exists() else None)


@st.cache_data(show_spinner=False)
def load_representation(tag: str = "representation") -> tuple[pd.DataFrame | None, dict | None]:
    c = RESULTS / f"representation_{tag}.csv"
    j = RESULTS / f"representation_{tag}.json"
    return (pd.read_csv(c) if c.exists() else None,
            json.loads(j.read_text()) if j.exists() else None)


@st.cache_data(show_spinner=False)
def load_efsa() -> dict | None:
    p = RESULTS / "efsa_coupling_study.json"
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data(show_spinner=False)
def load_us_deep(tag: str = "usdeep") -> tuple[pd.DataFrame | None, dict | None]:
    c = RESULTS / f"us_deep_summary_{tag}.csv"
    j = RESULTS / f"us_deep_headline_{tag}.json"
    return (pd.read_csv(c) if c.exists() else None,
            json.loads(j.read_text()) if j.exists() else None)


@st.cache_data(show_spinner=False)
def load_sweep(tag: str = "sweep") -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    c = RESULTS / f"sweep_comparisons_{tag}.csv"
    d = RESULTS / f"sweep_diagnostics_{tag}.csv"
    return (pd.read_csv(c) if c.exists() else None,
            pd.read_csv(d) if d.exists() else None)


def available_tags() -> list[str]:
    return sorted({p.name.replace("artifacts_", "")
                   for p in RESULTS.glob("artifacts_*") if p.is_dir()})


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def week_to_label(week: int) -> str:
    """Weeks are simulation time; show them as year/week for readability."""
    return f"Y{week // 52 + 1} W{week % 52 + 1}"


def styled(fig: go.Figure, height: int = 380, **kwargs) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=36, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12), legend=dict(orientation="h", y=1.12, x=0),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=PALETTE.grid, zeroline=False)
    fig.update_yaxes(gridcolor=PALETTE.grid, zeroline=False)
    return fig


def sector_ring_positions(n: int) -> tuple[np.ndarray, np.ndarray]:
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.cos(ang), np.sin(ang)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_alerts(art: dict) -> None:
    st.subheader("Active alerts")
    alerts = art["alerts"]
    if alerts.empty:
        st.info("No alerts were raised at the configured false-alarm budget.")
        return

    week_max = int(alerts["week"].max())
    window = st.slider("Show alerts raised in the last N weeks", 12, 208, 104, step=4)
    view = alerts[alerts["week"] >= week_max - window].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alerts in window", len(view))
    c2.metric("Regions affected", view["region"].nunique())
    c3.metric("Determinants", view["determinant"].nunique())
    crit = int((view.get("severity", pd.Series(dtype=str)) == "critical").sum())
    c4.metric("Critical", crit)

    cols = [c for c in ["week", "region", "determinant", "risk", "severity",
                        "lead_sector", "lead_sector_share", "expected_horizon_weeks",
                        "top_drivers"] if c in view.columns]
    table = view[cols].sort_values(["week", "risk"], ascending=[False, False]).copy()
    table["week"] = table["week"].map(week_to_label)
    st.dataframe(
        table.rename(columns={
            "week": "Raised", "region": "Region", "determinant": "Determinant",
            "risk": "Risk", "severity": "Severity", "lead_sector": "Lead sector",
            "lead_sector_share": "Sector share",
            "expected_horizon_weeks": "Horizon (wk)", "top_drivers": "Top drivers",
        }),
        width="stretch", hide_index=True,
        column_config={
            "Risk": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
            "Sector share": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )

    st.caption(
        "Risk is the calibrated probability that resistance begins to establish in "
        "human medicine within the alerting horizon. It is a prompt to investigate, "
        "not a diagnosis and not an instruction to change prescribing."
    )


def tab_map(art: dict) -> None:
    st.subheader("Regional risk (schematic)")
    st.caption("Simulated regions are laid out on the trade ring used to generate "
               "them. They are not real places and are deliberately not drawn on a map.")

    scores = art["scores"]
    score_col = "score_ensemble" if "score_ensemble" in scores.columns else "score_fusion"
    weeks = sorted(scores["week"].unique())
    week = st.select_slider("Week", options=weeks, value=weeks[-1],
                            format_func=week_to_label)
    dets = sorted(scores["determinant"].dropna().unique())
    det = st.selectbox("Determinant", dets)

    snap = scores[(scores["week"] == week) & (scores["determinant"] == det)]
    n = int(art["trade"].shape[0])
    x, y = sector_ring_positions(n)
    risk = np.zeros(n)
    for _, r in snap.iterrows():
        risk[int(r["region"])] = float(r[score_col])

    fig = go.Figure()
    adj = art["trade"]
    thresh = np.quantile(adj, 0.92)
    for i in range(n):
        for j in range(n):
            if i < j and (adj[i, j] > thresh or adj[j, i] > thresh):
                fig.add_trace(go.Scatter(
                    x=[x[i], x[j]], y=[y[i], y[j]], mode="lines",
                    line=dict(color="rgba(160,174,192,0.45)", width=1),
                    hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers+text",
        marker=dict(size=18 + 42 * risk, color=risk, colorscale="OrRd",
                    cmin=0, cmax=max(risk.max(), 0.05),
                    line=dict(color="white", width=1.5),
                    colorbar=dict(title="risk")),
        text=[str(i) for i in range(n)], textposition="middle center",
        textfont=dict(size=9, color="white"),
        hovertemplate="region %{text}<br>risk %{marker.color:.3f}<extra></extra>",
        showlegend=False))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    st.plotly_chart(styled(fig, 460, title=f"{det} — {week_to_label(week)}"),
                    width="stretch")

    st.markdown("**Risk over time, all regions**")
    pivot = (scores[scores["determinant"] == det]
             .pivot_table(index="region", columns="week", values=score_col))
    heat = go.Figure(go.Heatmap(z=pivot.to_numpy(), x=pivot.columns, y=pivot.index,
                                colorscale="OrRd", colorbar=dict(title="risk")))
    heat.add_vline(x=VALID_END_WEEK, line=dict(color=PALETTE.truth, dash="dash"))
    heat.update_yaxes(title="region")
    heat.update_xaxes(title="week (dashed line = start of test window)")
    st.plotly_chart(styled(heat, 360), width="stretch")


def tab_explorer(art: dict) -> None:
    st.subheader("Signal explorer")
    panel, scores = art["panel"], art["scores"]
    meta = art["meta"]

    events = [e for e in meta["events"] if e["is_true_event"] and e["human_emergence_week"]]
    labels = {f"region {e['region']} — {e['determinant']} (emergence {week_to_label(e['human_emergence_week'])})": e
              for e in sorted(events, key=lambda e: -e["human_emergence_week"])}
    choice = st.selectbox("Emergence event", list(labels))
    ev = labels[choice]

    sel = (panel["region"] == ev["region"]) & (panel["determinant"] == ev["determinant"])
    block = panel[sel].sort_values("week")
    sblock = scores[(scores["region"] == ev["region"])
                    & (scores["determinant"] == ev["determinant"])].sort_values("week")
    score_col = "score_ensemble" if "score_ensemble" in scores.columns else "score_fusion"

    fig = go.Figure()
    for col, name, sector in (("human_p", "human clinical", "human"),
                              ("animal_p", "animal", "animal"),
                              ("food_p", "food", "food")):
        if col in block:
            fig.add_trace(go.Scatter(x=block["week"], y=block[col], name=name,
                                     line=dict(color=SECTOR_COLOURS[sector], width=2)))
    if "env_x" in block:
        e = block["env_x"]
        scaled = (e - e.min()) / max(e.max() - e.min(), 1e-9) * block["human_p"].max()
        fig.add_trace(go.Scatter(x=block["week"], y=scaled, name="wastewater (rescaled)",
                                 line=dict(color=SECTOR_COLOURS["environment"],
                                           width=1.6, dash="dot")))
    fig.add_vline(x=ev["human_emergence_week"], line=dict(color=PALETTE.truth, width=2))
    fig.add_annotation(x=ev["human_emergence_week"], y=1.02, yref="paper",
                       text="human emergence (ground truth)", showarrow=False,
                       font=dict(size=11))
    st.plotly_chart(styled(fig, 380, title="One Health sector signals",
                           yaxis_title="resistant proportion"), width="stretch")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=sblock["week"], y=sblock[score_col], name="One Health risk",
                              line=dict(color=PALETTE.alert, width=2)))
    if "score_temporal_sd" in sblock:
        lo = (sblock[score_col] - sblock["score_temporal_sd"]).clip(lower=0)
        hi = (sblock[score_col] + sblock["score_temporal_sd"]).clip(upper=1)
        fig2.add_trace(go.Scatter(
            x=pd.concat([sblock["week"], sblock["week"][::-1]]),
            y=pd.concat([hi, lo[::-1]]), fill="toself", name="MC-dropout spread",
            fillcolor="rgba(197,48,48,0.16)", line=dict(width=0), hoverinfo="skip"))
    fig2.add_hline(y=meta["threshold"], line=dict(color=PALETTE.baseline, dash="dash"),
                   annotation_text="alert threshold")
    fig2.add_vline(x=ev["human_emergence_week"], line=dict(color=PALETTE.truth, width=2))

    my = art["alerts"]
    my = my[(my["region"] == ev["region"]) & (my["determinant"] == ev["determinant"])]
    for _, a in my.iterrows():
        fig2.add_vline(x=a["week"], line=dict(color=PALETTE.alert, width=1.2))
    bl = art["baseline_alarms"]
    bl = bl[(bl["region"] == ev["region"]) & (bl["determinant"] == ev["determinant"])]
    for _, a in bl.iterrows():
        fig2.add_vline(x=a["week"], line=dict(color=PALETTE.baseline, width=1, dash="dot"))
    st.plotly_chart(styled(fig2, 340, title="Risk score, system alerts (red) and "
                                            "conventional-chart alarms (grey dotted)",
                           yaxis_title="risk"), width="stretch")

    if len(my):
        first = int(my["week"].min())
        lead = ev["human_emergence_week"] - first
        st.metric("Lead time on this event", f"{lead:+d} weeks",
                  help="Positive means the system alerted before resistance "
                       "established in human medicine.")
    else:
        st.warning("The system raised no alert for this event inside the match window.")


def tab_explain(art: dict) -> None:
    st.subheader("Why did the system alert?")
    alerts = art["alerts"]
    shares = art.get("sector_shares")
    if alerts.empty:
        st.info("No alerts to explain.")
        return

    labels = {f"{week_to_label(int(r['week']))} — region {int(r['region'])} — {r['determinant']}"
              f" (risk {r['risk']:.2f})": i for i, r in alerts.iterrows()}
    pick = st.selectbox("Alert", list(labels))
    row = alerts.loc[labels[pick]]

    share_cols = [c for c in alerts.columns if c.startswith("share_")]
    if share_cols:
        vals = row[share_cols].astype(float).fillna(0.0)
        fig = go.Figure(go.Bar(
            x=vals.to_numpy() * 100,
            y=[c.replace("share_", "") for c in share_cols],
            orientation="h",
            marker_color=[SECTOR_COLOURS.get(c.replace("share_", ""), PALETTE.baseline)
                          for c in share_cols]))
        fig.update_xaxes(title="share of positive attribution (%)")
        st.plotly_chart(styled(fig, 300, title="Which sector raised this alert"),
                        width="stretch")

    if "top_drivers" in row and isinstance(row["top_drivers"], str) and row["top_drivers"]:
        st.markdown("**Top contributing features (SHAP)**")
        for part in row["top_drivers"].split("; "):
            st.markdown(f"- `{part}`")

    st.caption(f"Attribution method: {art['meta'].get('shap_method', 'unknown')}. "
               "Attributions explain the model's reasoning, not biological causation.")

    if shares is not None and not shares.empty:
        st.markdown("**Sector attribution over time for this series**")
        sub = shares[(shares["region"] == row["region"])
                     & (shares["determinant"] == row["determinant"])].sort_values("week")
        fig = go.Figure()
        for c in [c for c in sub.columns if c.startswith("share_")]:
            sector = c.replace("share_", "")
            fig.add_trace(go.Scatter(x=sub["week"], y=sub[c], name=sector,
                                     stackgroup="one",
                                     line=dict(width=0.5,
                                               color=SECTOR_COLOURS.get(sector, PALETTE.baseline))))
        fig.add_vline(x=row["week"], line=dict(color=PALETTE.truth, width=2))
        st.plotly_chart(styled(fig, 300, yaxis_title="attribution share"),
                        width="stretch")


def tab_evaluation(summary: pd.DataFrame | None, headline: dict | None,
                   per_event: pd.DataFrame | None, art: dict | None) -> None:
    st.subheader("Does it actually warn earlier?")
    if summary is None:
        st.info("Run `python -m amrews.eval.experiments --full` to populate this tab.")
        return

    c1, c2 = st.columns([2, 1])
    with c1:
        policy = st.radio(
            "Threshold policy", ["prospective", "matched_far"], horizontal=True,
            help="prospective: threshold calibrated on the validation window and "
                 "transferred, as a deployed system must. matched_far: threshold set "
                 "on the test window so every method realises the same false-alarm "
                 "rate -- not deployable, but it isolates detection timing from "
                 "threshold transfer.",
        )
    budgets = sorted(summary["budget"].unique()) if "budget" in summary else [None]
    with c2:
        budget = st.select_slider(
            "False alarms per series-year", options=budgets,
            value=budgets[min(1, len(budgets) - 1)],
            help="The alarm budget is a policy choice: a screening tool that "
                 "directs a sampling round tolerates more alarms than one that "
                 "triggers a prescribing change. Every method is thresholded to "
                 "the same budget so none can buy lead time by alarming more.",
        ) if budgets[0] is not None else None

    rows = summary[summary["policy"] == policy].copy()
    if budget is not None:
        rows = rows[np.isclose(rows["budget"], budget)]

    if headline:
        ref = headline.get("reference_method", {})
        # Keys are "<policy>@<budget>" since results span the budget sweep.
        ref = ref.get(f"{policy}@{budget}", ref.get(policy)) if isinstance(ref, dict) else ref
        comps = {k: v for k, v in headline.get("comparisons", {}).items()
                 if v.get("policy") == policy
                 and (budget is None or np.isclose(v.get("budget", budget), budget))}
        best = max(comps.values(), key=lambda c: c["mean_delta_weeks"], default=None)
        if best:
            c1, c2, c3 = st.columns(3)
            c1.metric(f"Best method vs {ref}", f"{best['mean_delta_weeks']:+.1f} weeks",
                      help="Paired per-event difference in lead time, misses penalised.")
            c2.metric("95% bootstrap CI",
                      f"[{best['ci_low']:+.1f}, {best['ci_high']:+.1f}]")
            c3.metric("Events compared", best["n_paired_events"])
            st.caption(f"Best method: `{best['method']}`; Wilcoxon signed-rank "
                       f"p = {best['wilcoxon']['p_value']:.3g}.")

    show = rows[["method", "tier", "sensitivity", "mean_lead_time",
                 "mean_lead_time_penalised", "false_alarms_per_series_year",
                 "n_events", "n_detected"]].sort_values(
        "mean_lead_time_penalised", ascending=False)
    st.dataframe(show.rename(columns={
        "method": "Method", "tier": "Tier", "sensitivity": "Sensitivity",
        "mean_lead_time": "Lead (detected)", "mean_lead_time_penalised": "Lead (penalised)",
        "false_alarms_per_series_year": "False alarms / series-yr",
        "n_events": "Events", "n_detected": "Detected"}),
        width="stretch", hide_index=True)

    if per_event is not None:
        ev = per_event[(per_event["policy"] == policy) & per_event["lead_time"].notna()]
        if budget is not None and "budget" in ev:
            ev = ev[np.isclose(ev["budget"], budget)]
        ev = ev[~ev["method"].str.startswith("ablation_")]
        fig = go.Figure()
        for method in ev["method"].unique():
            vals = ev.loc[ev["method"] == method, "lead_time"]
            fig.add_trace(go.Box(x=vals, name=method, orientation="h",
                                 marker_color=PALETTE.baseline if method.startswith("baseline")
                                 else PALETTE.human, boxpoints="all", jitter=0.4,
                                 pointpos=0, marker_size=4, line_width=1.2))
        fig.add_vline(x=0, line=dict(color=PALETTE.truth, width=2))
        fig.update_xaxes(title="lead time (weeks); positive = early warning")
        st.plotly_chart(styled(fig, 420, showlegend=False,
                               title="Lead time per event"), width="stretch")

    abl = rows[rows["tier"].isin(["ablation", "fusion"])]
    full = abl.loc[abl["method"] == "fusion", "mean_lead_time_penalised"]
    if not abl.empty and len(full):
        # Plotted as the cost of removing a stream, which is what an ablation
        # measures; absolute scores hide a near-zero contribution as an
        # invisible bar.
        cost = abl[abl["method"] != "fusion"].copy()
        cost["cost"] = float(full.iloc[0]) - cost["mean_lead_time_penalised"]
        cost = cost.sort_values("cost")
        names = {"ablation_clinical_only": "all One Health streams",
                 "ablation_no_animal": "animal / veterinary AMU",
                 "ablation_no_food": "food chain",
                 "ablation_no_environment": "wastewater",
                 "ablation_no_network": "trade network"}
        colours = {"ablation_no_animal": PALETTE.animal, "ablation_no_food": PALETTE.food,
                   "ablation_no_environment": PALETTE.environment,
                   "ablation_no_network": "#805ad5",
                   "ablation_clinical_only": PALETTE.alert}
        fig = go.Figure(go.Bar(
            x=cost["cost"], y=[names.get(m, m) for m in cost["method"]], orientation="h",
            marker_color=[colours.get(m, PALETTE.baseline) for m in cost["method"]],
            text=[f"{v:+.1f} wk" for v in cost["cost"]], textposition="outside"))
        fig.update_xaxes(title="weeks of lead time lost when this stream is removed")
        st.plotly_chart(styled(fig, 340, title="Sector ablations: what each stream is worth"),
                        width="stretch")

    if art is not None and "reliability" in art and not art["reliability"].empty:
        rel = art["reliability"]
        fig = go.Figure()
        lim = float(max(rel["predicted"].max(), rel["observed"].max())) * 1.1
        fig.add_trace(go.Scatter(x=[0, lim], y=[0, lim], mode="lines", name="perfect",
                                 line=dict(color=PALETTE.baseline, dash="dash")))
        fig.add_trace(go.Scatter(x=rel["predicted"], y=rel["observed"], mode="lines+markers",
                                 name="observed", line=dict(color=PALETTE.human)))
        fig.update_xaxes(title="predicted probability")
        fig.update_yaxes(title="observed frequency")
        st.plotly_chart(styled(fig, 340, title="Alert-score calibration"),
                        width="stretch")


def tab_real_data(study: dict | None) -> None:
    st.subheader("Real-data check: NCBI Pathogen Detection")
    if study is None:
        st.info("Run `python -m amrews.data.ncbi` to populate this tab.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Isolates", f"{study['n_isolates']:,}")
    c2.metric("Series analysed", study["n_analyses"])
    c3.metric("Reservoir leads", study["n_reservoir_leading"])
    c4.metric("Median best lag", f"{study['median_best_lag_quarters']:.0f} q"
              if study.get("median_best_lag_quarters") is not None else "n/a")

    counts = pd.Series(study.get("sector_counts", {})).sort_values(ascending=False)
    fig = go.Figure(go.Bar(x=counts.index, y=counts.to_numpy(),
                           marker_color=[SECTOR_COLOURS.get(s, PALETTE.baseline)
                                         for s in counts.index]))
    fig.update_yaxes(title="isolates", type="log")
    st.plotly_chart(styled(fig, 300, title=f"{study.get('organism','')} isolates by "
                                           "One Health sector"), width="stretch")

    findings = pd.DataFrame(study["findings"])
    if not findings.empty:
        fig = go.Figure()
        for sector in ("animal", "food", "environment"):
            sub = findings[findings["reservoir"] == sector]
            if sub.empty:
                continue
            fig.add_trace(go.Histogram(x=sub["best_lag_quarters"], name=sector,
                                       marker_color=SECTOR_COLOURS[sector], opacity=0.75))
        fig.update_layout(barmode="overlay")
        fig.update_xaxes(title="lag of strongest correlation with the human series (quarters)")
        st.plotly_chart(styled(fig, 340, title="Does the reservoir signal lead?"),
                        width="stretch")
        st.dataframe(findings.head(25), width="stretch", hide_index=True)

    st.caption(
        "This is an observational, differenced cross-correlation on an archive whose "
        "sampling programmes changed over time. A positive lag is consistent with the "
        "reservoir leading the clinic; it is not causal evidence, and the sequencing "
        "effort behind each sector is uneven."
    )



def tab_spatial(comparisons: pd.DataFrame | None, diagnostics: pd.DataFrame | None) -> None:
    st.subheader("Does the spatial model earn its place?")
    if comparisons is None:
        st.info("Run `python -m amrews.eval.experiments --sweep` to populate this tab.")
        return

    st.markdown(
        "The graph tier originally **underperformed** the plain temporal tier. The "
        "cause turned out to be the simulated world rather than the model: trade "
        "spillover perturbed a neighbour's animal reservoir but never seeded an "
        "emergence event there, so a neighbour's signal carried no information "
        "about this region. This sweep varies how much genuine spatial "
        "transmission the world contains. **The control at 0.0 reproduces the "
        "original world** — a gain that appears only above it is attributable to "
        "spatial structure, not to the architecture having been handed an easier "
        "problem."
    )

    fig = go.Figure()
    styles = {"all": PALETTE.truth, "trade_seeded": PALETTE.alert,
              "independent": PALETTE.baseline}
    for label, colour in styles.items():
        sub = comparisons[comparisons["event_type"] == label].sort_values("spatial_fraction")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["spatial_fraction"], y=sub["mean_delta_weeks"], name=label.replace("_", "-"),
            mode="lines+markers", line=dict(color=colour, width=2),
            error_y=dict(type="data", symmetric=False,
                         array=(sub["ci_high"] - sub["mean_delta_weeks"]).to_numpy(),
                         arrayminus=(sub["mean_delta_weeks"] - sub["ci_low"]).to_numpy())))
    fig.add_hline(y=0, line=dict(color=PALETTE.truth, width=2))
    fig.update_xaxes(title="share of events seeded along the trade network")
    fig.update_yaxes(title="graph tier minus temporal tier (weeks)")
    st.plotly_chart(styled(fig, 380, title="Lead-time gain from the graph tier"),
                    width="stretch")

    if diagnostics is not None and "gate" in diagnostics:
        c1, c2 = st.columns(2)
        g = diagnostics.groupby("spatial_fraction")["gate"].agg(["mean", "std"]).reset_index()
        f2 = go.Figure(go.Scatter(
            x=g["spatial_fraction"], y=g["mean"], mode="lines+markers",
            line=dict(color=PALETTE.environment, width=2),
            error_y=dict(type="data", array=g["std"].fillna(0).to_numpy())))
        f2.update_yaxes(title="learned gate", range=[0, 1])
        f2.update_xaxes(title="spatial transmission")
        c1.plotly_chart(styled(f2, 320, title="How much the model used the graph"),
                        width="stretch")

        lag_cols = [c for c in diagnostics.columns if c.startswith("lag_")]
        if lag_cols:
            m = diagnostics.groupby("spatial_fraction")[lag_cols].mean().reset_index()
            f3 = go.Figure()
            for c in lag_cols:
                f3.add_trace(go.Scatter(x=m["spatial_fraction"], y=m[c],
                                        name=f"{c.replace('lag_', '')} wk lag",
                                        stackgroup="one", line=dict(width=0.5)))
            f3.update_yaxes(title="weight on neighbour lag")
            f3.update_xaxes(title="spatial transmission")
            c2.plotly_chart(styled(f3, 320, title="Which neighbour lag it relied on"),
                            width="stretch")

        st.dataframe(diagnostics.groupby("spatial_fraction").mean(numeric_only=True)
                     .round(3).reset_index(), width="stretch", hide_index=True)


def tab_real_eval(summary: pd.DataFrame | None, headline: dict | None,
                  glass: dict | None) -> None:
    st.subheader("Does it hold up on real data?")

    if summary is None:
        st.info("Run `python -m amrews.eval.experiments --real` to populate this tab.")
    else:
        h = headline or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Real series", h.get("n_series", "n/a"))
        c2.metric("Countries", h.get("n_countries", "n/a"))
        c3.metric("Episodes", h.get("n_events", "n/a"))
        c4.metric("In test window", h.get("n_events_test", "n/a"))

        budgets = sorted(summary["budget"].unique())
        budget = st.select_slider("False alarms per series-year", options=budgets,
                                  value=budgets[min(1, len(budgets) - 1)])
        sub = summary[np.isclose(summary["budget"], budget)].sort_values("mean_lead_time")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sub["mean_lead_time"], y=sub["method"], orientation="h",
            marker_color=[PALETTE.human if m == "fusion" else PALETTE.baseline
                          for m in sub["method"]],
            text=[f"{v:+.2f}" for v in sub["mean_lead_time"]], textposition="outside"))
        fig.add_vline(x=0, line=dict(color=PALETTE.truth, width=2))
        fig.update_xaxes(title="mean lead time when detected (quarters)")
        st.plotly_chart(styled(fig, 340, title="Warning time on real NCBI data"),
                        width="stretch")

        st.dataframe(sub[["method", "sensitivity", "mean_lead_time",
                          "mean_lead_time_penalised", "false_alarms_per_series_year",
                          "n_detected", "n_events"]].round(3),
                     width="stretch", hide_index=True)

        st.warning(h.get("label_caveat", ""), icon="⚠️")
        st.markdown(
            "**How to read this.** The One Health model warns *earlier when it fires* "
            "but fires far less often, so on miss-penalised lead time it loses to the "
            "clinical charts. The simulated advantage does not replicate here. Part of "
            "that is the label above; part is that quarterly national data is a much "
            "coarser instrument than the weekly regional panel the simulation assumes."
        )

    st.divider()
    efsa = load_efsa()
    if efsa and efsa.get("n_pairs"):
        st.markdown("### Are the sectors coupled? (EFSA/ECDC, designed sampling)")
        dt_c1, dt_c2, dt_c3 = st.columns(3)
        dt_c1.metric("Positive couplings",
                     f"{efsa['n_positive']}/{efsa['n_pairs']}")
        dt_c2.metric("Sign test p", f"{efsa['sign_test_p']:.1e}")
        dt_c3.metric("Median rho", f"{efsa['median_rho']:+.2f}")
        findings = pd.DataFrame(efsa["findings"])
        fig = go.Figure(go.Histogram(x=findings["spearman_rho"], nbinsx=20,
                                     marker_color=PALETTE.animal))
        fig.add_vline(x=0, line=dict(color=PALETTE.truth, width=2))
        fig.update_xaxes(title="Spearman rho, animal vs human resistance across Member States")
        st.plotly_chart(styled(fig, 300,
                               title="EU harmonised monitoring: the sectors are coupled"),
                        width="stretch")
        st.dataframe(findings.head(12).round(3), width="stretch", hide_index=True)
        st.caption(efsa.get("resolution_caveat", ""))
        st.divider()

    st.markdown("### Is the underlying premise real?")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Reservoir sectors leading the clinic** (NCBI, quarterly)")
        study = load_case_study()
        if study:
            dt = study.get("direction_test", {})
            st.metric("Reservoir leads", f"{dt.get('reservoir_leads_wins')}/{dt.get('n_series')}",
                      help="Symmetric lag scan; a coin flip under the null.")
            st.caption(f"one-sided binomial p = {dt.get('binomial_p_one_sided'):.3f} "
                       f"across {study.get('n_isolates', 0):,} isolates — **not** "
                       "significant. At quarterly resolution this archive cannot "
                       "resolve sector lead-lag either way.")
    with cols[1]:
        st.markdown("**Antimicrobial use leading resistance** (WHO GLASS, annual)")
        if glass:
            dt = glass.get("direction_test", {})
            st.metric("Consumption leads",
                      f"{dt.get('consumption_leads_wins')}/{dt.get('n_series')}")
            st.caption(f"one-sided binomial p = {dt.get('binomial_p_one_sided'):.3f} "
                       f"across {glass.get('n_countries')} countries — **significant**. "
                       "The driver relationship is real at national scale, even though "
                       "the sector lead-lag is not resolvable above.")
            st.caption(glass.get("resolution_caveat", ""))



def tab_what_works() -> None:
    """The controlled comparisons: what actually drives performance, and what does not."""
    st.subheader("What actually drives performance?")
    st.markdown(
        "Three controlled experiments, each isolating one factor that the headline "
        "comparison changes together with others. Each overturned something we had "
        "previously asserted."
    )

    rep_df, rep_c = load_representation()
    st.markdown("### 1. Is it the model or the engineered features?")
    if rep_df is None:
        st.info("Run `python -m amrews.eval.experiments --representation` to populate this.")
    else:
        agg = (rep_df.groupby(["model", "representation"])["mean_lead_time_penalised"]
               .mean().unstack().reindex(index=["boosted_trees", "neural_net"],
                                         columns=["engineered", "raw"]))
        fig = go.Figure()
        for rep, colour in (("engineered", PALETTE.human), ("raw", PALETTE.animal)):
            fig.add_trace(go.Bar(x=list(agg.index), y=agg[rep], name=f"{rep} features",
                                 marker_color=colour,
                                 text=[f"{v:+.2f}" for v in agg[rep]],
                                 textposition="outside"))
        fig.add_hline(y=0, line=dict(color=PALETTE.truth, width=2))
        fig.update_yaxes(title="mean lead time, misses penalised (weeks)")
        st.plotly_chart(styled(fig, 340, barmode="group",
                               title="The gap is between model classes, not representations"),
                        width="stretch")
        if rep_c:
            rows = [{"contrast": k.replace("_", " "), "weeks": v["mean_delta_weeks"],
                     "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                     "significant": v["significant"]} for k, v in rep_c.items()]
            st.dataframe(pd.DataFrame(rows).round(2), width="stretch", hide_index=True)
        st.success(
            "**It is the model class.** Boosted trees beat the neural network by 4-6 weeks "
            "in both representations. The feature engineering buys the winning model "
            "+0.00 weeks, and actively hurts the neural network. It earns its place for "
            "interpretability, not accuracy.", icon="✅")

    st.divider()
    bench_df, bench_info = load_benchmark()
    st.markdown("### 2. Does the spatial component do anything?")
    if bench_df is None:
        st.info("Run `python -m amrews.eval.benchmark` to populate this.")
    else:
        arms = ["temporal_tcn", "graph_stgnn", "graph_shuffled"]
        labels = {"temporal_tcn": "Temporal TCN", "graph_stgnn": "Graph (real adj.)",
                  "graph_shuffled": "Graph (shuffled adj.)"}
        colours = {"temporal_tcn": PALETTE.baseline, "graph_stgnn": PALETTE.alert,
                   "graph_shuffled": PALETTE.environment}
        fig = go.Figure()
        for arm in arms:
            sub = bench_df[bench_df["method"] == arm]
            fig.add_trace(go.Bar(x=sub["dataset"], y=sub["mean_lead_time_penalised"],
                                 name=labels[arm], marker_color=colours[arm],
                                 text=[f"{v:+.2f}" for v in sub["mean_lead_time_penalised"]],
                                 textposition="outside"))
        fig.update_yaxes(title="mean lead time, misses penalised (weeks)")
        st.plotly_chart(styled(fig, 360, barmode="group",
                               title="Permutation control on three epidemic benchmarks"),
                        width="stretch")
        st.error(
            "**The spatial component is inert.** A randomly shuffled adjacency matches or "
            "beats the real one on all three benchmarks, including the one where the graph "
            "tier appears to win. Its learned weights look sensible and do not change the "
            "predictions.", icon="⚠️")

    st.divider()
    us_rob, bench_rob = load_robustness()
    st.markdown("### 3. How much depends on where we cut the data?")
    if us_rob is None:
        st.info("Run `python scripts/robustness.py` to populate this.")
    else:
        grid = us_rob.pivot_table(index="split", columns="seed", values="graph_vs_temporal")
        fig = go.Figure(go.Heatmap(z=grid.to_numpy(), x=[f"seed {c}" for c in grid.columns],
                                   y=grid.index, colorscale="RdBu", zmid=0,
                                   text=grid.round(2).to_numpy(), texttemplate="%{text}",
                                   colorbar=dict(title="months")))
        st.plotly_chart(styled(fig, 300,
                               title="Graph minus temporal on the real US panel, by split and seed"),
                        width="stretch")
        g = us_rob["graph_vs_temporal"].dropna()
        c1, c2, c3 = st.columns(3)
        c1.metric("Range across configurations", f"{g.min():+.1f} to {g.max():+.1f} mo")
        c2.metric("Favouring the graph tier", f"{int((g > 0).sum())}/{len(g)}")
        c3.metric("Significantly so", f"{int(us_rob['graph_sig'].sum())}/{len(us_rob)}")
        if bench_rob is not None:
            d = bench_rob["graph_vs_shuffled"]
            st.caption(
                f"By contrast the permutation control stays within {d.abs().max():.2f} weeks "
                f"of zero across {len(bench_rob)} seeds (mean {d.mean():+.2f}). Split choice "
                "can move *which tier is better*; it cannot move *whether the graph does "
                "anything*, because that contrast holds the split fixed and varies only "
                "the adjacency."
            )
        st.warning(
            "**The real US panel is too small to rank tiers reliably.** A split could be "
            "chosen that makes the spatial model look good; the full grid is shown so that "
            "the temptation is visible.", icon="⚠️")


def tab_governance() -> None:
    st.subheader("Governance, uncertainty and limitations")
    st.markdown(
        """
**What this system is.** A decision-support prototype that ranks
region-determinant pairs by the probability that resistance is about to
establish in human medicine. It is intended to direct scarce investigative
capacity -- a targeted sampling round, a stewardship audit -- not to change
treatment for any individual patient.

**What it is not.** It is not a diagnostic, it does not see patient records, and
it cannot confirm an outbreak. Every alert requires human epidemiological review
before any action.

**Known limitations, stated up front.**

- *Surveillance capacity is confounded with risk.* Signal strength tracks how
  much sequencing and sampling a region does. A region that tests little will
  generate few alerts, and that is a property of the data, not of its safety.
  The simulator reproduces this deliberately by giving regions uneven laboratory
  capacity.
- *Thresholds transfer imperfectly.* Alarm thresholds are calibrated on a past
  window; when score distributions drift, the realised false-alarm rate moves
  away from its budget. Both policies are reported rather than only the
  flattering one.
- *Lead time is measured against a simulated ground truth.* Real archives do not
  record when resistance began to establish. The real-data tab tests the premise
  the simulation encodes, but it cannot validate the lead-time figure itself.
- *Attribution is model explanation, not causation.* A high environmental share
  means the model leaned on wastewater features, not that the wastewater caused
  the clinical rise.

**Consequences of acting on an alert.** A false alarm is not free: it can
trigger unnecessary restriction of a veterinary antimicrobial, stigmatise a farm,
a region or an exporting country, and carry real trade and livelihood costs. The
false-alarm budget is therefore a policy choice, visible and adjustable, rather
than a hidden hyperparameter.
        """
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🧫 AMR One Health Early Warning System")
    st.caption("Integrating human, animal, food and environmental surveillance to "
               "detect emerging antimicrobial resistance before conventional "
               "clinical surveillance does.")

    tags = available_tags()
    with st.sidebar:
        st.header("Data")
        if not tags:
            st.error("No results found. Run `python -m amrews.eval.experiments --full`.")
            tag = "full"
        else:
            tag = st.selectbox("Result set", tags, index=len(tags) - 1)
        st.divider()
        st.markdown(
            f"**Protocol**\n\n"
            f"- train: weeks 0–{TRAIN_END_WEEK - 1}\n"
            f"- validate: {TRAIN_END_WEEK}–{VALID_END_WEEK - 1}\n"
            f"- test: {VALID_END_WEEK}+\n\n"
            "Models are frozen after training; the test window is scored once."
        )

    art = load_artifacts(tag)
    summary, headline, per_event = load_summary(tag)
    study = load_case_study()

    names = ["Alerts", "Regional risk", "Signal explorer", "Explainability",
             "Evaluation", "What works", "Spatial model", "Real data", "Governance"]
    tabs = st.tabs(names)
    with tabs[0]:
        tab_alerts(art) if art else st.info("No artifacts for this result set.")
    with tabs[1]:
        tab_map(art) if art else st.info("No artifacts for this result set.")
    with tabs[2]:
        tab_explorer(art) if art else st.info("No artifacts for this result set.")
    with tabs[3]:
        tab_explain(art) if art else st.info("No artifacts for this result set.")
    with tabs[4]:
        tab_evaluation(summary, headline, per_event, art)
    with tabs[5]:
        tab_what_works()
    with tabs[6]:
        tab_spatial(*load_sweep())
    with tabs[7]:
        real_summary, real_headline = load_real_eval()
        tab_real_eval(real_summary, real_headline, load_glass())
        st.divider()
        tab_real_data(study)
    with tabs[8]:
        tab_governance()


main()
