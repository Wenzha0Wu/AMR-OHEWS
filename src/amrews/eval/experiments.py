"""Experiment driver: run every tier over many seeds and export the results.

One seed = one synthetic world with its own emergence events. Because a single
world can be lucky, every headline number in the report is a paired comparison
aggregated over seeds, with a bootstrap interval attached.

Protocol per seed, in strict temporal order::

    weeks   0-311   train     model fitting only
    weeks 312-415   validate  probability calibration, stacking, alarm thresholds
    weeks 416-519   test      the only window results are reported from

Models are fitted once on the training window and then frozen, so scoring the
full archive afterwards leaks nothing: an alarm in week 400 is produced by a
model that never saw week 312 onwards.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict

import numpy as np
import pandas as pd

from ..config import (
    DATA_PROCESSED,
    FALSE_ALARM_BUDGET_PER_SERIES_YEAR,
    FALSE_ALARM_BUDGET_SWEEP,
    N_SEEDS_SWEEP,
    SPATIAL_FRACTION_SWEEP,
    N_SEEDS_FULL,
    N_SEEDS_QUICK,
    PRIMARY_HORIZON,
    RESULTS,
    TRAIN_END_WEEK,
    VALID_END_WEEK,
    SimConfig,
)
from ..data.features import FEATURE_GROUPS, build_features, feature_columns
from ..data.simulate import simulate
from ..models.baselines import BASELINE_NAMES, run_baselines
from ..models.ensemble import AlertEnsemble, generate_alerts
from ..models.fusion import FusionModel, sector_attribution
from ..models.graph import GraphModel
from ..models.temporal import TemporalModel
from .metrics import (
    alarms_from_scores,
    bootstrap_ci,
    calibrate_threshold,
    evaluate_scores,
    paired_lead_time_delta,
    probabilistic_metrics,
    reliability_curve,
    wilcoxon_delta,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#: Sector ablations. Each drops one feature group from the Tier-1 model to
#: measure what that surveillance stream is actually worth.
ABLATIONS = {
    "clinical_only": ("animal", "food", "environment", "network"),
    "no_environment": ("environment",),
    "no_animal": ("animal",),
    "no_food": ("food",),
    "no_network": ("network",),
}


def _split(features: pd.DataFrame):
    train = features[features["week"] < TRAIN_END_WEEK]
    valid = features[(features["week"] >= TRAIN_END_WEEK) & (features["week"] < VALID_END_WEEK)]
    test = features[features["week"] >= VALID_END_WEEK]
    return train, valid, test


def _record(results, per_event, scores, col, events, name, tier, budget, seed,
            sweep: tuple[float, ...] = FALSE_ALARM_BUDGET_SWEEP):
    """Evaluate one method across the budget sweep and both threshold policies.

    The full grid would be wasteful: the matched-false-alarm policy exists to
    isolate detection timing from threshold transfer, and one budget is enough
    for that. So the sweep runs under the prospective policy, and the matched
    policy is evaluated only at the primary budget.
    """
    jobs = [(b, False) for b in sweep] + [(budget, True)]
    for b, matched in jobs:
        res = _eval_method(scores, col, events, name, b, matched=matched)
        policy = "matched_far" if matched else "prospective"
        results.append({**res.summary(), "seed": seed, "tier": tier,
                        "policy": policy, "budget": b})
        # ``seed`` must travel with every event row: the same (region,
        # determinant, emergence_week) triple recurs across seeds, and without
        # it the paired join would cross-match events from different worlds.
        per_event.append(res.per_event.assign(policy=policy, budget=b, seed=seed))


def _assert_no_leakage(features: pd.DataFrame) -> None:
    """Guard the two mistakes that would silently invalidate every result."""
    # 1. Labels must look forward only.
    bad = features[(features[f"label_h{PRIMARY_HORIZON}"] == 1)
                   & (features["emergence_week"] <= features["week"])]
    assert bad.empty, "label marks an emergence that has already happened"
    # 2. No latent ground-truth column may appear among the model inputs.
    leaked = set(feature_columns()) & {"latent_human", "latent_animal", "emergence_week"}
    assert not leaked, f"ground truth leaked into features: {leaked}"


def _eval_method(scores, col, events, name, budget, matched: bool = False):
    """Score one method over the test window at a false-alarm-budgeted threshold.

    Two threshold policies are reported, and the difference between them is
    itself a result.

    ``matched=False`` (primary, prospective): the threshold is calibrated on the
    validation window and transferred unchanged to the test window. This is what
    a deployed system must do, and it is honest about the cost -- score
    distributions drift, so the realised test false-alarm rate can miss its
    budget.

    ``matched=True`` (secondary, oracle): the threshold is chosen on the test
    window itself so that every method realises the same false-alarm rate. It
    cannot be deployed, but it isolates the question "which method detects
    earlier?" from the separate question "whose threshold transfers better?",
    which would otherwise be confounded in a single number.
    """
    n_weeks = int(scores["week"].max()) + 1
    if matched:
        thr = calibrate_threshold(scores, col, events, VALID_END_WEEK, n_weeks, budget)
    else:
        thr = calibrate_threshold(scores, col, events, TRAIN_END_WEEK, VALID_END_WEEK, budget)
    return evaluate_scores(scores, col, events, name, thr, VALID_END_WEEK, n_weeks)


# --------------------------------------------------------------------------- #
# One seed
# --------------------------------------------------------------------------- #
def run_seed(
    seed: int,
    horizon: int = PRIMARY_HORIZON,
    with_deep: bool = True,
    budget: float = FALSE_ALARM_BUDGET_PER_SERIES_YEAR,
    keep_artifacts: bool = False,
    epochs: int = 25,
    spatial_fraction: float | None = None,
    deep_only: bool = False,
) -> dict:
    t0 = time.time()
    overrides = {} if spatial_fraction is None else {"spatial_fraction": spatial_fraction}
    cfg = SimConfig(seed=seed, **overrides)
    panel, events, trade = simulate(seed, cfg)
    features = build_features(panel, events, trade)
    _assert_no_leakage(features)

    train, valid, test = _split(features)
    results: list[dict] = []
    per_event: list[pd.DataFrame] = []
    prob_metrics: dict[str, dict] = {}

    # ---- Tier 0: conventional clinical-only baselines --------------------- #
    base_scores = run_baselines(features)
    for name in BASELINE_NAMES:
        _record(results, per_event, base_scores, f"score_{name}", events,
                f"baseline_{name}", "baseline", budget, seed)

    # ---- Tier 1: cross-sector fusion -------------------------------------- #
    cols = feature_columns()
    fusion = FusionModel(horizon=horizon, random_state=seed).fit(train, cols, valid=valid)
    fusion_scores = fusion.score_frame(features)
    _record(results, per_event, fusion_scores, "score_fusion", events, "fusion",
            "fusion", budget, seed)
    prob_metrics["fusion"] = probabilistic_metrics(
        test.loc[~test["post_event"], f"label_h{horizon}"].to_numpy(),
        fusion.predict_proba(test[~test["post_event"]]),
    )

    # ---- Sector ablations (Tier 1 feature groups) ------------------------- #
    for ab_name, drop in ({} if deep_only else ABLATIONS).items():
        ab_cols = feature_columns(drop_sectors=drop)
        ab = FusionModel(horizon=horizon, random_state=seed).fit(train, ab_cols, valid=valid)
        sc = features[["week", "region", "determinant"]].copy()
        sc[f"score_{ab_name}"] = ab.predict_proba(features, ab_cols)
        _record(results, per_event, sc, f"score_{ab_name}", events,
                f"ablation_{ab_name}", "ablation", budget, seed)

    all_scores = features[["week", "region", "determinant", f"label_h{horizon}", "post_event"]].merge(
        fusion_scores, on=["week", "region", "determinant"], how="left"
    )

    # ---- Tiers 2 and 3: deep temporal and spatio-temporal graph ----------- #
    temporal = graph = None
    graph_diag: dict = {}
    if with_deep:
        temporal = TemporalModel(
            horizon=horizon, n_regions=cfg.n_regions, n_dets=len(cfg.determinants),
            epochs=epochs, seed=seed,
        ).fit(train, valid=valid)
        # Match the graph tier's optimizer-step budget to the temporal tier's.
        # Their batches differ in kind -- the graph consumes whole region-stacks
        # -- so equal epochs mean very unequal training, and the first version of
        # this comparison gave the graph tier 610 steps against 1,525.
        temporal_steps = temporal.history[-1]["steps"]
        tsc = temporal.predict_frame(features, with_uncertainty=True)
        _record(results, per_event, tsc, "score_temporal", events, "temporal_tcn",
                "temporal", budget, seed)
        all_scores = all_scores.merge(tsc, on=["week", "region", "determinant"], how="left")

        graph = GraphModel(
            trade, horizon=horizon, n_dets=len(cfg.determinants), epochs=epochs, seed=seed,
            target_steps=temporal_steps,
        ).fit(train, valid=valid)
        graph_diag = graph.diagnostics()
        assert abs(graph.steps_run - temporal_steps) <= max(graph.steps_run * 0.1, 30), (
            f"training budgets not matched: temporal {temporal_steps}, graph {graph.steps_run}"
        )
        gsc = graph.predict_frame(features)
        _record(results, per_event, gsc, "score_graph", events, "graph_stgnn",
                "graph", budget, seed)
        all_scores = all_scores.merge(gsc, on=["week", "region", "determinant"], how="left")

    # ---- Ensemble ---------------------------------------------------------- #
    members = [c for c in ("score_fusion", "score_temporal", "score_graph") if c in all_scores]
    ens_summary = {}
    if len(members) > 1:
        valid_scores = all_scores[(all_scores["week"] >= TRAIN_END_WEEK)
                                  & (all_scores["week"] < VALID_END_WEEK)]
        ens = AlertEnsemble(horizon=horizon).fit(valid_scores, members)
        ens_frame = ens.score_frame(all_scores)
        _record(results, per_event, ens_frame, "score_ensemble", events, "ensemble",
                "ensemble", budget, seed)
        ens_summary = {"mode": ens.mode, "weights": ens.weights_}
        all_scores = all_scores.merge(ens_frame, on=["week", "region", "determinant"], how="left")
        prob_metrics["ensemble"] = probabilistic_metrics(
            all_scores.loc[(all_scores["week"] >= VALID_END_WEEK) & (~all_scores["post_event"]),
                           f"label_h{horizon}"].to_numpy(),
            all_scores.loc[(all_scores["week"] >= VALID_END_WEEK) & (~all_scores["post_event"]),
                           "score_ensemble"].to_numpy(),
        )

    out = {
        "seed": seed,
        "runtime_s": round(time.time() - t0, 1),
        "results": results,
        "per_event": pd.concat(per_event, ignore_index=True),
        "prob_metrics": prob_metrics,
        "ensemble": ens_summary,
        "graph_diagnostics": graph_diag,
        "spatial_fraction": cfg.spatial_fraction,
        "n_seeded_events": int(sum(1 for e in events if e.seeded_by_trade and e.is_true_event)),
        "n_events_test": int(sum(
            1 for e in events
            if e.is_true_event and e.human_emergence_week is not None
            and e.human_emergence_week >= VALID_END_WEEK
        )),
    }

    if keep_artifacts:
        out["artifacts"] = _build_artifacts(
            seed, cfg, features, events, trade, all_scores, base_scores, fusion,
            temporal, horizon, budget,
        )
    return out


# --------------------------------------------------------------------------- #
# Dashboard artifacts (one representative seed)
# --------------------------------------------------------------------------- #
def _build_artifacts(seed, cfg, features, events, trade, all_scores, base_scores,
                     fusion, temporal, horizon, budget) -> dict:
    """Export everything the Streamlit dashboard renders, so it needs no models."""
    score_col = "score_ensemble" if "score_ensemble" in all_scores else "score_fusion"
    n_weeks = int(features["week"].max()) + 1
    thr = calibrate_threshold(all_scores, score_col, events, TRAIN_END_WEEK, VALID_END_WEEK, budget)
    alarms = alarms_from_scores(all_scores, score_col, thr, start_week=TRAIN_END_WEEK,
                                end_week=n_weeks)

    shap_df = fusion.explain(features[features["week"] >= TRAIN_END_WEEK])
    shares = sector_attribution(shap_df, FEATURE_GROUPS)

    merged = all_scores.copy()
    if temporal is not None:
        tsd = temporal.predict_frame(features, with_uncertainty=True)
        if "score_temporal_sd" in tsd:
            merged = merged.merge(tsd[["week", "region", "determinant", "score_temporal_sd"]],
                                  on=["week", "region", "determinant"], how="left")

    alerts = generate_alerts(
        alarms, merged, score_col, sector_shares=shares, shap_df=shap_df,
        uncertainty_col="score_temporal_sd" if "score_temporal_sd" in merged else None,
        horizon=horizon,
    )

    baseline_thr = {
        n: calibrate_threshold(base_scores, f"score_{n}", events, TRAIN_END_WEEK,
                               VALID_END_WEEK, budget)
        for n in BASELINE_NAMES
    }
    baseline_alarms = pd.concat([
        alarms_from_scores(base_scores, f"score_{n}", baseline_thr[n],
                           start_week=TRAIN_END_WEEK, end_week=n_weeks).assign(method=n)
        for n in BASELINE_NAMES
    ], ignore_index=True)

    keep_cols = [
        "week", "region", "determinant", "human_p", "animal_p", "food_p", "env_x",
        "amu_vet", "amu_human", "human_n", "human_resistant", "lab_capacity",
        "latent_human", "latent_animal", f"label_h{horizon}", "post_event",
        "emergence_week", "is_decoy_series",
    ]
    reliability = reliability_curve(
        all_scores.loc[(all_scores["week"] >= VALID_END_WEEK) & (~all_scores["post_event"]),
                       f"label_h{horizon}"].to_numpy().astype(int),
        all_scores.loc[(all_scores["week"] >= VALID_END_WEEK) & (~all_scores["post_event"]),
                       score_col].to_numpy(dtype=float),
    )
    return {
        "panel": features[[c for c in keep_cols if c in features]],
        "scores": merged,
        "baseline_scores": base_scores,
        "baseline_alarms": baseline_alarms,
        "alerts": alerts,
        "sector_shares": shares,
        "events": [asdict(e) for e in events],
        "trade": trade,
        "threshold": thr,
        "baseline_thresholds": baseline_thr,
        "shap_method": shap_df.attrs.get("shap_method", "unknown"),
        "reliability": reliability,
    }


# --------------------------------------------------------------------------- #
# Real-data evaluation (NCBI Pathogen Detection)
# --------------------------------------------------------------------------- #
#: Quarterly analogues of the week-based evaluation constants.
REAL_HORIZON = 4          # predict emergence within the next year
REAL_MATCH_BEFORE = 4     # an alarm up to a year early is credited
REAL_MATCH_AFTER = 2
REAL_REFRACTORY = 3
REAL_PERIODS_PER_YEAR = 4
REAL_TRAIN_END_YEAR = 2018
REAL_VALID_END_YEAR = 2021


def run_real(budgets: tuple[float, ...] = FALSE_ALARM_BUDGET_SWEEP,
             horizon: int = REAL_HORIZON) -> dict:
    """Run the early-warning comparison on real NCBI Pathogen Detection data.

    Same protocol as the simulated evaluation -- clinical-only baselines against
    a One Health fusion model, matched false-alarm budgets, calendar-time split,
    paired per-event bootstrap -- on a panel of real country-determinant series.

    Two tiers only. The deep temporal and graph models need far more
    series-periods than 221 quarterly series provide, and there is no trade
    matrix between countries for the graph to propagate over. Running them here
    would produce numbers, not evidence.

    The emergence label is constructed from the observed human series, which is
    what the clinical baselines are built to detect; see the module docstring of
    :mod:`amrews.data.real_panel`. The comparison is therefore biased *towards*
    the baselines, and any One Health advantage is a conservative estimate.
    """
    from ..data.real_panel import load_real_panel

    features, events, _ = load_real_panel(horizon=horizon)
    year = features["week"] // 4
    train = features[year <= REAL_TRAIN_END_YEAR]
    valid = features[(year > REAL_TRAIN_END_YEAR) & (year <= REAL_VALID_END_YEAR)]

    valid_start = int(features.loc[year > REAL_TRAIN_END_YEAR, "week"].min())
    test_start = int(features.loc[year > REAL_VALID_END_YEAR, "week"].min())
    end = int(features["week"].max()) + 1

    eval_kw = dict(match_before=REAL_MATCH_BEFORE, match_after=REAL_MATCH_AFTER,
                   refractory=REAL_REFRACTORY, periods_per_year=REAL_PERIODS_PER_YEAR)

    cols = feature_columns()
    fusion = FusionModel(horizon=horizon, random_state=0).fit(train, cols, valid=valid)
    fusion_scores = features[["week", "region", "determinant"]].copy()
    fusion_scores["score_fusion"] = fusion.predict_proba(features, cols)

    # Three years of baseline: a single year is four quarterly observations,
    # too few for a control chart to have any idea what normal looks like.
    base_scores = run_baselines(features, periods_per_year=REAL_PERIODS_PER_YEAR,
                                baseline_years=3)

    rows, per_event = [], []
    for budget in budgets:
        for name, frame, col, tier in (
            *[(f"baseline_{n}", base_scores, f"score_{n}", "baseline") for n in BASELINE_NAMES],
            ("fusion", fusion_scores, "score_fusion", "fusion"),
        ):
            thr = calibrate_threshold(frame, col, events, valid_start, test_start,
                                      budget, **eval_kw)
            res = evaluate_scores(frame, col, events, name, thr, test_start, end, **eval_kw)
            rows.append({**res.summary(), "budget": budget, "tier": tier,
                         "policy": "prospective"})
            per_event.append(res.per_event.assign(budget=budget, policy="prospective",
                                                  seed=0))

    summary = pd.DataFrame(rows)
    ev = pd.concat(per_event, ignore_index=True)

    comparisons = {}
    for budget in budgets:
        sub = summary[(summary["tier"] == "baseline") & (summary["budget"] == budget)]
        if sub.empty:
            continue
        ref = sub.sort_values("mean_lead_time_penalised", ascending=False).iloc[0]["method"]
        ref_rows = ev[(ev["method"] == ref) & (ev["budget"] == budget)]
        model_rows = ev[(ev["method"] == "fusion") & (ev["budget"] == budget)]
        joined = paired_lead_time_delta(model_rows, ref_rows)
        if joined.empty:
            continue
        mean, lo, hi = bootstrap_ci(joined["delta"].to_numpy())
        comparisons[str(budget)] = {
            "vs": ref, "budget": budget, "n_paired_events": int(len(joined)),
            "mean_delta_quarters": mean, "ci_low": lo, "ci_high": hi,
            "significant": bool(np.isfinite(lo) and lo > 0),
            "wilcoxon": wilcoxon_delta(joined["delta"].to_numpy()),
        }

    test_mask = (year > REAL_VALID_END_YEAR) & (~features["post_event"])
    prob = probabilistic_metrics(
        features.loc[test_mask, f"label_h{horizon}"].to_numpy(),
        fusion.predict_proba(features[test_mask], cols),
    )
    return {
        "summary": summary, "per_event": ev, "comparisons": comparisons,
        "prob_metrics": prob,
        "n_series": int(features.groupby(["region", "determinant"]).ngroups),
        "n_countries": int(features["region"].nunique()),
        "n_events": len(events),
        "n_events_test": int(sum(1 for e in events
                                 if e.human_emergence_week // 4 > REAL_VALID_END_YEAR)),
        "label_caveat": (
            "Emergence is defined as a sustained rise in the observed human clinical "
            "series -- the very signal the clinical-only baselines are designed to "
            "detect. The comparison is therefore biased in favour of the baselines, "
            "and any One Health advantage reported here is a conservative estimate."
        ),
    }


# --------------------------------------------------------------------------- #
# Representation vs model class: the 2x2
# --------------------------------------------------------------------------- #
def run_representation_grid(n_seeds: int = 20, epochs: int = 25,
                            budget: float = FALSE_ALARM_BUDGET_PER_SERIES_YEAR,
                            horizon: int = PRIMARY_HORIZON) -> dict:
    """Separate the effect of the feature representation from the model class.

    Tier 1 (boosted trees on engineered features) beats Tier 2 (a convolutional
    network on raw series), but those differ in *two* ways at once, so the
    advantage cannot be assigned to either. This fills the two missing cells:

    ========================  ======================  ====================
    .                         Engineered features     Raw lagged series
    ========================  ======================  ====================
    Gradient-boosted trees    ``gbm_engineered``      ``gbm_raw``
    Neural network            ``nn_engineered``       ``nn_raw`` (Tier 2)
    ========================  ======================  ====================

    ``gbm_raw`` receives the same channels the network sees, sampled at lags 0 to
    48 in steps of four and flattened -- no smoothing, no slopes, no z-scores.
    ``nn_engineered`` is a multilayer perceptron over the engineered feature
    matrix, trained with the same optimiser, class weighting and checkpoint rule
    as the convolutional network, so the contrast is architecture rather than
    training recipe.
    """
    from ..data.features import add_raw_lag_features
    from ..models.dense import DenseModel
    from ..models.temporal import CHANNELS

    rows, per_event = [], []
    for seed in range(n_seeds):
        cfg = SimConfig(seed=seed)
        panel, events, trade = simulate(seed, cfg)
        features = build_features(panel, events, trade)
        raw_cols = add_raw_lag_features(features, list(CHANNELS))
        eng_cols = feature_columns()
        train, valid, test = _split(features)
        n_weeks = int(features["week"].max()) + 1
        n_dets = len(cfg.determinants)

        def record(frame, col, method, representation, model):
            thr = calibrate_threshold(frame, col, events, TRAIN_END_WEEK,
                                      VALID_END_WEEK, budget)
            res = evaluate_scores(frame, col, events, method, thr,
                                  VALID_END_WEEK, n_weeks)
            rows.append({**res.summary(), "seed": seed, "representation": representation,
                         "model": model})
            per_event.append(res.per_event.assign(seed=seed))

        for name, cols, representation in (("gbm_engineered", eng_cols, "engineered"),
                                           ("gbm_raw", raw_cols, "raw")):
            m = FusionModel(horizon=horizon, random_state=seed).fit(train, cols, valid=valid)
            frame = features[["week", "region", "determinant"]].copy()
            frame[f"score_{name}"] = m.predict_proba(features, cols)
            record(frame, f"score_{name}", name, representation, "boosted_trees")

        dense = DenseModel(horizon=horizon, n_regions=cfg.n_regions, n_dets=n_dets,
                           epochs=epochs, seed=seed).fit(train, eng_cols, valid=valid)
        frame = features[["week", "region", "determinant"]].copy()
        frame["score_nn_engineered"] = dense.predict_proba(features, eng_cols)
        record(frame, "score_nn_engineered", "nn_engineered", "engineered", "neural_net")

        temporal = TemporalModel(horizon=horizon, n_regions=cfg.n_regions, n_dets=n_dets,
                                 epochs=epochs, seed=seed).fit(train, valid=valid)
        record(temporal.predict_frame(features, with_uncertainty=False),
               "score_temporal", "nn_raw", "raw", "neural_net")
        print(f"  seed {seed} done", flush=True)

    summary = pd.DataFrame(rows)
    ev = pd.concat(per_event, ignore_index=True)

    # Paired contrasts that hold one factor fixed and vary the other.
    contrasts = {}
    for a, b, label in (
        ("gbm_engineered", "gbm_raw", "representation_effect_within_trees"),
        ("nn_engineered", "nn_raw", "representation_effect_within_nn"),
        ("gbm_engineered", "nn_engineered", "model_effect_on_engineered"),
        ("gbm_raw", "nn_raw", "model_effect_on_raw"),
    ):
        joined = paired_lead_time_delta(ev[ev["method"] == a], ev[ev["method"] == b])
        if joined.empty:
            continue
        mean, lo, hi = bootstrap_ci(joined["delta"].to_numpy())
        contrasts[label] = {"a": a, "b": b, "mean_delta_weeks": mean,
                            "ci_low": lo, "ci_high": hi,
                            "n_paired_events": int(len(joined)),
                            "significant": bool(np.isfinite(lo) and lo > 0),
                            "wilcoxon_p": wilcoxon_delta(joined["delta"].to_numpy())["p_value"]}
    return {"summary": summary, "per_event": ev, "contrasts": contrasts,
            "n_seeds": n_seeds}


# --------------------------------------------------------------------------- #
# Deep tiers on the real US state x month panel
# --------------------------------------------------------------------------- #
#: Monthly analogues of the evaluation constants.
US_HORIZON = 6
US_MATCH_BEFORE = 6
US_MATCH_AFTER = 3
US_REFRACTORY = 4
US_PERIODS_PER_YEAR = 12
US_WINDOW = 24            # two years of monthly history per window


def run_us_deep(min_months: int = 36, min_states: int = 20,
                budget: float = FALSE_ALARM_BUDGET_PER_SERIES_YEAR,
                epochs: int = 25, seed: int = 0,
                split: tuple[float, float] = (0.6, 0.8),
                quiet: bool = False) -> dict:
    """Do the temporal and graph tiers work when the data has the shape they need?

    Every previous test of Tiers 2 and 3 was on unfavourable ground: a simulated
    world of ~60 events, or 221 country-quarter series with no usable adjacency.
    This runs them on the most favourable real testbed available -- US states at
    monthly resolution, with genuine land-border adjacency rather than a
    synthetic ring.

    A negative result here is worth more than the simulated sweep, because the
    excuses available for the simulation (too few events, artificial geography)
    do not apply.
    """
    from ..data.real_panel import build_real_features, find_real_events
    from ..data.us_panel import build_state_month_panel, load_us_isolates
    from ..models.graph import us_state_relations

    panel = pd.read_csv(DATA_PROCESSED / "us_state_month_panel.csv")

    # Keep the dense core: states and determinants with enough months of human
    # data to define events and enough reservoir coverage to predict from.
    human = panel[panel["sector"] == "human"]
    counts = human.groupby("state")["period"].nunique()
    states = sorted(counts[counts >= min_months].index)
    if len(states) < min_states:
        states = sorted(counts.sort_values(ascending=False).head(min_states).index)
    panel = panel[panel["state"].isin(states)]

    wide = panel.pivot_table(index=["state", "determinant", "period"],
                             columns="sector", values=["prevalence", "n"], observed=True)
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index().rename(columns={
        "prevalence__human": "human_p", "prevalence__animal": "animal_p",
        "prevalence__food": "food_p", "prevalence__environment": "env_x",
        "n__human": "human_n", "n__animal": "animal_n",
        "n__food": "food_n", "n__environment": "environment_n",
        "state": "country",
    })
    for col in ("human_p", "animal_p", "food_p", "env_x",
                "human_n", "animal_n", "food_n", "environment_n"):
        if col not in wide:
            wide[col] = np.nan

    # Contiguous month grid per series, as the country pipeline does per quarter.
    filled = []
    for (st, det), block in wide.groupby(["country", "determinant"], observed=True):
        grid = pd.DataFrame({"period": range(int(block["period"].min()),
                                             int(block["period"].max()) + 1)})
        grid["country"], grid["determinant"] = st, det
        filled.append(grid.merge(block, on=["country", "determinant", "period"], how="left"))
    wide = pd.concat(filled, ignore_index=True)
    wide["year"] = wide["period"] // 12

    events = find_real_events(wide, baseline_periods=12, sustain=2, refractory=18)
    features = build_real_features(wide, events, horizon=US_HORIZON)

    year = features["week"] // 12
    split_train, split_valid = year.quantile(split[0]), year.quantile(split[1])
    train = features[year <= split_train]
    valid = features[(year > split_train) & (year <= split_valid)]
    valid_start = int(features.loc[year > split_train, "week"].min())
    test_start = int(features.loc[year > split_valid, "week"].min())
    end = int(features["week"].max()) + 1

    eval_kw = dict(match_before=US_MATCH_BEFORE, match_after=US_MATCH_AFTER,
                   refractory=US_REFRACTORY, periods_per_year=US_PERIODS_PER_YEAR)
    info = {
        "states": len(states), "months": int(features["week"].nunique()),
        "series": int(features.groupby(["region", "determinant"]).ngroups),
        "events": len(events),
        "events_test": int(sum(1 for e in events if e.human_emergence_week >= test_start)),
        "train_rows": int(len(train)), "valid_rows": int(len(valid)),
    }
    if not quiet:
        print("US deep testbed:", info)
    if info["events_test"] < 10:
        return {"status": "insufficient_test_events", **info}

    rows, per_event = [], []

    def record(frame, col, name, tier):
        thr = calibrate_threshold(frame, col, events, valid_start, test_start,
                                  budget, **eval_kw)
        res = evaluate_scores(frame, col, events, name, thr, test_start, end, **eval_kw)
        rows.append({**res.summary(), "tier": tier})
        per_event.append(res.per_event.assign(seed=seed))
        if not quiet:
            print(f"   {name:16s} sens={res.sensitivity:.2f} lead={res.mean_lead_time:6.2f} "
                  f"pen={res.mean_lead_time_penalised:6.2f} far={res.false_alarms_per_series_year:.2f}")

    base = run_baselines(features, periods_per_year=US_PERIODS_PER_YEAR, baseline_years=3)
    for n in BASELINE_NAMES:
        record(base, f"score_{n}", f"baseline_{n}", "baseline")

    cols = feature_columns()
    fusion = FusionModel(horizon=US_HORIZON, random_state=seed).fit(train, cols, valid=valid)
    fsc = features[["week", "region", "determinant"]].copy()
    fsc["score_fusion"] = fusion.predict_proba(features, cols)
    record(fsc, "score_fusion", "fusion", "fusion")

    n_dets = features["determinant"].nunique()
    # Real sector series only: no antimicrobial-use or weather covariates exist
    # here. Both deep tiers get the identical channel set so that any difference
    # between them is the graph term and nothing else.
    us_channels = ["human_p", "animal_p", "food_p", "env_x"]

    # The deep tiers index regions as integers (they carry a learned embedding);
    # here the region is a state code. Map to positions, model, and map back.
    region_index = {s: i for i, s in enumerate(states)}
    inverse = {i: s for s, i in region_index.items()}

    def to_numeric(df):
        out = df.copy()
        out["region"] = out["region"].map(region_index)
        return out.dropna(subset=["region"]).astype({"region": int})

    num_features, num_train, num_valid = (to_numeric(train), to_numeric(train),
                                          to_numeric(valid))
    num_features = to_numeric(features)

    def to_state(frame):
        out = frame.copy()
        out["region"] = out["region"].map(inverse)
        return out

    temporal = TemporalModel(horizon=US_HORIZON, n_regions=len(states), n_dets=n_dets,
                             epochs=epochs, seed=seed, channels=us_channels,
                             window=US_WINDOW).fit(num_train, valid=num_valid)
    tsc = to_state(temporal.predict_frame(num_features, with_uncertainty=False))
    record(tsc, "score_temporal", "temporal_tcn", "temporal")

    # Real land-border adjacency in place of the simulated ring.
    relations = us_state_relations(states)
    graph = GraphModel(np.zeros((len(states), len(states)), dtype=np.float32),
                       horizon=US_HORIZON, n_dets=n_dets, epochs=epochs, seed=seed,
                       target_steps=temporal.history[-1]["steps"], channels=us_channels,
                       window=US_WINDOW)
    graph.relation_names = list(relations)
    graph.relations_np = np.stack([relations[k] for k in graph.relation_names])
    graph.adj_np = graph.relations_np.mean(axis=0)
    graph.fit(num_train, valid=num_valid)
    gsc = to_state(graph.predict_frame(num_features))
    record(gsc, "score_graph", "graph_stgnn", "graph")

    summary = pd.DataFrame(rows)
    ev = pd.concat(per_event, ignore_index=True)
    comparisons = {}
    for a, b in (("temporal_tcn", "fusion"), ("graph_stgnn", "temporal_tcn"),
                 ("fusion", "baseline_ewma")):
        joined = paired_lead_time_delta(ev[ev["method"] == a], ev[ev["method"] == b])
        if joined.empty:
            continue
        mean, lo, hi = bootstrap_ci(joined["delta"].to_numpy())
        comparisons[f"{a}_vs_{b}"] = {
            "mean_delta_months": mean, "ci_low": lo, "ci_high": hi,
            "n_paired_events": int(len(joined)),
            "significant": bool(np.isfinite(lo) and lo > 0),
        }
    return {"status": "ok", **info, "summary": summary, "per_event": ev,
            "comparisons": comparisons, "graph_diagnostics": graph.diagnostics()}


# --------------------------------------------------------------------------- #
# Spatial-transmission sweep
# --------------------------------------------------------------------------- #
def run_spatial_sweep(n_seeds: int = N_SEEDS_SWEEP, epochs: int = 25,
                      budget: float = FALSE_ALARM_BUDGET_PER_SERIES_YEAR) -> dict:
    """Measure what the graph tier is worth as a function of spatial transmission.

    The point of the sweep is the control at ``spatial_fraction = 0``. Fixing a
    simulator until a favoured model wins is the easiest way to produce a
    meaningless result, so the setting that reproduces the original world is run
    alongside the others: if the graph tier shows no advantage there and a clear
    one at higher settings, the advantage is attributable to spatial structure
    rather than to the architecture having been given a friendlier world.
    """
    rows, diagnostics, per_event = [], [], []
    for sf in SPATIAL_FRACTION_SWEEP:
        for seed in range(n_seeds):
            out = run_seed(seed, with_deep=True, epochs=epochs, budget=budget,
                           spatial_fraction=sf, deep_only=True)
            for r in out["results"]:
                rows.append({**r, "spatial_fraction": sf})
            per_event.append(out["per_event"].assign(spatial_fraction=sf))
            diagnostics.append({"spatial_fraction": sf, "seed": seed,
                                "n_seeded_events": out["n_seeded_events"],
                                **{k: v for k, v in out["graph_diagnostics"].items()
                                   if not isinstance(v, dict)},
                                **{f"rel_{k}": v for k, v in
                                   out["graph_diagnostics"].get("relation_weights", {}).items()},
                                **{f"lag_{k}": v for k, v in
                                   out["graph_diagnostics"].get("lag_weights", {}).items()}})
            print(f"  sf={sf} seed={seed}: {out['runtime_s']}s, "
                  f"{out['n_seeded_events']} trade-seeded events, "
                  f"gate={out['graph_diagnostics'].get('gate', float('nan')):.3f}", flush=True)

    summary = pd.DataFrame(rows)
    events = pd.concat(per_event, ignore_index=True)
    diag = pd.DataFrame(diagnostics)

    # Paired graph-vs-temporal comparison within each setting, overall and split
    # by whether the event was trade-seeded.
    comparisons = []
    for sf in SPATIAL_FRACTION_SWEEP:
        sub = events[(events["spatial_fraction"] == sf)
                     & (events["policy"] == "prospective")
                     & (np.isclose(events["budget"], budget))]
        g = sub[sub["method"] == "graph_stgnn"]
        t = sub[sub["method"] == "temporal_tcn"]
        for label, mask in (("all", None), ("trade_seeded", True), ("independent", False)):
            gg, tt = (g, t) if mask is None else (
                g[g["seeded_by_trade"] == mask], t[t["seeded_by_trade"] == mask])
            joined = paired_lead_time_delta(gg, tt)
            if joined.empty:
                continue
            mean, lo, hi = bootstrap_ci(joined["delta"].to_numpy())
            comparisons.append({
                "spatial_fraction": sf, "event_type": label,
                "n_paired_events": int(len(joined)), "mean_delta_weeks": mean,
                "ci_low": lo, "ci_high": hi,
                "significant": bool(np.isfinite(lo) and lo > 0),
                "wilcoxon_p": wilcoxon_delta(joined["delta"].to_numpy())["p_value"],
            })

    return {"summary": summary, "per_event": events, "diagnostics": diag,
            "comparisons": pd.DataFrame(comparisons)}


# --------------------------------------------------------------------------- #
# Aggregation across seeds
# --------------------------------------------------------------------------- #
def aggregate(seed_outputs: list[dict], reference: str = "best_baseline") -> dict:
    """Pool per-seed results into the tables and intervals the report quotes."""
    summary = pd.DataFrame([r for o in seed_outputs for r in o["results"]])
    events = pd.concat([o["per_event"] for o in seed_outputs], ignore_index=True)

    by_method = (
        summary.groupby(["policy", "budget", "method"])
        .agg(
            tier=("tier", "first"),
            n_seeds=("seed", "nunique"),
            sensitivity=("sensitivity", "mean"),
            mean_lead_time=("mean_lead_time", "mean"),
            median_lead_time=("median_lead_time", "mean"),
            mean_lead_time_penalised=("mean_lead_time_penalised", "mean"),
            false_alarms_per_series_year=("false_alarms_per_series_year", "mean"),
            n_events=("n_events", "sum"),
            n_detected=("n_detected", "sum"),
        )
        .reset_index()
        .sort_values(["policy", "budget", "mean_lead_time_penalised"],
                     ascending=[True, True, False])
    )

    comparisons: dict[str, dict] = {}
    reference_by_policy: dict[str, str] = {}
    for (policy, budget), pol_rows in by_method.groupby(["policy", "budget"]):
        # The reference is whichever conventional baseline does best on penalised
        # lead time under this policy -- comparing against the weakest baseline
        # would be a soft target.
        baselines = pol_rows[pol_rows["tier"] == "baseline"]
        ref_method = (
            baselines.sort_values("mean_lead_time_penalised", ascending=False).iloc[0]["method"]
            if reference == "best_baseline" and len(baselines) else reference
        )
        reference_by_policy[f"{policy}@{budget}"] = ref_method
        pol_events = events[(events["policy"] == policy) & (events["budget"] == budget)]
        ref_rows = pol_events[pol_events["method"] == ref_method]

        for method in pol_events["method"].unique():
            if method == ref_method:
                continue
            joined = paired_lead_time_delta(pol_events[pol_events["method"] == method], ref_rows)
            if joined.empty:
                continue
            mean, lo, hi = bootstrap_ci(joined["delta"].to_numpy())
            comparisons[f"{policy}@{budget}::{method}"] = {
                "policy": policy,
                "budget": budget,
                "method": method,
                "vs": ref_method,
                "n_paired_events": int(len(joined)),
                "mean_delta_weeks": mean,
                "ci_low": lo,
                "ci_high": hi,
                "wilcoxon": wilcoxon_delta(joined["delta"].to_numpy()),
                "significant": bool(np.isfinite(lo) and lo > 0),
            }

    return {
        "reference_method": reference_by_policy,
        "by_method": by_method,
        "comparisons": comparisons,
        "per_seed": summary,
        "per_event": events,
        "prob_metrics": {
            k: {kk: float(np.nanmean([o["prob_metrics"].get(k, {}).get(kk, np.nan)
                                      for o in seed_outputs]))
                for kk in ("auc_pr", "roc_auc", "brier", "base_rate")}
            for k in {kk for o in seed_outputs for kk in o["prob_metrics"]}
        },
        "n_events_test_total": int(sum(o["n_events_test"] for o in seed_outputs)),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Run the AMR-OHEWS evaluation.")
    ap.add_argument("--quick", action="store_true", help=f"{N_SEEDS_QUICK} seeds, no deep models")
    ap.add_argument("--full", action="store_true", help=f"{N_SEEDS_FULL} seeds, all tiers")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="spatial-transmission sweep (graph tier vs temporal tier)")
    ap.add_argument("--representation", action="store_true",
                    help="2x2 of feature representation against model class")
    ap.add_argument("--us-deep", action="store_true",
                    help="run the deep tiers on the real US state x month panel")
    ap.add_argument("--real", action="store_true",
                    help="run the evaluation on real NCBI Pathogen Detection data")
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()

    if args.representation:
        out = run_representation_grid(n_seeds=args.seeds or 20, epochs=args.epochs)
        tag = args.tag or "representation"
        out["summary"].to_csv(RESULTS / f"representation_{tag}.csv", index=False)
        (RESULTS / f"representation_{tag}.json").write_text(
            json.dumps(out["contrasts"], indent=2, default=float))
        agg = (out["summary"].groupby(["model", "representation"])
               .agg(sensitivity=("sensitivity", "mean"),
                    lead=("mean_lead_time", "mean"),
                    penalised=("mean_lead_time_penalised", "mean"),
                    far=("false_alarms_per_series_year", "mean")).reset_index())
        print(f"\n=== representation x model, {out['n_seeds']} seeds ===")
        print(agg.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        print("\npaired contrasts (positive favours the first):")
        for label, c in out["contrasts"].items():
            star = "*" if c["significant"] else " "
            print(f" {star} {label:36s} {c['a']:16s} vs {c['b']:16s} "
                  f"{c['mean_delta_weeks']:+6.2f} wk [{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]"
                  f"  p={c['wilcoxon_p']:.3g}")
        return

    if args.us_deep:
        out = run_us_deep(epochs=args.epochs)
        tag = args.tag or "usdeep"
        if out.get("status") != "ok":
            print("stopping:", out); return
        out["summary"].to_csv(RESULTS / f"us_deep_summary_{tag}.csv", index=False)
        out["per_event"].to_csv(RESULTS / f"us_deep_per_event_{tag}.csv", index=False)
        (RESULTS / f"us_deep_headline_{tag}.json").write_text(json.dumps(
            {k: v for k, v in out.items() if k not in ("summary", "per_event")},
            indent=2, default=float))
        print("\n=== deep tiers on real US state x month data ===")
        print(out["summary"][["method", "tier", "sensitivity", "mean_lead_time",
                              "mean_lead_time_penalised", "false_alarms_per_series_year"]]
              .sort_values("mean_lead_time_penalised", ascending=False)
              .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
        for k, c in out["comparisons"].items():
            star = "*" if c["significant"] else " "
            print(f" {star} {k:28s} {c['mean_delta_months']:+.2f} months "
                  f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}] n={c['n_paired_events']}")
        print("\ngraph diagnostics:", out["graph_diagnostics"])
        return

    if args.real:
        out = run_real()
        tag = args.tag or "real"
        out["summary"].to_csv(RESULTS / f"real_summary_{tag}.csv", index=False)
        out["per_event"].to_csv(RESULTS / f"real_per_event_{tag}.csv", index=False)
        (RESULTS / f"real_headline_{tag}.json").write_text(json.dumps(
            {k: v for k, v in out.items() if k not in ("summary", "per_event")},
            indent=2, default=float))
        print(f"real panel: {out['n_series']} series, {out['n_countries']} countries, "
              f"{out['n_events']} events ({out['n_events_test']} in the test window)")
        for budget in sorted(out["summary"]["budget"].unique()):
            print(f"\n=== false-alarm budget {budget}/series-year (quarters) ===")
            sub = out["summary"][out["summary"]["budget"] == budget]
            print(sub[["method", "sensitivity", "mean_lead_time",
                       "mean_lead_time_penalised", "false_alarms_per_series_year",
                       "n_detected", "n_events"]].sort_values(
                "mean_lead_time_penalised", ascending=False).to_string(
                index=False, float_format=lambda v: f"{v:.2f}"))
            c = out["comparisons"].get(str(budget))
            if c:
                star = "*" if c["significant"] else " "
                print(f" {star} fusion vs {c['vs']}: {c['mean_delta_quarters']:+.2f} quarters "
                      f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]  n={c['n_paired_events']}  "
                      f"p={c['wilcoxon']['p_value']:.4g}")
        print(f"\nAUC-PR {out['prob_metrics']['auc_pr']:.3f} "
              f"(base rate {out['prob_metrics']['base_rate']:.3f})")
        print(f"\nCaveat: {out['label_caveat']}")
        return

    if args.sweep:
        agg = run_spatial_sweep(n_seeds=args.seeds or N_SEEDS_SWEEP, epochs=args.epochs)
        tag = args.tag or "sweep"
        agg["summary"].to_csv(RESULTS / f"sweep_summary_{tag}.csv", index=False)
        agg["per_event"].to_csv(RESULTS / f"sweep_per_event_{tag}.csv", index=False)
        agg["diagnostics"].to_csv(RESULTS / f"sweep_diagnostics_{tag}.csv", index=False)
        agg["comparisons"].to_csv(RESULTS / f"sweep_comparisons_{tag}.csv", index=False)
        print("\n=== graph tier minus temporal tier, by spatial transmission ===")
        print(agg["comparisons"].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print("\n=== learned graph diagnostics (mean over seeds) ===")
        num = agg["diagnostics"].select_dtypes("number")
        print(num.groupby(agg["diagnostics"]["spatial_fraction"]).mean().round(3).to_string())
        print(f"\nwrote results/sweep_*_{tag}.csv")
        return

    n_seeds = args.seeds or (N_SEEDS_FULL if args.full else N_SEEDS_QUICK)
    with_deep = not args.no_deep and not args.quick
    tag = args.tag or ("full" if args.full else "quick")

    outputs = []
    for seed in range(n_seeds):
        keep = seed == 0
        out = run_seed(seed, with_deep=with_deep, keep_artifacts=keep, epochs=args.epochs)
        if keep and "artifacts" in out:
            _save_artifacts(out.pop("artifacts"), tag)
        outputs.append(out)
        best = max((r for r in out["results"] if r["tier"] != "baseline"),
                   key=lambda r: r["mean_lead_time_penalised"], default=None)
        print(f"seed {seed}: {out['runtime_s']}s, {out['n_events_test']} test events, "
              f"best non-baseline = {best['method'] if best else 'n/a'} "
              f"({best['mean_lead_time_penalised']:.1f} wk)" if best else f"seed {seed} done")

    agg = aggregate(outputs)
    _save_aggregate(agg, tag)

    bm = agg["by_method"]
    for (policy, budget), rows in bm.groupby(["policy", "budget"]):
        print(f"\n=== {policy}, false-alarm budget {budget}/series-year "
              f"(mean over seeds, test window) ===")
        print(rows.drop(columns=["policy", "budget"]).to_string(
            index=False, float_format=lambda v: f"{v:.2f}"))
        print(f"reference baseline: {agg['reference_method'].get(f'{policy}@{budget}')}")
        comps = {k: c for k, c in agg["comparisons"].items()
                 if c["policy"] == policy and c["budget"] == budget}
        for _, c in sorted(comps.items(), key=lambda kv: -(kv[1]["mean_delta_weeks"] or 0)):
            star = "*" if c["significant"] else " "
            print(f" {star} {c['method']:24s} {c['mean_delta_weeks']:+6.2f} wk "
                  f"[{c['ci_low']:+.2f}, {c['ci_high']:+.2f}]  n={c['n_paired_events']}  "
                  f"p={c['wilcoxon']['p_value']:.4g}")


def _save_aggregate(agg: dict, tag: str) -> None:
    agg["by_method"].to_csv(RESULTS / f"summary_{tag}.csv", index=False)
    agg["per_seed"].to_csv(RESULTS / f"per_seed_{tag}.csv", index=False)
    agg["per_event"].to_csv(RESULTS / f"per_event_{tag}.csv", index=False)
    payload = {
        "reference_method": agg["reference_method"],
        "comparisons": agg["comparisons"],
        "prob_metrics": agg["prob_metrics"],
        "n_events_test_total": agg["n_events_test_total"],
        "by_method": agg["by_method"].to_dict(orient="records"),
    }
    (RESULTS / f"headline_{tag}.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote results/summary_{tag}.csv and results/headline_{tag}.json")


def _save_artifacts(art: dict, tag: str) -> None:
    d = RESULTS / f"artifacts_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    for key in ("panel", "scores", "baseline_scores", "baseline_alarms", "alerts",
                "sector_shares", "reliability"):
        art[key].to_csv(d / f"{key}.csv", index=False)
    np.save(d / "trade.npy", art["trade"])
    (d / "meta.json").write_text(json.dumps({
        "events": art["events"],
        "threshold": art["threshold"],
        "baseline_thresholds": art["baseline_thresholds"],
        "shap_method": art["shap_method"],
    }, indent=2, default=float))
    print(f"wrote dashboard artifacts to {d}")


if __name__ == "__main__":
    main()
