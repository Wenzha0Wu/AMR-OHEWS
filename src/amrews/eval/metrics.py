"""Evaluation: alarms, detection lead time, false-alarm budgets, uncertainty.

The single number this project lives or dies by is **lead time**: how many weeks
before resistance establishes in human medicine did a method raise its flag?
Getting that number right requires three things that are easy to get wrong.

1. **Equal false-alarm budgets.** Any detector can be made "earlier" by lowering
   its threshold until it alarms constantly. Every method here is therefore
   thresholded to the same number of false alarms per series-year, calibrated on
   a validation window that is disjoint from the test window.
2. **A refractory period.** A sustained exceedance is one alarm, not forty. Once
   a series alarms it is silenced for :data:`REFRACTORY_WEEKS`, so a method that
   simply stays above threshold cannot inflate its false-alarm count or claim
   credit for a later event with an older alarm.
3. **Honest handling of misses.** Reporting the mean lead time only over detected
   events flatters a method that detects two easy events and misses the rest.
   Both ``lead_time_detected`` and a miss-penalised variant are reported, along
   with sensitivity, and the headline comparison is *paired per event*.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import (
    BOOTSTRAP_RESAMPLES,
    FALSE_ALARM_BUDGET_PER_SERIES_YEAR,
    MATCH_WINDOW_AFTER,
    MATCH_WINDOW_BEFORE,
)

#: Weeks a series is silenced after raising an alarm.
#:
#: Deliberately shorter than :data:`MATCH_WINDOW_BEFORE`. When the two were
#: equal, an alarm fired just outside the match window -- too early to be
#: credited -- silenced the series for the whole window in which a creditable
#: alarm could have been raised, so being early was punished twice.
REFRACTORY_WEEKS = 13


# --------------------------------------------------------------------------- #
# Alarm generation
# --------------------------------------------------------------------------- #
def prepare_series(scores: pd.DataFrame, score_col: str) -> list[tuple]:
    """Group a score frame into per-series numpy arrays, once.

    Threshold calibration sweeps sixty candidate thresholds, each of which needs
    a full pass over every series. Re-running pandas ``groupby`` inside that
    sweep dominated the runtime, so the grouping is hoisted out and the sweep
    works on plain arrays.
    """
    df = scores.sort_values(["region", "determinant", "week"])
    out = []
    for (r, d), block in df.groupby(["region", "determinant"], observed=True, sort=False):
        weeks = block["week"].to_numpy(dtype=np.int64)
        vals = np.nan_to_num(block[score_col].to_numpy(dtype=float), nan=-np.inf)
        out.append((r, d, weeks, vals))
    return out


def alarms_from_prepared(
    prepared: list[tuple],
    threshold: float,
    start_week: int = 0,
    end_week: int | None = None,
    refractory: int = REFRACTORY_WEEKS,
) -> pd.DataFrame:
    """Discrete, de-duplicated alarms from pre-grouped series arrays."""
    rows = []
    for r, d, weeks, vals in prepared:
        sel = (weeks >= start_week) if end_week is None else (
            (weeks >= start_week) & (weeks < end_week))
        w, v = weeks[sel], vals[sel]
        # Only exceedances can start an alarm, so walk those rather than every week.
        hits = np.flatnonzero(v >= threshold)
        last = -np.inf
        for i in hits:
            if w[i] - last >= refractory:
                rows.append({"region": r, "determinant": d, "week": int(w[i]),
                             "score": float(v[i])})
                last = w[i]
    return pd.DataFrame(rows, columns=["region", "determinant", "week", "score"])


def alarms_from_scores(
    scores: pd.DataFrame,
    score_col: str,
    threshold: float,
    start_week: int = 0,
    end_week: int | None = None,
    refractory: int = REFRACTORY_WEEKS,
) -> pd.DataFrame:
    """Convert a continuous score series into discrete, de-duplicated alarms."""
    return alarms_from_prepared(
        prepare_series(scores, score_col), threshold, start_week, end_week, refractory
    )


# --------------------------------------------------------------------------- #
# Scoring one method
# --------------------------------------------------------------------------- #
@dataclass
class EvalResult:
    """Per-method evaluation summary plus the per-event rows behind it."""

    method: str
    sensitivity: float
    n_events: int
    n_detected: int
    mean_lead_time: float
    median_lead_time: float
    mean_lead_time_penalised: float
    false_alarms_per_series_year: float
    threshold: float
    per_event: pd.DataFrame

    def summary(self) -> dict:
        return {
            "method": self.method,
            "sensitivity": self.sensitivity,
            "n_events": self.n_events,
            "n_detected": self.n_detected,
            "mean_lead_time": self.mean_lead_time,
            "median_lead_time": self.median_lead_time,
            "mean_lead_time_penalised": self.mean_lead_time_penalised,
            "false_alarms_per_series_year": self.false_alarms_per_series_year,
            "threshold": self.threshold,
        }


def scoreable_events(events: list, lo: int, hi: int) -> list:
    """True events whose human-emergence week falls inside ``[lo, hi)``."""
    return [
        e for e in events
        if getattr(e, "is_true_event", False)
        and getattr(e, "human_emergence_week", None) is not None
        and lo <= e.human_emergence_week < hi
    ]


def _eligible_series_weeks(
    scores: pd.DataFrame, events: list, start_week: int, end_week: int,
    match_before: int = MATCH_WINDOW_BEFORE, periods_per_year: float = 52.0,
    match_after: int = MATCH_WINDOW_AFTER,
) -> int:
    """Series-weeks that may legitimately produce a false alarm.

    Weeks inside an event's match window are excluded (an alarm there is a
    detection), and so are weeks after an event has fully established (an alarm
    there is a late detection, not a false positive). Decoy series contribute
    every week -- that is what they are for.
    """
    # A series may carry more than one event (the real-data panel does), so the
    # truth map holds a list. Collapsing it to one week would silently count
    # weeks inside a second event's window as eligible for false alarms.
    truth: dict[tuple, list[int]] = {}
    for e in events:
        if getattr(e, "is_true_event", False) and getattr(e, "human_emergence_week", None) is not None:
            truth.setdefault((e.region, e.determinant), []).append(e.human_emergence_week)

    total = 0
    for (r, d), block in scores.groupby(["region", "determinant"], observed=True, sort=False):
        weeks = block["week"].to_numpy()
        weeks = weeks[(weeks >= start_week) & (weeks < end_week)]
        stars = truth.get((r, d))
        if not stars:
            total += len(weeks)
            continue
        # A week is eligible to produce a false alarm exactly when it lies
        # outside every event's match window -- the same condition the numerator
        # uses to decide that an alarm is false. Keeping the two definitions
        # identical is the whole point.
        #
        # This previously read `eligible &= weeks < w_star - match_before`,
        # conjoined over events. On a series with several events that admits
        # only the weeks preceding the *first* one, which on the real panels was
        # often none at all and sent the false-alarm rate to infinity. It was
        # also inconsistent with the numerator, which does count an alarm landing
        # after an event as a false alarm while the denominator refused to count
        # the week it landed in.
        eligible = np.ones(len(weeks), dtype=bool)
        for w_star in stars:
            eligible &= ~((weeks >= w_star - match_before) & (weeks <= w_star + match_after))
        total += int(eligible.sum())
    return total


def evaluate_scores(
    scores: pd.DataFrame,
    score_col: str,
    events: list,
    method: str,
    threshold: float,
    start_week: int,
    end_week: int,
    prepared: list[tuple] | None = None,
    eligible: int | None = None,
    match_before: int = MATCH_WINDOW_BEFORE,
    match_after: int = MATCH_WINDOW_AFTER,
    refractory: int = REFRACTORY_WEEKS,
    periods_per_year: float = 52.0,
) -> EvalResult:
    """Score one method at one threshold over ``[start_week, end_week)``.

    ``prepared`` and ``eligible`` are optional pre-computations; passing them
    lets the threshold sweep avoid redoing identical work sixty times.

    The window and rate parameters are in whatever time unit the ``week`` column
    holds. They default to weeks for the simulated evaluation; the real-data
    evaluation works in quarters and passes its own values, which is why these
    are arguments rather than module constants.
    """
    if prepared is None:
        prepared = prepare_series(scores, score_col)
    # Alarms may legitimately precede the test window -- an early warning is the
    # point -- so alarm generation starts a full match window earlier.
    alarms = alarms_from_prepared(
        prepared, threshold, start_week=start_week - match_before,
        end_week=end_week, refractory=refractory,
    )
    targets = scoreable_events(events, start_week, end_week)

    rows, matched_alarm_ids = [], set()
    for ev in targets:
        w_star = ev.human_emergence_week
        cand = alarms[
            (alarms["region"] == ev.region)
            & (alarms["determinant"] == ev.determinant)
            & (alarms["week"] >= w_star - match_before)
            & (alarms["week"] <= w_star + match_after)
        ]
        if len(cand):
            first = cand.sort_values("week").iloc[0]
            matched_alarm_ids.add((first["region"], first["determinant"], int(first["week"])))
            detected, det_week = True, int(first["week"])
            lead = float(w_star - det_week)
        else:
            detected, det_week, lead = False, None, np.nan
        rows.append({
            "method": method, "region": ev.region, "determinant": ev.determinant,
            "emergence_week": w_star, "detected": detected,
            "detection_week": det_week, "lead_time": lead,
            # Carried through so lead time can be split by event type: the graph
            # tier is supposed to help on trade-seeded events specifically, and
            # an average over both kinds would hide that.
            "seeded_by_trade": bool(getattr(ev, "seeded_by_trade", False)),
        })
    per_event = pd.DataFrame(rows)

    # False alarms: every alarm inside the scoring window that matched no event.
    fa = 0
    for _, a in alarms.iterrows():
        if a["week"] < start_week:
            continue
        if (a["region"], a["determinant"], int(a["week"])) in matched_alarm_ids:
            continue
        if _inside_any_window(a, targets, match_before, match_after):
            continue
        fa += 1
    if eligible is None:
        eligible = _eligible_series_weeks(scores, events, start_week, end_week,
                                          match_before, match_after=match_after)
    fa_rate = fa / max(eligible / periods_per_year, 1e-9)

    n_events = len(per_event)
    n_det = int(per_event["detected"].sum()) if n_events else 0
    leads = per_event.loc[per_event["detected"], "lead_time"] if n_events else pd.Series(dtype=float)
    # Miss-penalised lead time: a missed event is scored at the worst value the
    # match window allows, so sensitivity cannot be traded away for lead time.
    penalised = per_event["lead_time"].fillna(-float(match_after)) if n_events else pd.Series(dtype=float)

    return EvalResult(
        method=method,
        sensitivity=n_det / n_events if n_events else np.nan,
        n_events=n_events,
        n_detected=n_det,
        mean_lead_time=float(leads.mean()) if len(leads) else np.nan,
        median_lead_time=float(leads.median()) if len(leads) else np.nan,
        mean_lead_time_penalised=float(penalised.mean()) if n_events else np.nan,
        false_alarms_per_series_year=float(fa_rate),
        threshold=float(threshold),
        per_event=per_event,
    )


def _inside_any_window(alarm: pd.Series, targets: list,
                       match_before: int = MATCH_WINDOW_BEFORE,
                       match_after: int = MATCH_WINDOW_AFTER) -> bool:
    for ev in targets:
        if ev.region == alarm["region"] and ev.determinant == alarm["determinant"]:
            if (ev.human_emergence_week - match_before
                    <= alarm["week"] <= ev.human_emergence_week + match_after):
                return True
    return False


# --------------------------------------------------------------------------- #
# Threshold calibration
# --------------------------------------------------------------------------- #
def calibrate_threshold(
    scores: pd.DataFrame,
    score_col: str,
    events: list,
    start_week: int,
    end_week: int,
    budget: float = FALSE_ALARM_BUDGET_PER_SERIES_YEAR,
    n_candidates: int = 2000,
    **eval_kw,
) -> float:
    """Lowest threshold whose false-alarm rate stays within ``budget``.

    Calibration runs on the validation window only. Using the test window here
    would be the classic outbreak-detection mistake: tuning the alarm rate on the
    same data used to report lead time.

    The false-alarm rate is monotone non-increasing in the threshold, so the
    search is a bisection over a dense candidate grid rather than a linear scan.
    That matters for more than speed: a coarse grid over a score distribution
    with a long flat tail -- exactly what a calibrated classifier produces, where
    most weeks sit near zero -- jumps straight past the budget and silently
    hands the method a far stricter threshold than it was allowed, costing it
    sensitivity it should have had.
    """
    vals = scores[score_col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.inf

    prepared = prepare_series(scores, score_col)
    eligible = _eligible_series_weeks(
        scores, events, start_week, end_week,
        eval_kw.get("match_before", MATCH_WINDOW_BEFORE),
        match_after=eval_kw.get("match_after", MATCH_WINDOW_AFTER))

    def far(thr: float) -> float:
        return evaluate_scores(scores, score_col, events, "calib", float(thr),
                               start_week, end_week, prepared=prepared,
                               eligible=eligible, **eval_kw).false_alarms_per_series_year

    # Dense grid over the upper half of the score distribution.
    grid = np.unique(np.quantile(vals, np.linspace(0.5, 1.0, n_candidates)))
    if far(grid[0]) <= budget:          # even the loosest threshold is within budget
        return float(grid[0])
    if far(grid[-1]) > budget:          # even the strictest is not
        return float(grid[-1])

    lo, hi = 0, len(grid) - 1           # far(grid[lo]) > budget >= far(grid[hi])
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if far(grid[mid]) <= budget:
            hi = mid
        else:
            lo = mid
    return float(grid[hi])


# --------------------------------------------------------------------------- #
# Uncertainty
# --------------------------------------------------------------------------- #
def paired_lead_time_delta(
    model_events: pd.DataFrame, baseline_events: pd.DataFrame
) -> pd.DataFrame:
    """Join two methods' per-event rows so lead times can be compared pairwise."""
    keys = [k for k in ("seed", "region", "determinant", "emergence_week")
            if k in model_events.columns]
    drop = [c for c in ("method", "policy") if c in model_events.columns]
    m = model_events.drop(columns=drop).set_index(keys)
    b = baseline_events.drop(columns=drop).set_index(keys)
    joined = m.join(b, how="inner", lsuffix="_model", rsuffix="_baseline").reset_index()
    # A method that misses an event is credited with the worst lead the window
    # allows, so misses count against it rather than silently dropping out.
    worst = -float(MATCH_WINDOW_AFTER)
    joined["lead_model"] = joined["lead_time_model"].fillna(worst)
    joined["lead_baseline"] = joined["lead_time_baseline"].fillna(worst)
    joined["delta"] = joined["lead_model"] - joined["lead_baseline"]
    return joined


def bootstrap_ci(
    values: np.ndarray, n_resamples: int = BOOTSTRAP_RESAMPLES, alpha: float = 0.05,
    seed: int = 12345,
) -> tuple[float, float, float]:
    """Percentile bootstrap mean and confidence interval."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_resamples, v.size), replace=True).mean(axis=1)
    return (float(v.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def wilcoxon_delta(values: np.ndarray) -> dict:
    """Wilcoxon signed-rank test that paired lead-time deltas are centred on zero."""
    from scipy.stats import wilcoxon

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 6 or np.allclose(v, 0):
        return {"statistic": np.nan, "p_value": np.nan, "n": int(v.size)}
    stat, p = wilcoxon(v, zero_method="zsplit", alternative="greater")
    return {"statistic": float(stat), "p_value": float(p), "n": int(v.size)}


# --------------------------------------------------------------------------- #
# Probabilistic quality (for the supervised models)
# --------------------------------------------------------------------------- #
def probabilistic_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """AUC-PR, ROC-AUC and Brier score, with the positive-class base rate."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0, 1)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return {"auc_pr": np.nan, "roc_auc": np.nan, "brier": np.nan,
                "base_rate": float(y_true.mean())}
    return {
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "base_rate": float(y_true.mean()),
    }


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Binned observed frequency against predicted probability, for calibration plots."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.quantile(y_prob, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() < 5:
            continue
        rows.append({"bin": b, "predicted": float(y_prob[m].mean()),
                     "observed": float(y_true[m].mean()), "n": int(m.sum())})
    return pd.DataFrame(rows)
