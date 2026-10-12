"""Tier 0: conventional clinical-only surveillance baselines.

These stand in for what an AMR surveillance programme does today: watch the
human clinical resistant-proportion series for one pathogen-drug pair and raise
a flag when it departs from its recent history. They see *only* the human
stream, and only with its reporting delay, which is the comparison the research
question is built around.

Four methods are implemented, spanning the range of what is actually deployed:

``ewma``
    Exponentially weighted moving-average control chart -- the workhorse of
    laboratory quality control and of several national AMR dashboards.
``cusum``
    Cumulative-sum chart on the logit proportion, designed to accumulate small
    persistent shifts rather than react to single spikes. Historically the most
    sensitive of the classical charts to slow drifts, which is what emerging
    resistance looks like.
``farrington``
    A Farrington-style seasonal reference-window method (Farrington et al.,
    1996), the algorithm behind much of European infectious-disease outbreak
    detection. The expected value for week *t* is estimated from the same weeks
    of preceding years, with a linear trend term and quasi-Poisson overdispersion.
``manual``
    A deliberately coarse "three consecutive rising quarters" rule, approximating
    the annual/quarterly human review of surveillance reports that still triggers
    much real-world action.

Every method returns a **continuous score**, never a binary flag. Thresholds are
chosen downstream so that each method is compared at an identical false-alarm
budget -- otherwise a trigger-happy method would look like an early one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BASELINE_NAMES = ("ewma", "cusum", "farrington", "manual")


def _min_periods(window: int) -> int:
    """Observations required before a chart will emit a statistic.

    Half the window, capped at twelve and never exceeding the window itself --
    the last clause matters because a quarterly panel's review interval is a
    single period, and pandas rejects ``min_periods > window``.
    """
    return max(1, min(12, window // 2, window))


# --------------------------------------------------------------------------- #
# Individual charts (operate on one region x determinant series)
# --------------------------------------------------------------------------- #
def ewma_score(x: pd.Series, lam: float = 0.2, baseline: int = 52) -> np.ndarray:
    """Standardised EWMA exceedance over a trailing baseline."""
    v = x.to_numpy(dtype=float)
    v = _fill(v)
    ewma = pd.Series(v).ewm(alpha=lam, adjust=False).mean().to_numpy()
    # min_periods scales with the window: a quarterly panel's baseline is a
    # handful of periods, and a constant of 12 is simply invalid there.
    mp = _min_periods(baseline)
    mu = pd.Series(v).rolling(baseline, min_periods=mp).mean().shift(1).to_numpy()
    sd = pd.Series(v).rolling(baseline, min_periods=mp).std().shift(1).to_numpy()
    sd = np.where((sd is np.nan) | (sd == 0), np.nan, sd)
    # The EWMA of an i.i.d. series has variance sigma^2 * lam/(2-lam).
    sd_ewma = sd * np.sqrt(lam / (2 - lam))
    return np.nan_to_num((ewma - mu) / sd_ewma, nan=0.0, posinf=0.0, neginf=0.0)


def cusum_score(x: pd.Series, k: float = 0.5, baseline: int = 52) -> np.ndarray:
    """One-sided CUSUM on the logit proportion, restarted after each exceedance."""
    v = _fill(x.to_numpy(dtype=float))
    logit = np.log(np.clip(v, 1e-4, 1 - 1e-4) / (1 - np.clip(v, 1e-4, 1 - 1e-4)))
    s = pd.Series(logit)
    mp = _min_periods(baseline)
    mu = s.rolling(baseline, min_periods=mp).mean().shift(1).to_numpy()
    sd = s.rolling(baseline, min_periods=mp).std().shift(1).to_numpy()
    z = np.nan_to_num((logit - mu) / np.where(sd == 0, np.nan, sd), nan=0.0,
                      posinf=0.0, neginf=0.0)
    out = np.zeros_like(z)
    acc = 0.0
    for i, zi in enumerate(z):
        acc = max(0.0, acc + zi - k)
        # Cap the statistic: an unbounded CUSUM stays "alarmed" forever after one
        # large event, which would make its lead time on later events meaningless.
        acc = min(acc, 25.0)
        out[i] = acc
    return out


def farrington_score(x: pd.Series, n_years: int = 3, half_window: int = 3,
                     periods_per_year: int = 52) -> np.ndarray:
    """Farrington-style exceedance against same-season weeks of previous years.

    For period *t* the reference set is periods
    ``t - periods_per_year*k +/- half_window`` for ``k = 1..n_years``. A linear trend is removed by least squares and the
    residual scale is inflated for overdispersion, giving a standardised
    exceedance comparable across series.
    """
    v = _fill(x.to_numpy(dtype=float))
    n = len(v)
    out = np.zeros(n)
    for t in range(n):
        idx: list[int] = []
        for k in range(1, n_years + 1):
            centre = t - periods_per_year * k
            if centre < 0:
                continue
            idx.extend(range(max(0, centre - half_window), min(n, centre + half_window + 1)))
        if len(idx) < 8:
            continue
        ref_t = np.asarray(idx, dtype=float)
        ref_v = v[idx]
        if not np.isfinite(ref_v).any():
            continue
        # Least-squares trend through the reference window, extrapolated to t.
        a, b = np.polyfit(ref_t, ref_v, 1)
        expected = a * t + b
        resid = ref_v - (a * ref_t + b)
        sd = resid.std(ddof=1)
        # Quasi-Poisson style overdispersion floor: never trust a suspiciously
        # tight reference window.
        sd = max(sd, 0.25 * np.sqrt(max(expected, 1e-4)), 1e-3)
        out[t] = (v[t] - expected) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def manual_review_score(x: pd.Series, quarter: int = 13) -> np.ndarray:
    """Count of consecutive rising quarters -- a coarse stand-in for human review.

    The value only changes once a quarter, which is the point: it reproduces the
    step-like latency of a process driven by periodic reports rather than by
    continuous monitoring.
    """
    v = _fill(x.to_numpy(dtype=float))
    n = len(v)
    q_mean = pd.Series(v).rolling(quarter, min_periods=_min_periods(quarter)).mean().to_numpy()
    out = np.zeros(n)
    run = 0
    last_q = np.nan
    for t in range(n):
        if t % quarter == 0 and t >= quarter:
            cur = q_mean[t]
            if np.isfinite(cur) and np.isfinite(last_q):
                run = run + 1 if cur > last_q * 1.05 else 0
            last_q = cur
        out[t] = run
    return out


def _fill(v: np.ndarray) -> np.ndarray:
    """Forward/backward-fill NaNs so the charts run on gappy surveillance series."""
    s = pd.Series(v).ffill().bfill()
    return s.fillna(0.0).to_numpy()


# --------------------------------------------------------------------------- #
# Panel-level driver
# --------------------------------------------------------------------------- #
def _scorers(periods_per_year: int, baseline_years: int = 1) -> dict:
    """Bind the chart windows to the panel's time unit.

    The simulated panel is weekly and the real panel is quarterly. A control
    chart with a 52-*period* baseline means one year on the first and thirteen
    on the second, so every window is expressed in years here and converted.
    """
    ppy = periods_per_year
    baseline = max(4, int(ppy * baseline_years))
    return {
        "ewma": lambda x: ewma_score(x, baseline=baseline),
        "cusum": lambda x: cusum_score(x, baseline=baseline),
        "farrington": lambda x: farrington_score(
            x, half_window=max(1, ppy // 17), periods_per_year=ppy),
        # Periodic human review: the review interval is one quarter of a year,
        # but never shorter than a single observation period.
        "manual": lambda x: manual_review_score(x, quarter=max(1, round(ppy / 4))),
    }


def run_baselines(features: pd.DataFrame, column: str = "human_p",
                  periods_per_year: int = 52, baseline_years: int = 1) -> pd.DataFrame:
    """Score every (region, determinant) series with all four baselines.

    ``column`` defaults to the **delay-adjusted** human clinical proportion, so
    the baselines are handicapped by exactly the reporting latency a real
    programme faces -- no more and no less.
    """
    out = features[["week", "region", "determinant"]].copy()
    for name in BASELINE_NAMES:
        out[f"score_{name}"] = np.nan

    scorers = _scorers(periods_per_year, baseline_years)
    for (r, d), block in features.groupby(["region", "determinant"], observed=True, sort=False):
        block = block.sort_values("week")
        mask = (out["region"] == r) & (out["determinant"] == d)
        order = out.index[mask]
        for name, fn in scorers.items():
            out.loc[order, f"score_{name}"] = fn(block[column])
    return out
