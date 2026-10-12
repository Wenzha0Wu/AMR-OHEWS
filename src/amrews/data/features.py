"""Harmonisation and feature engineering for the One Health panel.

Two things in this module do the real work of making the evaluation honest.

**Reporting delays are applied before anything else.** A model asked to make a
call in week *t* may only see stream *s* up to week ``t - REPORTING_DELAY_WEEKS[s]``.
Human clinical data are the slowest stream, which is a large part of why
conventional surveillance is late; a pipeline that quietly used same-week
clinical counts would manufacture its own answer.

**Labels are forward-looking and post-event rows are withheld.** The supervised
target at (region, determinant, week *t*) is "does human-clinical emergence begin
in (t, t+H]?". Weeks after an event's emergence week are marked ``post_event``
and dropped from training and from false-alarm accounting -- an alarm raised two
weeks *after* resistance established is a late detection, not a false positive,
and scoring it either way would be wrong.

Feature groups map one-to-one onto sectors so that :mod:`amrews.eval.experiments`
can ablate a whole sector by dropping its group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import HORIZONS, REPORTING_DELAY_WEEKS

# Rolling windows (weeks) used throughout.
SHORT, MEDIUM, LONG = 8, 16, 52


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _shrunk_proportion(res: pd.Series, n: pd.Series) -> pd.Series:
    """Jeffreys-shrunk resistant proportion; NaN where nothing was tested.

    Raw ``resistant / n`` is unusable at the denominators real AMR surveillance
    works with -- a single resistant isolate out of three reads as 33%. The
    Beta(1/2, 1/2) prior keeps small samples from dominating every derived
    feature, and NaN-then-forward-fill preserves the fact that a week carried no
    information at all.
    """
    out = (res + 0.5) / (n + 1.0)
    return out.where(n > 0, np.nan)


def _available(df: pd.DataFrame, cols: list[str], delay: int) -> pd.DataFrame:
    """Shift ``cols`` forward by ``delay`` weeks within each series.

    After this, row *t* holds only values a decision-maker could actually have
    read in week *t*.
    """
    return df.groupby(["region", "determinant"], observed=True)[cols].shift(delay)


def _roll(s: pd.Series, window: int, fn: str) -> pd.Series:
    r = s.rolling(window, min_periods=max(2, window // 4))
    return getattr(r, fn)()


def _slope(s: pd.Series, window: int) -> pd.Series:
    """Ordinary-least-squares slope over a trailing window, per week.

    Written as a single dot product against fixed regression weights rather than
    ``rolling().apply()``: the pipeline fits this over ~30 simulated worlds, and
    the Python-level callback version cost more than the rest of feature
    engineering put together. Partial windows at the start of a series are
    fitted on whatever history exists rather than discarded -- a surveillance
    system does not get to skip its first year.
    """
    v = s.to_numpy(dtype=float)
    n = len(v)
    filled = pd.Series(v).ffill().bfill().to_numpy()
    filled = np.nan_to_num(filled, nan=0.0)
    out = np.full(n, np.nan)
    min_periods = max(3, window // 3)

    # Leading partial windows: a handful of small fits, done directly.
    for t in range(min_periods - 1, min(window - 1, n)):
        seg = filled[: t + 1]
        x = np.arange(len(seg), dtype=float)
        x -= x.mean()
        denom = (x**2).sum()
        out[t] = (x @ (seg - seg.mean())) / denom if denom else 0.0

    if n >= window:
        x = np.arange(window, dtype=float)
        x -= x.mean()
        weights = x / (x**2).sum()
        windows = np.lib.stride_tricks.sliding_window_view(filled, window)
        out[window - 1:] = windows @ weights

    # Where the underlying series was never observed, no slope is defined.
    out[np.isnan(v) & (pd.Series(v).ffill().isna().to_numpy())] = np.nan
    return pd.Series(out, index=s.index)


def _zscore(s: pd.Series, window: int) -> pd.Series:
    """Deviation from a trailing baseline, in trailing standard deviations."""
    mu = _roll(s, window, "mean")
    sd = _roll(s, window, "std")
    return (s - mu) / sd.replace(0.0, np.nan)


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
#: Feature groups, keyed by the sector they come from. Ablations drop whole groups.
FEATURE_GROUPS: dict[str, list[str]] = {}


def build_features(
    panel: pd.DataFrame,
    events: list,
    trade: np.ndarray,
    horizons: tuple[int, ...] = HORIZONS,
) -> pd.DataFrame:
    """Turn the raw observation panel into a causal, labelled feature table."""
    df = panel.sort_values(["region", "determinant", "week"]).reset_index(drop=True).copy()
    g = ["region", "determinant"]

    # ---- 1. raw sector signals ------------------------------------------- #
    df["human_p"] = _shrunk_proportion(df["human_resistant"], df["human_n"])
    df["animal_p"] = _shrunk_proportion(df["animal_resistant"], df["animal_n"])
    df["food_p"] = _shrunk_proportion(df["food_resistant"], df["food_n"])
    df["env_x"] = df["environment_log_abundance"]

    # Animal and food are sampled monthly: carry the last measurement forward,
    # which is exactly what an analyst reading a monthly report would do.
    for c in ("animal_p", "food_p", "env_x", "human_p"):
        df[c] = df.groupby(g, observed=True)[c].ffill()

    # ---- 2. apply reporting delays --------------------------------------- #
    delayed = {
        "human_p": REPORTING_DELAY_WEEKS["human"],
        "animal_p": REPORTING_DELAY_WEEKS["animal"],
        "food_p": REPORTING_DELAY_WEEKS["food"],
        "env_x": REPORTING_DELAY_WEEKS["environment"],
        "amu_vet": REPORTING_DELAY_WEEKS["amu_vet"],
        "amu_human": REPORTING_DELAY_WEEKS["amu_human"],
    }
    for col, delay in delayed.items():
        df[col] = _available(df, [col], delay)[col]

    # ---- 3. derived features per sector ---------------------------------- #
    FEATURE_GROUPS.clear()
    grp = df.groupby(g, observed=True)

    def add(sector: str, name: str, series: pd.Series) -> None:
        df[name] = series.to_numpy()
        FEATURE_GROUPS.setdefault(sector, []).append(name)

    for sector, col in (("human", "human_p"), ("animal", "animal_p"),
                        ("food", "food_p"), ("environment", "env_x")):
        add(sector, f"{sector}_level", grp[col].transform(lambda s: _roll(s, SHORT, "mean")))
        add(sector, f"{sector}_level_long", grp[col].transform(lambda s: _roll(s, LONG, "mean")))
        add(sector, f"{sector}_slope_s", grp[col].transform(lambda s: _slope(s, SHORT)))
        add(sector, f"{sector}_slope_m", grp[col].transform(lambda s: _slope(s, MEDIUM)))
        add(sector, f"{sector}_z", grp[col].transform(lambda s: _zscore(s, LONG)))
        add(sector, f"{sector}_accel",
            grp[col].transform(lambda s: _slope(s, SHORT) - _slope(s, MEDIUM)))

    # Antimicrobial-use pressure drives selection; include level and half-year change.
    for col in ("amu_vet", "amu_human"):
        sector = "animal" if col == "amu_vet" else "human"
        add(sector, f"{col}_level", grp[col].transform(lambda s: _roll(s, MEDIUM, "mean")))
        add(sector, f"{col}_change", grp[col].transform(lambda s: s - s.shift(26)))

    # Cross-sector leads: the gap between a reservoir sector and human clinical is
    # the quantity the whole system is built on, so give the model it directly.
    for sector in ("animal", "food", "environment"):
        add(sector, f"{sector}_minus_human_z", df[f"{sector}_z"] - df["human_z"])

    # ---- 4. spatial / network features ----------------------------------- #
    # Trade-weighted average of neighbouring regions' reservoir anomalies,
    # lagged so that a neighbour's uptick can only help *after* it was reported.
    # Both the anomaly z-scores (below) and the raw levels (here) are produced:
    # the deep encoders prefer levels, from which they can learn their own lag.
    for col in ("human_p", "animal_p", "food_p", "env_x"):
        wide_l = df.pivot_table(index=["week", "determinant"], columns="region",
                                values=col, observed=True)
        wide_l = wide_l.reindex(columns=range(trade.shape[0]))
        neigh_l = pd.DataFrame(
            np.nan_to_num(wide_l.to_numpy()) @ trade.T,
            index=wide_l.index, columns=wide_l.columns,
        )
        name = f"neighbour_{col}"
        long_l = neigh_l.stack().rename(name).reset_index()
        long_l.columns = ["week", "determinant", "region", name]
        df = df.merge(long_l, on=["week", "determinant", "region"], how="left")

    for sector in ("animal", "environment"):
        col = f"{sector}_z"
        wide = df.pivot_table(index=["week", "determinant"], columns="region",
                              values=col, observed=True)
        wide = wide.reindex(columns=range(trade.shape[0]))
        neigh = pd.DataFrame(
            np.nan_to_num(wide.to_numpy()) @ trade.T,
            index=wide.index, columns=wide.columns,
        )
        long = neigh.stack().rename(f"neighbour_{sector}_z").reset_index()
        long.columns = ["week", "determinant", "region", f"neighbour_{sector}_z"]
        df = df.merge(long, on=["week", "determinant", "region"], how="left")
        df[f"neighbour_{sector}_z"] = (
            df.sort_values(["region", "determinant", "week"])
              .groupby(g, observed=True)[f"neighbour_{sector}_z"].shift(4).to_numpy()
        )
        FEATURE_GROUPS.setdefault("network", []).append(f"neighbour_{sector}_z")

    # ---- 5. context: seasonality and surveillance capacity ---------------- #
    df["sin52"] = np.sin(2 * np.pi * df["week"] / 52.0)
    df["cos52"] = np.cos(2 * np.pi * df["week"] / 52.0)
    df["log_human_n"] = np.log1p(df["human_n"])
    FEATURE_GROUPS["context"] = ["sin52", "cos52", "temperature", "rainfall",
                                 "lab_capacity", "log_human_n"]

    # ---- 6. labels -------------------------------------------------------- #
    truth = {
        (e.region, e.determinant): e.human_emergence_week
        for e in events
        if getattr(e, "is_true_event", False) and getattr(e, "human_emergence_week", None) is not None
    }
    key = list(zip(df["region"], df["determinant"]))
    emergence = np.array([truth.get(k, -1) for k in key], dtype=float)
    emergence[emergence < 0] = np.nan
    df["emergence_week"] = emergence

    week = df["week"].to_numpy()
    for h in horizons:
        df[f"label_h{h}"] = (
            (~np.isnan(emergence)) & (emergence > week) & (emergence <= week + h)
        ).astype(int)

    # Rows at or after the emergence week: the event is no longer "emerging", it
    # has emerged. Excluded from training and from false-alarm counting.
    df["post_event"] = (~np.isnan(emergence)) & (week >= emergence)
    df["is_decoy_series"] = df.apply(
        lambda r: (r["region"], r["determinant"]) in {
            (e.region, e.determinant) for e in events if not getattr(e, "is_true_event", True)
        }, axis=1,
    )

    df = df.sort_values(["region", "determinant", "week"]).reset_index(drop=True)
    return df


def feature_columns(groups: dict[str, list[str]] | None = None,
                    drop_sectors: tuple[str, ...] = ()) -> list[str]:
    """Flatten :data:`FEATURE_GROUPS`, optionally omitting whole sectors (ablations)."""
    groups = groups or FEATURE_GROUPS
    cols: list[str] = []
    for sector, names in groups.items():
        if sector in drop_sectors:
            continue
        cols.extend(names)
    return cols


def clean_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Feature matrix with non-finite values replaced by 0 (models see a flat signal)."""
    x = df[cols].to_numpy(dtype=float)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------------------------------- #
# Raw lagged representation (for the representation-versus-model comparison)
# --------------------------------------------------------------------------- #
def add_raw_lag_features(df: pd.DataFrame, channels: list[str],
                         lags: tuple[int, ...] = tuple(range(0, 52, 4))) -> list[str]:
    """Expose the raw delay-adjusted series at fixed lags as plain columns.

    This is the representation the temporal network sees -- a window of raw
    sector series -- flattened so that a gradient-boosted model can consume the
    same information. It exists so that model class and input representation can
    be varied independently: Tier 1 changes *both* relative to Tier 2, so its
    advantage cannot otherwise be attributed to either.

    Deliberately no smoothing, no slopes, no z-scores. Whatever structure the
    trees find here, they find in the raw signal.
    """
    g = ["region", "determinant"]
    created: list[str] = []
    grouped = df.groupby(g, observed=True)
    for channel in channels:
        if channel not in df.columns:
            continue
        for lag in lags:
            name = f"rawlag_{channel}_{lag}"
            df[name] = grouped[channel].shift(lag).to_numpy() if lag else df[channel].to_numpy()
            created.append(name)
    return created
