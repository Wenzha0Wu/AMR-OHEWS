"""Real-data One Health panel and emergence labels from NCBI Pathogen Detection.

This module promotes real data from a premise check into the evaluation itself.
It builds the same kind of object :mod:`amrews.data.features` builds for the
simulator -- a causal feature table with forward-looking labels -- so that
:func:`amrews.models.baselines.run_baselines` and
:class:`amrews.models.fusion.FusionModel` run on it unchanged.

Three differences from the simulated pipeline, all forced by what real data is:

**Quarters, not weeks.** Isolate collection dates are dense enough for quarterly
prevalence per country and sector, and not for weekly. Every window, lag and
match tolerance below is in quarters.

**Countries, not regions.** The unit is (country, determinant). There is no
trade matrix and no plausible substitute, so the network feature group is empty
and the graph tier does not run here.

**A constructed label.** The simulator knows when resistance began to establish
because it put it there. Real archives do not record that, so the label is
defined as a sustained rise in the observed human clinical series.

> That last point is the central caveat and it must travel with every number
> this module produces. The label is built from exactly the signal the Tier-0
> clinical baselines are designed to detect, which **structurally advantages the
> comparator**. A One Health model that still wins against it is winning against
> a favourable opponent; one that does not has not necessarily failed. The bias
> is stated rather than corrected, because there is no honest way to correct it
> without inventing the ground truth we lack.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.features import FEATURE_GROUPS, _roll, _slope, _zscore

# Windows in quarters: one year, two years, five years.
SHORT, MEDIUM, LONG = 4, 8, 20

#: Reporting delay in quarters. Genomic surveillance submits in batches, and
#: clinical isolates reach public archives later than programme-sampled food and
#: animal isolates do.
REAL_DELAYS = {"human": 1, "animal": 1, "food": 1, "environment": 0}

SECTOR_COLS = {"human": "human_p", "animal": "animal_p",
               "food": "food_p", "environment": "env_x"}


@dataclass
class RealEvent:
    """An emergence episode detected in a real human clinical series.

    Field names deliberately mirror :class:`~amrews.data.simulate.EmergenceEvent`
    so the evaluation machinery in :mod:`amrews.eval.metrics` accepts both.
    """

    region: str                 # country
    determinant: str
    human_emergence_week: int   # period index, not a week
    is_true_event: bool = True
    seeded_by_trade: bool = False
    baseline_prevalence: float = float("nan")
    peak_prevalence: float = float("nan")


# --------------------------------------------------------------------------- #
# Panel construction
# --------------------------------------------------------------------------- #
def build_wide_panel(panel: pd.DataFrame, min_human: int = 12,
                     min_reservoir: int = 8) -> pd.DataFrame:
    """Pivot the long NCBI panel into one row per (country, determinant, period).

    Series are kept only where the human sector and at least one reservoir
    sector are both observed often enough to support a comparison; a series with
    two animal observations cannot inform anything and would only add noise to
    the false-alarm denominator.
    """
    wide = panel.pivot_table(index=["country", "determinant", "period"],
                             columns="sector", values=["prevalence", "n"],
                             observed=True)
    wide.columns = [f"{a}__{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    rename = {f"prevalence__{s}": col for s, col in SECTOR_COLS.items()}
    rename.update({f"n__{s}": f"{s}_n" for s in SECTOR_COLS})
    wide = wide.rename(columns=rename)
    for col in list(SECTOR_COLS.values()) + [f"{s}_n" for s in SECTOR_COLS]:
        if col not in wide:
            wide[col] = np.nan

    counts = wide.groupby(["country", "determinant"]).agg(
        n_human=("human_p", lambda s: s.notna().sum()),
        n_animal=("animal_p", lambda s: s.notna().sum()),
        n_food=("food_p", lambda s: s.notna().sum()),
        n_env=("env_x", lambda s: s.notna().sum()),
    ).reset_index()
    counts["n_reservoir"] = counts[["n_animal", "n_food", "n_env"]].max(axis=1)
    keep = counts[(counts["n_human"] >= min_human)
                  & (counts["n_reservoir"] >= min_reservoir)][["country", "determinant"]]
    wide = wide.merge(keep, on=["country", "determinant"], how="inner")

    # Reindex each series onto a contiguous period grid: rolling windows and
    # lags are meaningless across a gap that is silently closed up.
    filled = []
    for (country, det), block in wide.groupby(["country", "determinant"], observed=True):
        grid = pd.DataFrame({"period": range(int(block["period"].min()),
                                             int(block["period"].max()) + 1)})
        grid["country"], grid["determinant"] = country, det
        filled.append(grid.merge(block, on=["country", "determinant", "period"], how="left"))
    wide = pd.concat(filled, ignore_index=True)

    wide["year"] = wide["period"] // 4
    wide["quarter"] = wide["period"] % 4 + 1
    return wide.sort_values(["country", "determinant", "period"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def find_real_events(wide: pd.DataFrame, baseline_periods: int = 8,
                     k_sd: float = 2.0, min_absolute_rise: float = 0.01,
                     sustain: int = 2, refractory: int = 12) -> list[RealEvent]:
    """Locate sustained rises in each real human clinical series.

    An emergence is the first period at which the smoothed human prevalence
    stands more than ``k_sd`` trailing standard deviations *and* one percentage
    point above its own trailing baseline, and stays there for ``sustain``
    consecutive periods. Requiring both a relative and an absolute rise matters:
    a series that has been flat at 0.3% will clear any z-score threshold on
    noise alone.

    Unlike the simulator, a series may carry **several** episodes separated by at
    least ``refractory`` quarters. Over twenty-five years of real surveillance a
    country-determinant pair genuinely can see resistance rise more than once,
    and taking only the first episode per series put almost every event in the
    distant past -- leaving nine events in the held-out window, far too few to
    put an interval around.
    """
    events: list[RealEvent] = []
    for (country, det), block in wide.groupby(["country", "determinant"], observed=True):
        block = block.sort_values("period")
        series = block["human_p"].astype(float)
        if series.notna().sum() < baseline_periods + sustain + 2:
            continue
        smooth = series.interpolate(limit_direction="both").rolling(
            3, min_periods=1, center=False).mean()
        base_mu = smooth.shift(1).rolling(baseline_periods, min_periods=4).mean()
        base_sd = smooth.shift(1).rolling(baseline_periods, min_periods=4).std()

        above = ((smooth > base_mu + k_sd * base_sd.fillna(np.inf))
                 & (smooth > base_mu + min_absolute_rise)).to_numpy()
        periods = block["period"].to_numpy()
        i, n = 0, len(above)
        while i <= n - sustain:
            if above[i:i + sustain].all():
                events.append(RealEvent(
                    region=country, determinant=det,
                    human_emergence_week=int(periods[i]),
                    baseline_prevalence=float(np.nan_to_num(base_mu.iloc[i])),
                    peak_prevalence=float(np.nanmax(smooth.iloc[i:])),
                ))
                i += refractory
            else:
                i += 1
    return events


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def build_real_features(wide: pd.DataFrame, events: list[RealEvent],
                        horizon: int = 4) -> pd.DataFrame:
    """Causal features and forward labels, matching the simulated schema.

    Column names are deliberately those the rest of the codebase expects --
    ``week`` for the time index and ``region`` for the unit -- so the baselines,
    the fusion model and the whole of :mod:`amrews.eval.metrics` run unmodified.
    The time unit here is a quarter.
    """
    df = wide.rename(columns={"period": "week", "country": "region"}).copy()
    df = df.sort_values(["region", "determinant", "week"]).reset_index(drop=True)
    g = ["region", "determinant"]

    for col in SECTOR_COLS.values():
        df[col] = df.groupby(g, observed=True)[col].ffill()

    # Reporting delays first, so no feature can contain unpublished data.
    for sector, col in SECTOR_COLS.items():
        delay = REAL_DELAYS[sector]
        if delay:
            df[col] = df.groupby(g, observed=True)[col].shift(delay)

    FEATURE_GROUPS.clear()
    grp = df.groupby(g, observed=True)

    def add(sector: str, name: str, series: pd.Series) -> None:
        df[name] = series.to_numpy()
        FEATURE_GROUPS.setdefault(sector, []).append(name)

    for sector, col in SECTOR_COLS.items():
        add(sector, f"{sector}_level", grp[col].transform(lambda s: _roll(s, SHORT, "mean")))
        add(sector, f"{sector}_level_long", grp[col].transform(lambda s: _roll(s, LONG, "mean")))
        add(sector, f"{sector}_slope_s", grp[col].transform(lambda s: _slope(s, SHORT)))
        add(sector, f"{sector}_slope_m", grp[col].transform(lambda s: _slope(s, MEDIUM)))
        add(sector, f"{sector}_z", grp[col].transform(lambda s: _zscore(s, LONG)))

    for sector in ("animal", "food", "environment"):
        add(sector, f"{sector}_minus_human_z", df[f"{sector}_z"] - df["human_z"])

    df["log_human_n"] = np.log1p(df["human_n"].fillna(0))
    df["quarter_sin"] = np.sin(2 * np.pi * (df["week"] % 4) / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * (df["week"] % 4) / 4)
    FEATURE_GROUPS["context"] = ["log_human_n", "quarter_sin", "quarter_cos"]
    # No trade matrix exists for countries here, so there is no network group and
    # the graph tier is not applicable to the real panel.

    # A series can carry several episodes, so the label is "is any emergence due
    # in the next H quarters" and post-event masking is windowed rather than
    # permanent -- after an episode has passed, the series becomes eligible again.
    by_series: dict[tuple[str, str], list[int]] = {}
    for e in events:
        by_series.setdefault((e.region, e.determinant), []).append(e.human_emergence_week)

    week = df["week"].to_numpy()
    key = list(zip(df["region"], df["determinant"]))
    label = np.zeros(len(df), dtype=int)
    post = np.zeros(len(df), dtype=bool)
    next_emergence = np.full(len(df), np.nan)
    for i, k in enumerate(key):
        weeks_of_events = by_series.get(k)
        if not weeks_of_events:
            continue
        t = week[i]
        upcoming = [w for w in weeks_of_events if w > t]
        if upcoming:
            next_emergence[i] = min(upcoming)
            label[i] = int(min(upcoming) <= t + horizon)
        post[i] = any(w <= t < w + horizon for w in weeks_of_events)
    df["emergence_week"] = next_emergence
    df[f"label_h{horizon}"] = label
    df["post_event"] = post
    return df


def load_real_panel(path=None, horizon: int = 4, **kw):
    """Convenience: long NCBI panel on disk -> (features, events, wide panel)."""
    from ..config import DATA_PROCESSED

    path = path or DATA_PROCESSED / "ncbi_panel.csv"
    panel = pd.read_csv(path)
    wide = build_wide_panel(panel, **kw)
    events = find_real_events(wide)
    features = build_real_features(wide, events, horizon=horizon)
    return features, events, wide
