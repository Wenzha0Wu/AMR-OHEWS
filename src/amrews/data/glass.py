"""WHO GLASS: national resistance and antimicrobial consumption, 2016-2023.

This module does a different job from :mod:`amrews.data.ncbi`, and keeping the
two apart is the point.

NCBI Pathogen Detection gives dated, sector-labelled isolates, which is what a
lead-time analysis needs -- but in practice only two countries contribute enough
multi-sector data to analyse. WHO GLASS gives about a hundred countries, but
**annually and nationally**.

Annual data cannot support a claim measured in weeks, and no result from this
module is used as one. What it can test is a different and equally load-bearing
One Health premise: that the antimicrobial-use *driver* leads the resistance
*outcome*. If national antibiotic consumption this year predicts national
resistance in later years, the mechanism the whole system assumes is operating
at population scale, across a hundred countries rather than two.

Indicators (WHO Global Health Observatory OData API, no credentials required):

``AMR_INFECT_ECOLI``
    Percentage of E. coli bloodstream infections resistant to third-generation
    cephalosporins.
``AMR_INFECT_MRSA``
    Percentage of S. aureus bloodstream infections that are methicillin-resistant.
``GLASSAMC_TC``
    Total antibacterial consumption, defined daily doses per 1000 inhabitants
    per day.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED, DATA_RAW, RESULTS

GHO_BASE = "https://ghoapi.azureedge.net/api"

RESISTANCE_INDICATORS = {
    "ecoli_3gc_resistance_pct": "AMR_INFECT_ECOLI",
    "mrsa_resistance_pct": "AMR_INFECT_MRSA",
}
CONSUMPTION_INDICATOR = "GLASSAMC_TC"


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
def fetch_indicator(code: str, force: bool = False) -> pd.DataFrame:
    """Download one GHO indicator, caching the raw JSON under ``data/raw``."""
    import requests

    cache = DATA_RAW / f"gho_{code}.json"
    if cache.exists() and not force and cache.stat().st_size > 100:
        payload = json.loads(cache.read_text())
    else:
        r = requests.get(f"{GHO_BASE}/{code}", timeout=180)
        r.raise_for_status()
        payload = r.json()
        cache.write_text(json.dumps(payload))
        print(f"fetched {code}: {len(payload['value'])} rows -> {cache.name}")

    rows = payload["value"]
    df = pd.DataFrame(rows)
    keep = {"SpatialDim": "iso3", "TimeDim": "year", "NumericValue": "value",
            "Dim1": "dim1", "ParentLocation": "who_region"}
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    df = df[df.get("iso3", pd.Series(dtype=str)).notna()]
    # Country rows only: GHO mixes in regional and global aggregates, which
    # would otherwise be double-counted alongside their member states.
    df = df[df["iso3"].str.len() == 3]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["year", "value"])


def build_panel(force: bool = False) -> pd.DataFrame:
    """Country-year panel of resistance outcomes plus antimicrobial consumption."""
    amc = fetch_indicator(CONSUMPTION_INDICATOR, force)
    # The consumption indicator is broken out by antibiotic class in Dim1;
    # total use is the row with no class dimension, else the sum across classes.
    if "dim1" in amc.columns and amc["dim1"].notna().any():
        total = amc[amc["dim1"].isna()]
        amc = total if len(total) else (
            amc.groupby(["iso3", "year"], as_index=False)["value"].sum())
    amc = (amc.groupby(["iso3", "year"], as_index=False)["value"].mean()
              .rename(columns={"value": "consumption_ddd_per_1000_per_day"}))

    frames = []
    for name, code in RESISTANCE_INDICATORS.items():
        res = fetch_indicator(code, force)
        res = (res.groupby(["iso3", "year"], as_index=False)["value"].mean()
                  .rename(columns={"value": "resistance_pct"}))
        res["indicator"] = name
        frames.append(res)
    resistance = pd.concat(frames, ignore_index=True)

    panel = resistance.merge(amc, on=["iso3", "year"], how="left")
    return panel.sort_values(["indicator", "iso3", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Driver analysis
# --------------------------------------------------------------------------- #
def consumption_leads_resistance(panel: pd.DataFrame, max_lag: int = 3,
                                 min_years: int = 5) -> dict:
    """Does national antibiotic consumption lead national resistance?

    Per country and indicator, correlate year-on-year *changes* in consumption
    against changes in resistance at lags from ``-max_lag`` to ``+max_lag``.
    Differencing removes the shared secular trend, without which almost any two
    rising surveillance series correlate at every lag.

    The scan is symmetric for the same reason as in :mod:`amrews.data.ncbi`: a
    one-sided scan makes "the driver leads" true by construction. The reported
    test is whether the consumption-leads direction beats the reverse direction
    more often than a coin flip.
    """
    findings = []
    for (indicator, iso3), block in panel.groupby(["indicator", "iso3"]):
        block = block.dropna(subset=["resistance_pct", "consumption_ddd_per_1000_per_day"])
        block = block.sort_values("year")
        if len(block) < min_years:
            continue
        rd = block["resistance_pct"].diff().dropna()
        cd = block["consumption_ddd_per_1000_per_day"].diff().dropna()
        n = min(len(rd), len(cd))
        if n < 4:
            continue
        rd, cd = rd.to_numpy()[-n:], cd.to_numpy()[-n:]

        by_lag = {}
        for lag in range(-max_lag, max_lag + 1):
            a, b = (cd, rd) if lag == 0 else (
                (cd[:len(cd) - lag], rd[lag:]) if lag > 0
                else (cd[-lag:], rd[:len(rd) + lag]))
            m = min(len(a), len(b))
            if m < 3 or np.std(a[:m]) == 0 or np.std(b[:m]) == 0:
                continue
            by_lag[lag] = float(np.corrcoef(a[:m], b[:m])[0, 1])
        if not by_lag:
            continue
        pos = [v for k, v in by_lag.items() if k > 0]
        neg = [v for k, v in by_lag.items() if k < 0]
        best = max(by_lag, key=lambda k: by_lag[k])
        findings.append({
            "indicator": indicator, "iso3": iso3, "n_years": int(len(block)),
            "best_lag_years": int(best), "best_correlation": round(by_lag[best], 3),
            "max_corr_consumption_leads": round(max(pos), 3) if pos else None,
            "max_corr_resistance_leads": round(max(neg), 3) if neg else None,
        })

    paired = [(f["max_corr_consumption_leads"], f["max_corr_resistance_leads"])
              for f in findings
              if f["max_corr_consumption_leads"] is not None
              and f["max_corr_resistance_leads"] is not None]
    wins = sum(1 for a, b in paired if a > b)
    p = None
    if paired:
        from scipy.stats import binomtest

        p = float(binomtest(wins, len(paired), 0.5, alternative="greater").pvalue)

    return {
        "n_series": len(findings),
        "n_countries": len({f["iso3"] for f in findings}),
        "direction_test": {"n_series": len(paired), "consumption_leads_wins": wins,
                           "binomial_p_one_sided": p},
        "median_best_lag_years": float(np.median([f["best_lag_years"] for f in findings]))
        if findings else None,
        "findings": sorted(findings, key=lambda f: -f["best_correlation"])[:60],
        "resolution_caveat": (
            "Annual, national data. This tests whether the antimicrobial-use driver "
            "leads the resistance outcome; it cannot and does not support a lead-time "
            "claim measured in weeks."
        ),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="WHO GLASS driver analysis.")
    ap.add_argument("--force", action="store_true", help="re-download indicators")
    args = ap.parse_args()

    panel = build_panel(force=args.force)
    panel.to_csv(DATA_PROCESSED / "glass_panel.csv", index=False)
    print(f"panel: {len(panel):,} country-year rows, "
          f"{panel['iso3'].nunique()} countries, "
          f"{int(panel['year'].min())}-{int(panel['year'].max())}")
    print(panel.groupby("indicator")["iso3"].nunique().to_string())

    study = consumption_leads_resistance(panel)
    (RESULTS / "glass_driver_study.json").write_text(json.dumps(study, indent=2, default=float))

    dt = study["direction_test"]
    print(f"\n{study['n_series']} country-indicator series across "
          f"{study['n_countries']} countries")
    print(f"consumption-leads beats resistance-leads in "
          f"{dt['consumption_leads_wins']}/{dt['n_series']} series, "
          f"one-sided binomial p = {dt['binomial_p_one_sided']:.4g}")
    print(f"median best lag: {study['median_best_lag_years']} years")
    print(f"\n{study['resolution_caveat']}")


if __name__ == "__main__":
    main()
