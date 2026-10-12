"""US state x month panel from NCBI Pathogen Detection.

Built to answer one question the rest of the project could not: **would the deep
temporal and graph tiers work on real data if the data had the shape they need?**

Tiers 2 and 3 have failed everywhere so far, but never on favourable ground.
They failed on a simulated world of ~60 events, and on a real panel of 221
country-quarter series with no usable adjacency between countries. Neither is a
fair test of an architecture whose whole premise is many nodes, long series and
real spatial structure.

NCBI carries geography and dates that the country-level pipeline discards. Of
the isolates this module retains, **82,090 have both a US state and a
month-precision collection date** across the three organisms -- roughly fifty
nodes, two decades of months, and genuine geography for which real adjacency and
real livestock movement exist. That is the most favourable real testbed this
project can construct, which is exactly why it is the right place to settle the
question.

The normalisation is doing real work: the `geo_loc_name` field carries 390
distinct subnational strings for the US, mixing two-letter codes ('CA'), full
names ('California'), bare cities ('Houston'), city-state pairs ('Pittsburgh,
PA'), counties ('Maricopa County, Arizona') and regions ('Northeast'). Anything
that cannot be resolved to exactly one state is dropped rather than guessed.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED, DATA_RAW
from .ncbi import DETERMINANT_PATTERNS, USECOLS, assign_sector

STATE_CODES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin",
    "WY": "Wyoming", "DC": "District of Columbia",
}
NAME_TO_CODE = {v.lower(): k for k, v in STATE_CODES.items()}

#: Cities frequent enough in the archive to be worth resolving. Anything not
#: listed and not otherwise resolvable is dropped.
CITY_TO_CODE = {
    "houston": "TX", "chicago": "IL", "boston": "MA", "new york city": "NY",
    "pittsburgh": "PA", "san diego": "CA", "ann arbor": "MI", "seattle": "WA",
    "atlanta": "GA", "philadelphia": "PA", "los angeles": "CA", "baltimore": "MD",
    "denver": "CO", "minneapolis": "MN", "detroit": "MI", "phoenix": "AZ",
}


def normalise_state(value: str) -> str | None:
    """Resolve a free-text subnational string to a two-letter state code, or None."""
    if not isinstance(value, str):
        return None
    s = value.strip().strip(".")
    if not s:
        return None

    # 'Pittsburgh, PA' / 'Maricopa County, Arizona' -- the state is the last part.
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for candidate in reversed(parts):
        c = candidate.strip()
        if c.upper() in STATE_CODES:
            return c.upper()
        if c.lower() in NAME_TO_CODE:
            return NAME_TO_CODE[c.lower()]
        cleaned = re.sub(r"\s+(county|parish|city)$", "", c, flags=re.I).strip().lower()
        if cleaned in NAME_TO_CODE:
            return NAME_TO_CODE[cleaned]
        if cleaned in CITY_TO_CODE:
            return CITY_TO_CODE[cleaned]
    return None


def load_us_isolates(organisms: dict[str, Path] | None = None) -> pd.DataFrame:
    """Read the cached NCBI archives and keep US isolates with month-precision dates."""
    if organisms is None:
        organisms = {}
        for p in sorted(DATA_RAW.glob("*.amr.metadata.tsv")):
            organisms.setdefault(p.name.split("_")[0], p)

    frames = []
    for organism, path in organisms.items():
        df = pd.read_csv(path, sep="\t", usecols=lambda c: c in USECOLS,
                         dtype=str, on_bad_lines="skip")
        geo = df["geo_loc_name"].fillna("")
        df = df[geo.str.startswith("USA")].copy()
        df["state"] = df["geo_loc_name"].str.split(":").str[1].map(normalise_state)

        date = df["collection_date"].fillna("")
        month = date.str.extract(r"^(\d{4})-(\d{2})")
        df["year"] = pd.to_numeric(month[0], errors="coerce")
        df["month"] = pd.to_numeric(month[1], errors="coerce")

        df = df.dropna(subset=["state", "year", "month"])
        df = df[(df["year"] >= 1995) & (df["year"] <= 2030)
                & (df["month"] >= 1) & (df["month"] <= 12)]
        df["organism"] = organism
        frames.append(df)
        print(f"  {organism}: {len(df):,} US isolates with state and month")

    out = pd.concat(frames, ignore_index=True)
    out["period"] = (out["year"].astype(int) * 12 + out["month"].astype(int) - 1)
    for col in ("host", "epi_type", "isolation_source"):
        if col not in out.columns:
            out[col] = ""
    out["sector"] = [assign_sector(h, e, s) for h, e, s
                     in zip(out["host"], out["epi_type"], out["isolation_source"])]
    out["AMR_genotypes"] = out["AMR_genotypes"].fillna("")
    return out


def build_state_month_panel(isolates: pd.DataFrame, min_isolates: int = 5) -> pd.DataFrame:
    """Long panel of state x month x determinant x sector resistance prevalence."""
    rows = []
    for det, pattern in DETERMINANT_PATTERNS.items():
        present = isolates["AMR_genotypes"].str.contains(pattern).astype(int)
        tmp = isolates.assign(present=present)
        g = (tmp.groupby(["state", "sector", "period", "year", "month"], observed=True)
                .agg(n=("present", "size"), n_present=("present", "sum"))
                .reset_index())
        g["determinant"] = det
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["sector"] != "unknown"]
    panel = panel[panel["n"] >= min_isolates]
    panel["prevalence"] = (panel["n_present"] + 0.5) / (panel["n"] + 1.0)
    return panel.reset_index(drop=True)


def describe(panel: pd.DataFrame) -> dict:
    """Node, timestep and density counts -- checked before any modelling."""
    return {
        "rows": int(len(panel)),
        "states": int(panel["state"].nunique()),
        "months": int(panel["period"].nunique()),
        "year_range": [int(panel["year"].min()), int(panel["year"].max())],
        "determinants": int(panel["determinant"].nunique()),
        "sectors": panel["sector"].value_counts().to_dict(),
        "median_n_per_cell": float(panel["n"].median()),
        "state_month_cells": int(panel.groupby(["state", "period"]).ngroups),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the US state x month panel.")
    ap.add_argument("--min-isolates", type=int, default=5)
    args = ap.parse_args()

    iso = load_us_isolates()
    print(f"\n{len(iso):,} US isolates with state and month-precision dates")
    print(iso["sector"].value_counts().to_string())

    panel = build_state_month_panel(iso, min_isolates=args.min_isolates)
    panel.to_csv(DATA_PROCESSED / "us_state_month_panel.csv", index=False)
    info = describe(panel)
    print(f"\npanel -> data/processed/us_state_month_panel.csv")
    for k, v in info.items():
        print(f"  {k}: {v}")

    per_sector = (panel.groupby(["state", "sector"]).size().unstack(fill_value=0))
    both = per_sector[(per_sector.get("human", 0) > 0)
                      & (per_sector.drop(columns=["human"], errors="ignore").sum(axis=1) > 0)]
    print(f"\nstates with both a human and a reservoir sector: {len(both)}")


if __name__ == "__main__":
    main()
