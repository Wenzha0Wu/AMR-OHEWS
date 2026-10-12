"""Real-data slice: NCBI Pathogen Detection as a One Health archive.

The simulator can prove that *a* multi-sector model warns earlier in a world
where the reservoir genuinely leads. It cannot prove that the real world is such
a world. This module tests the premise on genuine data.

NCBI Pathogen Detection is unusual and valuable here because a single table
carries isolates from every One Health sector -- human clinical specimens,
livestock, retail meat and environmental samples -- with collection dates,
geography and AMRFinderPlus resistance genotypes, all processed through one
analytic pipeline. Most alternatives (WHO GLASS, ECDC EARS-Net, ESVAC, EFSA)
publish each sector separately, at annual resolution, with incompatible
denominators; cross-sector lead time cannot be measured across them without
heroic assumptions.

The analysis is deliberately modest: per-sector quarterly prevalence of a
resistance determinant, then the cross-correlation lag at which a reservoir
sector best predicts the human sector. A positive lag is consistent with the
reservoir leading. It is observational, confounded by sampling programmes that
changed over time, and is reported as supporting evidence rather than proof.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED, DATA_RAW, NCBI_BASE, NCBI_ORGANISM, NCBI_SNAPSHOT, RESULTS

# --------------------------------------------------------------------------- #
# Sector assignment
# --------------------------------------------------------------------------- #
_FOOD_PAT = re.compile(
    r"retail|meat|carcass|chicken breast|ground |raw milk|cheese|produce|lettuce|"
    r"spinach|food|poultry product|processing plant", re.I)
_ENV_PAT = re.compile(
    r"water|river|stream|lake|sewage|wastewater|sludge|effluent|soil|sediment|"
    r"manure|lagoon|environment|surface", re.I)
_HUMAN_PAT = re.compile(r"stool|faeces of human|blood|urine|wound|csf|rectal swab|patient", re.I)
_ANIMAL_HOSTS = re.compile(
    r"gallus|bos taurus|sus scrofa|meleagris|ovis|capra|equus|canis|felis|turkey|"
    r"cattle|swine|pig|bovine|porcine|avian|broiler|layer|duck|sheep|goat", re.I)


def assign_sector(host: str, epi_type: str, source: str) -> str:
    """Map an isolate's free-text provenance onto a One Health sector.

    Order matters. Human host wins outright; otherwise the isolation source
    decides between food and environment (a chicken carcass at retail is food
    surveillance, a river sample is environmental) before falling back to the
    host species. Anything left over is ``unknown`` and is excluded rather than
    guessed -- silently binning ambiguous isolates would bias exactly the
    cross-sector comparison this analysis exists to make.
    """
    # Missing cells arrive from pandas as float NaN, not as None or "".
    host = "" if not isinstance(host, str) else host.strip()
    source = "" if not isinstance(source, str) else source.strip()
    epi = ("" if not isinstance(epi_type, str) else epi_type).strip().lower()

    if re.search(r"homo sapiens|human", host, re.I):
        return "human"
    if _HUMAN_PAT.search(source) and not _ANIMAL_HOSTS.search(host):
        return "human"
    if _ENV_PAT.search(source):
        return "environment"
    if _FOOD_PAT.search(source):
        return "food"
    if _ANIMAL_HOSTS.search(host) or _ANIMAL_HOSTS.search(source):
        return "animal"
    if epi == "clinical":
        return "human"
    return "unknown"


#: Resistance determinants tracked in the real-data case study, as regexes over
#: the AMRFinderPlus genotype strings.
DETERMINANT_PATTERNS = {
    # Campylobacter-led determinants (kept from the original case study)
    "FQ_gyrA_T86I": re.compile(r"gyrA_T86I", re.I),
    "TET_tet(O)": re.compile(r"\btet\(O\)", re.I),
    "MAC_erm(B)": re.compile(r"\berm\(B\)", re.I),
    "BLA_blaOXA-61": re.compile(r"blaOXA-61", re.I),
    "AMG_aph(3\')-III": re.compile(r"aph\(3'\)-III", re.I),
    # Enterobacteriaceae determinants, for Salmonella and Klebsiella. These are
    # the WHO-priority resistances with a genuine One Health transmission story:
    # ESBLs and plasmid-mediated colistin resistance both moved through food
    # animals before they mattered in hospitals.
    "ESBL_blaCTX-M": re.compile(r"blaCTX-M", re.I),
    "COL_mcr": re.compile(r"\bmcr-\d", re.I),
    "CARB_blaKPC": re.compile(r"blaKPC", re.I),
    "CARB_blaNDM": re.compile(r"blaNDM", re.I),
    "TET_tet(A)": re.compile(r"\btet\(A\)", re.I),
    "SUL_sul1": re.compile(r"\bsul1\b", re.I),
    "AMG_aac(3)-IV": re.compile(r"aac\(3\)-IV", re.I),
    "FQ_qnrS": re.compile(r"\bqnrS", re.I),
}


# --------------------------------------------------------------------------- #
# Download and parse
# --------------------------------------------------------------------------- #
def latest_snapshot(organism: str = NCBI_ORGANISM) -> str:
    """Resolve the newest ``PDG`` snapshot directory for an organism."""
    import requests

    r = requests.get(f"{NCBI_BASE}/{organism}/", timeout=60)
    r.raise_for_status()
    snaps = sorted(set(re.findall(r"(PDG\d+\.\d+)/", r.text)),
                   key=lambda s: [int(x) for x in s[3:].split(".")])
    if not snaps:
        raise RuntimeError(f"no snapshots listed for {organism}")

    # The newest directory is not necessarily usable: NCBI publishes the snapshot
    # folder before the AMR metadata file inside it is built, so the most recent
    # snapshot 404s for days at a time (observed for both Salmonella and
    # E. coli). Walk backwards until a snapshot whose AMR file actually exists.
    for snapshot in reversed(snaps[-8:]):
        url = f"{NCBI_BASE}/{organism}/{snapshot}/AMR/{snapshot}.amr.metadata.tsv"
        try:
            head = requests.head(url, timeout=60, allow_redirects=True)
        except requests.RequestException:
            continue
        if head.status_code == 200:
            if snapshot != snaps[-1]:
                print(f"{organism}: newest snapshot {snaps[-1]} has no AMR file yet, "
                      f"falling back to {snapshot}")
            return snapshot
    raise RuntimeError(f"no snapshot of {organism} has a published AMR metadata file")


def download(organism: str = NCBI_ORGANISM, snapshot: str | None = NCBI_SNAPSHOT,
             force: bool = False) -> tuple[Path, str]:
    """Fetch the AMR metadata TSV for a snapshot, caching it under ``data/raw``."""
    import requests

    snapshot = snapshot or latest_snapshot(organism)
    url = f"{NCBI_BASE}/{organism}/{snapshot}/AMR/{snapshot}.amr.metadata.tsv"
    dest = DATA_RAW / f"{organism}_{snapshot}.amr.metadata.tsv"
    if dest.exists() and not force and dest.stat().st_size > 1_000_000:
        print(f"using cached {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest, snapshot

    print(f"downloading {url}")
    # Download to a temporary name and move into place only on success, so an
    # interrupted transfer cannot leave a truncated file that the cache check
    # above would happily accept next time.
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 22):
                fh.write(chunk)
                done += len(chunk)
                if total and done % (1 << 28) < (1 << 22):
                    print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", flush=True)
    tmp.replace(dest)
    print(f"saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest, snapshot


#: Columns the analysis needs. The archives are wide (49-59 columns) and up to
#: 858 MB; everything else is dropped at parse time.
USECOLS = ["collection_date", "geo_loc_name", "host", "epi_type",
           "isolation_source", "AMR_genotypes", "target_acc"]


def load_isolates(path: Path, chunksize: int = 250_000) -> pd.DataFrame:
    """Read the TSV in chunks and reduce it to the columns the analysis needs.

    Chunked rather than one shot: the Salmonella archive is 858 MB across ~59
    columns, and reading it whole to keep seven of them is a needless multi-GB
    spike on a machine that may be running model training at the same time.
    """
    frames = []
    reader = pd.read_csv(path, sep="\t", usecols=lambda c: c in USECOLS,
                         dtype=str, on_bad_lines="skip", chunksize=chunksize)
    for chunk in reader:
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    del frames

    # Collection dates arrive as YYYY, YYYY-MM or YYYY-MM-DD, plus a long tail of
    # ranges and free text. Anything without a usable year is dropped.
    year = df["collection_date"].str.extract(r"(\d{4})")[0].astype("float")
    month = df["collection_date"].str.extract(r"^\d{4}-(\d{2})")[0].astype("float")
    df["year"] = year
    df["month"] = month.fillna(7.0)          # undated within a year -> mid-year
    df = df[(df["year"] >= 2000) & (df["year"] <= 2030)].copy()
    df["quarter"] = ((df["month"] - 1) // 3 + 1).astype(int)
    df["period"] = df["year"].astype(int) * 4 + df["quarter"] - 1

    df["country"] = df["geo_loc_name"].fillna("").str.split(":").str[0].str.strip()
    for col in ("host", "epi_type", "isolation_source"):
        if col not in df.columns:
            df[col] = ""
    df["sector"] = [
        assign_sector(h, e, s)
        for h, e, s in zip(df["host"], df["epi_type"], df["isolation_source"])
    ]
    df["AMR_genotypes"] = df["AMR_genotypes"].fillna("")
    return df


def build_panel(df: pd.DataFrame, min_isolates: int = 20) -> pd.DataFrame:
    """Quarterly determinant prevalence per country x sector."""
    rows = []
    for det, pat in DETERMINANT_PATTERNS.items():
        present = df["AMR_genotypes"].str.contains(pat).astype(int)
        tmp = df.assign(present=present)
        g = (tmp.groupby(["country", "sector", "period", "year", "quarter"], observed=True)
                .agg(n=("present", "size"), n_present=("present", "sum"))
                .reset_index())
        g["determinant"] = det
        rows.append(g)
    panel = pd.concat(rows, ignore_index=True)
    panel = panel[panel["sector"] != "unknown"]
    # Isolates with no geographic annotation carry an empty country string, which
    # groupby treats as a country in its own right -- it was competing for a slot
    # in the top-N country selection and displacing a real one, which made the
    # analysis count depend on whether the panel had been round-tripped through
    # CSV (where "" becomes NaN and is dropped instead).
    panel = panel[panel["country"].notna() & (panel["country"].str.strip() != "")]
    panel["prevalence"] = (panel["n_present"] + 0.5) / (panel["n"] + 1.0)
    # Quarters with tiny denominators are noise, not signal.
    return panel[panel["n"] >= min_isolates].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Case study: does the reservoir lead the clinic?
# --------------------------------------------------------------------------- #
def cross_correlation_lag(panel: pd.DataFrame, country: str, determinant: str,
                          reservoir: str, max_lag: int = 6) -> dict:
    """Lag (in quarters) at which ``reservoir`` prevalence best tracks human prevalence.

    A positive best lag means the reservoir series has to be *delayed* to line up
    with the human series -- i.e. the reservoir moved first.

    The scan is deliberately **symmetric**, covering ``-max_lag .. +max_lag``.
    Scanning only non-negative lags, as a first version of this did, makes the
    headline finding meaningless: with nine candidate lags and only one of them
    equal to zero, "the reservoir leads" comes out true about nine times in ten
    by construction. Against a symmetric window the null is a coin flip, and the
    count of positive-lag series carries information.
    """
    sub = panel[(panel["country"] == country) & (panel["determinant"] == determinant)]
    h = sub[sub["sector"] == "human"].set_index("period")["prevalence"].sort_index()
    r = sub[sub["sector"] == reservoir].set_index("period")["prevalence"].sort_index()
    if len(h) < 8 or len(r) < 8:
        return {"country": country, "determinant": determinant, "reservoir": reservoir,
                "status": "insufficient_data", "n_human": len(h), "n_reservoir": len(r)}

    idx = range(min(h.index.min(), r.index.min()), max(h.index.max(), r.index.max()) + 1)
    h = h.reindex(idx).interpolate(limit_direction="both")
    r = r.reindex(idx).interpolate(limit_direction="both")
    # Differencing removes the shared secular trend, without which almost any two
    # rising surveillance series correlate at every lag.
    hd, rd = h.diff().dropna(), r.diff().dropna()
    common = hd.index.intersection(rd.index)
    hd, rd = hd.loc[common], rd.loc[common]

    out = {}
    for lag in range(-max_lag, max_lag + 1):
        # lag > 0: reservoir leads (delay it to match the human series).
        # lag < 0: the human series leads, the control direction.
        a, b = (rd, hd) if lag == 0 else (
            (rd.iloc[:len(rd) - lag], hd.iloc[lag:]) if lag > 0
            else (rd.iloc[-lag:], hd.iloc[:len(hd) + lag])
        )
        if len(a) < 6 or len(b) < 6 or a.std() == 0 or b.std() == 0:
            continue
        n = min(len(a), len(b))
        out[lag] = float(np.corrcoef(a.to_numpy()[:n], b.to_numpy()[:n])[0, 1])
    if not out:
        return {"country": country, "determinant": determinant, "reservoir": reservoir,
                "status": "degenerate"}

    best = max(out, key=lambda k: out[k])
    pos = [v for k, v in out.items() if k > 0]
    neg = [v for k, v in out.items() if k < 0]
    return {
        "country": country, "determinant": determinant, "reservoir": reservoir,
        "status": "ok", "n_quarters": int(len(common)),
        "best_lag_quarters": int(best), "best_correlation": round(out[best], 3),
        "correlation_by_lag": {str(k): round(v, 3) for k, v in out.items()},
        "lag_zero_correlation": round(out.get(0, float("nan")), 3),
        # Head-to-head between the two directions: the asymmetry, not the raw
        # count of positive lags, is what carries evidence.
        "max_corr_reservoir_leads": round(max(pos), 3) if pos else None,
        "max_corr_human_leads": round(max(neg), 3) if neg else None,
    }


def run_case_study(panel: pd.DataFrame, min_quarters: int = 8, top_countries: int = 10) -> dict:
    """Run the lag analysis for every well-sampled country/determinant/reservoir."""
    counts = (panel[panel["sector"] == "human"]
              .groupby("country", observed=True)["n"].sum().sort_values(ascending=False))
    countries = [c for c in counts.index[:top_countries] if c]

    findings = []
    for country in countries:
        for det in DETERMINANT_PATTERNS:
            for reservoir in ("animal", "food", "environment"):
                res = cross_correlation_lag(panel, country, det, reservoir)
                if res.get("status") == "ok" and res["n_quarters"] >= min_quarters:
                    findings.append(res)

    lags = [f["best_lag_quarters"] for f in findings]
    leading = [f for f in findings if f["best_lag_quarters"] > 0]

    # Direction test: in how many series does the reservoir-leads direction beat
    # the human-leads direction? Under a null of no lead-lag structure this is
    # a coin flip, so a binomial test on it is interpretable.
    paired = [(f["max_corr_reservoir_leads"], f["max_corr_human_leads"]) for f in findings
              if f.get("max_corr_reservoir_leads") is not None
              and f.get("max_corr_human_leads") is not None]
    wins = sum(1 for a, b in paired if a > b)
    binom_p = None
    if paired:
        from scipy.stats import binomtest

        binom_p = float(binomtest(wins, len(paired), 0.5, alternative="greater").pvalue)

    return {
        "n_analyses": len(findings),
        "n_reservoir_leading": len(leading),
        "share_reservoir_leading": round(len(leading) / len(findings), 3) if findings else None,
        "median_best_lag_quarters": float(np.median(lags)) if lags else None,
        "direction_test": {
            "n_series": len(paired),
            "reservoir_leads_wins": wins,
            "binomial_p_one_sided": binom_p,
        },
        "countries": countries,
        # Keep every analysed series: the figure histograms these, and
        # truncating to the strongest 40 made the panel disagree with its own
        # caption about how many series the test covered.
        "findings": sorted(findings, key=lambda f: -f["best_correlation"]),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Build the real-data One Health slice.")
    ap.add_argument("--organism", nargs="+", default=[NCBI_ORGANISM],
                    help="one or more organisms; results are pooled")
    ap.add_argument("--snapshot", default=NCBI_SNAPSHOT, help="default: newest available")
    ap.add_argument("--force", action="store_true", help="re-download even if cached")
    ap.add_argument("--min-isolates", type=int, default=15)
    args = ap.parse_args()

    frames, snapshots = [], {}
    for organism in args.organism:
        path, snapshot = download(organism, args.snapshot, force=args.force)
        iso = load_isolates(path)
        iso["organism"] = organism
        snapshots[organism] = snapshot
        print(f"{organism}: {len(iso):,} dated isolates")
        frames.append(iso)
    iso = pd.concat(frames, ignore_index=True)
    del frames
    print(f"\n{len(iso):,} dated isolates total; sectors:\n"
          f"{iso['sector'].value_counts().to_string()}")

    panel = build_panel(iso, min_isolates=args.min_isolates)
    panel.to_csv(DATA_PROCESSED / "ncbi_panel.csv", index=False)
    print(f"panel: {len(panel):,} country x sector x quarter x determinant rows")

    study = run_case_study(panel)
    study["organism"] = ", ".join(args.organism)
    study["snapshot"] = snapshots
    study["n_isolates"] = int(len(iso))
    study["sector_counts"] = iso["sector"].value_counts().to_dict()
    (RESULTS / "real_case_study.json").write_text(json.dumps(study, indent=2, default=float))

    dt = study["direction_test"]
    print(f"\n{study['n_analyses']} country/determinant/reservoir analyses; "
          f"{study['n_reservoir_leading']} show the reservoir leading "
          f"(median best lag {study['median_best_lag_quarters']} quarters)")
    print(f"direction test: reservoir-leads beats human-leads in "
          f"{dt['reservoir_leads_wins']}/{dt['n_series']} series, "
          f"one-sided binomial p = {dt['binomial_p_one_sided']:.4g}")
    for f in study["findings"][:8]:
        print(f"  {f['country']:<18s} {f['determinant']:<16s} {f['reservoir']:<12s} "
              f"lag={f['best_lag_quarters']}q r={f['best_correlation']:+.2f}")


if __name__ == "__main__":
    main()
