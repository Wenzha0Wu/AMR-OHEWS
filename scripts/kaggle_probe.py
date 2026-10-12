"""Reproducible Kaggle survey for AMR datasets, with the exclusion decisions.

The project was asked to consider Kaggle as a data source. This script records
what was searched, what was found, and why almost none of it was used -- so the
decision is auditable rather than a claim in a report.

The short version: Kaggle's contribution to this project was a **pointer, not a
dataset**. One candidate carried raw WHO extracts whose indicator codes
(``AMR_INFECT_ECOLI``, ``GLASSAMC_TC``) revealed that WHO GLASS resistance and
consumption data are available directly from the WHO Global Health Observatory
API -- which is where :mod:`amrews.data.glass` now pulls them from, with proper
provenance. Everything else was redundant with NCBI Pathogen Detection, aimed at
a different problem, or failed a provenance check.

Authentication uses ``~/.kaggle/access_token`` as a bearer token. Note that the
``datasets/list/files`` endpoint rejects bearer auth even though downloads
accept it, so candidate inspection works by downloading the (small) zip.

Run with ``python scripts/kaggle_probe.py``; add ``--download`` to re-inspect
the small candidates.
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKEN_PATH = pathlib.Path.home() / ".kaggle" / "access_token"
API = "https://www.kaggle.com/api/v1"

QUERIES = [
    "antimicrobial resistance", "antibiotic resistance", "AMR surveillance",
    "one health", "salmonella resistance", "antibiotic consumption", "MIC bacteria",
]

#: A dataset is useful to this project only if it can support a lead-time or a
#: driver analysis. That needs dates, geography and a resistance outcome;
#: One Health work additionally needs a sector label.
INCLUSION_CRITERIA = ["dated", "geographic", "resistance_outcome", "sector_labelled"]

#: Decisions recorded from the survey, with reasons.
VERDICTS = {
    "legendxt/who-glass-amr-ecoli-panel": (
        "pointer",
        "Carries authentic WHO extracts (IND_CODE, GLASSAMC_TC, M49 geo codes). Its "
        "value was revealing the WHO indicator codes; we pull from the WHO GHO API "
        "directly instead, for provenance. See amrews/data/glass.py.",
    ),
    "hussnainmamoon1/global-antibiotic-resistance-2018-2024-glass": (
        "excluded",
        "Fails provenance. Narrative free-text columns ('Overuse in community drives "
        "resistance'), editorial trend_direction fields and suspiciously round values "
        "across only 138 rows. Presented as WHO GLASS but does not look like an "
        "authentic extract. Not used.",
    ),
    "drscarlat/driams": (
        "out_of_scope",
        "Real and well published (Weis et al., Nature Medicine 2022), but 8.85 GB of "
        "MALDI-TOF spectra, one file per isolate, for spectrum-to-resistance "
        "prediction. Single-sector clinical data for a different problem.",
    ),
    "cristiancaniu/salmonella": (
        "redundant",
        "769 MB, almost certainly a mirror of NCBI Pathogen Detection Salmonella, "
        "which we fetch at source and fresher (858 MB, current snapshot).",
    ),
    "willianoliveiragibin/antibiotic-use-in-livestock-in-europe": (
        "excluded",
        "Malformed single fixed-width column; titled 'Europe' but contains "
        "Afghanistan, Albania, Algeria. Provenance unclear.",
    ),
    "astukari/global-health-observatory-national-amr-capacity": (
        "duplicate_of_source",
        "24 rows of WHO region-year GLASS site counts, available directly from the "
        "GHO API.",
    ),
}


def token() -> str:
    if not TOKEN_PATH.exists():
        raise SystemExit(
            f"no Kaggle credentials at {TOKEN_PATH}. Either place kaggle.json / an "
            "access token there, or skip this script: nothing else in the project "
            "depends on it."
        )
    return TOKEN_PATH.read_text().strip()


def search(query: str, page_size: int = 12) -> list[dict]:
    import requests

    r = requests.get(f"{API}/datasets/list",
                     params={"search": query, "pageSize": page_size, "sortBy": "hottest"},
                     headers={"Authorization": f"Bearer {token()}"}, timeout=60)
    r.raise_for_status()
    return r.json()


def inspect(ref: str, max_bytes: int = 5_000_000) -> dict:
    """Download a small dataset and report its files and columns."""
    import requests

    r = requests.get(f"{API}/datasets/download/{ref}",
                     headers={"Authorization": f"Bearer {token()}"},
                     timeout=300, allow_redirects=True)
    r.raise_for_status()
    if len(r.content) > max_bytes:
        return {"ref": ref, "note": f"skipped inspection, {len(r.content)/1e6:.0f} MB"}
    out = {"ref": ref, "files": []}
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        for info in zf.infolist():
            entry = {"name": info.filename, "bytes": info.file_size}
            if info.filename.lower().endswith((".csv", ".tsv")):
                import pandas as pd

                df = pd.read_csv(io.BytesIO(zf.read(info.filename)), sep=None,
                                 engine="python", nrows=200)
                entry["shape_head"] = list(df.shape)
                entry["columns"] = list(df.columns)[:20]
            out["files"].append(entry)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Survey Kaggle for AMR datasets.")
    ap.add_argument("--download", action="store_true",
                    help="re-inspect the small candidates rather than only listing")
    args = ap.parse_args()

    survey = {"queries": QUERIES, "inclusion_criteria": INCLUSION_CRITERIA,
              "results": {}, "verdicts": VERDICTS}
    for q in QUERIES:
        rows = search(q)
        survey["results"][q] = [
            {"ref": x.get("refNullable") or x.get("ref"),
             "title": x.get("titleNullable"),
             "mb": round((x.get("totalBytesNullable") or 0) / 1e6, 2)}
            for x in rows
        ]
        print(f"=== {q} ===")
        for row in survey["results"][q]:
            verdict = VERDICTS.get(row["ref"], ("", ""))[0]
            print(f"  {row['mb']:9.2f} MB  {row['ref']:<62s} {verdict}")

    if args.download:
        survey["inspections"] = [inspect(ref) for ref in VERDICTS
                                 if VERDICTS[ref][0] in ("pointer", "excluded",
                                                         "duplicate_of_source")]

    out = ROOT / "results" / "kaggle_survey.json"
    out.write_text(json.dumps(survey, indent=2))
    print(f"\nwrote {out.relative_to(ROOT)}")
    print("\nVerdicts:")
    for ref, (verdict, why) in VERDICTS.items():
        print(f"  [{verdict}] {ref}\n      {why}")


if __name__ == "__main__":
    main()
