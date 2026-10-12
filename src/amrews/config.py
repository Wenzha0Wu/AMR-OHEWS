"""Central configuration for the AMR One Health Early Warning System (AMR-OHEWS).

Everything that a reader of the report might want to change -- horizons, sector
definitions, simulation size, file locations -- lives here rather than being
scattered through the modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# One Health sectors
# --------------------------------------------------------------------------- #
#: The four surveillance sectors the system integrates. "human" is the only one
#: conventional AMR surveillance (e.g. WHO GLASS, ECDC EARS-Net) routinely covers;
#: the research question is whether the other three buy us warning time.
SECTORS = ("human", "animal", "food", "environment")

#: Reporting delay in weeks for each stream, i.e. how stale the most recent
#: observation is at decision time. Human clinical data are slowest because they
#: pass through laboratory confirmation and national aggregation.
REPORTING_DELAY_WEEKS = {
    "human": 3,
    "animal": 2,
    "food": 2,
    "environment": 1,
    "amu_vet": 8,     # veterinary antimicrobial sales are reported quarterly/late
    "amu_human": 6,
}

# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimConfig:
    """Parameters of the synthetic One Health surveillance generator."""

    n_regions: int = 20
    n_weeks: int = 520                      # ten years of weekly data
    determinants: tuple[str, ...] = (
        "FQ_gyrA_T86I",                     # fluoroquinolone resistance
        "ESBL_blaCTX-M",                    # extended-spectrum beta-lactamase
        "COL_mcr-1",                        # plasmid-mediated colistin resistance
        "TET_tet(O)",                       # tetracycline resistance
        "MAC_erm(B)",                       # macrolide resistance
        "AMG_aac(6')-Ib-cr",                # aminoglycoside / fluoroquinolone
    )

    # -- emergence events ---------------------------------------------------
    n_true_events: int = 60                 # events that do reach human medicine
    n_decoy_events: int = 40                # reservoir-only signals that never do
    event_earliest_week: int = 40
    event_latest_week: int = 470
    logistic_amplitude: tuple[float, float] = (0.18, 0.45)   # prevalence gain
    logistic_rate: tuple[float, float] = (0.07, 0.22)        # per-week steepness
    #: Weeks between the reservoir take-off and the corresponding rise in human
    #: clinical prevalence. Drawn uniformly per event. The range is the whole
    #: basis of the early-warning claim, so it is set from the transmission
    #: literature rather than for convenience: farm-to-clinic establishment of a
    #: resistance determinant is a months-long chain (animal reservoir -> food
    #: chain or environmental route -> human colonisation -> clinical isolate),
    #: not a weeks-long one. Roughly three to six months.
    reservoir_to_human_lag: tuple[int, int] = (10, 26)

    # -- sampling intensity -------------------------------------------------
    human_isolates_per_week: tuple[int, int] = (25, 90)      # per region
    animal_isolates_per_month: tuple[int, int] = (18, 55)
    food_isolates_per_month: tuple[int, int] = (12, 40)
    missingness: float = 0.06               # probability a sector-week is missing

    # -- baseline prevalence ------------------------------------------------
    baseline_prevalence: tuple[float, float] = (0.03, 0.14)

    # -- spatial spread -----------------------------------------------------
    trade_spillover: float = 0.35           # reservoir contamination from partners
    trade_lag_weeks: int = 5

    #: Share of events that are *trade-seeded secondaries* -- the same
    #: determinant arriving in a trade-connected region weeks later -- rather
    #: than independent index events.
    #:
    #: This is the parameter the spatial-model sweep varies. At 0.0 regions
    #: emerge independently and a graph model has nothing to learn, which is the
    #: control setting: it reproduces the original simulator exactly, so any
    #: graph-tier gain observed there must come from the architecture rather
    #: than from the world having been made friendlier.
    spatial_fraction: float = 0.5
    #: Weeks between a parent's onset and a secondary's, on top of the trade lag.
    secondary_onset_delay: tuple[int, int] = (8, 30)
    secondary_amplitude_scale: tuple[float, float] = (0.5, 0.9)

    seed: int = 0


#: Horizon (weeks) of the supervised task: "will human-clinical emergence begin
#: within the next H weeks?"  Two horizons are evaluated in the report.
#:
#: The primary horizon is set to 26 weeks to match both the reservoir-to-human
#: lag above and :data:`MATCH_WINDOW_BEFORE`. A shorter horizon caps the lead
#: time a supervised model can possibly achieve: trained to fire only in the 12
#: weeks before emergence, it cannot earn a 20-week warning no matter how clear
#: the environmental signal is.
HORIZONS = (12, 26)
PRIMARY_HORIZON = 26

#: A candidate alarm is matched to a true event if it falls in this window
#: around the event's human-emergence week. Alarms outside every window count
#: as false alarms.
#:
#: The window and the false-alarm budget below have to be chosen together. At a
#: budget of one alarm per series-year and a 52-week look-back, a method that
#: alarms at random still lands inside roughly every event window by chance, and
#: every method scores a spurious 8-20 week "lead". Halving the window and
#: quartering the budget puts the expected number of chance alarms per window at
#: about 0.18, so a detection has to be earned.
MATCH_WINDOW_BEFORE = 26
MATCH_WINDOW_AFTER = 12

#: Alarms are compared at an equal false-alarm budget so that a method cannot
#: "win" on lead time simply by alarming more often. Expressed per monitored
#: series (region x determinant) per year.
FALSE_ALARM_BUDGET_PER_SERIES_YEAR = 0.5

#: The budget is a policy choice, not a constant of nature: a screening tool
#: that directs an investigative sampling round can tolerate more alarms than
#: one that triggers a prescribing change. Results are reported across this
#: sweep so the sensitivity/lead-time trade-off is visible rather than buried
#: in one arbitrary setting.
FALSE_ALARM_BUDGET_SWEEP = (0.25, 0.5, 1.0)

# --------------------------------------------------------------------------- #
# Temporal split (weeks). Never random -- that would leak the future.
# --------------------------------------------------------------------------- #
TRAIN_END_WEEK = 312    # years 1-6   : model fitting
VALID_END_WEEK = 416    # years 7-8   : probability calibration + alarm thresholds
# test = weeks 416..519 (years 9-10) : reported results

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
N_SEEDS_QUICK = 3
N_SEEDS_FULL = 30
N_SEEDS_SWEEP = 12

#: Settings for the spatial-transmission sweep. 0.0 is the control.
SPATIAL_FRACTION_SWEEP = (0.0, 0.25, 0.5, 0.75)
BOOTSTRAP_RESAMPLES = 2000

# --------------------------------------------------------------------------- #
# Real-data slice (NCBI Pathogen Detection)
# --------------------------------------------------------------------------- #
NCBI_BASE = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results"
NCBI_ORGANISM = "Campylobacter"
#: ``None`` means "resolve the newest snapshot at run time". Pinning a snapshot
#: makes a run reproducible but goes stale fast: the first pinned value here was
#: three years old and stopped in 2019, which silently cost two-thirds of the
#: available isolates.
NCBI_SNAPSHOT: str | None = None


@dataclass
class Palette:
    """Colour-blind-safe sector palette shared by every figure and the dashboard."""

    human: str = "#2b6cb0"
    animal: str = "#b7791f"
    food: str = "#2f855a"
    environment: str = "#6b46c1"
    alert: str = "#c53030"
    baseline: str = "#718096"
    truth: str = "#1a202c"
    grid: str = "#e2e8f0"

    def sector(self, name: str) -> str:
        return getattr(self, name, self.baseline)


PALETTE = Palette()
