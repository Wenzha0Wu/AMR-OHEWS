"""Synthetic One Health surveillance generator with injected ground-truth events.

Why simulate at all?  The research question is comparative and temporal -- does a
multi-sector model warn *earlier* than clinical-only surveillance?  Answering it
requires knowing, for every emergence event, the week resistance actually began
to rise in the human population.  No real surveillance system records that: the
onset is exactly the thing surveillance is trying to find.  Real archives can
show that animal/environmental signals *correlate* with later human signals (see
:mod:`amrews.data.ncbi` for that analysis on genuine NCBI Pathogen Detection
data), but only a generative model with known onsets can quantify lead time in
weeks with confidence intervals.

Generative structure (per region r, determinant d, week t)::

    AMU_vet, AMU_hum  -- seasonal + AR(1) antimicrobial-use pressure
    animal(t)         -- endemic baseline + logistic emergence term + trade spillover
    environment(t)    -- fast readout of animal + human shedding, diluted by rainfall
    food(t)           -- animal reservoir delayed by the farm-to-retail chain
    human(t)          -- animal reservoir delayed by reservoir_to_human_lag,
                         amplified by human antimicrobial use

Each injected event is either
  * a **true event**  -- the determinant reaches human medicine, or
  * a **decoy**       -- it rises and falls in the reservoir without ever
                         establishing in humans.

Decoys matter: without them any method that simply reacts to an environmental
uptick would score a perfect lead time, and the evaluation would be meaningless.

The observation layer then degrades the latent truth the way real surveillance
does -- binomial sampling with realistic denominators, monthly rather than weekly
sampling for animal and food, measurement noise on wastewater, random missing
weeks, and stream-specific reporting delays (:data:`amrews.config.REPORTING_DELAY_WEEKS`).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import DATA_PROCESSED, SimConfig


# --------------------------------------------------------------------------- #
# Event bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class EmergenceEvent:
    """One injected resistance-emergence episode and its ground truth."""

    region: int
    determinant: str
    onset_week: int              # reservoir take-off (latent, unobservable)
    amplitude: float
    rate: float
    lag_to_human: int
    is_true_event: bool          # False => decoy: never establishes in humans
    human_emergence_week: int | None = None   # filled in after simulation
    #: Set when this event was seeded from another region along the trade
    #: network rather than arising independently. The graph tier exists to catch
    #: exactly these, so lead time is reported separately for them.
    seeded_by_trade: bool = False
    parent_region: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _logistic(t: np.ndarray, onset: float, rate: float) -> np.ndarray:
    """Logistic take-off curve, ~0 before ``onset`` and ->1 well after it."""
    return 1.0 / (1.0 + np.exp(-rate * (t - onset)))


def _ar1(rng: np.random.Generator, n: int, phi: float, sigma: float) -> np.ndarray:
    """Stationary AR(1) noise process, used for the antimicrobial-use series."""
    x = np.zeros(n)
    x[0] = rng.normal(0.0, sigma / np.sqrt(1 - phi**2))
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal(0.0, sigma)
    return x


def _seasonal(t: np.ndarray, amp: float, phase: float) -> np.ndarray:
    return amp * np.sin(2 * np.pi * (t / 52.0) + phase)


def _trade_matrix(rng: np.random.Generator, n_regions: int) -> np.ndarray:
    """Row-stochastic livestock-trade / connectivity matrix with no self-loops.

    Regions are laid out on a ring so that neighbours trade most heavily; a few
    random long-range links stand in for national distribution hubs. The same
    matrix is handed to the graph model as its adjacency, so what the STGNN can
    learn is genuinely present in the data-generating process.
    """
    w = np.zeros((n_regions, n_regions))
    for i in range(n_regions):
        for j in range(n_regions):
            if i == j:
                continue
            ring = min(abs(i - j), n_regions - abs(i - j))
            w[i, j] = np.exp(-ring)
    # a handful of long-range trade links
    for _ in range(n_regions // 2):
        i, j = rng.choice(n_regions, size=2, replace=False)
        w[i, j] += 0.6
    w += 1e-6
    return w / w.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------- #
# Main generator
# --------------------------------------------------------------------------- #
class OneHealthSimulator:
    """Generates a full synthetic One Health surveillance archive."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.weeks = np.arange(cfg.n_weeks)
        self.trade = _trade_matrix(self.rng, cfg.n_regions)

    # -- events ------------------------------------------------------------ #
    def _draw_events(self) -> list[EmergenceEvent]:
        """Draw independent (index) events, then trade-seeded secondary events.

        ``cfg.spatial_fraction`` of the events are *secondaries*: the same
        determinant arriving in a trade-connected region some weeks later,
        because a resistant clone travelled with a livestock consignment. Only
        secondaries give the graph tier anything to learn -- without them a
        neighbour's signal is uninformative about this region's label, which is
        precisely why the first version of this simulator made the spatial model
        look useless.

        The **total** number of events is held fixed and only the share that is
        trade-seeded varies. Letting the count grow with the transmission rate
        would change the number of scoreable test events per setting and make
        the sweep over ``spatial_fraction`` uninterpretable.
        """
        cfg, rng = self.cfg, self.rng
        n_total = cfg.n_true_events + cfg.n_decoy_events
        flags = np.array([True] * cfg.n_true_events + [False] * cfg.n_decoy_events)
        rng.shuffle(flags)

        n_secondary = int(round(cfg.spatial_fraction * n_total))
        n_index = n_total - n_secondary

        # One event per (region, determinant) series: overlapping episodes would
        # make the ground-truth onset ambiguous.
        taken: set[tuple[int, str]] = set()
        events: list[EmergenceEvent] = []

        def _new(region: int, determinant: str, onset: int, flag: bool,
                 amplitude: float, rate: float, parent: int | None) -> bool:
            if (region, determinant) in taken:
                return False
            if not (cfg.event_earliest_week <= onset < cfg.event_latest_week):
                return False
            taken.add((region, determinant))
            events.append(EmergenceEvent(
                region=int(region), determinant=determinant, onset_week=int(onset),
                amplitude=float(amplitude), rate=float(rate),
                lag_to_human=int(rng.integers(*cfg.reservoir_to_human_lag)),
                is_true_event=bool(flag), seeded_by_trade=parent is not None,
                parent_region=parent,
            ))
            return True

        # -- index events -----------------------------------------------------
        pairs = [(r, d) for r in range(cfg.n_regions) for d in cfg.determinants]
        order = rng.permutation(len(pairs))
        idx_flags = list(flags[:n_index])
        for pos in order:
            if not idx_flags:
                break
            region, determinant = pairs[pos]
            _new(region, determinant,
                 int(rng.integers(cfg.event_earliest_week, cfg.event_latest_week)),
                 idx_flags[0], float(rng.uniform(*cfg.logistic_amplitude)),
                 float(rng.uniform(*cfg.logistic_rate)), None) and idx_flags.pop(0)

        # -- trade-seeded secondary events ------------------------------------
        # A secondary inherits its parent's determinant and truth status (a decoy
        # spreads as a decoy: reservoir signal that still reaches nobody), starts
        # later by the trade lag plus an establishment delay, and is weaker.
        # Retry on collision rather than dropping the event. With 100 events over
        # 120 series, and secondaries concentrated on the trade-weighted
        # neighbours of existing events, a single-shot draw loses roughly three
        # quarters of them -- which would leave the sweep comparing worlds with
        # different numbers of events rather than different spatial structure.
        self.n_secondary_skipped = 0
        for flag in flags[n_index:]:
            candidates = [e for e in events if e.is_true_event == bool(flag)] or events
            if not candidates:
                break
            placed = False
            for _ in range(60):
                parent = candidates[rng.integers(len(candidates))]
                weights = self.trade[:, parent.region].copy()
                weights[parent.region] = 0.0
                if weights.sum() <= 0:
                    continue
                region = int(rng.choice(cfg.n_regions, p=weights / weights.sum()))
                onset = (parent.onset_week + cfg.trade_lag_weeks
                         + int(rng.integers(*cfg.secondary_onset_delay)))
                if _new(region, parent.determinant, onset, bool(flag),
                        parent.amplitude * float(rng.uniform(*cfg.secondary_amplitude_scale)),
                        parent.rate * float(rng.uniform(0.8, 1.2)), parent.region):
                    placed = True
                    break
            if not placed:
                # Fall back to an independent event so the event count stays
                # fixed across sweep settings; counted so the shortfall is visible.
                self.n_secondary_skipped += 1
                for _ in range(200):
                    region = int(rng.integers(cfg.n_regions))
                    determinant = cfg.determinants[rng.integers(len(cfg.determinants))]
                    if _new(region, determinant,
                            int(rng.integers(cfg.event_earliest_week, cfg.event_latest_week)),
                            bool(flag), float(rng.uniform(*cfg.logistic_amplitude)),
                            float(rng.uniform(*cfg.logistic_rate)), None):
                        break
        return events

    # -- latent dynamics --------------------------------------------------- #
    def _simulate_latent(self, events: list[EmergenceEvent]) -> dict[str, np.ndarray]:
        """Return latent prevalence arrays shaped (n_regions, n_determinants, n_weeks)."""
        cfg, rng, t = self.cfg, self.rng, self.weeks
        R, D, T = cfg.n_regions, len(cfg.determinants), cfg.n_weeks

        # Antimicrobial-use pressure is region-level (shared by all determinants).
        amu_vet = np.zeros((R, T))
        amu_hum = np.zeros((R, T))
        for r in range(R):
            amu_vet[r] = (
                rng.uniform(0.8, 1.4)
                + _seasonal(t, rng.uniform(0.05, 0.18), rng.uniform(0, 2 * np.pi))
                + _ar1(rng, T, 0.85, 0.05)
            )
            amu_hum[r] = (
                rng.uniform(0.7, 1.3)
                + _seasonal(t, rng.uniform(0.08, 0.22), rng.uniform(0, 2 * np.pi))
                + _ar1(rng, T, 0.85, 0.06)
            )
        rain = np.stack([
            1.0 + _seasonal(t, 0.35, rng.uniform(0, 2 * np.pi)) + _ar1(rng, T, 0.6, 0.12)
            for _ in range(R)
        ])
        temperature = np.stack([
            15 + _seasonal(t, 9.0, -np.pi / 2 + rng.normal(0, 0.2)) + _ar1(rng, T, 0.7, 1.0)
            for _ in range(R)
        ])

        # Endemic baselines with slow drift.
        base = rng.uniform(*cfg.baseline_prevalence, size=(R, D))
        drift = np.cumsum(rng.normal(0, 0.0006, size=(R, D, T)), axis=2)
        animal = base[:, :, None] + drift + 0.03 * (amu_vet[:, None, :] - amu_vet.mean())

        # Emergence terms in the animal reservoir.
        for ev in events:
            d = cfg.determinants.index(ev.determinant)
            curve = _logistic(t, ev.onset_week, ev.rate)
            if not ev.is_true_event:
                # Decoy: rises, then is cleared (e.g. the farm changes supplier or
                # a veterinary stewardship intervention lands) -- reservoir-only.
                decay_start = ev.onset_week + rng.integers(25, 60)
                curve = curve * (1.0 - _logistic(t, decay_start, ev.rate * 0.8))
            animal[ev.region, d] += ev.amplitude * curve

        # Between-region seeding along the trade network.
        spill = np.einsum("rs,sdt->rdt", self.trade, _shift_axis(animal, cfg.trade_lag_weeks))
        animal = animal + cfg.trade_spillover * spill
        animal = np.clip(animal, 1e-4, 0.97)

        # Human clinical prevalence: the reservoir signal, delayed and modulated
        # by human antimicrobial use. Decoys contribute nothing here, which is
        # what makes them decoys.
        human = np.zeros_like(animal)
        human_from_events = np.zeros_like(animal)
        for ev in events:
            if not ev.is_true_event:
                continue
            d = cfg.determinants.index(ev.determinant)
            curve = _logistic(t, ev.onset_week + ev.lag_to_human, ev.rate * rng.uniform(0.7, 1.0))
            human_from_events[ev.region, d] += ev.amplitude * rng.uniform(0.6, 1.0) * curve

        human_base = rng.uniform(*cfg.baseline_prevalence, size=(R, D))[:, :, None] + drift * 0.5
        human = (
            human_base
            + human_from_events
            + 0.025 * (amu_hum[:, None, :] - amu_hum.mean())
        )
        human = np.clip(human, 1e-4, 0.97)

        # Environment (wastewater resistome): fast, noisy readout of both the
        # animal reservoir and human shedding, diluted by rainfall.
        environment = (
            0.55 * _shift_axis(animal, 1)
            + 0.30 * _shift_axis(human, 1)
            + 0.15 * base[:, :, None]
        ) / np.clip(rain[:, None, :], 0.4, None) ** 0.5
        environment = np.clip(environment, 1e-4, None)

        # Food / retail meat: the reservoir delayed by the farm-to-retail chain.
        food = np.clip(_shift_axis(animal, int(rng.integers(2, 7))) * rng.uniform(0.75, 1.1), 1e-4, 0.97)

        return {
            "animal": animal,
            "human": human,
            "human_from_events": human_from_events,
            "environment": environment,
            "food": food,
            "amu_vet": amu_vet,
            "amu_hum": amu_hum,
            "rain": rain,
            "temperature": temperature,
        }

    # -- ground truth ------------------------------------------------------- #
    def _label_events(self, events: list[EmergenceEvent], latent: dict) -> list[EmergenceEvent]:
        """Define each true event's human-emergence week from the latent signal.

        The ground-truth week is the first week at which the event's contribution
        to human prevalence exceeds an absolute floor (1 percentage point) *and*
        a quarter of its eventual plateau -- i.e. the point at which resistance
        has genuinely started to establish in human medicine, well before it is
        visible above sampling noise. Every lead time in the evaluation is
        measured against this week.
        """
        contrib = latent["human_from_events"]
        for ev in events:
            if not ev.is_true_event:
                continue
            d = self.cfg.determinants.index(ev.determinant)
            series = contrib[ev.region, d]
            plateau = series[-1] if series[-1] > 0 else series.max()
            threshold = max(0.01, 0.25 * plateau)
            hits = np.flatnonzero(series >= threshold)
            ev.human_emergence_week = int(hits[0]) if hits.size else None
            if ev.human_emergence_week is not None and ev.human_emergence_week >= self.cfg.n_weeks - 8:
                # Too close to the end of the archive to be scoreable.
                ev.human_emergence_week = None
        return events

    # -- observation layer -------------------------------------------------- #
    def _observe(self, latent: dict) -> pd.DataFrame:
        """Degrade the latent truth into what surveillance would actually record."""
        cfg, rng, t = self.cfg, self.rng, self.weeks
        R, D, T = cfg.n_regions, len(cfg.determinants), cfg.n_weeks
        rows = []

        # Region-specific laboratory capacity. This is deliberately uneven: it is
        # the in-silico analogue of the surveillance-capacity bias discussed in
        # the ethics section -- low-capacity regions get noisier, later signals.
        capacity = rng.uniform(0.45, 1.0, size=R)

        for r in range(R):
            week_season = 1.0 + 0.25 * np.sin(2 * np.pi * t / 52.0)
            n_human = rng.integers(*cfg.human_isolates_per_week, size=T)
            n_human = np.maximum(3, (n_human * capacity[r] * week_season).astype(int))

            for d, det in enumerate(cfg.determinants):
                p_h = latent["human"][r, d]
                res_h = rng.binomial(n_human, np.clip(p_h, 0, 1))

                # Animal and food surveillance report monthly (every 4th week).
                monthly = (t % 4) == 0
                n_a = np.where(monthly, rng.integers(*cfg.animal_isolates_per_month, size=T), 0)
                n_a = (n_a * capacity[r]).astype(int)
                res_a = rng.binomial(np.maximum(n_a, 0), np.clip(latent["animal"][r, d], 0, 1))

                n_f = np.where(monthly, rng.integers(*cfg.food_isolates_per_month, size=T), 0)
                n_f = (n_f * capacity[r]).astype(int)
                res_f = rng.binomial(np.maximum(n_f, 0), np.clip(latent["food"][r, d], 0, 1))

                # Wastewater: continuous log10 gene copies per 16S rRNA copy.
                env_true = np.log10(np.clip(latent["environment"][r, d], 1e-6, None))
                env_obs = env_true + rng.normal(0, 0.18 / capacity[r], size=T)

                # Random missing weeks, independently per stream.
                miss = lambda: rng.random(T) < cfg.missingness   # noqa: E731
                m_h, m_a, m_f, m_e = miss(), miss(), miss(), miss()

                rows.append(
                    pd.DataFrame(
                        {
                            "week": t,
                            "region": r,
                            "determinant": det,
                            "lab_capacity": capacity[r],
                            # human clinical
                            "human_n": np.where(m_h, 0, n_human),
                            "human_resistant": np.where(m_h, 0, res_h),
                            # animal / veterinary
                            "animal_n": np.where(m_a, 0, n_a),
                            "animal_resistant": np.where(m_a, 0, res_a),
                            # food / retail meat
                            "food_n": np.where(m_f, 0, n_f),
                            "food_resistant": np.where(m_f, 0, res_f),
                            # environment / wastewater
                            "environment_log_abundance": np.where(m_e, np.nan, env_obs),
                            # drivers
                            "amu_vet": latent["amu_vet"][r],
                            "amu_human": latent["amu_hum"][r],
                            "rainfall": latent["rain"][r],
                            "temperature": latent["temperature"][r],
                            # latent truth, kept for diagnostics and plots only --
                            # never used as a model input (see features.py).
                            "latent_human": latent["human"][r, d],
                            "latent_animal": latent["animal"][r, d],
                        }
                    )
                )
        return pd.concat(rows, ignore_index=True)

    # -- public API --------------------------------------------------------- #
    @staticmethod
    def _check_causal_seeding(events: list[EmergenceEvent]) -> None:
        """A secondary event must start strictly after the parent it came from.

        Cheap to check and worth checking: a secondary that preceded its parent
        would let the graph tier "predict" an event from its own consequence.
        """
        by_region_det: dict[tuple[int, str], EmergenceEvent] = {
            (e.region, e.determinant): e for e in events
        }
        for e in events:
            if not e.seeded_by_trade:
                continue
            parent = by_region_det.get((e.parent_region, e.determinant))
            assert parent is not None, "secondary event lost its parent"
            assert e.onset_week > parent.onset_week, (
                f"secondary onset {e.onset_week} precedes parent {parent.onset_week}"
            )

    def run(self) -> tuple[pd.DataFrame, list[EmergenceEvent], np.ndarray]:
        events = self._draw_events()
        self._check_causal_seeding(events)
        latent = self._simulate_latent(events)
        events = self._label_events(events, latent)
        panel = self._observe(latent)
        return panel, events, self.trade


def _shift_axis(x: np.ndarray, lag: int) -> np.ndarray:
    """Delay along the final (time) axis of a 3-D array."""
    if lag <= 0:
        return x.copy()
    out = np.empty_like(x)
    out[..., :lag] = x[..., :1]
    out[..., lag:] = x[..., :-lag]
    return out


def simulate(seed: int = 0, cfg: SimConfig | None = None, **overrides):
    """Convenience wrapper: build a simulator for ``seed`` and run it.

    ``overrides`` set any :class:`~amrews.config.SimConfig` field, which is how
    the spatial-transmission sweep varies ``spatial_fraction`` without threading
    a config object through every call site.
    """
    cfg = cfg or SimConfig()
    fields = {**asdict(cfg), "seed": seed, **overrides}
    unknown = set(overrides) - set(asdict(cfg))
    if unknown:
        raise TypeError(f"unknown SimConfig fields: {sorted(unknown)}")
    cfg = SimConfig(**fields)
    return OneHealthSimulator(cfg).run()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic One Health surveillance data.")
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds to generate")
    ap.add_argument("--out", type=str, default=str(DATA_PROCESSED))
    ap.add_argument("--plot", action="store_true", help="write sanity-check figures")
    args = ap.parse_args()

    out = DATA_PROCESSED if args.out is None else __import__("pathlib").Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for seed in range(args.seeds):
        panel, events, trade = simulate(seed)
        panel.to_parquet(out / f"panel_seed{seed}.parquet") if _has_parquet() else panel.to_csv(
            out / f"panel_seed{seed}.csv.gz", index=False
        )
        manifest = {
            "seed": seed,
            "events": [e.to_dict() for e in events],
            "n_scoreable_events": sum(
                1 for e in events if e.is_true_event and e.human_emergence_week is not None
            ),
            "n_decoys": sum(1 for e in events if not e.is_true_event),
        }
        (out / f"events_seed{seed}.json").write_text(json.dumps(manifest, indent=2))
        np.save(out / f"trade_seed{seed}.npy", trade)
        print(
            f"seed {seed}: panel {panel.shape}, "
            f"{manifest['n_scoreable_events']} scoreable events, "
            f"{manifest['n_decoys']} decoys"
        )

    if args.plot:
        from ..plots import sanity_plots

        sanity_plots(seed=0)


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
