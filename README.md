# AMR-OHEWS — One Health Early Warning for Antimicrobial Resistance

A working prototype that integrates human clinical, animal/veterinary, food-chain and
environmental (wastewater) surveillance to warn of emerging antimicrobial resistance
before conventional clinical-only surveillance detects it — and an evaluation harness
that measures, in weeks, whether it actually does.

Course project for CE8001. The written proposal is in `docs/proposal.md`
(rendered to `docs/proposal.docx` and `docs/proposal.pdf`).

## Quickstart

    make setup       # .venv + pinned dependencies
    make quick       # 3 seeds, no deep models (~7 min) — sanity check
    make setup-gpu-volta   # only if your GPU is a V100 (see requirements.txt)
    make real        # download NCBI Pathogen Detection, run the premise test
    make full        # 30 seeds, all four tiers + ablations (~1 h with a GPU)
    make figures     # render report figures into figures/
    make dashboard   # Streamlit dashboard on :8501
    make docs        # build proposal.docx and proposal.pdf

## What the system does

For every *series* — one region crossed with one resistance determinant — and every
week, it estimates the probability that resistance begins to establish in the human
clinical population within the next H weeks, using only data an analyst could actually
have read that week.

Four model tiers, each a different answer to "what should the model see":

| Tier | Model | Sees |
| --- | --- | --- |
| 0 | EWMA, CUSUM, Farrington-style, quarterly manual review | Human clinical only — the comparator |
| 1 | Gradient-boosted trees + SHAP | All sectors, engineered features |
| 2 | Dilated causal TCN + MC dropout | All sectors, raw series, learns its own filters |
| 3 | TCN + GraphSAGE over the region trade graph | The above, plus spatial spillover |

Plus a logistic stacker over tiers 1–3 and an alerting layer that calibrates to a
probability, thresholds against an explicit false-alarm budget, and attributes each
alert to a sector.

## Headline result

Over 30 independent simulated worlds and 294 held-out emergence events, at a matched
budget of 0.5 false alarms per series-year with thresholds transferred from the
validation window:

| Method | Sensitivity | Mean lead time | vs best baseline | 95% CI |
| --- | --- | --- | --- | --- |
| One Health fusion (Tier 1) | 0.80 | **+8.6 weeks** | **+3.1 weeks** | +1.3 to +5.0 |
| Ensemble (Tiers 1–3) | 0.78 | +8.4 weeks | +2.6 weeks | +0.7 to +4.5 |
| Temporal TCN (Tier 2) | 0.66 | +9.2 weeks | −0.2 weeks | −2.0 to +1.8 |
| Spatio-temporal GNN (Tier 3) | 0.59 | +9.1 weeks | −1.5 weeks | −3.4 to +0.4 |
| EWMA chart (clinical only) | 0.86 | +2.2 weeks | — | — |

Removing the One Health streams from the Tier-1 model costs **10.6 weeks** and drops
sensitivity from 0.80 to 0.34.

## Is it the model or the features?

Tier 1 changes two things at once relative to Tier 2 — boosted trees instead of a neural
net, *and* engineered features instead of raw series — so its win could not be attributed
to either. Filling in the missing cells of the 2x2, over 20 seeds:

| | Engineered features | Raw lagged series |
| --- | --- | --- |
| **Gradient-boosted trees** | **+4.88 wk** | +3.64 wk |
| **Neural network** | −1.03 wk | +1.30 wk |

| Contrast | Effect | 95% CI |
| --- | --- | --- |
| Trees vs net, engineered | **+6.20 wk** | +4.50 to +7.98 |
| Trees vs net, raw | **+4.03 wk** | +1.92 to +6.15 |
| Engineered vs raw, within trees | +0.00 wk | −2.00 to +2.18 |
| Engineered vs raw, within net | −2.17 wk | −3.98 to −0.31 |

**It is the model class.** Trees beat the network by 4–6 weeks in *both* representations.
The feature engineering buys the winning model **nothing** (+0.00 wk) and actively hurts
the neural network. It earns its place for interpretability — SHAP attributions and sector
ablations are only legible because the features have names — not for accuracy.

This does not weaken the One Health result: the ablations remove entire *data streams*,
and the raw representation contains every sector too.

## The spatial tier does not work, and we can prove it

The graph tier failed everywhere, and for most of this project we blamed the data — too
few events, an artificial ring topology, a world without spatial transmission. We tested
each of those and each was wrong. The explanation that was right came from a control we
ran last: **replace the adjacency matrix with a randomly permuted one.**

On three standard epidemic-forecasting benchmarks where spatial autocorrelation is known
to be strong:

| Benchmark | Graph vs Temporal | **Real graph vs shuffled graph** |
| --- | --- | --- |
| US-States ILINet (49 × 360) | +0.4 wk | **−0.3 wk** |
| Japan-Prefectures (47 × 348) | −1.0 wk | **−2.6 wk** |
| Chickenpox Hungary (20 × 521) | **+2.4 wk** (p = 4e-4) | **+0.1 wk** |

Where the graph tier appears to win, a random graph wins by the same amount. **The
spatial component is inert.**

This survives a split-sensitivity challenge that the absolute numbers do not. Across
3 splits x 4 seeds on the real US panel, graph-minus-temporal ranges from **-4.3 to
+2.5 months** (3/12 configurations favour the graph tier, none significantly) — so a
split could be chosen to make it look good. But real-minus-shuffled stays within
**0.5 weeks of zero in every seed**, because that contrast holds the split fixed and
varies only the graph. Its learned weights look sensible — 0.98 on real two-step
adjacency, 0.002 on a uniform null relation, 0.43 on an 8-month neighbour lag — and
those weights do not change the predictions. Inspecting model internals is not a test;
perturbing the input is.

## What the real data says

| Question | Source | Result |
| --- | --- | --- |
| Are the One Health sectors coupled? | EFSA/ECDC EU Summary Report, 31 countries, designed sampling | **Yes** — 78/97 pairs positive, **p = 5.8e-10**. Ciprofloxacin in broiler *C. jejuni* vs humans: **ρ = +0.76, 25 countries** |
| Does the driver lead the outcome? | WHO GLASS, 25 countries | **Yes** — 33/50 series, **p = 0.016** |
| Do reservoir sectors temporally lead the clinic? | NCBI, 195 series, quarterly | **No** — 96/195, p = 0.61 |
| ...at finer resolution? | NCBI, 351 US state series, **monthly** | **No** — 139/351, p = 1.0 |
| Does the early-warning advantage replicate? | 221 real country-determinant series | **No** — longest warning, fires too rarely |

**The signal is real; its temporal ordering is not recoverable.** No existing source
combines the three properties such a system needs: sector linkage in a common geography,
sub-monthly cadence, and recorded rather than inferred denominators. EFSA has two.
NCBI has one.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/amrews/config.py` | Every tunable: horizons, splits, budgets, simulator parameters |
| `src/amrews/data/simulate.py` | One Health generator with injected ground-truth events and decoys |
| `src/amrews/data/features.py` | Reporting delays, harmonisation, causal features, labels |
| `src/amrews/data/ncbi.py` | Real NCBI Pathogen Detection slice and the lead-lag premise test |
| `src/amrews/models/` | `baselines.py`, `fusion.py`, `temporal.py`, `graph.py`, `ensemble.py` |
| `src/amrews/eval/metrics.py` | Alarms, lead time, false-alarm budgets, bootstrap CIs |
| `src/amrews/eval/experiments.py` | Per-seed protocol, ablations, aggregation |
| `src/amrews/plots.py` | Report figures |
| `src/amrews/dashboard/app.py` | Streamlit dashboard |
| `scripts/build_docs.py` | Markdown to docx/LaTeX/plain text (no pandoc on this host) |

## Evaluation protocol

Weeks 0–311 train, 312–415 calibrate probabilities and alarm thresholds, 416–519 test.
Models are frozen after training. Two assertions run every seed: no label may mark an
emergence that has already happened, and no latent ground-truth column may enter the
feature matrix.

All methods are compared at matched false-alarm budgets (0.25, 0.5 and 1.0 alarms per
series-year), under two threshold policies — *prospective*, where the threshold is
transferred from the validation window as a deployed system must do, and *matched-FAR*,
where it is set on the test window so every method realises the same alarm rate. Every
headline number is a paired per-event comparison over 30 independent simulated worlds
with a bootstrap confidence interval.

## Development log: what went wrong and what we did about it

The problems worth recording were almost all evaluation-design problems rather than
modelling problems. Several of them produced *better-looking* results than the truth,
which is the dangerous direction.

**Every method scored a spurious 8–20 week lead.** With a 52-week match window and a
budget of one alarm per series-year, a method alarming at random lands inside roughly
every event window by chance — so even the crudest quarterly-review baseline looked
like an early-warning system. Fixed by choosing the window and the budget together: a
26-week window at 0.25–1.0 alarms per series-year puts expected chance matches near 0.2
per window. The lesson generalises: a detection metric with a generous matching window
is a lottery unless the alarm rate is tight enough to make winning unlikely.

**The real-data finding was an artifact of a one-sided scan.** Our first lead-lag
analysis scanned lags 0–8 quarters and reported that 14 of 15 series showed the
reservoir leading the clinic. With nine candidate lags and only one of them zero, that
result is nearly guaranteed by construction. Making the scan symmetric (−6 to +6) and
adding a binomial direction test changed the conclusion completely. We report the
symmetric result.

**Isotonic calibration made the alarm budget unsatisfiable.** Isotonic regression maps
scores onto a step function with a handful of distinct levels. Thousands of weeks then
share the top level, so no threshold exists between "no alarms" and "five times over
budget" — one ablation ran at 1.25 alarms per series-year against a 0.25 budget before
we caught it. Switched to Platt scaling, which is strictly monotone and, with a few
dozen validation positives, the better-behaved choice anyway.

**Threshold calibration overshot its budget.** The initial linear sweep over 60
quantiles was too coarse in the tail of a calibrated-probability distribution and
silently handed methods far stricter thresholds than they were allowed. Replaced with a
bisection over a 2000-point grid, exploiting the monotonicity of false-alarm rate in
the threshold.

**The refractory period punished being early twice.** With refractory and match window
both 26 weeks, an alarm fired just outside the window — too early to be credited —
silenced the series for exactly the period in which a creditable alarm could have been
raised. Shortened the refractory to 13 weeks.

**The supervised horizon capped the achievable lead time.** Training the model to fire
only in the 12 weeks before emergence made a 20-week warning impossible to earn no
matter how clear the environmental signal was. Horizon raised to 26 weeks to match both
the reservoir-to-human lag and the match window.

**Feature engineering took 36 seconds per seed.** `rolling().apply()` with a Python
callback dominated the whole pipeline. Rewritten as a dot product against fixed
regression weights over a `sliding_window_view`: 36 s to 2.6 s, which is what made a
30-seed bootstrap practical.

**The pinned NCBI snapshot was three years stale.** It stopped in 2019 and yielded
40,367 dated isolates; the current snapshot yields 146,795. Snapshot resolution is now
automatic, with the pin available for reproducibility.

**The graph tier "did not earn its place" — and the cause was the simulator.**
Tier 3 underperformed Tier 2, which looked like a clean negative result about
spatial models. It was not. Trade spillover perturbed a neighbour's animal
reservoir but never seeded an emergence event there, so a neighbour's signal was
uninformative about this region's label: `animal_z` correlated 0.144 with the
label while `neighbour_animal_z` managed 0.008, and across 2,772 same-determinant
event pairs the correlation between trade weight and emergence-week gap was
−0.004. The graph tier was being asked to learn structure that did not exist.
Fixed by giving the simulator genuine trade-seeded secondary events — and by
sweeping how many there are, with the original zero-transmission world kept as
the control, so the fix cannot be mistaken for tuning the world until the model
wins.

**The graph tier was also undertrained, which nobody would have noticed.** Its
batches are whole region-stacks, so equal epochs meant 610 optimizer steps
against the temporal tier's 1,525. Step budgets are now matched and asserted in
code. Separately, the gate and the relation/lag mixing weights are a handful of
scalars that did not move at all at the shared learning rate — the gate sat at
its initial value even in worlds with strong spatial signal — so they get their
own parameter group at 20x the base rate with no weight decay.

**The pinned NCBI snapshot was three years stale, and the fix had a second bug.**
The pin stopped in 2019 and yielded 40,367 isolates; resolving the newest
snapshot automatically raised that to 146,795. But the newest *directory* is not
always usable: NCBI publishes the folder before the AMR metadata inside it is
built, so Salmonella and E. coli both 404ed. The loader now walks backwards to
the newest snapshot whose file actually exists. With three organisms the archive
is 1,101,779 dated isolates.

**I misread the WHO API and nearly dropped the best real-data source.** An early
search of the GHO indicator list concluded it held "capacity indicators, not
resistance rates". It does hold resistance rates — I had truncated the output at
25 of 42 matches. `AMR_INFECT_ECOLI` covers 100 countries, 2016–2023. Ironically
it was a Kaggle dataset that exposed the indicator codes, which is the whole of
Kaggle's contribution to this project.

**One candidate dataset appears synthetic.** A Kaggle dataset presented as WHO
GLASS carried narrative free-text columns ("Overuse in community drives
resistance"), editorial trend fields and suspiciously round values across 138
rows. Excluded, and recorded in `scripts/kaggle_probe.py` — nothing downstream
would have flagged it, which is the point.

**Hard-coded week assumptions surfaced the moment real data arrived.** The
control charts used a 52-period baseline and `min_periods=12`; on a quarterly
panel that is thirteen years of baseline and an invalid window. Windows are now
expressed in years and converted, and the match-window constants in `metrics.py`
became parameters so the same evaluation code runs in weeks or quarters.

**The false-alarm denominator disagreed with its own numerator.**
`eligible &= weeks < w_star - match_before`, conjoined over events, so on a series
carrying several events only the weeks before the *first* one counted — often none,
which sent the false-alarm rate to infinity on the real panels. The numerator counted
an alarm landing after an event as false while the denominator refused to count the
week it landed in. Since thresholds are calibrated to a false-alarm budget, this moved
every number downstream of it; all results were recomputed after the fix.

**The gated residual did not do what it was designed to do.** Tier 3 was built with a
gate on the graph branch specifically so it could never underperform the Tier 2 it
contains. But its local pathway ran through a 48-d projection that Tier 2's head does
not have, so the guarantee never held. On the Japan ILI benchmark it cost the graph tier
an AUC-PR of 0.87 against Tier 2's 0.36 with an identical encoder. Both defects survived
a long time because our checks inspected model internals rather than perturbing inputs
and watching the output.

**PyTorch on the available GPUs.** `python3 -m venv` failed for lack of `ensurepip`
(venv now created from the conda interpreter); the default CPU wheel took ~13 minutes
per seed; the CUDA 13 wheel reported the V100s as available and then failed inside the
first convolution because it ships no `sm_70` kernels. Installed the cu126 build and
made `default_device()` check the card's compute capability against the build's
architecture list rather than trusting `torch.cuda.is_available()`. 84 s per seed.

**A `pos_weight` tensor left on the CPU** crashed both deep tiers on their first
backward pass — the ordinary tax on moving a model to a device.

## Data

- **Simulated** (primary evaluation): generated on demand, seeded, no download.
- **Real** (premise test): NCBI Pathogen Detection *Campylobacter* AMR metadata,
  fetched by `make real` and cached in `data/raw/` (~128 MB).

## Requirements

Python 3.13, a CUDA GPU optional but recommended for the deep tiers. See
`requirements.txt` for pinned versions.
