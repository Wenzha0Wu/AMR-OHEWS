"""Tier 1: cross-sector fusion model (gradient boosting + SHAP attribution).

This is the workhorse of the system and the direct answer to the research
question. It consumes the same human clinical stream the Tier-0 baselines see,
plus lagged animal, food, environmental, antimicrobial-use and trade-network
features, and predicts a single quantity:

    P(human-clinical emergence begins within the next H weeks | data available now)

Gradient-boosted trees are a deliberate choice over something larger. The signal
is a few hundred positive weeks in a table of a few tens of thousands of rows --
a regime where boosted trees are hard to beat, train in seconds (so a 30-seed
bootstrap is affordable), and, critically for a public-health tool, come with
exact per-alert Shapley attributions. A public-health officer will not act on an
alert that cannot say *which sector* raised it.

Class imbalance (~1% positive) is handled by sample weighting rather than
resampling, so that the calibration step downstream sees the true base rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator

from ..data.features import clean_matrix


class FusionModel:
    """Gradient-boosted One Health fusion classifier with probability calibration."""

    def __init__(
        self,
        horizon: int = 12,
        max_iter: int = 400,
        learning_rate: float = 0.06,
        max_depth: int | None = 6,
        min_samples_leaf: int = 40,
        l2_regularization: float = 1.0,
        random_state: int = 0,
    ):
        self.horizon = horizon
        self.params = dict(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            early_stopping=False,
            random_state=random_state,
        )
        self.model: HistGradientBoostingClassifier | None = None
        self.calibrated: CalibratedClassifierCV | None = None
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, feature_cols: list[str],
            valid: pd.DataFrame | None = None) -> "FusionModel":
        """Fit on ``train``; if ``valid`` is given, isotonically calibrate on it.

        Calibration matters here beyond tidiness: alert thresholds are set from a
        false-alarm budget, and an uncalibrated boosted-tree score makes that
        budget meaningless across seeds.
        """
        self.feature_names = list(feature_cols)
        y_col = f"label_h{self.horizon}"

        # Post-emergence rows are dropped: the question "will it emerge?" is not
        # defined once it has.
        tr = train[~train["post_event"]]
        x = clean_matrix(tr, feature_cols)
        y = tr[y_col].to_numpy().astype(int)

        pos = max(int(y.sum()), 1)
        w = np.where(y == 1, len(y) / (2.0 * pos), len(y) / (2.0 * max(len(y) - pos, 1)))

        self.model = HistGradientBoostingClassifier(**self.params)
        self.model.fit(x, y, sample_weight=w)

        if valid is not None and len(valid):
            va = valid[~valid["post_event"]]
            yv = va[y_col].to_numpy().astype(int)
            if yv.sum() >= 5:
                # Platt (sigmoid) scaling, not isotonic. Isotonic is the usual
                # default and was the first choice here, but it maps scores onto
                # a step function with only a handful of distinct levels. With
                # alarm thresholds derived from a false-alarm budget, those ties
                # are fatal: thousands of weeks share the top level, so no
                # threshold exists between "no alarms" and "far over budget",
                # and the calibration silently overshoots its budget several-fold.
                # Platt scaling is strictly monotone and continuous, and with a
                # few dozen validation positives it is the better-behaved choice
                # anyway -- isotonic wants thousands.
                self.calibrated = CalibratedClassifierCV(
                    FrozenEstimator(self.model), method="sigmoid"
                )
                self.calibrated.fit(clean_matrix(va, feature_cols), yv)
        return self

    # ------------------------------------------------------------------ #
    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
        cols = feature_cols or self.feature_names
        x = clean_matrix(df, cols)
        est = self.calibrated or self.model
        assert est is not None, "model must be fitted first"
        return est.predict_proba(x)[:, 1]

    # ------------------------------------------------------------------ #
    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the standard (week, region, determinant, score) alarm frame."""
        out = df[["week", "region", "determinant"]].copy()
        out["score_fusion"] = self.predict_proba(df)
        return out

    # ------------------------------------------------------------------ #
    def explain(self, df: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
        """Per-row SHAP values, with graceful degradation.

        SHAP's tree path-dependent explainer does not support every scikit-learn
        version's histogram booster, so this falls back to a model-agnostic
        permutation explainer, and finally to global permutation importance
        broadcast across rows. The dashboard always gets *something* to show,
        and the fallback used is recorded in the returned frame's attributes.
        """
        cols = self.feature_names
        sample = df.iloc[:max_rows]
        x = clean_matrix(sample, cols)
        method = "none"
        values = None

        try:
            import shap

            explainer = shap.TreeExplainer(self.model)
            values = explainer.shap_values(x)
            if isinstance(values, list):
                values = values[1]
            if values.ndim == 3:                      # (n, features, classes)
                values = values[:, :, -1]
            method = "tree_shap"
        except Exception:
            try:
                import shap

                background = shap.sample(clean_matrix(df, cols), 100, random_state=0)
                explainer = shap.explainers.Permutation(
                    lambda z: self.model.predict_proba(z)[:, 1], background
                )
                values = explainer(x[: min(len(x), 300)], max_evals=1200).values
                sample = sample.iloc[: values.shape[0]]
                method = "permutation_shap"
            except Exception:
                from sklearn.inspection import permutation_importance

                y = sample[f"label_h{self.horizon}"].to_numpy().astype(int)
                if y.sum() > 0:
                    imp = permutation_importance(
                        self.model, x, y, n_repeats=3, random_state=0, scoring="average_precision"
                    ).importances_mean
                else:
                    imp = np.zeros(len(cols))
                values = np.tile(imp, (len(sample), 1))
                method = "global_permutation"

        out = pd.DataFrame(values, columns=cols, index=sample.index)
        for k in ("week", "region", "determinant"):
            out[k] = sample[k].to_numpy()
        out.attrs["shap_method"] = method
        return out


def sector_attribution(shap_df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Collapse per-feature SHAP values into per-sector contributions.

    "The environmental stream drove 46% of this alert" is the sentence an
    epidemiologist can act on; a ranked list of sixty engineered feature names is
    not. Signed sums are kept so that a sector can also argue *against* an alert.
    """
    meta = ["week", "region", "determinant"]
    out = shap_df[meta].copy()
    for sector, cols in groups.items():
        present = [c for c in cols if c in shap_df.columns]
        out[f"shap_{sector}"] = shap_df[present].sum(axis=1) if present else 0.0
    sector_cols = [c for c in out.columns if c.startswith("shap_")]
    pos = out[sector_cols].clip(lower=0)
    total = pos.sum(axis=1).replace(0, np.nan)
    for c in sector_cols:
        out[c.replace("shap_", "share_")] = (pos[c] / total).fillna(0.0)
    return out
