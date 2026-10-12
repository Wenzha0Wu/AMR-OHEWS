"""Alert layer: combine the model tiers and turn probabilities into alerts.

Each tier sees the problem differently -- engineered cross-sector features
(Tier 1), learned temporal filters (Tier 2), spatial spillover (Tier 3) -- and
they fail differently too, which is the usual argument for stacking them.

The stacker is a plain logistic regression fitted on the validation window, and
it falls back to a rank average when the validation window holds too few
positive weeks to fit anything. That fallback is not a formality: with ~1%
positives, a stacker trained on a handful of events will happily learn to ignore
two of its three inputs, and the rank average is the more honest default.

An alert is then more than a number. Public-health action needs to know which
sector raised the flag, how confident the system is, and how much warning it
claims to be giving, so :func:`generate_alerts` attaches all three.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..config import PRIMARY_HORIZON


class AlertEnsemble:
    """Stacks tier scores into one calibrated risk and emits alert records."""

    def __init__(self, horizon: int = PRIMARY_HORIZON, min_positives: int = 15):
        self.horizon = horizon
        self.min_positives = min_positives
        self.stacker: LogisticRegression | None = None
        self.calibrator: LogisticRegression | None = None
        self.members: list[str] = []
        self.mode = "rank_average"
        self.weights_: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    @staticmethod
    def _rank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(np.argsort(x))
        return order / max(len(x) - 1, 1)

    def _combine_raw(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.members].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0)
        if self.mode == "stacked" and self.stacker is not None:
            return self.stacker.predict_proba(x)[:, 1]
        return np.column_stack([self._rank(x[:, i]) for i in range(x.shape[1])]).mean(axis=1)

    # ------------------------------------------------------------------ #
    def fit(self, valid_scores: pd.DataFrame, members: list[str]) -> "AlertEnsemble":
        """Fit the stacker and the isotonic calibrator on the validation window."""
        self.members = list(members)
        y = valid_scores[f"label_h{self.horizon}"].to_numpy().astype(int)
        keep = ~valid_scores["post_event"].to_numpy().astype(bool)
        x = np.nan_to_num(valid_scores[self.members].to_numpy(dtype=float), nan=0.0)

        if y[keep].sum() >= self.min_positives and len(self.members) > 1:
            self.stacker = LogisticRegression(max_iter=1000, class_weight="balanced")
            self.stacker.fit(x[keep], y[keep])
            self.mode = "stacked"
            self.weights_ = dict(zip(self.members, self.stacker.coef_[0].round(4).tolist()))
        else:
            self.mode = "rank_average"
            self.weights_ = {m: round(1 / len(self.members), 4) for m in self.members}

        raw = self._combine_raw(valid_scores)
        if y[keep].sum() >= 5:
            # Platt scaling on the combined score. See the note in
            # :class:`~amrews.models.fusion.FusionModel`: an isotonic calibrator
            # here collapses the score into a few tied levels and makes a
            # false-alarm-budgeted threshold impossible to place.
            self.calibrator = LogisticRegression(max_iter=1000)
            self.calibrator.fit(raw[keep].reshape(-1, 1), y[keep])
        return self

    # ------------------------------------------------------------------ #
    def predict(self, scores: pd.DataFrame) -> np.ndarray:
        raw = self._combine_raw(scores)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]

    def score_frame(self, scores: pd.DataFrame) -> pd.DataFrame:
        out = scores[["week", "region", "determinant"]].copy()
        out["score_ensemble"] = self.predict(scores)
        return out


# --------------------------------------------------------------------------- #
# Alert records
# --------------------------------------------------------------------------- #
def generate_alerts(
    alarms: pd.DataFrame,
    scores: pd.DataFrame,
    score_col: str,
    sector_shares: pd.DataFrame | None = None,
    shap_df: pd.DataFrame | None = None,
    uncertainty_col: str | None = None,
    horizon: int = PRIMARY_HORIZON,
    top_k_drivers: int = 4,
) -> pd.DataFrame:
    """Attach risk, uncertainty, sector attribution and top drivers to each alarm."""
    if alarms.empty:
        return pd.DataFrame(columns=["week", "region", "determinant", "risk"])

    keys = ["week", "region", "determinant"]
    out = alarms.merge(scores, on=keys, how="left", suffixes=("", "_s"))
    out["risk"] = out[score_col].astype(float)

    if uncertainty_col and uncertainty_col in out.columns:
        out["risk_sd"] = out[uncertainty_col].astype(float)
        # Wilson-free, deliberately crude band: MC-dropout spread is a relative
        # confidence signal, not a coverage guarantee, and is labelled as such.
        out["risk_lower"] = (out["risk"] - out["risk_sd"]).clip(lower=0)
        out["risk_upper"] = (out["risk"] + out["risk_sd"]).clip(upper=1)

    if sector_shares is not None and not sector_shares.empty:
        share_cols = [c for c in sector_shares.columns if c.startswith("share_")]
        out = out.merge(sector_shares[keys + share_cols], on=keys, how="left")
        if share_cols:
            arr = out[share_cols].to_numpy(dtype=float)
            arr = np.nan_to_num(arr)
            out["lead_sector"] = [share_cols[i].replace("share_", "") for i in arr.argmax(axis=1)]
            out["lead_sector_share"] = arr.max(axis=1)

    if shap_df is not None and not shap_df.empty:
        feat_cols = [c for c in shap_df.columns if c not in keys]
        merged = out.merge(shap_df, on=keys, how="left", suffixes=("", "_shap"))
        drivers = []
        for _, row in merged.iterrows():
            vals = row[feat_cols].astype(float)
            if vals.isna().all():
                drivers.append("")
                continue
            top = vals.sort_values(ascending=False).head(top_k_drivers)
            drivers.append("; ".join(f"{k} (+{v:.3f})" for k, v in top.items() if v > 0))
        out["top_drivers"] = drivers

    out["expected_horizon_weeks"] = horizon
    out["severity"] = pd.cut(
        out["risk"], bins=[-0.01, 0.25, 0.5, 0.75, 1.01],
        labels=["watch", "elevated", "high", "critical"],
    )
    return out.sort_values(["week", "risk"], ascending=[True, False]).reset_index(drop=True)
