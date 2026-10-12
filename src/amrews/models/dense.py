"""A neural network over engineered features.

The missing cell in the representation-versus-model comparison. Tier 1 is
gradient-boosted trees on engineered features; Tier 2 is a convolutional network
on raw series. Those differ in two ways at once, so Tier 1's advantage cannot be
attributed to the model class or to the feature representation without holding
one of them fixed.

This model holds the representation fixed at the engineered features and swaps
the model class for a neural network, using the same training protocol as
:class:`~amrews.models.temporal.TemporalModel` -- same optimiser, same class
weighting, same validation-AUC-PR checkpoint selection, same step budget -- so
that the comparison is about architecture rather than about training recipe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..data.features import clean_matrix
from .temporal import default_device


class DenseClassifier(nn.Module):
    """Multilayer perceptron with region and determinant embeddings."""

    def __init__(self, n_features: int, n_regions: int, n_dets: int,
                 hidden: int = 128, emb: int = 8, dropout: float = 0.2):
        super().__init__()
        self.region_emb = nn.Embedding(n_regions, emb)
        self.det_emb = nn.Embedding(n_dets, emb)
        self.net = nn.Sequential(
            nn.Linear(n_features + 2 * emb, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x, region, det):
        z = torch.cat([x, self.region_emb(region), self.det_emb(det)], dim=1)
        return self.net(z).squeeze(-1)


class DenseModel:
    """Neural network on a fixed feature matrix, with the shared fit/predict API."""

    def __init__(self, horizon: int = 12, n_regions: int = 20, n_dets: int = 6,
                 epochs: int = 25, batch_size: int = 512, lr: float = 3e-3,
                 dropout: float = 0.2, seed: int = 0, device: str | None = None):
        self.horizon, self.epochs = horizon, epochs
        self.n_regions, self.n_dets = n_regions, n_dets
        self.batch_size, self.lr, self.dropout, self.seed = batch_size, lr, dropout, seed
        self.device = torch.device(device or default_device())
        self.net: DenseClassifier | None = None
        self.feature_names: list[str] = []
        self.stats = None
        self.det_index: dict[str, int] = {}
        self.history: list[dict] = []

    def _prepare(self, df: pd.DataFrame, cols: list[str]):
        x = clean_matrix(df, cols)
        if self.stats is None:
            mu, sd = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True) + 1e-6
            self.stats = (mu, sd)
        x = (x - self.stats[0]) / self.stats[1]
        region = df["region"].to_numpy()
        det = np.array([self.det_index.get(d, 0) for d in df["determinant"]])
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(region, dtype=torch.long),
                torch.tensor(det, dtype=torch.long))

    def fit(self, train: pd.DataFrame, feature_cols: list[str],
            valid: pd.DataFrame | None = None) -> "DenseModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.feature_names = list(feature_cols)
        self.det_index = {d: i for i, d in enumerate(sorted(train["determinant"].unique()))}

        tr = train[~train["post_event"]]
        x, reg, det = self._prepare(tr, feature_cols)
        y = torch.tensor(tr[f"label_h{self.horizon}"].to_numpy(), dtype=torch.float32)

        pos = float(y.sum().item())
        pos_weight = torch.tensor([(len(y) - pos) / max(pos, 1.0)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.net = DenseClassifier(x.shape[1], self.n_regions, self.n_dets,
                                   dropout=self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        va = None
        if valid is not None and len(valid):
            vb = valid[~valid["post_event"]]
            if len(vb):
                va = (vb, *self._prepare(vb, feature_cols))

        best_state, best_score, steps = None, -np.inf, 0
        n = len(y)
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i: i + self.batch_size]
                opt.zero_grad()
                logits = self.net(x[idx].to(self.device), reg[idx].to(self.device),
                                  det[idx].to(self.device))
                loss = loss_fn(logits, y[idx].to(self.device))
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                total += float(loss.item()) * len(idx)
                steps += 1
            sched.step()

            score = np.nan
            if va is not None:
                from sklearn.metrics import average_precision_score

                vb, vx, vreg, vdet = va
                yv = vb[f"label_h{self.horizon}"].to_numpy()
                if yv.sum() >= 3:
                    p = self._forward(vx, vreg, vdet)
                    score = float(average_precision_score(yv, p))
                    if score > best_score:
                        best_score = score
                        best_state = {k: v.detach().clone()
                                      for k, v in self.net.state_dict().items()}
            self.history.append({"epoch": epoch, "loss": total / n,
                                 "valid_auc_pr": score, "steps": steps})

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _forward(self, x, reg, det) -> np.ndarray:
        assert self.net is not None
        self.net.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                logits = self.net(x[i: i + 4096].to(self.device),
                                  reg[i: i + 4096].to(self.device),
                                  det[i: i + 4096].to(self.device))
                outs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outs)

    def predict_proba(self, df: pd.DataFrame, feature_cols: list[str] | None = None):
        return self._forward(*self._prepare(df, feature_cols or self.feature_names))

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["week", "region", "determinant"]].copy()
        out["score_dense"] = self.predict_proba(df)
        return out
