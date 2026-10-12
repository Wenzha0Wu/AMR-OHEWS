"""Tier 2: dilated temporal convolutional network over the raw sector series.

The Tier-1 fusion model is handed engineered features -- rolling means, slopes,
z-scores -- that encode a human analyst's prior about what an emerging signal
looks like. Tier 2 removes that prior: it sees the delay-adjusted sector series
themselves and learns its own temporal filters over a 52-week window.

A dilated *causal* TCN is used rather than an LSTM or a Transformer:

* **Causal.** Padding is left-sided only, so the receptive field at week *t*
  cannot contain week *t+1*. With a recurrent or attention model it is easy to
  leak the future through a careless mask; here it is structural.
* **Dilated.** Dilations 1-2-4-8-16 with kernel size 3 reach a full year of
  history in five layers, which matters because AMR emergence is slow and
  seasonal.
* **Small.** A few hundred positive windows cannot support a Transformer. This
  network has ~60k parameters and trains in under a minute on CPU, which is what
  makes a 30-seed bootstrap feasible.

Dropout is kept active at inference to give **Monte-Carlo dropout** uncertainty:
each alert carries a predictive standard deviation, so the dashboard can show a
confident warning differently from a marginal one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#: Raw (delay-adjusted) channels the deep models consume.
CHANNELS = [
    "human_p", "animal_p", "food_p", "env_x",
    "amu_vet", "amu_human", "temperature", "rainfall",
    "neighbour_animal_z", "neighbour_environment_z",
    # Trade-weighted neighbour *levels*, unlagged. The z-score channels above
    # carry a fixed 4-week lag baked in; these let the encoder learn its own lag
    # from 52 weeks of neighbour history instead, which matters because the
    # transmission delay is itself variable.
    "neighbour_human_p", "neighbour_animal_p", "neighbour_food_p", "neighbour_env_x",
]

WINDOW = 52


def default_device() -> str:
    """Prefer a GPU when one is present *and* this build has kernels for it.

    The networks are small, but the evaluation fits two of them per simulated
    world across thirty worlds. On CPU that is hours; the same work on one GPU
    is minutes, which is the difference between a bootstrap over thirty seeds
    and a single anecdote.

    ``torch.cuda.is_available()`` is not sufficient: a wheel built for newer
    architectures reports an older card as available and then fails deep inside
    the first convolution with "unable to find an engine to execute this
    computation". Checking the card's compute capability against the build's
    architecture list turns that into a clean fallback.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        supported = [int(a.split("_")[1]) for a in torch.cuda.get_arch_list()
                     if a.startswith("sm_")]
        if supported and (major * 10 + minor) < min(supported):
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


# --------------------------------------------------------------------------- #
# Windowing
# --------------------------------------------------------------------------- #
def build_windows(
    df: pd.DataFrame,
    horizon: int,
    window: int = WINDOW,
    channels: list[str] | None = None,
    stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Slice the feature table into per-series sliding windows.

    Returns arrays ``x`` (N, C, window), ``y``, ``week``, ``region``,
    ``det_idx``, ``post_event`` plus the standardisation ``stats``. Statistics
    are computed on the training call and passed back in for validation/test so
    that no test-period scale information reaches the model.
    """
    channels = channels or CHANNELS
    df = df.sort_values(["region", "determinant", "week"])
    dets = sorted(df["determinant"].unique())
    det_index = {d: i for i, d in enumerate(dets)}

    xs, ys, weeks, regions, det_ids, post = [], [], [], [], [], []
    for (r, d), block in df.groupby(["region", "determinant"], observed=True, sort=False):
        arr = block[channels].to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        y = block[f"label_h{horizon}"].to_numpy().astype(np.float32)
        w = block["week"].to_numpy()
        pe = block["post_event"].to_numpy().astype(bool)
        n = len(block)
        if n <= window:
            continue
        # Stride tricks: a (n-window+1, window, C) view with no copy.
        strided = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)  # (n-w+1, C, w)
        xs.append(strided)
        ys.append(y[window - 1:])
        weeks.append(w[window - 1:])
        regions.append(np.full(len(strided), r, dtype=np.int64))
        det_ids.append(np.full(len(strided), det_index[d], dtype=np.int64))
        post.append(pe[window - 1:])

    x = np.concatenate(xs).astype(np.float32)
    if stats is None:
        mu = x.mean(axis=(0, 2), keepdims=True)
        sd = x.std(axis=(0, 2), keepdims=True) + 1e-6
        stats = (mu, sd)
    x = (x - stats[0]) / stats[1]

    return {
        "x": x,
        "y": np.concatenate(ys),
        "week": np.concatenate(weeks),
        "region": np.concatenate(regions),
        "det_idx": np.concatenate(det_ids),
        "post_event": np.concatenate(post),
        "stats": stats,
        "det_index": det_index,
    }


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
class CausalBlock(nn.Module):
    """Residual dilated causal convolution block."""

    def __init__(self, ch: int, dilation: int, kernel: int = 3, dropout: float = 0.2):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv1 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, kernel, dilation=dilation)
        self.norm = nn.GroupNorm(4, ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        h = self.drop(h)
        h = self.act(self.conv2(nn.functional.pad(h, (self.pad, 0))))
        return self.norm(x + self.drop(h))


class TCNEncoder(nn.Module):
    """Maps a (B, C, T) window to a (B, 2*hidden) embedding of its final week."""

    def __init__(self, in_ch: int, hidden: int = 48, dilations=(1, 2, 4, 8, 16),
                 dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([CausalBlock(hidden, d, dropout=dropout) for d in dilations])
        self.hidden = hidden
        self.out_dim = hidden * 2

    def forward_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """Per-timestep hidden states, shape (B, hidden, T).

        The convolution stack computes every timestep anyway, so exposing the
        sequence costs nothing and lets the graph tier read a neighbour's state
        as it stood *k weeks ago* -- which is what spatial transmission along a
        trade route actually looks like. Causal padding guarantees position
        ``t`` still depends only on inputs up to ``t``.
        """
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_sequence(x)
        # Concatenate the final timestep (the "now" state) with a mean over the
        # window (the "recent regime"); emergence needs both.
        return torch.cat([h[:, :, -1], h.mean(dim=2)], dim=1)


class TCNClassifier(nn.Module):
    """TCN encoder + region/determinant embeddings + MLP head."""

    def __init__(self, in_ch: int, n_regions: int, n_dets: int, hidden: int = 48,
                 emb: int = 8, dropout: float = 0.2):
        super().__init__()
        self.encoder = TCNEncoder(in_ch, hidden=hidden, dropout=dropout)
        self.region_emb = nn.Embedding(n_regions, emb)
        self.det_emb = nn.Embedding(n_dets, emb)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.out_dim + 2 * emb, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x, region, det):
        z = self.encoder(x)
        z = torch.cat([z, self.region_emb(region), self.det_emb(det)], dim=1)
        return self.head(z).squeeze(-1)


# --------------------------------------------------------------------------- #
# Wrapper with the shared fit / predict_proba interface
# --------------------------------------------------------------------------- #
class TemporalModel:
    """Trainable wrapper exposing the same API as :class:`FusionModel`."""

    def __init__(self, horizon: int = 12, n_regions: int = 20, n_dets: int = 6,
                 epochs: int = 25, batch_size: int = 512, lr: float = 3e-3,
                 dropout: float = 0.2, mc_samples: int = 10, seed: int = 0,
                 device: str | None = None, channels: list[str] | None = None,
                 window: int = WINDOW):
        # The default channel list is simulation-specific (antimicrobial use,
        # weather, neighbour anomalies). Real panels carry a different set, so
        # the caller supplies it.
        self.channels = list(channels) if channels else list(CHANNELS)
        # 52 periods is one year of weekly data. On a monthly panel that is more
        # than four years and no validation slice is long enough to yield a
        # single window, so the caller shortens it.
        self.window = window
        self.horizon = horizon
        self.n_regions, self.n_dets = n_regions, n_dets
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.dropout, self.mc_samples, self.seed = dropout, mc_samples, seed
        self.device = torch.device(device or default_device())
        self.net: TCNClassifier | None = None
        self.stats = None
        self.det_index: dict[str, int] = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, feature_cols=None, valid: pd.DataFrame | None = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        tr = build_windows(train, self.horizon, window=self.window, channels=self.channels)
        self.stats, self.det_index = tr["stats"], tr["det_index"]
        keep = ~tr["post_event"]
        x = torch.tensor(tr["x"][keep])
        y = torch.tensor(tr["y"][keep])
        reg = torch.tensor(tr["region"][keep])
        det = torch.tensor(tr["det_idx"][keep])

        pos = float(y.sum().item())
        pos_weight = torch.tensor([(len(y) - pos) / max(pos, 1.0)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.net = TCNClassifier(x.shape[1], self.n_regions, self.n_dets,
                                 dropout=self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        va = None
        if valid is not None and len(valid):
            va = build_windows(valid, self.horizon, window=self.window, channels=self.channels, stats=self.stats)

        best_state, best_score = None, -np.inf
        n = len(y)
        steps_done = 0
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
            sched.step()

            score = np.nan
            if va is not None and va["y"].sum() >= 3:
                from sklearn.metrics import average_precision_score

                p = self._forward_probs(va, stochastic=False)
                m = ~va["post_event"]
                if va["y"][m].sum() >= 3:
                    score = float(average_precision_score(va["y"][m], p[m]))
                    if score > best_score:
                        best_score = score
                        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            steps_done += (n + self.batch_size - 1) // self.batch_size
            self.history.append({"epoch": epoch, "loss": total / n,
                                 "valid_auc_pr": score, "steps": steps_done})

        # Keep the best validation checkpoint rather than the last epoch: with so
        # few positives the final epoch is frequently not the best one.
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ #
    def _forward_probs(self, win: dict, stochastic: bool) -> np.ndarray:
        assert self.net is not None
        self.net.train(stochastic)          # dropout active iff stochastic
        outs = []
        x = torch.tensor(win["x"])
        reg = torch.tensor(win["region"])
        det = torch.tensor(win["det_idx"])
        with torch.no_grad():
            for i in range(0, len(x), 4096):
                logits = self.net(x[i: i + 4096].to(self.device),
                                  reg[i: i + 4096].to(self.device),
                                  det[i: i + 4096].to(self.device))
                outs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(outs)

    # ------------------------------------------------------------------ #
    def predict_frame(self, df: pd.DataFrame, with_uncertainty: bool = True) -> pd.DataFrame:
        """Score a feature table, returning weeks that have a full 52-week history."""
        win = build_windows(df, self.horizon, window=self.window, channels=self.channels, stats=self.stats)
        mean = self._forward_probs(win, stochastic=False)
        out = pd.DataFrame({
            "week": win["week"],
            "region": win["region"],
            "determinant": [None] * len(mean),
            "score_temporal": mean,
        })
        inv = {v: k for k, v in self.det_index.items()}
        out["determinant"] = [inv[i] for i in win["det_idx"]]

        if with_uncertainty and self.mc_samples > 1:
            draws = np.stack([self._forward_probs(win, stochastic=True)
                              for _ in range(self.mc_samples)])
            out["score_temporal_sd"] = draws.std(axis=0)
        return out

    def predict_proba(self, df: pd.DataFrame, feature_cols=None) -> np.ndarray:
        """Align window-level predictions back onto every row of ``df``.

        The first 51 weeks of each series have no full window; they are scored 0,
        which is the honest answer -- the model has no basis for an alert there.
        """
        preds = self.predict_frame(df, with_uncertainty=False)
        key = df[["week", "region", "determinant"]].merge(
            preds, on=["week", "region", "determinant"], how="left"
        )
        return key["score_temporal"].fillna(0.0).to_numpy()
