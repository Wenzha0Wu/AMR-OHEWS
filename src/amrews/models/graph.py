"""Tier 3: spatio-temporal graph network over the region connectivity graph.

Tiers 1 and 2 treat each region as an island. Resistance does not behave that
way: it travels with livestock consignments, along shared watersheds and through
patient referral between hospitals. When a neighbouring region's animal reservoir
lit up two months ago, that is evidence about *this* region today -- evidence no
per-region model can express.

The architecture is deliberately the Tier-2 encoder plus one idea. Each region's
52-week window is encoded by the same causal TCN; the resulting embeddings for
all regions in a given (determinant, week) are then stacked into a graph signal
and propagated over a normalised adjacency built from trade volume, watershed
sharing and referral flow. Because the encoder is shared with Tier 2, any
difference in the results is attributable to the graph term rather than to a
bigger network -- which is what makes the ablation in the report interpretable.

The graph layers are written out longhand with a dense adjacency matrix multiply
rather than pulled in from a graph-learning library. At twenty nodes the dense
form is faster, adds no dependency, and -- the deciding reason for a report that
has to explain its own methods -- fits in six readable lines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .temporal import WINDOW, TCNEncoder, build_windows, default_device


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
#: Land borders between US states. Static, well known, and hard-coding it avoids
#: a geospatial dependency for fifty nodes. Used when the graph tier runs on the
#: real US state panel rather than the simulated region ring.
US_STATE_BORDERS = {
    "AL": "FL GA MS TN", "AZ": "CA CO NM NV UT", "AR": "LA MS MO OK TN TX",
    "CA": "AZ NV OR", "CO": "AZ KS NE NM OK UT WY", "CT": "MA NY RI",
    "DE": "MD NJ PA", "DC": "MD VA", "FL": "AL GA", "GA": "AL FL NC SC TN",
    "ID": "MT NV OR UT WA WY", "IL": "IN IA KY MO WI", "IN": "IL KY MI OH",
    "IA": "IL MN MO NE SD WI", "KS": "CO MO NE OK", "KY": "IL IN MO OH TN VA WV",
    "LA": "AR MS TX", "ME": "NH", "MD": "DE DC PA VA WV", "MA": "CT NH NY RI VT",
    "MI": "IN OH WI", "MN": "IA ND SD WI", "MS": "AL AR LA TN",
    "MO": "AR IL IA KS KY NE OK TN", "MT": "ID ND SD WY", "NE": "CO IA KS MO SD WY",
    "NV": "AZ CA ID OR UT", "NH": "ME MA VT", "NJ": "DE NY PA",
    "NM": "AZ CO OK TX UT", "NY": "CT MA NJ PA VT", "NC": "GA SC TN VA",
    "ND": "MN MT SD", "OH": "IN KY MI PA WV", "OK": "AR CO KS MO NM TX",
    "OR": "CA ID NV WA", "PA": "DE MD NJ NY OH WV", "RI": "CT MA",
    "SC": "GA NC", "SD": "IA MN MT NE ND WY", "TN": "AL AR GA KY MS MO NC VA",
    "TX": "AR LA NM OK", "UT": "AZ CO ID NM NV WY", "VT": "MA NH NY",
    "VA": "DC KY MD NC TN WV", "WA": "ID OR", "WV": "KY MD OH PA VA",
    "WI": "IA IL MI MN", "WY": "CO ID MT NE SD UT",
}


def us_state_relations(states: list[str]) -> dict[str, np.ndarray]:
    """Real relations between US states: shared borders and a distance kernel.

    The simulated ring in :func:`build_relations` is a stand-in. When the graph
    tier is run on the real US panel it gets real geography instead, so a
    failure there cannot be blamed on synthetic adjacency.
    """
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)

    border = np.zeros((n, n), dtype=np.float32)
    for a, neighbours in US_STATE_BORDERS.items():
        if a not in idx:
            continue
        for b in neighbours.split():
            if b in idx:
                border[idx[a], idx[b]] = 1.0

    # Second-order adjacency: neighbours-of-neighbours, as a coarse proxy for
    # the multi-step livestock and food movement the border matrix misses.
    two_step = ((border @ border) > 0).astype(np.float32)
    np.fill_diagonal(two_step, 0.0)

    out = {}
    for name, a in (("border", border), ("two_step", two_step),
                    ("uniform", np.ones((n, n), dtype=np.float32) / max(n, 1))):
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 0.0)
        a = a + np.eye(n, dtype=np.float32)
        d = np.power(np.clip(a.sum(axis=1), 1e-9, None), -0.5)
        out[name] = (a * d[:, None] * d[None, :]).astype(np.float32)
    return out


def build_relations(trade: np.ndarray, watershed_decay: float = 1.0,
                    referral_weight: float = 0.3) -> dict[str, np.ndarray]:
    """Three separately normalised region adjacencies, one per real coupling.

    * **trade** -- livestock movement, the dominant pathway for resistant
      commensals moving between regions;
    * **watershed** -- ring adjacency, standing in for shared surface water and
      wastewater catchments;
    * **referral** -- a weak all-to-all term for patient movement.

    These used to be summed into one fixed matrix before normalisation, which
    silently decided their relative importance in advance. Keeping them separate
    lets the model learn the mix, and the learned weights are then reportable:
    they say which pathway the data actually supports.
    """
    r = trade.shape[0]
    idx = np.arange(r)
    ring = np.minimum(np.abs(idx[:, None] - idx[None, :]), r - np.abs(idx[:, None] - idx[None, :]))
    watershed = np.exp(-watershed_decay * ring)
    referral = referral_weight * np.ones((r, r))

    out = {}
    for name, a in (("trade", trade.copy()), ("watershed", watershed),
                    ("referral", referral)):
        a = 0.5 * (a + a.T)
        np.fill_diagonal(a, 0.0)
        a = a + np.eye(r)                  # self-loops: keep a region's own signal
        d = np.power(np.clip(a.sum(axis=1), 1e-9, None), -0.5)
        out[name] = (a * d[:, None] * d[None, :]).astype(np.float32)   # D^-1/2 A D^-1/2
    return out


def build_adjacency(trade: np.ndarray, **kw) -> np.ndarray:
    """Backwards-compatible single matrix: the mean of the three relations."""
    rel = build_relations(trade, **kw)
    return np.mean(list(rel.values()), axis=0).astype(np.float32)


class GraphConv(nn.Module):
    """One GraphSAGE-style layer: separate self and neighbourhood transforms."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # h: (B, R, F)  adj: (R, R)
        neigh = torch.einsum("rs,bsf->brf", adj, h)
        return self.drop(self.act(self.norm(self.self_lin(h) + self.neigh_lin(neigh))))


class STGNN(nn.Module):
    """Shared TCN encoder -> multi-lag graph propagation -> gated per-node logit.

    Three deliberate design choices, each addressing a way the first version of
    this tier was set up to fail.

    **Multi-lag propagation.** Transmission along a trade route takes months, so
    mixing only contemporaneous neighbour embeddings asks the graph to see an
    effect before its cause has shown up. Neighbour states are read at several
    lags from the encoder's sequence output and combined with learned weights.

    **Multi-relational adjacency.** Trade, watershed and referral are propagated
    under a learned softmax mix rather than a hand-picked sum.

    **Gated residual.** The graph branch is scaled by ``sigmoid(gate)``, starting
    small so the network begins close to Tier 2 and has to earn its way to using
    the graph. That makes "the graph tier is worse" nearly impossible by
    construction, and turns the learned gate into a readable measure of how much
    the spatial pathway was worth.

    The gate starts at ``sigmoid(-1) = 0.27`` rather than the ``sigmoid(-3) =
    0.05`` first tried. At 0.05 the graph branch is scaled so far down that the
    gradient reaching it -- and reaching the gate itself -- is negligible, and
    the gate never moved off its initial value even in worlds with strong
    spatial signal. The mixing scalars also get their own optimizer group (see
    :meth:`GraphModel.fit`): a handful of scalars need a far larger step size
    than weight matrices do.
    """

    def __init__(self, in_ch: int, n_regions: int, n_dets: int, hidden: int = 48,
                 graph_dim: int = 48, emb: int = 8, dropout: float = 0.2,
                 n_relations: int = 3, lags: tuple[int, ...] = (0, 4, 8, 12),
                 gate_init: float = -1.0):
        super().__init__()
        self.encoder = TCNEncoder(in_ch, hidden=hidden, dropout=dropout)
        self.region_emb = nn.Embedding(n_regions, emb)
        self.det_emb = nn.Embedding(n_dets, emb)
        self.lags = lags
        self.local_dim = self.encoder.out_dim + 2 * emb
        self.proj = nn.Linear(self.local_dim, graph_dim)
        self.lag_proj = nn.Linear(hidden + 2 * emb, graph_dim)
        self.g1 = GraphConv(graph_dim, graph_dim, dropout)
        self.g2 = GraphConv(graph_dim, graph_dim, dropout)
        self.lag_logits = nn.Parameter(torch.zeros(len(lags)))
        self.relation_logits = nn.Parameter(torch.zeros(n_relations))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        # The head sees the **full** local representation, not the projected one.
        # It previously took [proj(local), gated_graph], which squeezed the local
        # path through a graph_dim bottleneck that Tier 2's head does not have.
        # The gated residual was supposed to guarantee this tier could always
        # fall back to Tier 2; with a lossier local path it could not, and on the
        # Japan ILI benchmark the graph tier scored AUC-PR 0.36 against Tier 2's
        # 0.87 using the identical encoder. Concatenating the unprojected local
        # representation restores the guarantee.
        self.head = nn.Sequential(
            nn.Linear(self.local_dim + graph_dim, 64), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def mixed_adjacency(self, relations: torch.Tensor) -> torch.Tensor:
        """Learned softmax mix of the stacked (n_relations, R, R) adjacencies."""
        w = torch.softmax(self.relation_logits, dim=0)
        return torch.einsum("k,krs->rs", w, relations)

    def forward(self, x, region, det, relations):
        # x: (B, R, C, T)
        b, r, c, t = x.shape
        seq = self.encoder.forward_sequence(x.reshape(b * r, c, t))    # (B*R, hid, T)
        reg_e = self.region_emb(region.reshape(-1))
        det_e = self.det_emb(det.reshape(-1))

        local = torch.cat([seq[:, :, -1], seq.mean(dim=2), reg_e, det_e], dim=1)
        h0 = self.proj(local).reshape(b, r, -1)            # graph-space copy
        local_full = local.reshape(b, r, -1)               # what Tier 2's head sees

        adj = self.mixed_adjacency(relations)
        lag_w = torch.softmax(self.lag_logits, dim=0)
        mixed = 0.0
        for i, lag in enumerate(self.lags):
            step = seq[:, :, max(0, t - 1 - lag)]
            h_lag = self.lag_proj(torch.cat([step, reg_e, det_e], dim=1)).reshape(b, r, -1)
            mixed = mixed + lag_w[i] * self.g1(h_lag, adj)
        h = self.g2(mixed, adj)

        gated = torch.sigmoid(self.gate) * h
        return self.head(torch.cat([local_full, gated], dim=-1)).squeeze(-1)   # (B, R)


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #
class GraphModel:
    """Spatio-temporal graph model with the shared fit / predict interface."""

    def __init__(self, trade: np.ndarray, horizon: int = 12, n_dets: int = 6,
                 epochs: int = 25, batch_items: int = 64, lr: float = 3e-3,
                 dropout: float = 0.2, seed: int = 0, device: str | None = None,
                 target_steps: int | None = None, channels: list[str] | None = None,
                 window: int = WINDOW):
        from .temporal import CHANNELS
        self.channels = list(channels) if channels else list(CHANNELS)
        self.window = window
        #: Minimum reporting nodes for a (determinant, week) item to be usable.
        self.min_present = 2
        self.relation_names = list(build_relations(trade))
        self.relations_np = np.stack([build_relations(trade)[k] for k in self.relation_names])
        self.adj_np = self.relations_np.mean(axis=0)
        self.n_regions = trade.shape[0]
        #: When set, epochs are chosen so this many optimizer steps are taken.
        #: The graph tier used to run ~610 steps against the temporal tier's
        #: ~1,525 simply because its batches are whole region-stacks, so part of
        #: the gap between them was undertraining rather than architecture.
        self.target_steps = target_steps
        self.horizon, self.n_dets = horizon, n_dets
        self.epochs, self.batch_items, self.lr = epochs, batch_items, lr
        self.dropout, self.seed = dropout, seed
        self.device = torch.device(device or default_device())
        self.net: STGNN | None = None
        self.stats = None
        self.det_index: dict[str, int] = {}
        self.history: list[dict] = []

    # ------------------------------------------------------------------ #
    def _group(self, win: dict) -> tuple:
        """Regroup per-series windows into (determinant, week) items of all regions.

        Returns ``item_x`` (I, R, C, T), ``item_y`` (I, R), the determinant index
        and week of each item, a post-event mask and a presence mask.

        Real spatial panels are never complete -- no month has every US state
        reporting every determinant -- so absent nodes are zero-filled and
        flagged rather than the whole item being dropped. They are excluded from
        the loss and from scoring, but the graph still propagates over the full
        node set, which is what lets a present node borrow from a present
        neighbour. Requiring complete items instead discarded the entire real
        panel.
        """
        key = win["det_idx"].astype(np.int64) * 1000000 + win["week"].astype(np.int64)
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        bounds = np.flatnonzero(np.diff(key_s)) + 1
        groups = np.split(order, bounds)

        n_ch, n_t = win["x"].shape[1], win["x"].shape[2]
        xs, ys, dets, weeks, posts, presents = [], [], [], [], [], []
        for g in groups:
            if len(g) < self.min_present:
                continue
            regions = win["region"][g]
            valid = (regions >= 0) & (regions < self.n_regions)
            g, regions = g[valid], regions[valid]
            if len(g) < self.min_present:
                continue
            x_item = np.zeros((self.n_regions, n_ch, n_t), dtype=np.float32)
            y_item = np.zeros(self.n_regions, dtype=np.float32)
            post_item = np.ones(self.n_regions, dtype=bool)   # absent => masked out
            present = np.zeros(self.n_regions, dtype=bool)
            x_item[regions] = win["x"][g]
            y_item[regions] = win["y"][g]
            post_item[regions] = win["post_event"][g]
            present[regions] = True
            xs.append(x_item); ys.append(y_item); posts.append(post_item)
            presents.append(present)
            dets.append(win["det_idx"][g][0]); weeks.append(win["week"][g][0])

        if not xs:
            raise ValueError("no (determinant, week) items met the minimum node count")
        return (np.stack(xs), np.stack(ys), np.asarray(dets), np.asarray(weeks),
                np.stack(posts), np.stack(presents))

    # ------------------------------------------------------------------ #
    def fit(self, train: pd.DataFrame, feature_cols=None, valid: pd.DataFrame | None = None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        win = build_windows(train, self.horizon, window=self.window, channels=self.channels)
        self.stats, self.det_index = win["stats"], win["det_index"]
        x, y, det, week, post, present = self._group(win)

        relations = torch.tensor(self.relations_np, device=self.device)
        region_idx = torch.arange(self.n_regions).repeat(self.batch_items, 1)

        pos = float(y[~post].sum())
        total = float((~post).sum())
        pos_weight = torch.tensor([(total - pos) / max(pos, 1.0)], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        self.net = STGNN(x.shape[2], self.n_regions, self.n_dets, dropout=self.dropout).to(self.device)
        # The gate and the relation/lag mixing logits are a handful of scalars
        # that have to travel a long way; at the shared learning rate they do not
        # move at all. Separate group, much larger step, no weight decay (decay
        # on a mixing logit just pulls it back towards a uniform mix).
        mixing = {"gate", "relation_logits", "lag_logits"}
        mix_params = [v for k, v in self.net.named_parameters() if k in mixing]
        rest = [v for k, v in self.net.named_parameters() if k not in mixing]
        opt = torch.optim.AdamW(
            [{"params": rest, "lr": self.lr, "weight_decay": 1e-4},
             {"params": mix_params, "lr": self.lr * 20, "weight_decay": 0.0}])

        xt = torch.tensor(x)
        yt = torch.tensor(y)
        dt = torch.tensor(det)
        # Post-emergence nodes are masked out of the loss rather than deleted,
        # because the graph layer still needs the full region stack as input.
        mask = torch.tensor(~post, dtype=torch.float32)

        va_pack = None
        if valid is not None and len(valid):
            vw = build_windows(valid, self.horizon, window=self.window, channels=self.channels, stats=self.stats)
            va_pack = self._group(vw)

        best_state, best_score = None, -np.inf
        n_items = len(xt)
        steps_per_epoch = (n_items + self.batch_items - 1) // self.batch_items
        epochs = self.epochs
        if self.target_steps:
            epochs = max(1, int(round(self.target_steps / max(steps_per_epoch, 1))))
        self.epochs_run, self.steps_run = epochs, epochs * steps_per_epoch
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        steps_done = 0
        for epoch in range(epochs):
            self.net.train()
            perm = torch.randperm(n_items)
            running = 0.0
            for i in range(0, n_items, self.batch_items):
                idx = perm[i: i + self.batch_items]
                b = len(idx)
                opt.zero_grad()
                logits = self.net(
                    xt[idx].to(self.device),
                    region_idx[:b].to(self.device),
                    dt[idx][:, None].repeat(1, self.n_regions).to(self.device),
                    relations,
                )
                raw = loss_fn(logits, yt[idx].to(self.device))
                m = mask[idx].to(self.device)
                loss = (raw * m).sum() / m.sum().clamp(min=1.0)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                running += float(loss.item()) * b
            sched.step()

            score = np.nan
            if va_pack is not None:
                from sklearn.metrics import average_precision_score

                vx, vy, vdet, vweek, vpost, vpresent = va_pack
                p = self._forward(vx, vdet)
                keep = (~vpost.reshape(-1)) & vpresent.reshape(-1)
                yv = vy.reshape(-1)[keep]
                if yv.sum() >= 3:
                    score = float(average_precision_score(yv, p.reshape(-1)[keep]))
                    if score > best_score:
                        best_score = score
                        best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            steps_done += steps_per_epoch
            self.history.append({"epoch": epoch, "loss": running / n_items,
                                 "valid_auc_pr": score, "steps": steps_done})

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    # ------------------------------------------------------------------ #
    def _forward(self, x: np.ndarray, det: np.ndarray, stochastic: bool = False) -> np.ndarray:
        assert self.net is not None
        self.net.train(stochastic)
        relations = torch.tensor(self.relations_np, device=self.device)
        outs = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_items):
                xb = torch.tensor(x[i: i + self.batch_items]).to(self.device)
                b = len(xb)
                reg = torch.arange(self.n_regions).repeat(b, 1).to(self.device)
                dd = torch.tensor(det[i: i + self.batch_items])[:, None].repeat(1, self.n_regions).to(self.device)
                outs.append(torch.sigmoid(self.net(xb, reg, dd, relations)).cpu().numpy())
        return np.concatenate(outs)

    # ------------------------------------------------------------------ #
    def predict_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        win = build_windows(df, self.horizon, window=self.window, channels=self.channels, stats=self.stats)
        x, y, det, week, post, present = self._group(win)
        probs = self._forward(x, det)
        inv = {v: k for k, v in self.det_index.items()}
        frame = pd.DataFrame({
            "week": np.repeat(week, self.n_regions),
            "region": np.tile(np.arange(self.n_regions), len(week)),
            "determinant": np.repeat([inv[d] for d in det], self.n_regions),
            "score_graph": probs.reshape(-1),
            "present": present.reshape(-1),
        })
        # Zero-filled absent nodes carry no information; emitting a score for
        # them would invent surveillance that never happened.
        return frame[frame["present"]].drop(columns="present").reset_index(drop=True)

    def predict_proba(self, df: pd.DataFrame, feature_cols=None) -> np.ndarray:
        preds = self.predict_frame(df)
        key = df[["week", "region", "determinant"]].merge(
            preds, on=["week", "region", "determinant"], how="left"
        )
        return key["score_graph"].fillna(0.0).to_numpy()

    # ------------------------------------------------------------------ #
    def diagnostics(self) -> dict:
        """Learned gate, relation mix and lag mix -- the interpretable outputs.

        The gate in particular answers the question this tier exists to settle:
        how much did the spatial pathway actually contribute? A gate that stays
        near its initial value means the model declined to use the graph.
        """
        if self.net is None:
            return {}
        with torch.no_grad():
            return {
                "gate": float(torch.sigmoid(self.net.gate).item()),
                "relation_weights": dict(zip(
                    self.relation_names,
                    torch.softmax(self.net.relation_logits, 0).cpu().numpy().round(4).tolist())),
                "lag_weights": dict(zip(
                    [str(l) for l in self.net.lags],
                    torch.softmax(self.net.lag_logits, 0).cpu().numpy().round(4).tolist())),
                "epochs_run": getattr(self, "epochs_run", None),
                "steps_run": getattr(self, "steps_run", None),
            }
