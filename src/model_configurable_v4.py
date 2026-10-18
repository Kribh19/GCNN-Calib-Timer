"""
model_configurable_v4.py — paper mechanics, our style
======================================================
Same explicit-tensor, no-DGL architecture as v3, upgraded with the four
mechanics discovered by reading the paper's released code:

CHANGE 1 — LOG-SPACE NET DELAY (kills the activation-floor problem for good)
  v3: NetDelayHead = softplus(mlp(x) - 4)  -> floor ~0.018 ns, ceiling games.
  v4: plain linear head predicts  log(0.0001 + delay_ns) + 7.6  (paper's exact
      transform). No positivity constraint needed: log space has no floor.
      For propagation we convert back:  d_ns = exp(pred - 7.6) - 0.0001.
  Train L_net in LOG space (see PATCH_train.md).

CHANGE 2 — TEACHER FORCING (paper's groundtruth mode)
  forward(g, groundtruth=True) propagates every level from the TRUE previous-
  level AT/slew where labels exist (falling back to predictions where they
  don't). Each level learns a clean one-hop mapping; errors cannot compound
  during training. Evaluate with groundtruth=False (full propagation).

CHANGE 3 — PI BOUNDARY CONDITIONS
  v3 initialized AT=0 and SLEW=0.15 everywhere. v4 initializes AT and SLEW at
  level-0 pins from labels when available (g["AT_true"], g["SLEW_true"] +
  g["slew_real_mask"]) — in BOTH training and inference, exactly like the
  paper injects true atslew at PI nodes. PI arrivals are design constraints,
  not predictions, so this is legitimate and must be stated in the paper.

CHANGE 4 — SLEW IN LOG SPACE (supervision side)
  The model still carries slew in ns internally (the LUT query needs ns), but
  exposes  out["SLEW_log"] = log(0.0001 + SLEW) + 3  so the loss can be taken
  in log space with weight ~1.0.

Kept from v3 (deliberately): Kronecker LUT module (equivalent to the paper's
LUT attention), sum_max aggregation, cell-correction MLP with clamp, load
clamp [0.001, 0.5] pF, slew factor range [1.0, 3.0], alpha residual skips.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_ND_EPS, LOG_ND_SHIFT = 1e-4, 7.6     # paper: log(0.0001 + d) + 7.6
LOG_SL_EPS, LOG_SL_SHIFT = 1e-4, 3.0     # paper: log(0.0001 + s) + 3


def nd_to_log(d_ns):   return torch.log(LOG_ND_EPS + d_ns) + LOG_ND_SHIFT
def nd_from_log(x):    return (torch.exp(x - LOG_ND_SHIFT) - LOG_ND_EPS).clamp(min=0.0)
def slew_to_log(s_ns): return torch.log(LOG_SL_EPS + s_ns) + LOG_SL_SHIFT


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    net_layers: int = 3
    aggregation: str = "sum_max"
    lut_k: int = 7
    cell_correction: bool = True

    def describe(self):
        return (f"hidden_dim={self.hidden_dim}  net_layers={self.net_layers}  "
                f"aggregation={self.aggregation}  lut_k={self.lut_k}  "
                f"cell_correction={self.cell_correction}")


def mlp(i, o, h):
    return nn.Sequential(
        nn.Linear(i, h), nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(h, h), nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(h, o))


NODE_FEATURES = 18
NET_EDGE_FEATS = 4


class NetEmbedding(nn.Module):
    """Same message passing as v3; only the delay head changed (CHANGE 1)."""
    def __init__(self, cfg: ModelConfig, in_dim=NODE_FEATURES, edge_dim=NET_EDGE_FEATS):
        super().__init__()
        H = cfg.hidden_dim
        self.cfg, self.H, self.edge_dim = cfg, H, edge_dim
        self.inp = nn.Linear(in_dim, H)
        pair_dim = 2 * H + edge_dim
        self.bcast = nn.ModuleList([mlp(pair_dim, H, H) for _ in range(cfg.net_layers)])
        self.msg   = nn.ModuleList([mlp(pair_dim, H, H) for _ in range(cfg.net_layers)])
        agg_width = 2 * H if cfg.aggregation == "sum_max" else H
        self.reduce = nn.ModuleList([mlp(H + agg_width, H, H) for _ in range(cfg.net_layers)])
        self.alpha = nn.Parameter(torch.full((cfg.net_layers,), 0.1))
        if cfg.aggregation == "attention":
            self.attn_w = nn.ModuleList([nn.Linear(2*H, H) for _ in range(cfg.net_layers)])
            self.attn_a = nn.ModuleList([nn.Linear(H, 1)   for _ in range(cfg.net_layers)])

        # CHANGE 1: plain linear head on LOG delay. No softplus, no shift, no clamp.
        self.head_netdelay_log = mlp(pair_dim, 1, H)
        self.head_load = mlp(H, 1, H)

    def _aggregate(self, msgs, net_src, N, layer_idx, h):
        H, device = self.H, msgs.device
        if self.cfg.aggregation == "mean":
            s = torch.zeros(N, H, device=device).index_add(0, net_src, msgs)
            c = torch.zeros(N, 1, device=device).index_add(
                0, net_src, torch.ones(len(net_src), 1, device=device)).clamp(min=1)
            return s / c
        if self.cfg.aggregation == "sum":
            return torch.zeros(N, H, device=device).index_add(0, net_src, msgs)
        if self.cfg.aggregation == "max":
            out = torch.full((N, H), -1e9, device=device)
            out = out.index_reduce(0, net_src, msgs, "amax", include_self=True)
            return torch.where(out < -1e8, torch.zeros_like(out), out)
        if self.cfg.aggregation == "sum_max":
            s_sum = torch.zeros(N, H, device=device).index_add(0, net_src, msgs)
            s_max = torch.full((N, H), -1e9, device=device)
            s_max = s_max.index_reduce(0, net_src, msgs, "amax", include_self=True)
            s_max = torch.where(s_max < -1e8, torch.zeros_like(s_max), s_max)
            return torch.cat([s_sum, s_max], dim=-1)
        if self.cfg.aggregation == "attention":
            src_h = h[net_src]
            scores = self.attn_a[layer_idx](torch.relu(
                self.attn_w[layer_idx](torch.cat([src_h, msgs], dim=-1)))).squeeze(-1)
            scores_exp = torch.exp(scores - scores.max())
            denom = torch.zeros(N, device=msgs.device).index_add(
                0, net_src, scores_exp).clamp(min=1e-9)
            weights = scores_exp / denom[net_src]
            return torch.zeros(N, H, device=msgs.device).index_add(
                0, net_src, msgs * weights.unsqueeze(-1))
        raise ValueError(f"unknown aggregation: {self.cfg.aggregation}")

    def forward(self, x, net_src, net_dst, edge_geo):
        h = torch.relu(self.inp(x))
        h_init = h.clone()
        N = x.shape[0]
        for l in range(self.cfg.net_layers):
            pair = torch.cat([h[net_src], h[net_dst], edge_geo], dim=-1)
            sink_new = self.bcast[l](pair)
            h_b = h.clone().index_copy(0, net_dst, sink_new)
            pair2 = torch.cat([h_b[net_src], h_b[net_dst], edge_geo], dim=-1)
            msgs = self.msg[l](pair2)
            agg = self._aggregate(msgs, net_src, N, l, h_b)
            drv_new = self.reduce[l](torch.cat([h_b, agg], dim=-1))
            h = torch.relu(h_b + drv_new + self.alpha[l] * h_init)

        pair = torch.cat([h[net_src], h[net_dst], edge_geo], dim=-1)
        net_delay_log = self.head_netdelay_log(pair).squeeze(-1)   # log space
        net_delay_ns = nd_from_log(net_delay_log)                  # for propagation
        net_load = self.head_load(h).squeeze(-1).clamp(min=0.001, max=0.5)
        return h, net_delay_log, net_delay_ns, net_load


class CellLUTModule(nn.Module):
    """Unchanged from v3 — equivalent role to the paper's LUT attention."""
    def __init__(self, cfg: ModelConfig, n_gates):
        super().__init__()
        self.k, self.n_tables = cfg.lut_k, 8
        H = cfg.hidden_dim
        self.coef_slew = nn.ModuleList(
            [mlp(1 + n_gates, self.k, H) for _ in range(self.n_tables)])
        self.coef_load = nn.ModuleList(
            [mlp(1 + n_gates, self.k, H) for _ in range(self.n_tables)])
        self.table_weight = mlp(1 + n_gates, self.n_tables, H)

    def forward(self, slew, load, gate_onehot, lut_delay, lut_slew, lut_idx=None):
        feat = torch.cat([slew.unsqueeze(-1), gate_onehot], dim=-1)
        tw = torch.softmax(self.table_weight(feat), dim=-1)
        d_out = torch.zeros(slew.shape[0], device=slew.device)
        s_out = torch.zeros(slew.shape[0], device=slew.device)
        for t in range(self.n_tables):
            a = torch.softmax(self.coef_slew[t](feat), dim=-1)
            b = torch.softmax(self.coef_load[t](
                torch.cat([load.unsqueeze(-1), gate_onehot], dim=-1)), dim=-1)
            coef = a.unsqueeze(-1) * b.unsqueeze(-2)
            d_out = d_out + tw[:, t] * (coef * lut_delay[:, t]).sum(dim=(-1, -2))
            s_out = s_out + tw[:, t] * (coef * lut_slew[:, t]).sum(dim=(-1, -2))
        return d_out, s_out


class TimingGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, n_gates,
                 in_dim=NODE_FEATURES, edge_dim=NET_EDGE_FEATS):
        super().__init__()
        self.cfg = cfg
        self.net_emb = NetEmbedding(cfg, in_dim=in_dim, edge_dim=edge_dim)
        self.lut = CellLUTModule(cfg, n_gates)
        if cfg.cell_correction:
            H2 = cfg.hidden_dim
            self.proj_i = nn.Linear(cfg.hidden_dim, H2)
            self.proj_o = nn.Linear(cfg.hidden_dim, H2)
            self.cellprop = mlp(H2 + H2 + 3, 2, cfg.hidden_dim)
        self.correction_scale = 0.2
        self.slew_net_mlp = nn.Sequential(
            nn.Linear(edge_dim, cfg.hidden_dim), nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1), nn.Sigmoid())
        self._slew_lo, self._slew_rng = 1.0, 2.0

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _teacher_src(pred_vec, true_vec, valid_mask, idx, on):
        """CHANGE 2: source values for propagation. In teacher-forcing mode,
        use TRUE values where a trusted label exists, else the prediction."""
        if not on or true_vec is None:
            return pred_vec[idx]
        if valid_mask is None:
            return true_vec[idx]
        take_true = valid_mask[idx]
        return torch.where(take_true, true_vec[idx], pred_vec[idx])

    def forward(self, g, groundtruth=False):
        h, net_delay_log, net_delay_ns, net_load = self.net_emb(
            g["x"], g["net_src"], g["net_dst"], g["edge_geo"])

        N, device = g["x"].shape[0], g["x"].device
        AT_true   = g.get("AT_true")            # may be None on unlabeled data
        SLEW_true = g.get("SLEW_true")
        slew_mask = g.get("slew_real_mask")

        AT   = torch.zeros(N, device=device)
        SLEW = torch.full((N,), 0.15, device=device)

        # CHANGE 3: PI boundary conditions from labels (train AND inference).
        lvl0 = g["levels"][0] if len(g["levels"]) else None
        pi = g.get("pi_nodes")
        if pi is None and lvl0 is not None:
            nb0 = g["net_by_level"].get(lvl0, [])
            ce0 = g["cell_by_level"].get(lvl0, [])
            srcs = []
            if len(nb0): srcs.append(g["net_src"][torch.tensor(nb0, dtype=torch.long, device=device)])
            if len(ce0): srcs.append(g["cell_src"][torch.tensor(ce0, dtype=torch.long, device=device)])
            pi = torch.unique(torch.cat(srcs)) if srcs else None
        if pi is not None:
            if AT_true is not None:
                AT = AT.index_copy(0, pi, AT_true[pi])
            if SLEW_true is not None:
                if slew_mask is not None:
                    ok = pi[slew_mask[pi]]
                    if len(ok): SLEW = SLEW.index_copy(0, ok, SLEW_true[ok])
                else:
                    SLEW = SLEW.index_copy(0, pi, SLEW_true[pi])

        pred_cell_delay = torch.zeros(len(g["cell_src"]), device=device)

        for lvl in g["levels"]:
            ne = g["net_by_level"].get(lvl, [])
            if len(ne) > 0:
                ne = torch.tensor(ne, dtype=torch.long, device=device)
                s, d = g["net_src"][ne], g["net_dst"][ne]
                src_at = self._teacher_src(AT, AT_true, None, s, groundtruth)
                src_sl = self._teacher_src(SLEW, SLEW_true, slew_mask, s, groundtruth)
                cand_at = src_at + net_delay_ns[ne]
                slew_factor = (self._slew_lo + self._slew_rng *
                               self.slew_net_mlp(g["edge_geo"][ne]).squeeze(-1))
                cand_sl = src_sl * slew_factor
                AT   = AT.index_reduce(0, d, cand_at, "amax", include_self=True)
                SLEW = SLEW.index_copy(0, d, cand_sl)

            ce = g["cell_by_level"].get(lvl, [])
            if len(ce) > 0:
                ce = torch.tensor(ce, dtype=torch.long, device=device)
                i_pins, o_pins = g["cell_src"][ce], g["cell_dst"][ce]
                load = net_load[o_pins]
                src_sl = self._teacher_src(SLEW, SLEW_true, slew_mask, i_pins, groundtruth)
                cd, so = self.lut(src_sl, load, g["gate_onehot"][ce],
                                  g["lut_delay"][ce], g["lut_slew"][ce])
                if self.cfg.cell_correction:
                    pi_c = torch.relu(self.proj_i(h[i_pins]))
                    po_c = torch.relu(self.proj_o(h[o_pins]))
                    corr = self.cellprop(torch.cat([
                        pi_c, po_c, cd.unsqueeze(-1), so.unsqueeze(-1),
                        load.unsqueeze(-1)], dim=-1))
                    cd = cd + self.correction_scale * corr[:, 0].clamp(-0.5, 0.5)
                    so = so + self.correction_scale * corr[:, 1].clamp(-0.5, 0.5)
                cd = cd.clamp(min=0.0)
                so = so.clamp(min=0.0, max=3.0)
                pred_cell_delay = pred_cell_delay.index_copy(0, ce, cd)
                src_at = self._teacher_src(AT, AT_true, None, i_pins, groundtruth)
                cand_at = src_at + cd
                AT   = AT.index_reduce(0, o_pins, cand_at, "amax", include_self=True)
                SLEW = SLEW.index_copy(0, o_pins, so)

        return {
            "AT": AT,
            "SLEW": SLEW,
            "SLEW_log": slew_to_log(SLEW),          # CHANGE 4: for log-space loss
            "net_delay": net_delay_ns,               # ns (back-compat)
            "net_delay_log": net_delay_log,          # CHANGE 1: for log-space loss
            "cell_delay": pred_cell_delay,
            "net_load": net_load,
        }
