"""
model_configurable.py — Fix v3 (+ CellLUTModule reverted)
============================================================
Changes vs v2 (unchanged from before, all verified working):

FIX 1 — correction_scale 0.1 -> 0.2
  v2's 0.1 was too conservative. Cell delay R2 was 0.11-0.58 on most designs
  meaning the LUT output alone is not accurate enough and the correction
  wasn't contributing enough to fix it. 0.2 with the +/-0.5ns clamp means
  max correction per arc = 0.2 x 0.5 = 0.1ns -- still LUT-dominant but
  enough room to actually correct LUT errors.

FIX 2 — net_load clamped to [0.001, 0.5] pF
  Previously raw MLP output used as load. If it goes negative or very large,
  the LUT interpolation becomes nonsense (LUT tables are for physical loads
  0.001-0.5 pF in sky130hd). Clamping keeps the LUT query physically valid.

FIX 3 — slew factor range widened to [1.0, 3.0]
  Old: lo=1.0, rng=1.0 -> factor in [1.0, 2.0]
  New: lo=1.0, rng=2.0 -> factor in [1.0, 3.0]
  synth_ram has max slew 2.13ns starting from 0.15ns init.
  Over 10 levels: 0.15 x avg_factor^10 = 2.13 -> avg_factor ~ 1.31
  Factor range [1.0, 3.0] easily covers this. [1.0, 2.0] was tight.

FIX 4 — hidden_dim for cell correction raised to full H (not H//2)
  Old: H2 = max(H//2, 8) = 16. proj_i/proj_o project 32->16 then cellprop
  is 16+16+3=35 -> 2.  Very narrow bottleneck.
  New: H2 = H = 32. proj_i/proj_o project 32->32, cellprop is 32+32+3=67->2.
  More capacity for the correction that's doing the heavy lifting on AT.

FIX 5 — residual skip connection in net embedding (alpha learned per layer)
  Existing: h = relu(h_b + drv_new + alpha*h_init)
  Already has skip -- this is fine. Keep as is.

*** REVERTED (this version) — CellLUTModule table mixing ***
  A prior version replaced simple equal-averaging across the 8 LUT tables
  with a learned per-gate softmax mixing weight (a `table_weight` MLP).
  Per project memory, that version caused training collapse (train R2
  stuck at 0.74) and was flagged as "should not be reintroduced without a
  clear fix." No such fix has been designed or verified in this session,
  so CellLUTModule below reverts to simple equal-averaging (unweighted
  mean of the 8 per-table outputs) -- the documented "proven approach."
  See CellLUTModule's docstring for detail.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    net_layers: int = 3
    aggregation: str = "sum_max"
    lut_k: int = 7
    cell_correction: bool = True

    def describe(self):
        return (
            f"hidden_dim={self.hidden_dim}  net_layers={self.net_layers}  "
            f"aggregation={self.aggregation}  lut_k={self.lut_k}  "
            f"cell_correction={self.cell_correction}"
        )


def mlp(i, o, h):
    return nn.Sequential(
        nn.Linear(i, h), nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(h, h), nn.LeakyReLU(negative_slope=0.2),
        nn.Linear(h, o)
    )


NODE_FEATURES = 18   # 14 original + 4 pin cap (ff/ss rise/fall) matching paper
# Node features:
#  0-1:   x, y (norm)
#  2-4:   is_PI, is_IN, is_OUT
#  5-8:   dist_left/right/bottom/top
#  9:     is_endpoint
#  10:    fanout_norm
#  11:    net_cap_norm (SPEF)
#  12:    logic_depth_norm
#  13:    net_res_norm (SPEF)
#  14-15: pin_cap_rise_ff, pin_cap_fall_ff (Liberty ff) <- paper feature
#  16-17: pin_cap_rise_ss, pin_cap_fall_ss (Liberty ss) <- paper feature
NET_EDGE_FEATS = 4


class NetDelayHead(nn.Module):
    """
    Net delay prediction head using softplus(x - shift).
    shift=4: softplus(-4) = 0.018 ns at init -- near zero, never dead.
    Ceiling: raw clamped to max=6 -> softplus(6-4)=2.13 ns hard ceiling.
    This covers synth_ram max (0.97 ns) with headroom.
    Lower ceiling than before (was 8->4.02 ns) to prevent saturation.
    """
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        H = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, H), nn.LeakyReLU(0.2),
            nn.Linear(H, H),      nn.LeakyReLU(0.2),
            nn.Linear(H, 1)
        )
        self.shift = 4.0

    def forward(self, x):
        raw = self.net(x).squeeze(-1).clamp(max=6.0)
        return F.softplus(raw - self.shift)


class NetEmbedding(nn.Module):
    def __init__(self, cfg: ModelConfig, in_dim=NODE_FEATURES, edge_dim=NET_EDGE_FEATS):
        super().__init__()
        H = cfg.hidden_dim
        self.cfg = cfg
        self.H = H
        self.edge_dim = edge_dim

        self.inp = nn.Linear(in_dim, H)
        pair_dim = 2 * H + edge_dim

        self.bcast  = nn.ModuleList([mlp(pair_dim, H, H) for _ in range(cfg.net_layers)])
        self.msg    = nn.ModuleList([mlp(pair_dim, H, H) for _ in range(cfg.net_layers)])

        agg_width = 2 * H if cfg.aggregation == "sum_max" else H
        self.reduce = nn.ModuleList([mlp(H + agg_width, H, H) for _ in range(cfg.net_layers)])

        self.alpha = nn.Parameter(torch.full((cfg.net_layers,), 0.1))

        if cfg.aggregation == "attention":
            self.attn_w = nn.ModuleList([nn.Linear(2*H, H) for _ in range(cfg.net_layers)])
            self.attn_a = nn.ModuleList([nn.Linear(H, 1)   for _ in range(cfg.net_layers)])

        self.head_netdelay = NetDelayHead(pair_dim, H)
        # FIX 2: net_load will be clamped in forward to physical pF range
        self.head_load = mlp(H, 1, H)

    def _aggregate(self, msgs, net_src, N, layer_idx, h):
        H = self.H
        device = msgs.device

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
            scores = self.attn_a[layer_idx](
                torch.relu(self.attn_w[layer_idx](torch.cat([src_h, msgs], dim=-1)))
            ).squeeze(-1)
            scores_exp = torch.exp(scores - scores.max())
            denom = torch.zeros(N, device=device).index_add(
                0, net_src, scores_exp).clamp(min=1e-9)
            weights = scores_exp / denom[net_src]
            weighted = msgs * weights.unsqueeze(-1)
            return torch.zeros(N, H, device=device).index_add(0, net_src, weighted)

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
        net_delay = self.head_netdelay(pair)

        # FIX 2: clamp load to physical pF range for sky130hd
        # sky130hd pin caps range from ~0.001 pF (small gate) to ~0.5 pF (large buf)
        # raw MLP can go negative which makes LUT interpolation invalid
        net_load = self.head_load(h).squeeze(-1).clamp(min=0.001, max=0.5)

        return h, net_delay, net_load


class CellLUTModule(nn.Module):
    """
    Kronecker LUT interpolation over 8 tables per arc.

    Paper Table 3: 8 LUTs per cell arc --
      ff corner: cell_rise, cell_fall, rise_transition, fall_transition  (idx 0-3)
      ss corner: cell_rise, cell_fall, rise_transition, fall_transition  (idx 4-7)

    *** REVERTED per project memory ***
    A prior version replaced simple equal-averaging across the 8 tables
    with a learned per-gate softmax mixing weight (self.table_weight MLP).
    That version is documented to have caused training collapse (train R2
    stuck at 0.74) and was flagged: "should not be reintroduced without a
    clear fix." No such fix has been designed or verified, so this version
    reverts to simple equal-averaging (1/8 each table) -- the documented
    "proven approach": simple coef_slew/coef_load Kronecker product per
    table, combined by unweighted mean.

    For each of the 8 tables we learn separate (slew, load) Kronecker
    coefficients. Final cell delay / slew = unweighted mean of the 8
    per-table outputs. No learned table-mixing weight.
    """
    def __init__(self, cfg: ModelConfig, n_gates):
        super().__init__()
        self.k       = cfg.lut_k   # 7
        self.n_tables= 8           # matches paper
        H            = cfg.hidden_dim

        # one set of Kronecker coefs per table
        self.coef_slew = nn.ModuleList(
            [mlp(1 + n_gates, self.k, H) for _ in range(self.n_tables)])
        self.coef_load = nn.ModuleList(
            [mlp(1 + n_gates, self.k, H) for _ in range(self.n_tables)])

        # NOTE: no self.table_weight -- reverted to simple equal-averaging,
        # see class docstring.

    def forward(self, slew, load, gate_onehot, lut_delay, lut_slew, lut_idx=None):
        """
        lut_delay: [E, 8, 7, 7]
        lut_slew:  [E, 8, 7, 7]
        """
        feat = torch.cat([slew.unsqueeze(-1), gate_onehot], dim=-1)

        d_out = torch.zeros(slew.shape[0], device=slew.device)
        s_out = torch.zeros(slew.shape[0], device=slew.device)

        for t in range(self.n_tables):
            a = torch.softmax(self.coef_slew[t](feat), dim=-1)   # [E, k]
            b = torch.softmax(self.coef_load[t](
                torch.cat([load.unsqueeze(-1), gate_onehot], dim=-1)), dim=-1)
            coef = a.unsqueeze(-1) * b.unsqueeze(-2)              # [E, k, k]
            d_t  = (coef * lut_delay[:, t, :, :]).sum(dim=(-1,-2))
            s_t  = (coef * lut_slew[:, t, :, :]).sum(dim=(-1,-2))
            d_out = d_out + d_t / self.n_tables
            s_out = s_out + s_t / self.n_tables

        return d_out, s_out


class TimingGNN(nn.Module):
    def __init__(self, cfg: ModelConfig, n_gates,
                 in_dim=NODE_FEATURES, edge_dim=NET_EDGE_FEATS):
        super().__init__()
        self.cfg = cfg
        self.net_emb = NetEmbedding(cfg, in_dim=in_dim, edge_dim=edge_dim)
        self.lut = CellLUTModule(cfg, n_gates)

        if cfg.cell_correction:
            # FIX 4: full H width (was H//2 = 16, now 32) -- more correction capacity
            H2 = cfg.hidden_dim
            self.proj_i = nn.Linear(cfg.hidden_dim, H2)
            self.proj_o = nn.Linear(cfg.hidden_dim, H2)
            self.cellprop = mlp(H2 + H2 + 3, 2, cfg.hidden_dim)

        # FIX 1: correction_scale 0.1 -> 0.2
        # 0.1 was too small -- cell R2 stuck at 0.11-0.58 on most designs
        # 0.2 with +/-0.5ns clamp = max 0.1ns correction per arc (still LUT-dominant)
        self.correction_scale = 0.2

        # FIX 3: slew factor range [1.0, 3.0] (was [1.0, 2.0])
        # synth_ram needs factor ~1.31 avg over 10 levels -> [1.0, 3.0] covers it
        self.slew_net_mlp = nn.Sequential(
            nn.Linear(edge_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
            nn.Sigmoid()
        )
        self._slew_lo  = 1.0
        self._slew_rng = 2.0  # was 1.0

    def forward(self, g):
        h, net_delay, net_load = self.net_emb(
            g["x"], g["net_src"], g["net_dst"], g["edge_geo"]
        )

        N = g["x"].shape[0]
        device = g["x"].device

        AT   = torch.zeros(N, device=device)
        SLEW = torch.full((N,), 0.15, device=device)
        pred_cell_delay = torch.zeros(len(g["cell_src"]), device=device)

        for lvl in g["levels"]:
            ne = g["net_by_level"].get(lvl, [])
            if len(ne) > 0:
                # ne is a list of edge indices -- convert to tensor for GPU indexing
                ne = torch.tensor(ne, dtype=torch.long, device=device)
                s = g["net_src"][ne]
                d = g["net_dst"][ne]

                cand_at = AT[s] + net_delay[ne]

                slew_factor = (
                    self._slew_lo +
                    self._slew_rng * self.slew_net_mlp(g["edge_geo"][ne]).squeeze(-1)
                )
                cand_sl = SLEW[s] * slew_factor

                AT   = AT.index_reduce(0, d, cand_at, "amax", include_self=True)
                SLEW = SLEW.index_copy(0, d, cand_sl)

            ce = g["cell_by_level"].get(lvl, [])
            if len(ce) > 0:
                # ce is a list of edge indices -- convert to tensor for GPU indexing
                ce = torch.tensor(ce, dtype=torch.long, device=device)
                i_pins = g["cell_src"][ce]
                o_pins = g["cell_dst"][ce]
                load   = net_load[o_pins]

                cd, so = self.lut(
                    SLEW[i_pins], load,
                    g["gate_onehot"][ce],
                    g["lut_delay"][ce],
                    g["lut_slew"][ce],
                )

                if self.cfg.cell_correction:
                    pi_c = torch.relu(self.proj_i(h[i_pins]))
                    po_c = torch.relu(self.proj_o(h[o_pins]))
                    corr = self.cellprop(torch.cat([
                        pi_c, po_c,
                        cd.unsqueeze(-1), so.unsqueeze(-1), load.unsqueeze(-1)
                    ], dim=-1))
                    # FIX 1: scale=0.2, clamp +/-0.5ns -> max correction = 0.1ns per arc
                    cd = cd + self.correction_scale * corr[:, 0].clamp(-0.5, 0.5)
                    so = so + self.correction_scale * corr[:, 1].clamp(-0.5, 0.5)

                cd = cd.clamp(min=0.0)
                so = so.clamp(min=0.0, max=3.0)

                pred_cell_delay = pred_cell_delay.index_copy(0, ce, cd)
                cand_at = AT[i_pins] + cd
                AT   = AT.index_reduce(0, o_pins, cand_at, "amax", include_self=True)
                SLEW = SLEW.index_copy(0, o_pins, so)

        return {
            "AT": AT,
            "SLEW": SLEW,
            "net_delay": net_delay,
            "cell_delay": pred_cell_delay,
            "net_load": net_load,
        }